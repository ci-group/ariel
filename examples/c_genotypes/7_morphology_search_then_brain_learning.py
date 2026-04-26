"""
Two-stage experiment:
1) Morphology-only TreeGenome evolution (descriptor fitness)
2) Brain learning on the best found morphology (multithreaded CMA-ES)

Goal: test whether morphologies discovered by morphology-only search can
actually learn locomotion when given a dedicated controller optimization phase.
"""

# Standard library
import argparse
import contextlib
import copy
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Third-party
import mujoco
import mujoco.viewer
import nevergrad as ng
import numpy as np
import torch
from rich.console import Console
from rich.progress import track
from rich.traceback import install
from torch import nn

# ARIEL imports
from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec import EA, EAOperation, EASettings, Individual, Population
from ariel.ec.genotypes.tree.operators import (
    _prune_invalid_edges,
    crossover_subtree,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
    validate_tree_depth,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from ariel.simulation.controllers.controller import Controller, Tracker
from ariel.simulation.controllers.utils.data_get import get_state_from_data
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.morphological_descriptor import MorphologicalMeasures
from ariel.utils.runners import thread_safe_runner

install()
console = Console()


# ============================================================================ #
#                                   CONFIG                                     #
# ============================================================================ #

parser = argparse.ArgumentParser(
    description="Morphology search first, then multithreaded brain learning on best morphology",
)

# Morphology search
parser.add_argument("--morph-budget", type=int, default=20, help="Morphology generations")
parser.add_argument("--morph-pop", type=int, default=30, help="Morphology population size")
parser.add_argument("--max-modules", type=int, default=20, help="Maximum modules in tree")
parser.add_argument("--max-depth", type=int, default=12, help="Maximum tree depth")

# Brain learning
parser.add_argument("--learn-budget", type=int, default=32, help="CMA iterations")
parser.add_argument("--learn-pop", type=int, default=32, help="CMA population")
parser.add_argument(
    "--learn-workers",
    type=int,
    default=max(1, os.cpu_count() or 1),
    help="Thread workers for candidate evaluation",
)
parser.add_argument("--dur", type=float, default=10.0, help="Active control duration")
parser.add_argument("--eval-delay", type=float, default=1.0, help="Warm-up before scoring")

# Runtime
parser.add_argument(
    "--visualize",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Visualize best learned controller",
)
parser.add_argument("--viewer-duration", type=float, default=10.0, help="Viewer replay duration")

args = parser.parse_args()

MORPH_BUDGET = args.morph_budget
MORPH_POP = args.morph_pop
MAX_MODULES = args.max_modules
MAX_DEPTH = args.max_depth

LEARN_BUDGET = args.learn_budget
LEARN_POP = args.learn_pop
LEARN_WORKERS = max(1, min(args.learn_workers, LEARN_POP))
DURATION = args.dur
EVAL_DELAY = max(0.0, args.eval_delay)

SEED = 42
RNG = np.random.default_rng(SEED)
torch.manual_seed(SEED)

SCRIPT_NAME = Path(__file__).stem
DATA = Path.cwd() / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)

SPAWN_POSITION = (-0.8, 0.0, 0.1)
TARGET_POSITIONS = [np.array([2.0, 0.0, 0.1], dtype=np.float32)]


class Network(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, hidden_size: int
    ) -> None:
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        self.hidden_activation = nn.ELU()
        self.output_activation = nn.Tanh()

        self.input = input_size

        # Disable gradients for all parameters
        for param in self.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def forward(self, model, data):
        robot_state = get_state_from_data(data)
        phase_inputs = np.array(
            [
                2.0 * np.sin(data.time * 2.0 * np.pi),
                2.0 * np.cos(data.time * 2.0 * np.pi),
            ],
            dtype=np.float32,
        )
        state = torch.tensor(
            np.concatenate([robot_state, phase_inputs]).astype(np.float32),
            dtype=torch.float32,
        )

        x = self.hidden_activation(self.fc1(state))
        x = self.hidden_activation(self.fc2(x))
        x = self.output_activation(self.fc4(x)) * (torch.pi / 2)

        return x.detach().numpy()

@torch.no_grad()
def fill_parameters(net: nn.Module, vector: torch.Tensor):
    """Fill the parameters of a torch module (net) from a vector.

    No gradient information is kept.

    The vector's length must be exactly the same with the number
    of parameters of the PyTorch module.

    Args:
        net: The torch module whose parameter values will be filled.
        vector: A 1-D torch tensor which stores the parameter values.
    """
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(vector[address : address + n], device=d.device)
        address += n

    if address != len(vector):
        raise IndexError("The parameter vector is larger than expected")



# ============================================================================ #
#                            MORPHOLOGY SEARCH                                 #
# ============================================================================ #


def morphology_fitness(genome: TreeGenome) -> float:
    """Descriptor-only morphology fitness (lower is better)."""
    try:
        graph = genome.to_networkx()
        if graph.number_of_nodes() == 0:
            return float("inf")
        m = MorphologicalMeasures(graph)
        # Keep same spirit as morphology-only script: maximize these descriptors,
        # then negate for minimization framework.
        score = (
            m.symmetry * 0.20
            + m.joints * 0.20
            + m.branching * 0.20
            + m.length_of_limbs * 0.20
            + m.module_diversity * 0.20
        )
        return -float(score)
    except Exception:
        return float("inf")


class MorphologySearch:
    def __init__(self) -> None:
        self.config = EASettings(
            is_maximisation=False,
            num_steps=MORPH_BUDGET,
            target_population_size=MORPH_POP,
            output_folder=DATA,
            db_file_name=f"morph_search_{int(time.time())}.db",
            db_handling="delete",
        )

    def _spawnable_joint_count(self, genome: TreeGenome) -> int:
        try:
            graph = genome.to_networkx()
            if graph.number_of_nodes() == 0:
                return 0
            spec = construct_mjspec_from_graph(graph).spec
            model = spec.compile()
            return model.nu
        except Exception:
            return 0

    def mutate_morphology(self, genome: TreeGenome) -> TreeGenome:
        new = copy.deepcopy(genome)
        mutation_type = RNG.choice(["point", "subtree", "shrink", "hoist"], p=[0.45, 0.35, 0.1, 0.1])

        if mutation_type == "point":
            mutate_replace_node(new)
        elif mutation_type == "subtree":
            mutate_subtree_replacement(new, max_modules=MAX_MODULES)
        elif mutation_type == "shrink":
            mutate_shrink(new)
        else:
            mutate_hoist(new)

        if RNG.random() < 0.25:
            noncore = [nid for nid in new.nodes if nid != IDX_OF_CORE]
            if noncore:
                nid = random.choice(noncore)
                mtype = ModuleType[new.nodes[nid]["type"]]
                rots = [r.name for r in ALLOWED_ROTATIONS[mtype]]
                if rots:
                    new.nodes[nid]["rotation"] = random.choice(rots)

        _prune_invalid_edges(new)
        with contextlib.suppress(ValueError):
            validate_genome_dict(new.to_dict())
        return new

    def create_individual(self) -> Individual:
        while True:
            genome = random_tree(MAX_MODULES)
            if self._spawnable_joint_count(genome) > 0 and validate_tree_depth(genome, MAX_DEPTH):
                break

        ind = Individual()
        ind.genotype = {"morph": genome.to_dict()}
        ind.tags = {"ps": False, "valid": True}
        return ind

    def parent_selection(self, population: Population) -> Population:
        population = population.sort(sort="min", attribute="fitness_")
        cutoff = len(population) // 2
        for i, ind in enumerate(population):
            ind.tags["ps"] = i < cutoff
        return population

    def reproduction(self, population: Population) -> Population:
        parents = [ind for ind in population if ind.tags.get("ps", False)]
        if not parents:
            parents = list(population)

        offspring: list[Individual] = []
        target_pool = self.config.target_population_size * 2

        while len(population) + len(offspring) < target_pool:
            use_sexual = len(parents) >= 2 and RNG.random() < 0.6
            if use_sexual:
                p1, p2 = random.sample(parents, 2)
                t1 = TreeGenome.from_dict(p1.genotype["morph"])
                t2 = TreeGenome.from_dict(p2.genotype["morph"])
                c1, c2 = crossover_subtree(t1, t2)
                child_morph = c1 if RNG.random() < 0.5 else c2
            else:
                parent = random.choice(parents)
                child_morph = TreeGenome.from_dict(parent.genotype["morph"])

            child_morph = self.mutate_morphology(child_morph)

            attempts = 0
            while attempts < 15:
                has_joints = self._spawnable_joint_count(child_morph) > 0
                valid_depth = validate_tree_depth(child_morph, MAX_DEPTH)
                if has_joints and valid_depth:
                    break
                child_morph = self.mutate_morphology(child_morph)
                attempts += 1

            child = Individual()
            child.genotype = {"morph": child_morph.to_dict()}
            child.tags = {"ps": False, "valid": True}
            child.requires_eval = True
            offspring.append(child)

        population.extend(offspring)
        return population

    def evaluate(self, population: Population) -> Population:
        to_eval = [
            ind
            for ind in population
            if ind.alive and ind.tags.get("valid") and ind.requires_eval
        ]
        if not to_eval:
            return population

        for ind in track(to_eval, description="Morphology search..."):
            genome = TreeGenome.from_dict(ind.genotype["morph"])
            ind.fitness = morphology_fitness(genome)
            ind.requires_eval = False

        return population

    def survivor_selection(self, population: Population) -> Population:
        population = population.sort(sort="min", attribute="fitness_")
        survivors = population[: self.config.target_population_size]
        for ind in population:
            if ind not in survivors:
                ind.alive = False
        return population

    def evolve(self) -> Individual | None:
        console.log("Initializing morphology population...")
        population = Population([self.create_individual() for _ in range(MORPH_POP)])
        population = self.evaluate(population)

        ops = [
            EAOperation(self.parent_selection),
            EAOperation(self.reproduction),
            EAOperation(self.evaluate),
            EAOperation(self.survivor_selection),
        ]

        ea = EA(
            population,
            operations=ops,
            num_steps=MORPH_BUDGET,
            db_file_path=self.config.db_file_path,
            db_handling=self.config.db_handling,
            quiet=self.config.quiet,
        )
        ea.run()
        return ea.get_solution("best", only_alive=False)


# ============================================================================ #
#                              BRAIN LEARNING                                  #
# ============================================================================ #


class BrainLearner:
    def __init__(self, genome_dict: dict) -> None:
        self.genome_dict = genome_dict

    def _spawn_with_fallback(self) -> tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]:
        def _build(correct_collision_with_floor: bool) -> tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]:
            genome = TreeGenome.from_dict(self.genome_dict)
            graph = genome.to_networkx()
            if graph.number_of_nodes() == 0:
                raise ValueError("Empty morphology")
            spec = construct_mjspec_from_graph(graph).spec
            world = SimpleFlatWorld()
            world.spawn(
                spec,
                position=SPAWN_POSITION,
                correct_collision_with_floor=correct_collision_with_floor,
            )
            model = world.spec.compile()
            data = mujoco.MjData(model)
            return world, model, data

        try:
            return _build(True)
        except Exception:
            return _build(False)

    def learn(self) -> tuple[float, list[float]]:
        try:
            warmup_world, warmup_model, warmup_data = self._spawn_with_fallback()
        except Exception:
            return float("inf"), []

        if warmup_model.nu == 0:
            return float("inf"), []

        input_size = len(get_state_from_data(warmup_data)) + 2
        dummy_net = Network(input_size=input_size, output_size=warmup_model.nu, hidden_size=32)
        num_params = sum(p.numel() for p in dummy_net.parameters())
        del warmup_world, warmup_model, warmup_data, dummy_net

        param = ng.p.Array(shape=(num_params,))
        cma_config = ng.optimizers.ParametrizedCMA(popsize=LEARN_POP)
        learner = cma_config(
            parametrization=param,
            budget=LEARN_BUDGET * LEARN_POP,
            num_workers=LEARN_POP,
        )

        thread_local = threading.local()

        def _build_context() -> dict:
            world, model, data = self._spawn_with_fallback()
            net = Network(input_size=input_size, output_size=model.nu, hidden_size=32)
            tracker = Tracker(name_to_bind="core", observable_attributes=["xpos"], quiet=True)
            tracker.setup(world.spec, data)
            controller = Controller(controller_callback_function=net.forward, tracker=tracker)
            return {
                "world": world,
                "model": model,
                "data": data,
                "net": net,
                "controller": controller,
            }

        def _get_context() -> dict:
            if not hasattr(thread_local, "ctx"):
                thread_local.ctx = _build_context()
            return thread_local.ctx

        def _evaluate_candidate(vec: np.ndarray) -> float:
            ctx = _get_context()
            model = ctx["model"]
            data = ctx["data"]
            net = ctx["net"]
            controller = ctx["controller"]

            fill_parameters(net, vec)

            total_fit = 0.0
            for target in TARGET_POSITIONS:
                mujoco.mj_resetData(model, data)
                if EVAL_DELAY > 0.0:
                    data.ctrl[:] = 0.0
                    delay_steps = int(EVAL_DELAY / model.opt.timestep)
                    if delay_steps > 0:
                        mujoco.mj_step(model, data, nstep=delay_steps)

                dist_start = float(np.linalg.norm(target - data.qpos[0:3]))
                thread_safe_runner(model, data, controller, duration=DURATION)
                dist_end = float(np.linalg.norm(target - data.qpos[0:3]))
                progress = dist_start - dist_end
                total_fit += -progress

            return total_fit / len(TARGET_POSITIONS)

        best_fit = float("inf")
        best_vec: list[float] = []

        with ThreadPoolExecutor(max_workers=LEARN_WORKERS) as executor:
            for _ in range(LEARN_BUDGET):
                candidates = [learner.ask() for _ in range(LEARN_POP)]
                future_to_idx = {
                    executor.submit(_evaluate_candidate, cand.value): idx
                    for idx, cand in enumerate(candidates)
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        fit = float(future.result())
                    except Exception:
                        fit = float("inf")

                    learner.tell(candidates[idx], fit)

                    if fit < best_fit:
                        best_fit = fit
                        best_vec = candidates[idx].value.tolist()

        return best_fit, best_vec

    def replay(self, brain_vec: list[float], duration: float) -> None:
        mujoco.set_mjcb_control(None)
        try:
            try:
                world, model, data = self._spawn_with_fallback()
            except Exception as exc:
                console.log(f"[red]Could not spawn morphology for replay: {exc}[/red]")
                return

            if model.nu == 0:
                console.log("[red]No actuators on morphology.[/red]")
                return

            net = Network(input_size=len(get_state_from_data(data)) + 2, output_size=model.nu, hidden_size=32)
            if brain_vec:
                fill_parameters(net, brain_vec)

            tracker = Tracker(name_to_bind="core", observable_attributes=["xpos"], quiet=True)
            tracker.setup(world.spec, data)
            controller = Controller(controller_callback_function=net.forward, tracker=tracker)

            mujoco.mj_resetData(model, data)

            if sys.platform == "darwin" or not hasattr(mujoco.viewer, "launch_passive"):
                console.log("[yellow]Using active MuJoCo viewer fallback (passive viewer unavailable).[/yellow]")
                console.log("[yellow]Close the viewer window to continue.[/yellow]")
                mujoco.set_mjcb_control(controller.set_control)
                mujoco.viewer.launch(model=model, data=data)
                return

            with mujoco.viewer.launch_passive(model, data) as v:
                sim_start = time.time()
                while v.is_running() and (time.time() - sim_start) < duration:
                    step_start = time.time()
                    controller.set_control(model, data)
                    mujoco.mj_step(model, data)
                    v.sync()
                    remaining = model.opt.timestep - (time.time() - step_start)
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            mujoco.set_mjcb_control(None)


# ============================================================================ #
#                                    MAIN                                      #
# ============================================================================ #


def main() -> None:
    console.rule("[bold magenta]Stage 1: Morphology Search[/bold magenta]")
    console.log(
        f"MorphPop={MORPH_POP}, MorphGens={MORPH_BUDGET}, MaxModules={MAX_MODULES}, MaxDepth={MAX_DEPTH}",
    )

    morph_search = MorphologySearch()
    t0 = time.time()
    best_morph = morph_search.evolve()
    t1 = time.time()

    if best_morph is None:
        console.log("[red]No morphology found.[/red]")
        return

    best_genome = TreeGenome.from_dict(best_morph.genotype["morph"])
    morph_path = DATA / "best_morphology.json"
    morph_path.write_text(json.dumps(best_genome.to_dict(), indent=2), encoding="utf-8")

    console.log(f"Best morphology fitness: {best_morph.fitness:.4f}")
    console.log(f"Morphology search elapsed: {t1 - t0:.2f}s")
    console.log(f"Saved best morphology to: {morph_path}")

    console.rule("[bold cyan]Stage 2: Brain Learning on Best Morphology[/bold cyan]")
    console.log(
        f"LearnBudget={LEARN_BUDGET}, LearnPop={LEARN_POP}, LearnWorkers={LEARN_WORKERS}, "
        f"Dur={DURATION}s, EvalDelay={EVAL_DELAY}s",
    )

    learner = BrainLearner(best_genome.to_dict())
    best_fit, best_vec = learner.learn()
    t2 = time.time()

    if not np.isfinite(best_fit):
        console.log("[red]Learning failed or morphology not controllable.[/red]")
        return

    brain_path = DATA / "best_brain.npy"
    np.save(brain_path, np.asarray(best_vec, dtype=np.float32))

    console.rule("[bold green]Final Result[/bold green]")
    console.log(f"Best learned locomotion fitness: {best_fit:.4f}")
    console.log(f"Brain learning elapsed: {t2 - t1:.2f}s")
    console.log(f"Total elapsed: {t2 - t0:.2f}s")
    console.log(f"Saved best brain to: {brain_path}")

    if args.visualize:
        learner.replay(best_vec, duration=args.viewer_duration)


if __name__ == "__main__":
    main()
