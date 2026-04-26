"""
Bootstrap a population using the two-stage pipeline, then evolve that population.

Stage A (bootstrap each individual):
1) Morphology-only TreeGenome search
2) Brain learning on that best morphology

Stage B (population evolution):
- Parent selection
- Reproduction (crossover + mutation on morphology)
- Re-evaluate offspring by brain learning
- Survivor selection
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
    description="Bootstrap a population with morph+brain pipeline, then evolve it",
)

# Bootstrap stage
parser.add_argument("--bootstrap-pop", type=int, default=8, help="Initial population size")
parser.add_argument(
    "--bootstrap-morph-budget",
    type=int,
    default=8,
    help="Morph-only generations for each bootstrapped individual",
)
parser.add_argument(
    "--bootstrap-morph-pop",
    type=int,
    default=12,
    help="Morph-only population size for each bootstrapped individual",
)
parser.add_argument(
    "--bootstrap-max-attempts",
    type=int,
    default=2,
    help="Retries per individual if bootstrap fails",
)

# Evolution stage
parser.add_argument("--evo-budget", type=int, default=8, help="Outer evolution generations")
parser.add_argument(
    "--outer-mutate-point",
    type=float,
    default=0.20,
    help="Outer evolution probability of point mutation",
)
parser.add_argument(
    "--outer-mutate-subtree",
    type=float,
    default=0.20,
    help="Outer evolution probability of subtree replacement mutation",
)
parser.add_argument(
    "--outer-mutate-shrink",
    type=float,
    default=0.30,
    help="Outer evolution probability of shrink mutation",
)
parser.add_argument(
    "--outer-mutate-hoist",
    type=float,
    default=0.30,
    help="Outer evolution probability of hoist mutation",
)

# Shared morphology constraints
parser.add_argument("--max-modules", type=int, default=20, help="Maximum modules in tree")
parser.add_argument("--max-depth", type=int, default=12, help="Maximum tree depth")

# Brain learning (used in bootstrap + outer evaluation)
parser.add_argument("--learn-budget", type=int, default=8, help="CMA iterations")
parser.add_argument("--learn-pop", type=int, default=16, help="CMA population")
parser.add_argument(
    "--learn-workers",
    type=int,
    default=max(1, os.cpu_count() or 1),
    help="Thread workers for candidate evaluation",
)
parser.add_argument("--dur", type=float, default=5.0, help="Active control duration")
parser.add_argument("--eval-delay", type=float, default=1.0, help="Warm-up before scoring")

# Runtime
parser.add_argument(
    "--visualize",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Visualize final best controller",
)
parser.add_argument("--viewer-duration", type=float, default=10.0, help="Viewer replay duration")

args = parser.parse_args()

BOOTSTRAP_POP = args.bootstrap_pop
BOOTSTRAP_MORPH_BUDGET = args.bootstrap_morph_budget
BOOTSTRAP_MORPH_POP = args.bootstrap_morph_pop
BOOTSTRAP_MAX_ATTEMPTS = max(1, args.bootstrap_max_attempts)
EVO_BUDGET = args.evo_budget
MAX_MODULES = args.max_modules
MAX_DEPTH = args.max_depth

_outer_mut_raw = np.array(
    [
        args.outer_mutate_point,
        args.outer_mutate_subtree,
        args.outer_mutate_shrink,
        args.outer_mutate_hoist,
    ],
    dtype=np.float64,
)
if np.any(_outer_mut_raw < 0):
    raise ValueError("Outer mutation probabilities must be >= 0")
if float(np.sum(_outer_mut_raw)) <= 0.0:
    raise ValueError("At least one outer mutation probability must be > 0")
OUTER_MUTATION_PROBS = (_outer_mut_raw / float(np.sum(_outer_mut_raw))).tolist()

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


# ============================================================================ #
#                                  NETWORK                                     #
# ============================================================================ #


class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.hidden_activation = nn.ELU()
        self.output_activation = nn.Tanh()

        for param in self.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def forward(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
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
        x = self.output_activation(self.fc_out(x)) * (torch.pi / 2)
        return x.detach().numpy()


@torch.no_grad()
def fill_parameters(net: nn.Module, vector: np.ndarray | list[float]) -> None:
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(vector[address : address + n], device=d.device)
        address += n


# ============================================================================ #
#                            MORPHOLOGY FITNESS                                #
# ============================================================================ #


def morphology_fitness(genome: TreeGenome) -> float:
    try:
        graph = genome.to_networkx()
        if graph.number_of_nodes() == 0:
            return float("inf")
        m = MorphologicalMeasures(graph)
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


# ============================================================================ #
#                      STAGE A: MORPH SEARCH (PER IND)                         #
# ============================================================================ #


class MorphologySearch:
    def __init__(self, num_steps: int, pop_size: int, run_tag: str) -> None:
        self.num_steps = num_steps
        self.pop_size = pop_size
        self.config = EASettings(
            is_maximisation=False,
            num_steps=num_steps,
            target_population_size=pop_size,
            output_folder=DATA,
            db_file_name=f"morph_bootstrap_{run_tag}_{int(time.time())}_{random.randint(0, 99999)}.db",
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
        mutation_type = RNG.choice(
            ["point", "subtree", "shrink", "hoist"],
            p=OUTER_MUTATION_PROBS,
        )

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

        for ind in to_eval:
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
        population = Population([self.create_individual() for _ in range(self.pop_size)])
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
            num_steps=self.num_steps,
            db_file_path=self.config.db_file_path,
            db_handling=self.config.db_handling,
            quiet=True,
        )
        ea.run()
        return ea.get_solution("best", only_alive=False)


# ============================================================================ #
#                     STAGE A/B: BRAIN LEARNING EVALUATOR                      #
# ============================================================================ #


class BrainEvaluator:
    def __init__(self) -> None:
        pass

    def _spawn_with_fallback(self, genome_dict: dict) -> tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]:
        def _build(correct_collision_with_floor: bool) -> tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]:
            genome = TreeGenome.from_dict(genome_dict)
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

    def learn(self, genome_dict: dict) -> tuple[float, list[float]]:
        try:
            warmup_world, warmup_model, warmup_data = self._spawn_with_fallback(genome_dict)
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
            world, model, data = self._spawn_with_fallback(genome_dict)
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

    def replay(self, genome_dict: dict, brain_vec: list[float], duration: float) -> None:
        mujoco.set_mjcb_control(None)
        try:
            try:
                world, model, data = self._spawn_with_fallback(genome_dict)
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
#                 STAGE B: POPULATION EVOLUTION AFTER BOOTSTRAP                #
# ============================================================================ #


class BootstrappedJointEvolution:
    def __init__(self) -> None:
        self.brain_eval = BrainEvaluator()
        self.config = EASettings(
            is_maximisation=False,
            num_steps=EVO_BUDGET,
            target_population_size=BOOTSTRAP_POP,
            output_folder=DATA,
            db_file_name=f"joint_evolution_{int(time.time())}.db",
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

    def _bootstrap_one(self, index: int) -> Individual:
        for attempt in range(1, BOOTSTRAP_MAX_ATTEMPTS + 1):
            run_tag = f"ind{index}_try{attempt}"
            morph_search = MorphologySearch(
                num_steps=BOOTSTRAP_MORPH_BUDGET,
                pop_size=BOOTSTRAP_MORPH_POP,
                run_tag=run_tag,
            )
            best_morph = morph_search.evolve()
            if best_morph is None:
                continue

            morph_dict = TreeGenome.from_dict(best_morph.genotype["morph"]).to_dict()
            fit, brain_vec = self.brain_eval.learn(morph_dict)
            if not np.isfinite(fit):
                continue

            ind = Individual()
            ind.genotype = {"morph": morph_dict}
            ind.tags = {"ps": False, "valid": True, "last_brain": brain_vec}
            ind.fitness = fit
            ind.requires_eval = False
            return ind

        fallback = Individual()
        fallback.genotype = {"morph": random_tree(MAX_MODULES).to_dict()}
        fallback.tags = {"ps": False, "valid": False, "last_brain": []}
        fallback.fitness = float("inf")
        fallback.requires_eval = False
        return fallback

    def bootstrap_population(self) -> Population:
        inds: list[Individual] = []
        for i in track(range(BOOTSTRAP_POP), description="Bootstrapping population (pipeline)..."):
            inds.append(self._bootstrap_one(i))
        return Population(inds)

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
            child.tags = {"ps": False, "valid": True, "last_brain": []}
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

        for ind in track(to_eval, description="Re-learning brains for offspring..."):
            fit, brain_vec = self.brain_eval.learn(ind.genotype["morph"])
            ind.fitness = fit
            ind.tags["last_brain"] = brain_vec
            ind.requires_eval = False

        return population

    def survivor_selection(self, population: Population) -> Population:
        population = population.sort(sort="min", attribute="fitness_")
        survivors = population[: self.config.target_population_size]
        for ind in population:
            if ind not in survivors:
                ind.alive = False

        finite_fits = [ind.fitness_ for ind in survivors if ind.alive and np.isfinite(ind.fitness_)]
        if finite_fits:
            console.log(
                f"Outer gen best={min(finite_fits):.4f}, mean={float(np.mean(finite_fits)):.4f}, n={len(finite_fits)}",
            )
        return population

    def run(self) -> Individual | None:
        console.rule("[bold magenta]Bootstrap Stage[/bold magenta]")
        console.log(
            f"BootstrapPop={BOOTSTRAP_POP}, MorphPop={BOOTSTRAP_MORPH_POP}, "
            f"MorphGens={BOOTSTRAP_MORPH_BUDGET}, LearnPop={LEARN_POP}, LearnBudget={LEARN_BUDGET}",
        )

        population = self.bootstrap_population()
        population = population.sort(sort="min", attribute="fitness_")

        ops = [
            EAOperation(self.parent_selection),
            EAOperation(self.reproduction),
            EAOperation(self.evaluate),
            EAOperation(self.survivor_selection),
        ]

        console.rule("[bold cyan]Outer Evolution Stage[/bold cyan]")
        console.log(
            "Outer mutation probs: "
            f"point={OUTER_MUTATION_PROBS[0]:.2f}, "
            f"subtree={OUTER_MUTATION_PROBS[1]:.2f}, "
            f"shrink={OUTER_MUTATION_PROBS[2]:.2f}, "
            f"hoist={OUTER_MUTATION_PROBS[3]:.2f}",
        )
        ea = EA(
            population,
            operations=ops,
            num_steps=EVO_BUDGET,
            db_file_path=self.config.db_file_path,
            db_handling=self.config.db_handling,
            quiet=False,
        )
        ea.run()
        return ea.get_solution("best", only_alive=False)


# ============================================================================ #
#                                    MAIN                                      #
# ============================================================================ #


def main() -> None:
    t0 = time.time()
    evo = BootstrappedJointEvolution()
    best = evo.run()
    t1 = time.time()

    if best is None:
        console.log("[red]No valid solution found.[/red]")
        return

    best_genome = TreeGenome.from_dict(best.genotype["morph"])
    best_brain = best.tags.get("last_brain", [])

    timestamp = int(time.time())
    morph_path = DATA / f"best_morphology_{timestamp}.json"
    brain_path = DATA / f"best_brain_{timestamp}.npy"

    morph_path.write_text(json.dumps(best_genome.to_dict(), indent=2), encoding="utf-8")
    np.save(brain_path, np.asarray(best_brain, dtype=np.float32))

    console.rule("[bold green]Final Result[/bold green]")
    console.log(f"Best fitness: {best.fitness:.4f}")
    console.log(f"Elapsed: {t1 - t0:.2f}s")
    console.log(f"Saved best morphology to: {morph_path}")
    console.log(f"Saved best brain to: {brain_path}")

    if args.visualize and best_brain:
        evo.brain_eval.replay(best_genome.to_dict(), best_brain, duration=args.viewer_duration)


if __name__ == "__main__":
    main()
