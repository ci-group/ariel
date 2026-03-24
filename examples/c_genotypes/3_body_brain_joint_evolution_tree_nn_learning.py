"""
Joint body-brain evolution with TreeGenome morphology and NN controller learning.

This example combines:
- Tree genotype morphology evolution (selection / crossover / mutation)
- Neural-network controller optimization via CMA-ES (Nevergrad)
- MuJoCo simulation for locomotion fitness

Fitness is the minimum distance to target after learning the controller for each
candidate morphology.
"""

# Standard library
import argparse
import contextlib
import copy
import random
import time
from pathlib import Path
from typing import Literal

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
from ariel.utils.runners import simple_runner

# Pretty logs
install()
console = Console()


# ============================================================================ #
#                               CONFIGURATION                                  #
# ============================================================================ #

parser = argparse.ArgumentParser(
    description="Tree body + NN brain joint evolution with learning",
)
parser.add_argument("--budget", type=int, default=5, help="EA generations")
parser.add_argument("--pop", type=int, default=10, help="EA population size")
parser.add_argument("--dur", type=float, default=3.0, help="Simulation duration")
parser.add_argument(
    "--learn-budget",
    type=int,
    default=6,
    help="CMA-ES iterations per individual",
)
parser.add_argument(
    "--learn-pop",
    type=int,
    default=12,
    help="CMA-ES population per iteration",
)
parser.add_argument(
    "--max-modules",
    type=int,
    default=20,
    help="Maximum tree modules",
)
parser.add_argument(
    "--max-depth",
    type=int,
    default=12,
    help="Maximum TreeGenome depth",
)
parser.add_argument(
    "--visualize",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Visualize best individual with MuJoCo viewer",
)
parser.add_argument(
    "--viewer-duration",
    type=float,
    default=10.0,
    help="Seconds to simulate in passive viewer mode",
)
args = parser.parse_args()

POP_SIZE = args.pop
BUDGET = args.budget
DURATION = args.dur
LEARN_BUDGET = args.learn_budget
LEARN_POP = args.learn_pop
NUM_MODULES = args.max_modules
MAX_DEPTH = args.max_depth

SEED = 42
RNG = np.random.default_rng(SEED)
torch.manual_seed(SEED)

SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)

SPAWN_POSITION = (-0.8, 0.0, 0.1)
TARGET_POSITION = np.array([2.0, 0.0, 0.1])

ViewerTypes = Literal["simple", "viewer"]


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
    def forward(self, model, data):
        state = torch.tensor(get_state_from_data(data), dtype=torch.float32)
        x = self.hidden_activation(self.fc1(state))
        x = self.hidden_activation(self.fc2(x))
        x = self.output_activation(self.fc_out(x)) * (torch.pi / 2)
        return x.detach().numpy()


@torch.no_grad()
def fill_parameters(net: nn.Module, vector: np.ndarray | list[float]) -> None:
    """Fill torch module parameters from a flat vector."""
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(vector[address : address + n], device=d.device)
        address += n


# ============================================================================ #
#                              EVOLUTION LOGIC                                 #
# ============================================================================ #


class JointTreeNNLearningEvolution:
    def __init__(self) -> None:
        self.config = EASettings(
            is_maximisation=False,
            num_steps=BUDGET,
            target_population_size=POP_SIZE,
            output_folder=DATA,
            db_file_name="database.db",
            db_handling="delete",
        )

    def map_genotype_to_body(self, genome_data: dict | TreeGenome) -> mujoco.MjSpec | None:
        """Decode TreeGenome to MuJoCo spec; return None for invalid morphologies."""
        genome = (
            TreeGenome.from_dict(genome_data)
            if isinstance(genome_data, dict)
            else genome_data
        )
        try:
            robot_graph = genome.to_networkx()
            if robot_graph.number_of_nodes() == 0:
                return None
            return construct_mjspec_from_graph(robot_graph).spec
        except Exception:
            return None

    def spawn_with_fallback(
        self,
        world: SimpleFlatWorld,
        spec: mujoco.MjSpec,
        position: tuple[float, float, float],
    ) -> None:
        """Spawn robot robustly, retrying without floor correction on failure."""
        try:
            world.spawn(
                spec,
                position=position,
                correct_collision_with_floor=True,
            )
        except Exception as exc:
            console.log(
                "[yellow]Spawn with floor-correction failed; retrying without correction:[/yellow] "
                f"{exc}",
            )
            world.spawn(
                spec,
                position=position,
                correct_collision_with_floor=False,
            )

    def get_joint_count(self, genome: TreeGenome) -> int:
        spec = self.map_genotype_to_body(genome)
        if spec is None:
            return 0
        try:
            model = spec.compile()
            return model.nu
        except Exception:
            return 0

    def mutate_morphology(self, genome: TreeGenome) -> TreeGenome:
        """Apply one of several GP-style mutations to a tree morphology."""
        new = copy.deepcopy(genome)

        mutation_type = RNG.choice(
            ["point", "subtree", "shrink", "hoist"],
            p=[0.45, 0.35, 0.1, 0.1],
        )

        if mutation_type == "point":
            mutate_replace_node(new)
        elif mutation_type == "subtree":
            mutate_subtree_replacement(new, max_modules=NUM_MODULES)
        elif mutation_type == "shrink":
            mutate_shrink(new)
        elif mutation_type == "hoist":
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
        """Create a valid individual containing only morphology genotype.

        The controller is learned during evaluation.
        """
        while True:
            genome = random_tree(NUM_MODULES)
            joints = self.get_joint_count(genome)
            if joints > 0 and validate_tree_depth(genome, MAX_DEPTH):
                break

        ind = Individual()
        ind.genotype = {"morph": genome.to_dict()}
        ind.tags = {
            "ps": False,
            "valid": True,
            "last_brain": [],
        }
        return ind

    def parent_selection(self, population: Population) -> Population:
        """Truncation parent selection on fitness."""
        population = population.sort(sort="min", attribute="fitness_")
        cutoff = len(population) // 2
        for i, ind in enumerate(population):
            ind.tags["ps"] = i < cutoff
        return population

    def crossover_morphologies(self, p1: Individual, p2: Individual) -> TreeGenome:
        t1 = TreeGenome.from_dict(p1.genotype["morph"])
        t2 = TreeGenome.from_dict(p2.genotype["morph"])
        c1, c2 = crossover_subtree(t1, t2)
        return c1 if RNG.random() < 0.5 else c2

    def reproduction(self, population: Population) -> Population:
        """Generate offspring by sexual/asexual reproduction + mutation."""
        parents = [ind for ind in population if ind.tags.get("ps", False)]
        if not parents:
            parents = list(population)

        offspring: list[Individual] = []
        target_pool = self.config.target_population_size * 2

        while len(population) + len(offspring) < target_pool:
            use_sexual = len(parents) >= 2 and RNG.random() < 0.6
            if use_sexual:
                p1, p2 = random.sample(parents, 2)
                child_morph = self.crossover_morphologies(p1, p2)
            else:
                parent = random.choice(parents)
                child_morph = TreeGenome.from_dict(parent.genotype["morph"])

            child_morph = self.mutate_morphology(child_morph)

            attempts = 0
            while attempts < 15:
                has_joints = self.get_joint_count(child_morph) > 0
                valid_depth = validate_tree_depth(child_morph, MAX_DEPTH)
                if has_joints and valid_depth:
                    break
                child_morph = self.mutate_morphology(child_morph)
                attempts += 1

            if self.get_joint_count(child_morph) == 0:
                child_morph = random_tree(NUM_MODULES)

            child = Individual()
            child.genotype = {"morph": child_morph.to_dict()}
            child.tags = {
                "ps": False,
                "valid": True,
                "last_brain": [],
            }
            child.requires_eval = True
            offspring.append(child)

        population.extend(offspring)
        return population

    def individual_learn(self, ind: Individual) -> tuple[float, list[float]]:
        """Learn NN controller for one morphology with CMA-ES."""
        spec = self.map_genotype_to_body(ind.genotype["morph"])
        if spec is None:
            return float("inf"), []

        mujoco.set_mjcb_control(None)
        world = SimpleFlatWorld()
        self.spawn_with_fallback(world, spec, SPAWN_POSITION)

        model = world.spec.compile()
        if model.nu == 0:
            return float("inf"), []

        data = mujoco.MjData(model)

        net = Network(
            input_size=len(get_state_from_data(data)),
            output_size=model.nu,
            hidden_size=32,
        )
        num_vars = sum(p.numel() for p in net.parameters())

        param = ng.p.Array(shape=(num_vars,))
        learner = ng.optimizers.registry["CMA"](
            parametrization=param,
            budget=LEARN_BUDGET * LEARN_POP,
        )

        tracker = Tracker(
            name_to_bind="core",
            observable_attributes=["xpos"],
            quiet=True,
        )
        tracker.setup(world.spec, data)

        controller = Controller(
            controller_callback_function=net.forward,
            tracker=tracker,
        )

        best_fit = float("inf")
        best_vec: list[float] = []

        for _ in range(LEARN_BUDGET):
            candidates = [learner.ask() for _ in range(LEARN_POP)]
            for candidate in candidates:
                vec = candidate.value
                fill_parameters(net, vec)

                mujoco.mj_resetData(model, data)
                mujoco.set_mjcb_control(controller.set_control)
                simple_runner(model, data, duration=DURATION)

                xc, yc, zc = data.qpos[0:3].copy()
                xt, yt, zt = TARGET_POSITION
                fitness = float(np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2))

                learner.tell(candidate, fitness)

                if fitness < best_fit:
                    best_fit = fitness
                    best_vec = vec.tolist()

        return best_fit, best_vec

    def evaluate(self, population: Population) -> Population:
        """Evaluate all unevaluated individuals via NN learning."""
        to_eval = [
            ind
            for ind in population
            if ind.alive and ind.tags.get("valid") and ind.requires_eval
        ]

        if not to_eval:
            return population

        for ind in track(to_eval, description="Learning + Evaluating..."):
            fit, best_vec = self.individual_learn(ind)
            ind.fitness = fit
            ind.tags["last_brain"] = best_vec
            ind.requires_eval = False

        return population

    def survivor_selection(self, population: Population) -> Population:
        """Truncation survivor selection."""
        population = population.sort(sort="min", attribute="fitness_")
        survivors = population[: self.config.target_population_size]

        for ind in population:
            if ind not in survivors:
                ind.alive = False

        finite_fits = [
            ind.fitness_
            for ind in survivors
            if ind.fitness_ is not None and np.isfinite(ind.fitness_)
        ]
        if finite_fits:
            console.log(
                "[green]Survivors:[/green] "
                f"avg={np.mean(finite_fits):.3f}, min={np.min(finite_fits):.3f}, max={np.max(finite_fits):.3f}",
            )

        return population

    def run_best(
        self,
        best: Individual,
        mode: ViewerTypes = "viewer",
        duration: float = 10.0,
    ) -> None:
        """Replay best morphology with its learned NN controller."""
        spec = self.map_genotype_to_body(best.genotype["morph"])
        if spec is None:
            console.log("[red]Cannot visualize invalid best morphology.[/red]")
            return

        world = SimpleFlatWorld()
        self.spawn_with_fallback(world, spec, SPAWN_POSITION)

        model = world.spec.compile()
        data = mujoco.MjData(model)

        if model.nu == 0:
            console.log("[red]Best morphology has no actuators; cannot simulate controller.[/red]")
            return

        net = Network(
            input_size=len(get_state_from_data(data)),
            output_size=model.nu,
            hidden_size=32,
        )

        brain_vec = best.tags.get("last_brain", [])
        if brain_vec:
            fill_parameters(net, brain_vec)

        tracker = Tracker(
            name_to_bind="core",
            observable_attributes=["xpos"],
            quiet=True,
        )
        tracker.setup(world.spec, data)

        controller = Controller(
            controller_callback_function=net.forward,
            tracker=tracker,
        )

        mujoco.mj_resetData(model, data)
        mujoco.set_mjcb_control(controller.set_control)

        if mode == "simple":
            simple_runner(model, data, duration=duration)
            return

        with mujoco.viewer.launch_passive(model, data) as v:
            sim_start = time.time()
            while v.is_running() and (time.time() - sim_start) < duration:
                step_start = time.time()
                mujoco.mj_step(model, data)
                v.sync()
                remaining = model.opt.timestep - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)

    def evolve(self) -> Individual | None:
        console.log("Initializing population...")
        population = Population([self.create_individual() for _ in range(POP_SIZE)])

        # Initial learning/eval
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
            num_steps=BUDGET,
            db_file_path=self.config.db_file_path,
            db_handling=self.config.db_handling,
            quiet=self.config.quiet,
        )
        ea.run()

        return ea.get_solution("best", only_alive=False)


def main() -> None:
    console.rule("[bold magenta]Tree Body + NN Brain Joint Evolution (with Learning)[/bold magenta]")
    console.log(
        f"Pop={POP_SIZE}, Gens={BUDGET}, LearnBudget={LEARN_BUDGET}, LearnPop={LEARN_POP}, Dur={DURATION}s",
    )

    evo = JointTreeNNLearningEvolution()
    start = time.time()
    best = evo.evolve()
    elapsed = time.time() - start

    if best is None:
        console.log("[red]No best individual found.[/red]")
        return

    console.rule("[bold green]Final Best[/bold green]")
    console.log(f"Best fitness (distance to target): {best.fitness:.4f}")
    console.log(f"Elapsed: {elapsed:.2f}s")

    if args.visualize:
        evo.run_best(best, mode="viewer", duration=args.viewer_duration)


if __name__ == "__main__":
    main()
