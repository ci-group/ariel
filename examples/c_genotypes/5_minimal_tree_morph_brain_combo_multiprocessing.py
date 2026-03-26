"""
Minimal joint tree-morphology + brain-learning example (multiprocessing).

This variant keeps morphology EA logic simple and evaluates each individual
in separate processes to avoid MuJoCo thread contention.

Lower fitness is better.
"""

# Standard library
import argparse
import contextlib
import copy
import multiprocessing as mp
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from ariel.utils.video_recorder import VideoRecorder

install()
console = Console()

parser = argparse.ArgumentParser(description="Minimal tree morphology + brain joint evolution (multiprocessing)")
parser.add_argument("--budget", type=int, default=20, help="Morphology generations")
parser.add_argument("--pop", type=int, default=16, help="Morphology population")
parser.add_argument("--dur", type=float, default=5.0, help="Active control duration")
parser.add_argument(
    "--eval-delay",
    type=float,
    default=2.0,
    help="No-control warm-up seconds before scoring (minimum enforced: 2.0)",
)
parser.add_argument(
    "--z-penalty-weight",
    type=float,
    default=2.0,
    help="Penalty weight for vertical (z-axis) motion during active control",
)
parser.add_argument("--learn-budget", type=int, default=4, help="CMA iterations per morphology")
parser.add_argument("--learn-pop", type=int, default=16, help="CMA population per iteration")
parser.add_argument(
    "--eval-workers",
    type=int,
    default=max(1, os.cpu_count() or 1),
    help="Worker processes for parallel individual evaluation",
)
parser.add_argument(
    "--eval-timeout",
    type=int,
    default=180,
    help="Per-generation timeout in seconds for process-based evaluation",
)
parser.add_argument("--max-modules", type=int, default=10, help="Max modules in tree")
parser.add_argument("--max-depth", type=int, default=12, help="Max tree depth")
parser.add_argument("--morph-weight", type=float, default=0.3, help="Weight of morphology term")
parser.add_argument(
    "--visualize",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Visualize the final best individual",
)
parser.add_argument(
    "--save-video",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Save a video of the final best individual",
)
parser.add_argument(
    "--video-duration",
    type=float,
    default=10.0,
    help="Duration of saved best-individual video in seconds",
)
args = parser.parse_args()

POP_SIZE = args.pop
BUDGET = args.budget
DURATION = args.dur
EVAL_DELAY = max(2.0, args.eval_delay)
Z_PENALTY_WEIGHT = max(0.0, args.z_penalty_weight)
LEARN_BUDGET = args.learn_budget
LEARN_POP = args.learn_pop
EVAL_WORKERS = max(1, min(args.eval_workers, POP_SIZE))
EVAL_TIMEOUT_S = max(10, int(args.eval_timeout))
NUM_MODULES = args.max_modules
MAX_DEPTH = args.max_depth
MORPH_WEIGHT = args.morph_weight

SEED = 42
RNG = np.random.default_rng(SEED)
torch.manual_seed(SEED)

SCRIPT_NAME = Path(__file__).stem
DATA = Path.cwd() / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)

SPAWN_POSITION = (-0.8, 0.0, 0.1)
TARGET_POSITION = np.array([2.0, 0.0, 0.1], dtype=np.float32)


class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.hidden_activation = nn.ELU()
        self.output_activation = nn.Tanh()

        for p in self.parameters():
            p.requires_grad = False

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


def morphology_fitness_term(genome: TreeGenome) -> float:
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
        return -float(score)/10
    except Exception:
        return float("inf")


def _spawn_with_fallback(
    genome_dict: dict,
    position: tuple[float, float, float],
) -> tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]:
    def _build(correct_collision_with_floor: bool) -> tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]:
        genome = TreeGenome.from_dict(genome_dict)
        graph = genome.to_networkx()
        if graph.number_of_nodes() == 0:
            raise ValueError("Could not decode morphology")

        spec = construct_mjspec_from_graph(graph).spec
        world = SimpleFlatWorld()
        world.spawn(
            spec,
            position=position,
            correct_collision_with_floor=correct_collision_with_floor,
        )
        model = world.spec.compile()
        data = mujoco.MjData(model)
        return world, model, data

    try:
        return _build(True)
    except Exception:
        return _build(False)


def _learn_brain_progress_for_genome(
    genome_dict: dict,
    duration: float,
    eval_delay: float,
    learn_budget: int,
    learn_pop: int,
    target_position: np.ndarray,
    z_penalty_weight: float,
) -> tuple[float, list[float]]:
    try:
        world, model, data = _spawn_with_fallback(genome_dict, SPAWN_POSITION)
    except Exception:
        return -float("inf"), []

    if model.nu == 0:
        return -float("inf"), []

    net = Network(
        input_size=len(get_state_from_data(data)) + 2,
        output_size=model.nu,
        hidden_size=16,
    )
    num_params = sum(p.numel() for p in net.parameters())

    param = ng.p.Array(shape=(num_params,))
    learner = ng.optimizers.registry["CMA"](
        parametrization=param,
        budget=learn_budget * learn_pop,
    )

    tracker = Tracker(name_to_bind="core", observable_attributes=["xpos"], quiet=True)
    tracker.setup(world.spec, data)
    controller = Controller(controller_callback_function=net.forward, tracker=tracker)

    best_score = -float("inf")
    best_vec: list[float] = []

    for _ in range(learn_budget):
        candidates = [learner.ask() for _ in range(learn_pop)]
        for candidate in candidates:
            vec = candidate.value
            fill_parameters(net, vec)

            mujoco.mj_resetData(model, data)
            if eval_delay > 0.0:
                data.ctrl[:] = 0.0
                delay_steps = int(eval_delay / model.opt.timestep)
                for _ in range(max(0, delay_steps)):
                    data.ctrl[:] = 0.0
                    mujoco.mj_step(model, data)

            dist_start = float(np.linalg.norm(target_position - data.qpos[0:3]))

            # Active control starts only after the no-control warm-up.
            z_ref = float(data.qpos[2])
            z_dev_accum = 0.0
            active_steps = max(1, int(duration / model.opt.timestep))
            for _ in range(active_steps):
                controller.set_control(model, data)
                mujoco.mj_step(model, data)
                z_dev_accum += abs(float(data.qpos[2]) - z_ref)

            dist_end = float(np.linalg.norm(target_position - data.qpos[0:3]))
            progress = dist_start - dist_end
            z_penalty = z_dev_accum / active_steps
            score = progress - (z_penalty_weight * z_penalty)

            learner.tell(candidate, -score)
            if score > best_score:
                best_score = score
                best_vec = vec.tolist()

    return best_score, best_vec


def _evaluate_individual_process(task: tuple[dict, float, float, int, int, float, float]) -> tuple[float, list[float]]:
    genome_dict, duration, eval_delay, learn_budget, learn_pop, morph_weight, z_penalty_weight = task
    try:
        genome = TreeGenome.from_dict(genome_dict)
        morph_term = morphology_fitness_term(genome)
        score, best_vec = _learn_brain_progress_for_genome(
            genome_dict,
            duration,
            eval_delay,
            learn_budget,
            learn_pop,
            TARGET_POSITION,
            z_penalty_weight,
        )

        if not np.isfinite(morph_term) or not np.isfinite(score):
            return float("inf"), best_vec

        fit = -score + morph_weight * morph_term
        return fit, best_vec
    except Exception:
        return float("inf"), []


class MinimalJointEvolution:
    def __init__(self) -> None:
        self.config = EASettings(
            is_maximisation=False,
            num_steps=BUDGET,
            target_population_size=POP_SIZE,
            output_folder=DATA,
            db_file_name=f"database_{int(time.time())}.db",
            db_handling="delete",
        )

    def map_genotype_to_body(self, genome_data: dict | TreeGenome) -> mujoco.MjSpec | None:
        genome = TreeGenome.from_dict(genome_data) if isinstance(genome_data, dict) else genome_data
        try:
            graph = genome.to_networkx()
            if graph.number_of_nodes() == 0:
                return None
            return construct_mjspec_from_graph(graph).spec
        except Exception:
            return None

    def spawn_with_fallback(
        self,
        genome_data: dict | TreeGenome,
        position: tuple[float, float, float],
    ) -> tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]:
        def _build(correct_collision_with_floor: bool) -> tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]:
            spec = self.map_genotype_to_body(genome_data)
            if spec is None:
                raise ValueError("Could not decode morphology")

            world = SimpleFlatWorld()
            world.spawn(
                spec,
                position=position,
                correct_collision_with_floor=correct_collision_with_floor,
            )
            model = world.spec.compile()
            data = mujoco.MjData(model)
            return world, model, data

        try:
            return _build(True)
        except Exception:
            return _build(False)

    def get_joint_count(self, genome: TreeGenome) -> int:
        spec = self.map_genotype_to_body(genome)
        if spec is None:
            return 0
        try:
            return spec.compile().nu
        except Exception:
            return 0

    def mutate_morphology(self, genome: TreeGenome) -> TreeGenome:
        new = copy.deepcopy(genome)
        mutation_type = RNG.choice(["point", "subtree", "shrink", "hoist"], p=[0.45, 0.35, 0.1, 0.1])

        if mutation_type == "point":
            mutate_replace_node(new)
        elif mutation_type == "subtree":
            mutate_subtree_replacement(new, max_modules=NUM_MODULES)
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
            genome = random_tree(NUM_MODULES)
            if self.get_joint_count(genome) > 0 and validate_tree_depth(genome, MAX_DEPTH):
                break

        ind = Individual()
        ind.genotype = {"morph": genome.to_dict()}
        ind.tags = {"ps": False, "valid": True, "last_brain": []}
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
                p = random.choice(parents)
                child_morph = TreeGenome.from_dict(p.genotype["morph"])

            child_morph = self.mutate_morphology(child_morph)

            attempts = 0
            while attempts < 12:
                if self.get_joint_count(child_morph) > 0 and validate_tree_depth(child_morph, MAX_DEPTH):
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

        tasks = [
            (
                ind.genotype["morph"],
                DURATION,
                EVAL_DELAY,
                LEARN_BUDGET,
                LEARN_POP,
                MORPH_WEIGHT,
                Z_PENALTY_WEIGHT,
            )
            for ind in to_eval
        ]

        if EVAL_WORKERS == 1:
            for ind, task in track(zip(to_eval, tasks), total=len(to_eval), description="Learning + Evaluating..."):
                fit, best_vec = _evaluate_individual_process(task)
                ind.fitness = fit
                ind.tags["last_brain"] = best_vec
                ind.requires_eval = False
            return population

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=EVAL_WORKERS, mp_context=ctx) as executor:
            future_to_ind = {
                executor.submit(_evaluate_individual_process, task): ind
                for ind, task in zip(to_eval, tasks)
            }

            pending = set(future_to_ind)
            start_t = time.monotonic()
            deadline = start_t + EVAL_TIMEOUT_S
            completed = 0

            while pending:
                now = time.monotonic()
                if now >= deadline:
                    console.log(
                        "[yellow]Evaluation timeout reached; assigning inf fitness to "
                        f"{len(pending)} pending individuals.[/yellow]",
                    )
                    for fut in pending:
                        fut.cancel()
                        ind = future_to_ind[fut]
                        ind.fitness = float("inf")
                        ind.tags["last_brain"] = []
                        ind.requires_eval = False
                    break

                done = [f for f in pending if f.done()]
                if not done:
                    time.sleep(0.1)
                    continue

                for fut in done:
                    pending.remove(fut)
                    ind = future_to_ind[fut]
                    try:
                        fit, best_vec = fut.result()
                    except Exception:
                        fit, best_vec = float("inf"), []
                    ind.fitness = fit
                    ind.tags["last_brain"] = best_vec
                    ind.requires_eval = False
                    completed += 1

            if completed < len(to_eval):
                console.log(
                    f"[yellow]Completed {completed}/{len(to_eval)} individual evaluations this generation.[/yellow]",
                )

        return population

    def survivor_selection(self, population: Population) -> Population:
        population = population.sort(sort="min", attribute="fitness_")
        survivors = population[: self.config.target_population_size]
        for ind in population:
            if ind not in survivors:
                ind.alive = False

        finite = [ind.fitness_ for ind in survivors if ind.fitness_ is not None and np.isfinite(ind.fitness_)]
        if finite:
            console.log(
                "[green]Survivors:[/green] "
                f"avg={np.mean(finite):.3f}, min={np.min(finite):.3f}, max={np.max(finite):.3f}",
            )
        return population

    def run_best(self, best: Individual, duration: float = 10.0) -> None:
        mujoco.set_mjcb_control(None)
        try:
            try:
                world, model, data = self.spawn_with_fallback(best.genotype["morph"], SPAWN_POSITION)
            except Exception as exc:
                console.log(f"[red]Could not spawn best morphology: {exc}[/red]")
                return

            if model.nu == 0:
                console.log("[red]No actuators on best morphology.[/red]")
                return

            net = Network(input_size=len(get_state_from_data(data)) + 2, output_size=model.nu, hidden_size=16)
            brain_vec = best.tags.get("last_brain", [])
            if brain_vec:
                fill_parameters(net, brain_vec)

            tracker = Tracker(name_to_bind="core", observable_attributes=["xpos"], quiet=True)
            tracker.setup(world.spec, data)
            controller = Controller(controller_callback_function=net.forward, tracker=tracker)

            mujoco.mj_resetData(model, data)
            total_duration = duration + EVAL_DELAY

            if sys.platform == "darwin" or not hasattr(mujoco.viewer, "launch_passive"):
                console.log("[yellow]Using active MuJoCo viewer fallback (passive viewer unavailable).[/yellow]")
                console.log("[yellow]Close the viewer window to continue.[/yellow]")

                def _delayed_control(model, data):
                    if data.time < EVAL_DELAY:
                        data.ctrl[:] = 0.0
                        return
                    controller.set_control(model, data)

                mujoco.set_mjcb_control(_delayed_control)
                mujoco.viewer.launch(model=model, data=data)
                return

            with mujoco.viewer.launch_passive(model, data) as v:
                sim_start = time.time()
                while v.is_running() and (time.time() - sim_start) < total_duration:
                    step_start = time.time()
                    if data.time < EVAL_DELAY:
                        data.ctrl[:] = 0.0
                    else:
                        controller.set_control(model, data)
                    mujoco.mj_step(model, data)
                    v.sync()
                    remaining = model.opt.timestep - (time.time() - step_start)
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            mujoco.set_mjcb_control(None)

    def save_best_video(self, best: Individual, duration: float = 10.0) -> None:
        mujoco.set_mjcb_control(None)
        try:
            try:
                world, model, data = self.spawn_with_fallback(best.genotype["morph"], SPAWN_POSITION)
            except Exception as exc:
                console.log(f"[red]Could not spawn best morphology for video: {exc}[/red]")
                return

            if model.nu == 0:
                console.log("[red]No actuators on best morphology (video skipped).[/red]")
                return

            net = Network(input_size=len(get_state_from_data(data)) + 2, output_size=model.nu, hidden_size=16)
            brain_vec = best.tags.get("last_brain", [])
            if brain_vec:
                fill_parameters(net, brain_vec)

            tracker = Tracker(name_to_bind="core", observable_attributes=["xpos"], quiet=True)
            tracker.setup(world.spec, data)
            controller = Controller(controller_callback_function=net.forward, tracker=tracker)

            videos_dir = DATA / "videos"
            videos_dir.mkdir(exist_ok=True, parents=True)
            video_recorder = VideoRecorder(file_name="best_individual", output_folder=videos_dir)

            mujoco.mj_resetData(model, data)
            total_duration = duration + EVAL_DELAY
            steps_per_frame = max(1, int(round(1.0 / (model.opt.timestep * video_recorder.fps))))

            with mujoco.Renderer(
                model,
                width=video_recorder.width,
                height=video_recorder.height,
            ) as renderer:
                while data.time < total_duration:
                    for _ in range(steps_per_frame):
                        if data.time < EVAL_DELAY:
                            data.ctrl[:] = 0.0
                        else:
                            controller.set_control(model, data)
                        mujoco.mj_step(model, data)
                        if data.time >= total_duration:
                            break

                    renderer.update_scene(data)
                    video_recorder.write(renderer.render())

            video_recorder.release()
            console.log(f"[green]Saved best-individual video to {videos_dir}[/green]")
        finally:
            mujoco.set_mjcb_control(None)

    def evolve(self) -> Individual | None:
        console.log("Initializing population...")
        population = Population([self.create_individual() for _ in range(POP_SIZE)])
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
    console.rule("[bold magenta]Minimal Tree Morph + Brain Evolution (Multiprocessing)[/bold magenta]")
    console.log(
        f"Pop={POP_SIZE}, Gens={BUDGET}, LearnBudget={LEARN_BUDGET}, LearnPop={LEARN_POP}, "
        f"EvalWorkers={EVAL_WORKERS}, EvalTimeout={EVAL_TIMEOUT_S}s, "
        f"Dur={DURATION}s, Delay={EVAL_DELAY}s, ZPenaltyWeight={Z_PENALTY_WEIGHT}, MorphWeight={MORPH_WEIGHT}",
    )

    evo = MinimalJointEvolution()
    start = time.time()
    best = evo.evolve()
    elapsed = time.time() - start

    if best is None:
        console.log("[red]No best individual found.[/red]")
        return

    console.rule("[bold green]Final Best[/bold green]")
    console.log(f"Best combined fitness: {best.fitness:.4f}")
    console.log(f"Elapsed: {elapsed:.2f}s")

    if args.save_video:
        evo.save_best_video(best, duration=args.video_duration)

    if args.visualize:
        evo.run_best(best, duration=10.0)


if __name__ == "__main__":
    main()
