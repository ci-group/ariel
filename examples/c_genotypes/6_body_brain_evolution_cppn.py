"""
Joint CPPN-morphology + brain-learning evolution (multiprocessing).

Builds on 5_body_evolution_cppn.py (CPPN-NEAT body genetics) and
4_body_brain_joint_evolution_tree_nn_learning_multiprocessing.py (brain learning).

Fitness = -(loco_weight * locomotion_score + morpho_weight * morpho_score)
Lower fitness is better.
"""

import json
import multiprocessing as mp
import os
import random
import sys
import time
import warnings
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import mujoco.viewer
import nevergrad as ng
import numpy as np
import torch
from rich.console import Console
from rich.progress import track
from rich.traceback import install
from torch import nn

from ariel.body_phenotypes.robogen_lite.config import NUM_OF_ROTATIONS, NUM_OF_TYPES_OF_MODULES
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome
from ariel.body_phenotypes.robogen_lite.cppn_neat.id_manager import IdManager
from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import MorphologyDecoderBestFirst
from ariel.ec import EA, EAOperation, EASettings, Individual, Population
from ariel.simulation.controllers.controller import Controller, Tracker
from ariel.simulation.controllers.utils.data_get import get_state_from_data
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.environments._simple_flat_with_target import SimpleFlatWorldWithTarget
from ariel.utils.morphological_descriptor import MorphologicalMeasures
from ariel.utils.video_recorder import VideoRecorder

install()
console = Console()

warnings.filterwarnings(
    "ignore",
    message="TPA: apparent inconsistency",
    category=UserWarning,
    module="cma",
)

parser = argparse.ArgumentParser(description="CPPN body + brain joint evolution (multiprocessing)")
parser.add_argument("--bud", type=int, default=10, help="Morphology generations")
parser.add_argument("--pop", type=int, default=10, help="Morphology population")
parser.add_argument("--dur", type=float, default=5.0, help="Active control duration (s)")
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
parser.add_argument("--learn-bud", type=int, default=10, help="CMA iterations per morphology")
parser.add_argument("--learn-pop", type=int, default=10, help="CMA population per iteration")
parser.add_argument(
    "--eval-workers",
    type=int,
    default=max(1, os.cpu_count() or 1),
    help="Worker processes for parallel individual evaluation",
)
parser.add_argument("--max-modules", type=int, default=15, help="Max CPPN-decoded modules")
parser.add_argument(
    "--loco-weight",
    type=float,
    default=0.7,
    help="Weight for locomotion score in combined fitness",
)
parser.add_argument(
    "--morpho-weight",
    type=float,
    default=0.3,
    help="Weight for morphological score in combined fitness",
)
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
parser.add_argument("--video-duration", type=float, default=10.0, help="Duration of saved video (s)")
args = parser.parse_args()

POP_SIZE = args.pop
BUDGET = args.bud
DURATION = args.dur
EVAL_DELAY = max(2.0, args.eval_delay)
Z_PENALTY_WEIGHT = max(0.0, args.z_penalty_weight)
LEARN_BUDGET = args.learn_bud
LEARN_POP = args.learn_pop
EVAL_WORKERS = max(1, min(args.eval_workers, POP_SIZE))
NUM_MODULES = args.max_modules
LOCO_WEIGHT = args.loco_weight
MORPHO_WEIGHT = args.morpho_weight

SEED = 42
RNG = np.random.default_rng(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

SCRIPT_NAME = Path(__file__).stem
DATA = Path.cwd() / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)

SPAWN_POSITION = (-0.8, 0.0, 0.1)
TARGET_POSITION = np.array([2.0, 0.0, 0.1], dtype=np.float32)

T = NUM_OF_TYPES_OF_MODULES
R = NUM_OF_ROTATIONS
NUM_CPPN_INPUTS = 6
NUM_CPPN_OUTPUTS = 1 + T + R

id_manager = IdManager(
    node_start=NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS - 1,
    innov_start=(NUM_CPPN_INPUTS * NUM_CPPN_OUTPUTS) - 1,
)


# ---------------------------------------------------------------------------
# Brain network (identical to example 4)
# ---------------------------------------------------------------------------

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
        d[:] = torch.as_tensor(vector[address: address + n], device=d.device)
        address += n


# ---------------------------------------------------------------------------
# Morphological fitness (static, no simulation)
# ---------------------------------------------------------------------------

def _morpho_score_from_graph(graph) -> float:
    try:
        m = MorphologicalMeasures(graph)
        return (
            m.symmetry * 0.20
            + m.joints * 0.20
            + m.branching * 0.20
            + m.length_of_limbs * 0.20
            + m.module_diversity * 0.20
        )
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# CPPN spawn helper (decode + collision-correction fallback)
# ---------------------------------------------------------------------------

def _spawn_with_fallback(
    genome_dict: dict,
    position: tuple[float, float, float],
) -> tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]:
    def _build(correct_collision_with_floor: bool) -> tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]:
        genome = Genome.from_dict(genome_dict)
        graph = MorphologyDecoderBestFirst(genome, NUM_MODULES).decode()
        if graph.number_of_nodes() == 0:
            raise ValueError("Empty CPPN decoded graph")
        spec = construct_mjspec_from_graph(graph).spec
        world = SimpleFlatWorld()
        world.spawn(spec, position=position, correct_collision_with_floor=correct_collision_with_floor)
        model = world.spec.compile()
        data = mujoco.MjData(model)
        return world, model, data

    try:
        return _build(True)
    except Exception:
        return _build(False)


# ---------------------------------------------------------------------------
# Brain learning inner loop (runs inside worker process)
# ---------------------------------------------------------------------------

def _learn_brain_for_genome(
    genome_dict: dict,
    duration: float,
    eval_delay: float,
    learn_budget: int,
    learn_pop: int,
    target_position: np.ndarray,
    z_penalty_weight: float,
) -> tuple[float, float, list[float], list[float]]:
    """Returns (loco_score, morpho_score, best_weight_vec, iteration_scores).

    loco_score and morpho_score are raw un-weighted values.
    """
    # Static morpho score — decode once upfront
    morpho_score = 0.0
    try:
        _g = Genome.from_dict(genome_dict)
        _graph = MorphologyDecoderBestFirst(_g, NUM_MODULES).decode()
        if _graph.number_of_nodes() > 0:
            morpho_score = _morpho_score_from_graph(_graph)
    except Exception:
        pass

    # Spawn simulation (decodes again inside, with fallback)
    try:
        world, model, data = _spawn_with_fallback(genome_dict, SPAWN_POSITION)
    except Exception:
        return -float("inf"), morpho_score, [], []

    if model.nu == 0:
        return -float("inf"), morpho_score, [], []

    net = Network(
        input_size=len(get_state_from_data(data)) + 2,
        output_size=model.nu,
        hidden_size=16,
    )
    num_params = sum(p.numel() for p in net.parameters())

    min_lambda = 4 + int(3 * np.log(max(num_params, 2)))
    learn_pop = max(learn_pop, min_lambda)
    if learn_pop % 2 != 0:
        learn_pop += 1

    param = ng.p.Array(shape=(num_params,)).set_mutation(sigma=0.5)
    learner = ng.optimizers.registry["CMA"](
        parametrization=param,
        budget=learn_budget * learn_pop,
        num_workers=learn_pop,
    )

    tracker = Tracker(name_to_bind="core", observable_attributes=["xpos"], quiet=True)
    tracker.setup(world.spec, data)
    controller = Controller(controller_callback_function=net.forward, tracker=tracker)

    best_loco = -float("inf")
    best_vec: list[float] = []
    iteration_scores: list[float] = []

    for _ in range(learn_budget):
        candidates = [learner.ask() for _ in range(learn_pop)]
        iter_best = -float("inf")
        for candidate in candidates:
            vec = candidate.value
            fill_parameters(net, vec)

            mujoco.mj_resetData(model, data)
            if eval_delay > 0.0:
                delay_steps = int(eval_delay / model.opt.timestep)
                for _ in range(max(0, delay_steps)):
                    data.ctrl[:] = 0.0
                    mujoco.mj_step(model, data)

            dist_start = float(np.linalg.norm(target_position - data.qpos[0:3]))
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
            if score > best_loco:
                best_loco = score
                best_vec = vec.tolist()
            if score > iter_best:
                iter_best = score

        iteration_scores.append(iter_best)

    return best_loco, morpho_score, best_vec, iteration_scores


def _evaluate_individual_process(
    task: tuple[dict, float, float, int, int, float, float, float],
) -> tuple[float, list[float], list[float]]:
    genome_dict, duration, eval_delay, learn_budget, learn_pop, z_penalty_weight, loco_weight, morpho_weight = task
    try:
        loco_score, morpho_score, best_vec, iteration_scores = _learn_brain_for_genome(
            genome_dict,
            duration,
            eval_delay,
            learn_budget,
            learn_pop,
            TARGET_POSITION,
            z_penalty_weight,
        )

        if not np.isfinite(loco_score):
            return float("inf"), [], []

        combined = loco_weight * loco_score + morpho_weight * morpho_score
        fit = -combined if np.isfinite(combined) else float("inf")

        deltas: list[float] = []
        prev = 0.0
        for s in iteration_scores:
            deltas.append(s - prev)
            prev = s

        return fit, best_vec, deltas
    except Exception:
        return float("inf"), [], []


# ---------------------------------------------------------------------------
# Evolution class
# ---------------------------------------------------------------------------

class CPPNBrainEvolution:
    def __init__(self) -> None:
        self.config = EASettings(
            is_maximisation=False,
            num_steps=BUDGET,
            target_population_size=POP_SIZE,
            output_folder=DATA,
            db_file_name=f"database_{int(time.time())}.db",
            db_handling="delete",
        )

    # --- Genome helpers -----------------------------------------------------

    def _create_random_genome(self) -> Genome:
        g = Genome.random(
            num_inputs=NUM_CPPN_INPUTS,
            num_outputs=NUM_CPPN_OUTPUTS,
            next_node_id=(NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS),
            next_innov_id=0,
        )
        for _ in range(3):
            g.mutate(0.6, 0.6, id_manager.get_next_innov_id, id_manager.get_next_node_id)
        return g

    def _mutate(self, genome: Genome) -> Genome:
        g = genome.copy()
        g.mutate(0.2, 0.3, id_manager.get_next_innov_id, id_manager.get_next_node_id)
        return g

    def _crossover(self, a: Genome, b: Genome) -> Genome:
        return a.crossover(b, is_maximisation=False)

    # --- EA operations ------------------------------------------------------

    def create_individual(self) -> Individual:
        ind = Individual()
        ind.genotype = {"cppn": self._create_random_genome().to_dict()}
        ind.tags = {"ps": False, "valid": True, "last_brain": [], "learning_deltas": []}
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
            if len(parents) >= 2 and RNG.random() < 0.5:
                p1, p2 = random.sample(parents, 2)
                g1 = Genome.from_dict(p1.genotype["cppn"])
                g1.fitness = p1.fitness if p1.fitness is not None else 0.0
                g2 = Genome.from_dict(p2.genotype["cppn"])
                g2.fitness = p2.fitness if p2.fitness is not None else 0.0
                child_genome = self._crossover(g1, g2)
            else:
                parent = random.choice(parents)
                child_genome = Genome.from_dict(parent.genotype["cppn"])

            child_genome = self._mutate(child_genome)
            ind = Individual()
            ind.genotype = {"cppn": child_genome.to_dict()}
            ind.tags = {"ps": False, "valid": True, "last_brain": [], "learning_deltas": []}
            ind.requires_eval = True
            offspring.append(ind)

        population.extend(offspring)
        return population

    def evaluate(self, population: Population) -> Population:
        to_eval = [
            ind for ind in population
            if ind.alive and ind.tags.get("valid") and ind.requires_eval
        ]
        if not to_eval:
            return population

        tasks = [
            (
                ind.genotype["cppn"],
                DURATION,
                EVAL_DELAY,
                LEARN_BUDGET,
                LEARN_POP,
                Z_PENALTY_WEIGHT,
                LOCO_WEIGHT,
                MORPHO_WEIGHT,
            )
            for ind in to_eval
        ]

        if EVAL_WORKERS == 1:
            for ind, task in track(
                zip(to_eval, tasks), total=len(to_eval), description="Learning + Evaluating..."
            ):
                fit, best_vec, deltas = _evaluate_individual_process(task)
                ind.fitness = fit
                ind.tags["last_brain"] = best_vec
                ind.tags["learning_deltas"] = deltas
                ind.requires_eval = False
            return population

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=EVAL_WORKERS, mp_context=ctx) as executor:
            future_to_ind = {
                executor.submit(_evaluate_individual_process, task): ind
                for ind, task in zip(to_eval, tasks)
            }
            for fut in as_completed(future_to_ind):
                ind = future_to_ind[fut]
                try:
                    fit, best_vec, deltas = fut.result()
                except Exception:
                    fit, best_vec, deltas = float("inf"), [], []
                ind.fitness = fit
                ind.tags["last_brain"] = best_vec
                ind.tags["learning_deltas"] = deltas
                ind.requires_eval = False

        return population

    def survivor_selection(self, population: Population) -> Population:
        population = population.sort(sort="min", attribute="fitness_")
        survivors = population[: self.config.target_population_size]
        for ind in population:
            if ind not in survivors:
                ind.alive = False

        finite = [
            ind.fitness_
            for ind in survivors
            if ind.fitness_ is not None and np.isfinite(ind.fitness_)
        ]
        if finite:
            console.log(
                "[green]Survivors:[/green] "
                f"avg={np.mean(finite):.3f}, min={np.min(finite):.3f}, max={np.max(finite):.3f}",
            )
        return population

    # --- Post-evolution visualization / video --------------------------------

    def _decode_best(self, best: Individual) -> tuple[mujoco.MjModel, mujoco.MjData, object, object]:
        """Decode best individual for visualization. Uses SimpleFlatWorldWithTarget."""
        cppn = Genome.from_dict(best.genotype["cppn"])
        graph = MorphologyDecoderBestFirst(cppn, NUM_MODULES).decode()
        if graph.number_of_nodes() == 0:
            raise ValueError("Empty graph for best individual")
        spec = construct_mjspec_from_graph(graph).spec
        world = SimpleFlatWorldWithTarget()
        world.spawn(spec, position=SPAWN_POSITION)
        model = world.spec.compile()
        data = mujoco.MjData(model)
        return world, model, data

    def run_best(self, best: Individual, duration: float = 10.0) -> None:
        mujoco.set_mjcb_control(None)
        try:
            try:
                world, model, data = self._decode_best(best)
            except Exception as exc:
                console.log(f"[red]Could not decode best morphology: {exc}[/red]")
                return

            if model.nu == 0:
                console.log("[red]No actuators on best morphology.[/red]")
                return

            net = Network(
                input_size=len(get_state_from_data(data)) + 2,
                output_size=model.nu,
                hidden_size=16,
            )
            brain_vec = best.tags.get("last_brain", [])
            if brain_vec:
                fill_parameters(net, brain_vec)

            tracker = Tracker(name_to_bind="core", observable_attributes=["xpos"], quiet=True)
            tracker.setup(world.spec, data)
            controller = Controller(controller_callback_function=net.forward, tracker=tracker)

            mujoco.mj_resetData(model, data)
            total_duration = duration + EVAL_DELAY

            if sys.platform == "darwin" or not hasattr(mujoco.viewer, "launch_passive"):
                console.log("[yellow]Using active MuJoCo viewer fallback.[/yellow]")
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
                world, model, data = self._decode_best(best)
            except Exception as exc:
                console.log(f"[red]Could not decode best morphology for video: {exc}[/red]")
                return

            if model.nu == 0:
                console.log("[red]No actuators on best morphology (video skipped).[/red]")
                return

            net = Network(
                input_size=len(get_state_from_data(data)) + 2,
                output_size=model.nu,
                hidden_size=16,
            )
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

    # --- Main evolution loop ------------------------------------------------

    def evolve(self) -> Individual | None:
        console.log("Initializing CPPN population...")
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    console.rule("[bold magenta]CPPN Body + Brain Joint Evolution (Multiprocessing)[/bold magenta]")
    console.log(
        f"Pop={POP_SIZE}, Gens={BUDGET}, LearnBudget={LEARN_BUDGET}, LearnPop={LEARN_POP}, "
        f"EvalWorkers={EVAL_WORKERS}, Dur={DURATION}s, Delay={EVAL_DELAY}s, "
        f"ZPenalty={Z_PENALTY_WEIGHT}, LocoW={LOCO_WEIGHT}, MorphoW={MORPHO_WEIGHT}",
    )

    evo = CPPNBrainEvolution()
    start = time.time()
    best = evo.evolve()
    elapsed = time.time() - start

    if best is None:
        console.log("[red]No best individual found.[/red]")
        return

    best_cppn = Genome.from_dict(best.genotype["cppn"])
    best_graph = MorphologyDecoderBestFirst(best_cppn, NUM_MODULES).decode()
    best_brain = best.tags.get("last_brain", [])
    timestamp = int(time.time())

    genome_path = DATA / f"best_cppn_genome_{timestamp}.json"
    brain_path = DATA / f"best_brain_{timestamp}.npy"
    genome_path.write_text(json.dumps(best_cppn.to_dict(), indent=2), encoding="utf-8")
    np.save(brain_path, np.asarray(best_brain, dtype=np.float32))

    console.rule("[bold green]Final Best[/bold green]")
    console.log(f"Best combined fitness: {best.fitness:.4f}")
    console.log(f"Elapsed: {elapsed:.2f}s")
    console.log(f"Modules decoded: {best_graph.number_of_nodes()}")
    console.log(f"Saved CPPN genome to: {genome_path}")
    console.log(f"Saved brain to: {brain_path}")

    if args.save_video:
        evo.save_best_video(best, duration=args.video_duration)

    if args.visualize:
        evo.run_best(best, duration=10.0)


if __name__ == "__main__":
    main()
