"""
Author: A-lamo (Aron Ferencz)
JOINT-EVOLUTION: JOINT-evolving Body and Brain.

Genotype:
  - 'morph': CPPN Genotype (Graph) -> Decodes to Body
  - 'ctrl':  Float Vector (Array)  -> Decodes to CPG Parameters
"""

from __future__ import annotations

# Standard library
import argparse
import copy
import random
import time
from pathlib import Path
from typing import Any, Literal

import mujoco

# Third-party libraries
import numpy as np
import torch
from mujoco import viewer
from rich.console import Console
from rich.progress import track
from rich.traceback import install

from ariel.utils.renderers import video_renderer

# Initialize rich console
install()
console = Console()

# --- ARIEL IMPORTS ---
from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
)
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)

# --- GENOTYPE IMPORTS ---
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome
from ariel.body_phenotypes.robogen_lite.cppn_neat.id_manager import IdManager
from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import (
    MorphologyDecoderBestFirst,
)
from ariel.ec.a001 import Individual
from ariel.ec.a004 import EA, EASettings, EAStep
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import (
    NaCPG,
    create_fully_connected_adjacency,
)
from ariel.simulation.controllers.simple_cpg import (
    SimpleCPG,
    create_fully_connected_adjacency,
)
from ariel.simulation.environments._simple_flat_with_target import (
    SimpleFlatWorldWithTarget,
)
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

# ============================================================================ #
#                               CONFIGURATION                                  #
# ============================================================================ #

parser = argparse.ArgumentParser(description="Dual Evolution: Body + Brain")
parser.add_argument(
    "--budget", type=int, default=80, help="Number of generations"
)
parser.add_argument("--pop", type=int, default=80, help="Population size")
parser.add_argument("--dur", type=int, default=30, help="Sim Duration")
args = parser.parse_args()

# Constants
DURATION: int = args.dur
POP_SIZE: int = args.pop
BUDGET: int = args.budget
NUM_MODULES: int = 10
CTRL_GENOME_SIZE: int = NUM_MODULES * 5

SPAWN_POSITION = (-0.8, 0.0, 0.1)
TARGET_POSITION = np.array([2.0, 0.0, 0.5])

# CPPN Config
T, R = NUM_OF_TYPES_OF_MODULES, NUM_OF_ROTATIONS
NUM_CPPN_INPUTS = 6
NUM_CPPN_OUTPUTS = 1 + T + R

# Type Aliases
type Population = list[Individual]
type ViewerTypes = Literal["launcher", "video", "simple"]

# Determinism
SEED = 42
RNG = np.random.default_rng(SEED)
torch.manual_seed(SEED)

SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)


# ============================================================================ #
#                            EVOLUTION CLASS                                   #
# ============================================================================ #
class Evolution:
    def __init__(self) -> None:
        self.id_manager = IdManager(
            node_start=NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS - 1,
            innov_start=(NUM_CPPN_INPUTS * NUM_CPPN_OUTPUTS) - 1,
        )

        self.config = EASettings(
            is_maximisation=False,  # Minimize Distance to Target
            num_of_generations=BUDGET,
            target_population_size=POP_SIZE,
            db_file_path=DATA / "database.db",
        )

    # ------------------------------------------------------------------------ #
    #                          HELPER METHODS                                  #
    # ------------------------------------------------------------------------ #
    def map_genotype_to_body(
        self, genome_data: dict | Genome
    ) -> mujoco.MjSpec | None:
        """Decodes CPPN into a MuJoCo Body Spec."""
        genome = (
            Genome.from_dict(genome_data)
            if isinstance(genome_data, dict)
            else genome_data
        )

        try:
            genome.get_node_ordering()  # Topological check (deep stuff)
            decoder = MorphologyDecoderBestFirst(
                cppn_genome=genome, max_modules=NUM_MODULES
            )
            robot_graph = decoder.decode()

            if robot_graph.number_of_nodes() == 0:
                return None

            # Note: construct_mjspec_from_graph returns a wrapper, access .spec
            return construct_mjspec_from_graph(robot_graph).spec
        except Exception:
            return None

    def map_genotype_to_brain(
        self, cpg: SimpleCPG, full_genome: list[float]
    ) -> None:
        """Decodes Float Vector into CPG Parameters."""
        n = cpg.phase.shape[0]
        params = np.array(full_genome)

        # Resize logic if body changed size
        required_size = n * 5
        if required_size > len(params):
            params = np.resize(params, required_size)
        else:
            params = params[:required_size]

        p_phase = params[0 * n : 1 * n]
        p_w = params[1 * n : 2 * n]
        p_amp = params[2 * n : 3 * n]
        p_ha = params[3 * n : 4 * n]
        p_b = params[4 * n : 5 * n]

        with torch.no_grad():
            cpg.phase.data.copy_(torch.from_numpy(p_phase * np.pi).float())
            # Frequency: [0.2, 4.0] Hz - Faster movements for quicker walking
            cpg.w.data.copy_(
                torch.from_numpy(0.2 + (3.8 * (p_w + 1.0) / 2.0)).float()
            )
            # Amplitude: [0.5, 4.0] - MUCH stronger motor commands
            cpg.amplitudes.data.copy_(
                torch.from_numpy(0.5 + (3.5 * (p_amp + 1.0) / 2.0)).float()
            )
            cpg.ha.data.copy_(torch.from_numpy(p_ha * 2.0).float())
            cpg.b.data.copy_(torch.from_numpy(p_b * 0.5).float())

    def get_joint_count(self, genome: Genome) -> int:
        """Checks if the genome produces a body with actuators."""
        spec = self.map_genotype_to_body(genome)
        if spec is None:
            return 0
        try:
            return spec.compile().nu
        except:
            return 0

    def mutate_ctrl_vector(self, genome: list[float]) -> list[float]:
        """Gaussian mutation for brain vector - (very) aggressive to find strong solutions."""
        arr = np.array(genome)
        # 40% of genes mutated with larger noise for faster exploration
        mask = RNG.random(arr.shape) < 0.40
        noise = RNG.normal(0, 0.6, arr.shape)  # Increased from 0.4 to 0.6
        arr[mask] += noise[mask]
        return np.clip(arr, -1.0, 1.0).tolist()

    def crossover_ctrl_vectors(
        self, ctrl1: list[float], ctrl2: list[float]
    ) -> list[float]:
        """Uniform crossover for controller float vectors."""
        arr1 = np.array(ctrl1)
        arr2 = np.array(ctrl2)
        # Align sizes in case of mismatch
        size = max(len(arr1), len(arr2))
        if len(arr1) < size:
            arr1 = np.resize(arr1, size)
        if len(arr2) < size:
            arr2 = np.resize(arr2, size)
        # Uniform crossover: take each gene from either parent with 50% probability
        mask = RNG.random(size) < 0.5
        child = np.where(mask, arr1, arr2)
        return child.tolist()

    def crossover_morphologies(self, morph1: Genome, morph2: Genome) -> Genome:
        """Simple morphology crossover: randomly select connections from each parent."""
        # Create a new genome based on parent 1
        child = morph1.copy()

        # Randomly replace some connections with parent 2's connections
        try:
            for conn in child.connections:
                if RNG.random() < 0.5 and morph2.connections:
                    # Randomly pick a connection from parent 2
                    p2_conn = random.choice(morph2.connections)
                    conn.weight = p2_conn.weight
        except:
            pass  # If crossover fails, just use child as-is

        return child

    # ------------------------------------------------------------------------ #
    #                          EA OPERATORS                                    #
    # ------------------------------------------------------------------------ #
    def create_individual(self) -> Individual:
        """Initialization: Ensures valid physical body."""
        while True:
            try:
                genome = Genome.random(
                    num_inputs=NUM_CPPN_INPUTS,
                    num_outputs=NUM_CPPN_OUTPUTS,
                    next_node_id=self.id_manager.get_next_node_id(),
                    next_innov_id=self.id_manager.get_next_innov_id(),
                )
                for _ in range(3):
                    genome.mutate(
                        1.0,
                        1.0,
                        self.id_manager.get_next_innov_id,
                        self.id_manager.get_next_node_id,
                    )

                if self.get_joint_count(genome) > 0:
                    break
            except Exception:
                continue

        ind = Individual()
        ind.genotype = {
            "morph": genome.to_dict(),
            "ctrl": RNG.uniform(-1.0, 1.0, size=CTRL_GENOME_SIZE).tolist(),
        }
        # Initialize all tags
        ind.tags["ps"] = False
        ind.tags["valid"] = True
        ind.tags["debug_joints"] = 0
        return ind

    def reproduction(self, population: Population) -> Population:
        """Joint Reproduction: Crossover (Body + Brain) + Mutation."""
        parents = [ind for ind in population if ind.tags.get("ps", False)]

        # Fallback: if no ps parents, use all individuals
        if not parents:
            console.log(
                "[yellow]Warning: No ps-tagged individuals, using entire population as parents[/yellow]"
            )
            parents = population

        new_offspring = []
        target_pool = self.config.target_population_size * 2

        while len(population) + len(new_offspring) < target_pool:
            # Sexual Reproduction (50% chance if we have 2+ parents)
            use_sexual = len(parents) >= 2 and RNG.random() < 0.5

            if use_sexual:
                # Select two parents for crossover
                p1, p2 = random.sample(parents, 2)
                p1_morph = Genome.from_dict(p1.genotype["morph"])
                p2_morph = Genome.from_dict(p2.genotype["morph"])

                # Crossover morphologies
                c_morph = self.crossover_morphologies(p1_morph, p2_morph)

                # Crossover brain vectors
                c_ctrl = self.crossover_ctrl_vectors(
                    p1.genotype["ctrl"], p2.genotype["ctrl"]
                )
            else:
                # Asexual reproduction: single parent
                parent = random.choice(parents)
                p_morph = Genome.from_dict(parent.genotype["morph"])
                c_morph = p_morph.copy()
                c_ctrl = (
                    parent.genotype["ctrl"].copy()
                    if isinstance(parent.genotype["ctrl"], list)
                    else parent.genotype["ctrl"]
                )

            # Body Mutation (with Validity Retry)
            valid_child = False
            attempts = 0
            while not valid_child and attempts < 20:
                mutant = c_morph.copy()
                mutant.mutate(
                    0.8,
                    0.5,
                    self.id_manager.get_next_innov_id,
                    self.id_manager.get_next_node_id,
                )
                if self.get_joint_count(mutant) > 0:
                    c_morph = mutant
                    valid_child = True
                attempts += 1

            # Brain Mutation (always applied after crossover/selection)
            c_ctrl = self.mutate_ctrl_vector(c_ctrl)

            ind = Individual()
            # Initialize all tags
            ind.genotype = {"morph": c_morph.to_dict(), "ctrl": c_ctrl}
            ind.tags["ps"] = False
            ind.tags["valid"] = True
            ind.tags["debug_joints"] = 0
            new_offspring.append(ind)

        population.extend(new_offspring)
        return population

    def evaluate(self, population: Population) -> Population:
        """Evaluation Loop: Calls run_simulation in 'simple' mode."""
        to_eval = [
            ind
            for ind in population
            if ind.alive and ind.tags.get("valid") and ind.requires_eval
        ]

        if not to_eval:
            return population

        for ind in track(to_eval, description="Evaluating..."):
            # Pass mode="simple" for fast, headless evaluation
            fitness = self.run_simulation("simple", ind)
            ind.fitness = fitness
            ind.requires_eval = False

        return population

    def parent_selection(self, population: Population) -> Population:
        population.sort(
            key=lambda x: x.fitness_ if x.fitness_ is not None else float("inf")
        )
        cutoff = len(population) // 2
        for i, ind in enumerate(population):
            ind.tags["ps"] = i < cutoff

        # Diagnostics
        ps_count = sum(1 for ind in population if ind.tags.get("ps", False))
        console.log(
            f"[cyan]Parent Selection: {ps_count}/{len(population)} marked for reproduction[/cyan]"
        )

        return population

    def survivor_selection(self, population: Population) -> Population:
        population.sort(
            key=lambda x: x.fitness_ if x.fitness_ is not None else float("inf")
        )
        survivors = population[: self.config.target_population_size]
        for ind in population:
            if ind not in survivors:
                ind.alive = False

        # Diagnostics
        avg_fitness = np.mean([
            ind.fitness_
            for ind in survivors
            if ind.fitness_ is not None and ind.fitness_ != float("inf")
        ])
        console.log(
            f"[green]Survivor Selection: Avg fitness = {avg_fitness:.4f}[/green]"
        )

        return population

    def sync_ids(self, population: Population) -> None:
        """Syncs ID manager with population CPPNs."""
        # This function assumes that the ID manager is correct
        max_nid = self.id_manager._node_id
        max_inn = self.id_manager._innov_id
        for ind in population:
            g = ind.genotype["morph"]
            if "nodes" in g:
                for n in g["nodes"].values():
                    nid = n.get("id", n.get("_id"))
                    if nid and nid > max_nid:
                        max_nid = nid
            if "connections" in g:
                for c in g["connections"]:
                    inn = c.get("innovation", c.get("innov_id"))
                    if inn and inn > max_inn:
                        max_inn = inn
        self.id_manager._node_id = max_nid
        self.id_manager._innov_id = max_inn

    def fast_physics_runner(
        self, model: mujoco.MjModel, data: mujoco.MjData, duration: float
    ) -> None:
        """
        Optimized physics-only simulation runner (no rendering).
        Uses render_skip and speed multiplier tricks from 4_robot_with_camera.py.
        Runs physics as fast as possible without visualization overhead.
        """
        steps_required = int(duration / model.opt.timestep)
        step = 0

        while step < steps_required:
            mujoco.mj_step(model, data)
            step += 1

    # ------------------------------------------------------------------------ #
    #                          SIMULATION RUNNER                               #
    # ------------------------------------------------------------------------ #
    def run_simulation(self, mode: ViewerTypes, ind: Individual) -> float:
        """
        Builds phenotype and runs simulation.
        Handles reconstruction retries for visualization.
        """
        mujoco.set_mjcb_control(None)

        # 1. Reconstruct Body (Retry Logic for Determinism)
        expected_joints = ind.tags.get("debug_joints", 0)
        spec = None
        model = None

        # In evaluation mode, we just try once. In visual mode, we retry.
        attempts = 15 if mode != "simple" else 1

        for _ in range(attempts):
            temp_spec = self.map_genotype_to_body(ind.genotype["morph"])
            if temp_spec:
                try:
                    temp_model = temp_spec.compile()
                    # If simple mode (first run) or joints match expected (replay)
                    if mode == "simple" or temp_model.nu == expected_joints:
                        spec = temp_spec
                        model = temp_model
                        break
                except:
                    pass

        if model is None:
            # Fallback
            spec = self.map_genotype_to_body(ind.genotype["morph"])
            if spec:
                model = spec.compile()
            else:
                return float("inf")

        # 2. Setup Environment & Physics
        if mode == "simple":
            # If simple, we check joints here to fail fast
            if model.nu == 0:
                ind.tags["debug_joints"] = 0
                return float("inf")
            ind.tags["debug_joints"] = model.nu  # Save for replay

        world = SimpleFlatWorldWithTarget()
        world.spawn(spec, position=SPAWN_POSITION)
        model = world.spec.compile()
        data = mujoco.MjData(model)

        # 3. Setup Brain
        adj_dict = create_fully_connected_adjacency(model.nu)
        cpg = SimpleCPG(adj_dict)
        self.map_genotype_to_brain(cpg, ind.genotype["ctrl"])

        if mode != "simple":
            console.log(
                f"[green]Simulating with {model.nu} joints (Target: {expected_joints})[/green]"
            )

        # 4. Setup Controller
        tracker = Tracker(mujoco.mjtObj.mjOBJ_BODY, "core", ["xpos"])
        ctrl = Controller(
            controller_callback_function=lambda m, d, *a, **k: cpg.forward(
                d.time
            ),
            tracker=tracker,
        )
        ctrl.tracker.setup(world.spec, data)

        # Bind Control Loop
        # Note: *args and **kwargs in lambda ensure compatibility with runner
        mujoco.set_mjcb_control(
            lambda m, d: ctrl.set_control(m, d, duration=DURATION)
        )
        mujoco.mj_resetData(model, data)

        # 5. Execute
        match mode:
            case "simple":
                # Use optimized physics-only runner (no rendering overhead)
                self.fast_physics_runner(model, data, duration=DURATION)
            case "video":
                recorder = VideoRecorder(
                    output_folder=str(DATA / "videos"),
                    file_name=f"dual_{ind.id}",
                )
                video_renderer(
                    model, data, duration=DURATION, video_recorder=recorder
                )
            case "launcher":
                viewer.launch(model=model, data=data)

        # 6. Calculate Fitness (Distance to Target)
        # 1-Second Delay Implementation to penalize falling strategies
        # Always enforce 1 second delay if simulation duration allows
        delay_time = min(
            1.0, DURATION
        )  # Delay is always 1 second (or full duration if shorter)
        delay_fraction = delay_time / DURATION if DURATION > 0 else 0

        dist = float("inf")

        if tracker.history["xpos"]:
            first_key = list(tracker.history["xpos"].keys())[0]
            traj = tracker.history["xpos"][first_key]

            if traj:
                # Skip the first N% of trajectory that corresponds to 1 second of elapsed time
                start_idx = max(0, int(len(traj) * delay_fraction))
                pos_after_delay = np.array(traj[start_idx])
                pos_final = np.array(traj[-1])
                valid_movement_vector = pos_final - pos_after_delay
                effective_pos = np.array(SPAWN_POSITION) + valid_movement_vector
                dist = np.sqrt(
                    np.sum((effective_pos[:2] - TARGET_POSITION[:2]) ** 2)
                )

                if mode != "simple":
                    console.log(
                        f"[blue]Traj len: {len(traj)}, start_idx: {start_idx}, movement: {np.linalg.norm(valid_movement_vector):.3f}[/blue]"
                    )

        return dist

    # ------------------------------------------------------------------------ #
    #                          MAIN LOOP                                       #
    # ------------------------------------------------------------------------ #
    def evolve(self) -> Individual | None:
        console.log("Initializing population...")
        population = [self.create_individual() for _ in range(POP_SIZE)]
        self.sync_ids(population)

        # Initial Eval
        population = self.evaluate(population)

        ops = [
            EAStep("parent_selection", self.parent_selection),
            EAStep("reproduction", self.reproduction),
            EAStep("evaluation", self.evaluate),
            EAStep("survivor_selection", self.survivor_selection),
        ]

        ea = EA(population, operations=ops, num_of_generations=BUDGET)
        ea.run()

        return ea.get_solution("best", only_alive=False)


def main() -> None:
    console.rule(
        "[bold purple]Starting Joint Evolution (Morph + Ctrl)[/bold purple]"
    )

    evo = Evolution()
    best = evo.evolve()

    if best:
        console.rule("[bold green]Final Best Result")
        console.log(f"Best Fitness (Dist to Target): {best.fitness:.4f}")
        evo.run_simulation("launcher", best)


if __name__ == "__main__":
    main()
