"""
Author: A-lamo (Aron Ferencz)
JOINT-EVOLUTION: JOINT-evolving Body and Brain.

Genotype:
  - 'morph': CPPN Genotype (Graph) -> Decodes to Body
  - 'ctrl':  Float Vector (Array)  -> Decodes to CPG Parameters
"""

# Standard library
import argparse
import csv
import random
from pathlib import Path
from typing import Literal

# Third-party libraries
import mujoco
import numpy as np
import torch
from mujoco import viewer
from rich.console import Console
from rich.progress import track
from rich.traceback import install

# --- ARIEL IMPORTS ---
from ariel.body_phenotypes.robogen_lite.collision_validation import (
    is_physically_valid,
)
from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
    ModuleType,
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
from ariel.ec import (
    EA,
    EAOperation,
    EASettings,
    Individual,
    Population,
)
from ariel.parameters.ariel_modules import ArielModulesConfig
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.simple_cpg import (
    SimpleCPG,
    create_fully_connected_adjacency,
)
from ariel.simulation.environments._simple_flat_with_target import (
    SimpleFlatWorldWithTarget,
)
from ariel.utils.renderers import video_renderer
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder


# Initialize rich console
install()
console = Console()


# ============================================================================ #
#                               CONFIGURATION                                  #
# ============================================================================ #

ariel_modules_config = ArielModulesConfig()


parser = argparse.ArgumentParser(
    description="Dual Evolution: Body + Brain",
)

parser.add_argument(
    "--budget",
    type=int,
    default=80,
    help="Number of generations",
)

parser.add_argument(
    "--pop",
    type=int,
    default=80,
    help="Population size",
)

parser.add_argument(
    "--dur",
    type=int,
    default=30,
    help="Simulation duration",
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed",
)

parser.add_argument(
    "--bone-mode",
    choices=[
        "fixed",
        "evolvable",
    ],
    default="evolvable",
    help=(
        "Use fixed-length bricks or "
        "evolvable variable-length bricks"
    ),
)

parser.add_argument(
    "--fixed-length",
    type=float,
    default=ariel_modules_config.BRICK_LENGTH_DEFAULT,
    help=(
        "Brick length in meters when "
        "--bone-mode=fixed"
    ),
)

parser.add_argument(
    "--max-modules",
    type=int,
    default=10,
    help="Maximum number of modules",
)

parser.add_argument(
    "--visualize",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Launch MuJoCo viewer for best individual",
)

args = parser.parse_args()


# ============================================================================ #
#                                  CONSTANTS                                   #
# ============================================================================ #

DURATION: int = args.dur
POP_SIZE: int = args.pop
BUDGET: int = args.budget
NUM_MODULES: int = args.max_modules

CTRL_GENOME_SIZE: int = (
    NUM_MODULES
    * 5
)


BONE_MODE = args.bone_mode
FIXED_BRICK_LENGTH = args.fixed_length


if not (
    ariel_modules_config.BRICK_LENGTH_MIN
    <= FIXED_BRICK_LENGTH
    <= ariel_modules_config.BRICK_LENGTH_MAX
):
    parser.error(
        "--fixed-length must be between "
        f"{ariel_modules_config.BRICK_LENGTH_MIN} and "
        f"{ariel_modules_config.BRICK_LENGTH_MAX} meters"
    )


SPAWN_POSITION = (
    -0.8,
    0.0,
    0.1,
)

TARGET_POSITION = np.array([
    2.0,
    0.0,
    0.5,
])


# --------------------------------------------------------------------------- #
# CPPN CONFIGURATION
# --------------------------------------------------------------------------- #

T = NUM_OF_TYPES_OF_MODULES
R = NUM_OF_ROTATIONS

NUM_CPPN_INPUTS = 6

# CPPN outputs:
#
# 0
#   Connection score
#
# next T
#   Module type scores
#
# next R
#   Rotation scores
#
# final output
#   Continuous brick length
#
NUM_CPPN_OUTPUTS = (
    1
    + T
    + R
    + 1
)


# --------------------------------------------------------------------------- #
# TYPE ALIASES
# --------------------------------------------------------------------------- #

type ViewerTypes = Literal[
    "launcher",
    "video",
    "simple",
]


# --------------------------------------------------------------------------- #
# DETERMINISM
# --------------------------------------------------------------------------- #

SEED = args.seed

RNG = np.random.default_rng(
    SEED
)

torch.manual_seed(
    SEED
)

random.seed(
    SEED
)


# --------------------------------------------------------------------------- #
# DATA SETUP
# --------------------------------------------------------------------------- #

SCRIPT_NAME = Path(
    __file__
).stem

CWD = Path.cwd()

DATA = (
    CWD
    / "__data__"
    / SCRIPT_NAME
    / f"{BONE_MODE}_seed_{SEED}"
)

DATA.mkdir(
    exist_ok=True,
    parents=True,
)


LOG_FILE = (
    DATA
    / "variable_bones.csv"
)


# Remove results from an earlier run with
# the same mode and seed.
#
# This prevents CSV rows from multiple smoke
# tests being mixed together.
if LOG_FILE.exists():
    LOG_FILE.unlink()


# ============================================================================ #
#                            EVOLUTION CLASS                                   #
# ============================================================================ #


class Evolution:
    def __init__(
        self,
    ) -> None:
        self.id_manager = IdManager(
            node_start=(
                NUM_CPPN_INPUTS
                + NUM_CPPN_OUTPUTS
                - 1
            ),
            innov_start=(
                NUM_CPPN_INPUTS
                * NUM_CPPN_OUTPUTS
            )
            - 1,
        )

        self.config = EASettings(
            is_maximisation=False,
            num_steps=BUDGET,
            target_population_size=POP_SIZE,
            output_folder=DATA,
            db_file_name="database.db",
        )

        # 0 = initial population.
        self.evaluation_round = 0

        # Used when Individual.id has not yet
        # been assigned by the EA/database.
        self.evaluation_counter = 0


    # ======================================================================== #
    #                    MORPHOLOGY / BONE HELPERS                             #
    # ======================================================================== #

    def apply_bone_mode(
        self,
        graph,
    ) -> None:
        """Apply the selected variable-bone experiment condition.

        In evolvable mode, CPPN-generated brick lengths remain unchanged.

        In fixed mode, all brick lengths are overwritten with
        FIXED_BRICK_LENGTH.

        The CPPN still contains the length output in both cases so that
        fixed and evolvable experiments use the same network architecture.
        """
        if (
            BONE_MODE
            != "fixed"
        ):
            return

        for _, node_data in graph.nodes(
            data=True
        ):
            if (
                node_data["type"]
                == ModuleType.BRICK.name
            ):
                node_data["length"] = (
                    FIXED_BRICK_LENGTH
                )


    def decode_morphology_graph(
        self,
        genome_data: dict | Genome,
    ):
        """Decode CPPN morphology and check physical validity."""
        genome = (
            Genome.from_dict(
                genome_data
            )
            if isinstance(
                genome_data,
                dict,
            )
            else genome_data
        )

        try:
            # Ensure the CPPN can be evaluated.
            genome.get_node_ordering()

            decoder = (
                MorphologyDecoderBestFirst(
                    cppn_genome=genome,
                    max_modules=NUM_MODULES,
                )
            )

            robot_graph = (
                decoder.decode()
            )

            if (
                robot_graph.number_of_nodes()
                == 0
            ):
                return None

            # Apply fixed/evolvable experiment.
            self.apply_bone_mode(
                robot_graph
            )

            # Reject geometric self-intersections.
            if not is_physically_valid(
                robot_graph
            ):
                return None

            return robot_graph

        except Exception:
            return None


    def get_morphology_statistics(
        self,
        graph,
    ) -> dict:
        """Calculate morphology and variable-bone statistics."""
        brick_lengths = [
            float(
                node_data[
                    "length"
                ]
            )
            for _, node_data
            in graph.nodes(
                data=True
            )
            if (
                node_data["type"]
                == ModuleType.BRICK.name
                and "length"
                in node_data
            )
        ]


        num_hinges = sum(
            1
            for _, node_data
            in graph.nodes(
                data=True
            )
            if (
                node_data["type"]
                == ModuleType.HINGE.name
            )
        )


        if brick_lengths:
            lengths = np.asarray(
                brick_lengths,
                dtype=float,
            )

            mean_length = float(
                np.mean(
                    lengths
                )
            )

            min_length = float(
                np.min(
                    lengths
                )
            )

            max_length = float(
                np.max(
                    lengths
                )
            )

            std_length = float(
                np.std(
                    lengths
                )
            )

        else:
            mean_length = float(
                "nan"
            )

            min_length = float(
                "nan"
            )

            max_length = float(
                "nan"
            )

            std_length = float(
                "nan"
            )


        return {
            "num_modules": (
                graph.number_of_nodes()
            ),
            "num_bricks": (
                len(
                    brick_lengths
                )
            ),
            "num_hinges": (
                num_hinges
            ),
            "mean_length": (
                mean_length
            ),
            "min_length": (
                min_length
            ),
            "max_length": (
                max_length
            ),
            "std_length": (
                std_length
            ),
        }


    def log_evaluation(
        self,
        generation: int,
        ind: Individual,
        graph,
        fitness: float,
    ) -> None:
        """Log locomotion fitness and morphology statistics."""
        stats = (
            self.get_morphology_statistics(
                graph
            )
        )


        # Individual IDs may not yet have
        # been allocated by the EA.
        individual_id = getattr(
            ind,
            "id",
            None,
        )

        if individual_id is None:
            individual_id = (
                self.evaluation_counter
            )


        file_exists = (
            LOG_FILE.exists()
        )


        with open(
            LOG_FILE,
            "a",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "generation",
                    "individual_id",
                    "seed",
                    "fitness",
                    "bone_mode",
                    "fixed_length_mm",
                    "num_modules",
                    "num_bricks",
                    "num_hinges",
                    "mean_length_mm",
                    "min_length_mm",
                    "max_length_mm",
                    "std_length_mm",
                ],
            )


            if not file_exists:
                writer.writeheader()


            writer.writerow({
                "generation": (
                    generation
                ),
                "individual_id": (
                    individual_id
                ),
                "seed": (
                    SEED
                ),
                "fitness": (
                    fitness
                ),
                "bone_mode": (
                    BONE_MODE
                ),
                "fixed_length_mm": (
                    FIXED_BRICK_LENGTH
                    * 1000
                    if (
                        BONE_MODE
                        == "fixed"
                    )
                    else ""
                ),
                "num_modules": (
                    stats[
                        "num_modules"
                    ]
                ),
                "num_bricks": (
                    stats[
                        "num_bricks"
                    ]
                ),
                "num_hinges": (
                    stats[
                        "num_hinges"
                    ]
                ),
                "mean_length_mm": (
                    stats[
                        "mean_length"
                    ]
                    * 1000
                ),
                "min_length_mm": (
                    stats[
                        "min_length"
                    ]
                    * 1000
                ),
                "max_length_mm": (
                    stats[
                        "max_length"
                    ]
                    * 1000
                ),
                "std_length_mm": (
                    stats[
                        "std_length"
                    ]
                    * 1000
                ),
            })


        self.evaluation_counter += 1


    # ======================================================================== #
    #                          BODY / BRAIN HELPERS                             #
    # ======================================================================== #

    def map_genotype_to_body(
        self,
        genome_data: dict | Genome,
    ) -> mujoco.MjSpec | None:
        """Decode CPPN into a physically valid MuJoCo body specification."""
        robot_graph = (
            self.decode_morphology_graph(
                genome_data
            )
        )

        if robot_graph is None:
            return None


        try:
            return (
                construct_mjspec_from_graph(
                    robot_graph
                ).spec
            )

        except Exception:
            return None


    def map_genotype_to_brain(
        self,
        cpg: SimpleCPG,
        full_genome: list[float],
    ) -> None:
        """Decode float vector into CPG parameters."""
        n = cpg.phase.shape[
            0
        ]


        params = np.array(
            full_genome
        )


        required_size = (
            n
            * 5
        )


        if (
            required_size
            > len(params)
        ):
            params = np.resize(
                params,
                required_size,
            )

        else:
            params = params[
                :required_size
            ]


        p_phase = params[
            0 * n : 1 * n
        ]

        p_w = params[
            1 * n : 2 * n
        ]

        p_amp = params[
            2 * n : 3 * n
        ]

        p_ha = params[
            3 * n : 4 * n
        ]

        p_b = params[
            4 * n : 5 * n
        ]


        with torch.no_grad():
            cpg.phase.data.copy_(
                torch.from_numpy(
                    p_phase
                    * np.pi
                ).float()
            )


            # Frequency:
            # [0.2, 4.0] Hz
            cpg.w.data.copy_(
                torch.from_numpy(
                    0.2
                    + (
                        3.8
                        * (
                            p_w
                            + 1.0
                        )
                        / 2.0
                    )
                ).float()
            )


            # Amplitude:
            # [0.5, 4.0]
            cpg.amplitudes.data.copy_(
                torch.from_numpy(
                    0.5
                    + (
                        3.5
                        * (
                            p_amp
                            + 1.0
                        )
                        / 2.0
                    )
                ).float()
            )


            cpg.ha.data.copy_(
                torch.from_numpy(
                    p_ha
                    * 2.0
                ).float()
            )


            cpg.b.data.copy_(
                torch.from_numpy(
                    p_b
                    * 0.5
                ).float()
            )


    def get_joint_count(
        self,
        genome: Genome,
    ) -> int:
        """Return actuator count for a physically valid morphology."""
        spec = (
            self.map_genotype_to_body(
                genome
            )
        )


        if spec is None:
            return 0


        try:
            model = (
                spec.compile()
            )

            return int(
                model.nu
            )

        except Exception:
            return 0


    def mutate_ctrl_vector(
        self,
        genome: list[float],
    ) -> list[float]:
        """Gaussian mutation for controller vector."""
        arr = np.array(
            genome
        )


        mask = (
            RNG.random(
                arr.shape
            )
            < 0.40
        )


        noise = RNG.normal(
            0,
            0.6,
            arr.shape,
        )


        arr[
            mask
        ] += noise[
            mask
        ]


        return np.clip(
            arr,
            -1.0,
            1.0,
        ).tolist()


    def crossover_ctrl_vectors(
        self,
        ctrl1: list[float],
        ctrl2: list[float],
    ) -> list[float]:
        """Uniform crossover for controller float vectors."""
        arr1 = np.array(
            ctrl1
        )

        arr2 = np.array(
            ctrl2
        )


        size = max(
            len(arr1),
            len(arr2),
        )


        if (
            len(arr1)
            < size
        ):
            arr1 = np.resize(
                arr1,
                size,
            )


        if (
            len(arr2)
            < size
        ):
            arr2 = np.resize(
                arr2,
                size,
            )


        mask = (
            RNG.random(
                size
            )
            < 0.5
        )


        child = np.where(
            mask,
            arr1,
            arr2,
        )


        return (
            child.tolist()
        )


    def crossover_morphologies(
        self,
        parent1: Individual,
        parent2: Individual,
    ) -> Genome:
        """Crossover two CPPN morphologies using NEAT crossover."""
        morph1 = Genome.from_dict(
            parent1.genotype[
                "morph"
            ]
        )

        morph2 = Genome.from_dict(
            parent2.genotype[
                "morph"
            ]
        )


        morph1.fitness = (
            parent1.fitness
            if (
                parent1.fitness
                is not None
            )
            else float(
                "inf"
            )
        )


        morph2.fitness = (
            parent2.fitness
            if (
                parent2.fitness
                is not None
            )
            else float(
                "inf"
            )
        )


        return morph1.crossover(
            morph2,
            is_maximisation=False,
        )


    # ======================================================================== #
    #                          EA OPERATORS                                    #
    # ======================================================================== #

    def create_individual(
        self,
    ) -> Individual:
        """Create an initial individual with at least one actuated joint."""
        while True:
            try:
                genome = Genome.random(
                    num_inputs=(
                        NUM_CPPN_INPUTS
                    ),
                    num_outputs=(
                        NUM_CPPN_OUTPUTS
                    ),
                    next_node_id=(
                        self.id_manager.get_next_node_id()
                    ),
                    next_innov_id=(
                        self.id_manager.get_next_innov_id()
                    ),
                )


                for _ in range(
                    3
                ):
                    genome.mutate(
                        1.0,
                        1.0,
                        self.id_manager.get_next_innov_id,
                        self.id_manager.get_next_node_id,
                    )


                joint_count = (
                    self.get_joint_count(
                        genome
                    )
                )


                if (
                    joint_count
                    > 0
                ):
                    break


            except Exception:
                continue


        ind = Individual()


        ind.genotype = {
            "morph": (
                genome.to_dict()
            ),
            "ctrl": (
                RNG.uniform(
                    -1.0,
                    1.0,
                    size=CTRL_GENOME_SIZE,
                ).tolist()
            ),
        }


        ind.tags[
            "ps"
        ] = False

        ind.tags[
            "valid"
        ] = True

        ind.tags[
            "debug_joints"
        ] = 0


        return ind


    def reproduction(
        self,
        population: Population,
    ) -> Population:
        """Joint reproduction with guaranteed actuated offspring."""
        parents = [
            ind
            for ind in population
            if ind.tags.get(
                "ps",
                False,
            )
        ]


        if not parents:
            console.log(
                "[yellow]"
                "Warning: No ps-tagged individuals, "
                "using entire population as parents"
                "[/yellow]"
            )

            parents = population


        new_offspring = []

        target_pool = (
            self.config.target_population_size
            * 2
        )


        while (
            len(population)
            + len(new_offspring)
            < target_pool
        ):
            use_sexual = (
                len(parents)
                >= 2
                and RNG.random()
                < 0.5
            )


            # --------------------------------------------------------------- #
            # CREATE BASE CHILD
            # --------------------------------------------------------------- #

            if use_sexual:
                p1, p2 = random.sample(
                    parents,
                    2,
                )


                c_morph = (
                    self.crossover_morphologies(
                        p1,
                        p2,
                    )
                )


                c_ctrl = (
                    self.crossover_ctrl_vectors(
                        p1.genotype[
                            "ctrl"
                        ],
                        p2.genotype[
                            "ctrl"
                        ],
                    )
                )


                # Keep the fitter parent as a
                # known fallback morphology.
                #
                # Lower fitness is better.
                p1_fitness = (
                    p1.fitness
                    if (
                        p1.fitness
                        is not None
                    )
                    else float(
                        "inf"
                    )
                )

                p2_fitness = (
                    p2.fitness
                    if (
                        p2.fitness
                        is not None
                    )
                    else float(
                        "inf"
                    )
                )


                fallback_parent = (
                    p1
                    if (
                        p1_fitness
                        <= p2_fitness
                    )
                    else p2
                )


                fallback_morph = (
                    Genome.from_dict(
                        fallback_parent.genotype[
                            "morph"
                        ]
                    )
                )


            else:
                parent = random.choice(
                    parents
                )


                p_morph = (
                    Genome.from_dict(
                        parent.genotype[
                            "morph"
                        ]
                    )
                )


                c_morph = (
                    p_morph.copy()
                )


                fallback_morph = (
                    p_morph.copy()
                )


                c_ctrl = (
                    parent.genotype[
                        "ctrl"
                    ].copy()
                    if isinstance(
                        parent.genotype[
                            "ctrl"
                        ],
                        list,
                    )
                    else parent.genotype[
                        "ctrl"
                    ]
                )


            # --------------------------------------------------------------- #
            # BODY MUTATION
            # --------------------------------------------------------------- #

            valid_child = False
            attempts = 0


            while (
                not valid_child
                and attempts
                < 20
            ):
                mutant = (
                    c_morph.copy()
                )


                mutant.mutate(
                    0.8,
                    0.5,
                    self.id_manager.get_next_innov_id,
                    self.id_manager.get_next_node_id,
                )


                joint_count = (
                    self.get_joint_count(
                        mutant
                    )
                )


                if (
                    joint_count
                    > 0
                ):
                    c_morph = (
                        mutant
                    )

                    valid_child = (
                        True
                    )


                attempts += 1


            # --------------------------------------------------------------- #
            # FALLBACK IF MUTATION FAILED
            # --------------------------------------------------------------- #

            if not valid_child:
                fallback_joint_count = (
                    self.get_joint_count(
                        fallback_morph
                    )
                )


                if (
                    fallback_joint_count
                    > 0
                ):
                    c_morph = (
                        fallback_morph.copy()
                    )

                    valid_child = (
                        True
                    )


            # Absolute safety:
            # never allow an unactuated morphology
            # to enter the offspring population.
            if (
                not valid_child
            ):
                continue


            final_joint_count = (
                self.get_joint_count(
                    c_morph
                )
            )


            if (
                final_joint_count
                <= 0
            ):
                continue


            # --------------------------------------------------------------- #
            # BRAIN MUTATION
            # --------------------------------------------------------------- #

            c_ctrl = (
                self.mutate_ctrl_vector(
                    c_ctrl
                )
            )


            # --------------------------------------------------------------- #
            # CREATE OFFSPRING INDIVIDUAL
            # --------------------------------------------------------------- #

            ind = Individual()


            ind.genotype = {
                "morph": (
                    c_morph.to_dict()
                ),
                "ctrl": (
                    c_ctrl
                ),
            }


            ind.tags[
                "ps"
            ] = False

            ind.tags[
                "valid"
            ] = True

            ind.tags[
                "debug_joints"
            ] = (
                final_joint_count
            )


            new_offspring.append(
                ind
            )


        population.extend(
            new_offspring
        )


        return population


    def evaluate(
        self,
        population: Population,
    ) -> Population:
        """Evaluate locomotion and log variable-bone statistics."""
        to_eval = [
            ind
            for ind in population
            if (
                ind.alive
                and ind.tags.get(
                    "valid"
                )
                and ind.requires_eval
            )
        ]


        if not to_eval:
            return population


        for ind in track(
            to_eval,
            description="Evaluating...",
        ):
            # --------------------------------------------------------------- #
            # VERIFY BODY BEFORE SIMULATION
            # --------------------------------------------------------------- #

            genome = Genome.from_dict(
                ind.genotype[
                    "morph"
                ]
            )


            joint_count = (
                self.get_joint_count(
                    genome
                )
            )


            if (
                joint_count
                <= 0
            ):
                ind.fitness = float(
                    "inf"
                )

                ind.requires_eval = (
                    False
                )

                console.log(
                    "[yellow]"
                    "Rejected individual with "
                    "zero actuated joints."
                    "[/yellow]"
                )

                continue


            ind.tags[
                "debug_joints"
            ] = (
                joint_count
            )


            # --------------------------------------------------------------- #
            # SIMULATE
            # --------------------------------------------------------------- #

            fitness = (
                self.run_simulation(
                    "simple",
                    ind,
                )
            )


            ind.fitness = (
                fitness
            )

            ind.requires_eval = (
                False
            )


            # --------------------------------------------------------------- #
            # LOG MORPHOLOGY
            # --------------------------------------------------------------- #

            graph = (
                self.decode_morphology_graph(
                    ind.genotype[
                        "morph"
                    ]
                )
            )


            if graph is not None:
                self.log_evaluation(
                    self.evaluation_round,
                    ind,
                    graph,
                    fitness,
                )


        self.evaluation_round += 1


        return population


    def parent_selection(
        self,
        population: Population,
    ) -> Population:
        """Select top half as parents."""
        population = population.sort(
            sort="min",
            attribute="fitness_",
        )


        cutoff = (
            len(population)
            // 2
        )


        for i, ind in enumerate(
            population
        ):
            ind.tags[
                "ps"
            ] = (
                i
                < cutoff
            )


        ps_count = sum(
            1
            for ind in population
            if ind.tags.get(
                "ps",
                False,
            )
        )


        console.log(
            f"[cyan]"
            f"Parent Selection: "
            f"{ps_count}/{len(population)} "
            f"marked for reproduction"
            f"[/cyan]"
        )


        return population


    def survivor_selection(
        self,
        population: Population,
    ) -> Population:
        """Keep the best POP_SIZE individuals."""
        population = population.sort(
            sort="min",
            attribute="fitness_",
        )


        survivors = population[
            : self.config.target_population_size
        ]


        for ind in population:
            if (
                ind
                not in survivors
            ):
                ind.alive = False


        scored = [
            ind.fitness_
            for ind in survivors
            if (
                ind.fitness_
                is not None
                and ind.fitness_
                != float(
                    "inf"
                )
            )
        ]


        if scored:
            console.log(
                "[green]"
                "Survivor Selection: "
                f"Avg fitness = "
                f"{np.mean(scored):.4f}, "
                f"Best = "
                f"{min(scored):.4f}, "
                f"Worst = "
                f"{max(scored):.4f}"
                "[/green]"
            )

        else:
            console.log(
                "[yellow]"
                "Survivor Selection: "
                "no finite fitness values "
                "this generation"
                "[/yellow]"
            )


        return population


    # ======================================================================== #
    #                            ID MANAGEMENT                                  #
    # ======================================================================== #

    def sync_ids(
        self,
        population: Population,
    ) -> None:
        """Sync ID manager with population CPPNs."""
        max_nid = (
            self.id_manager._node_id
        )

        max_inn = (
            self.id_manager._innov_id
        )


        for ind in population:
            g = ind.genotype[
                "morph"
            ]


            if (
                "nodes"
                in g
            ):
                for n in g[
                    "nodes"
                ].values():
                    nid = n.get(
                        "id",
                        n.get(
                            "_id"
                        ),
                    )


                    if (
                        nid
                        and nid
                        > max_nid
                    ):
                        max_nid = (
                            nid
                        )


            if (
                "connections"
                in g
            ):
                for c in g[
                    "connections"
                ]:
                    inn = c.get(
                        "innovation",
                        c.get(
                            "innov_id"
                        ),
                    )


                    if (
                        inn
                        and inn
                        > max_inn
                    ):
                        max_inn = (
                            inn
                        )


        self.id_manager._node_id = (
            max_nid
        )

        self.id_manager._innov_id = (
            max_inn
        )


    # ======================================================================== #
    #                            PHYSICS RUNNER                                 #
    # ======================================================================== #

    def fast_physics_runner(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        duration: float,
    ) -> None:
        """Run physics as fast as possible without rendering."""
        steps_required = int(
            duration
            / model.opt.timestep
        )


        step = 0


        while (
            step
            < steps_required
        ):
            mujoco.mj_step(
                model,
                data,
            )

            step += 1


    # ======================================================================== #
    #                          SIMULATION RUNNER                               #
    # ======================================================================== #

    def run_simulation(
        self,
        mode: ViewerTypes,
        ind: Individual,
    ) -> float:
        """Build phenotype and run locomotion simulation."""
        mujoco.set_mjcb_control(
            None
        )


        # -------------------------------------------------------------------- #
        # 1. RECONSTRUCT BODY
        # -------------------------------------------------------------------- #

        expected_joints = (
            ind.tags.get(
                "debug_joints",
                0,
            )
        )


        spec = None
        model = None


        attempts = (
            15
            if (
                mode
                != "simple"
            )
            else 1
        )


        for _ in range(
            attempts
        ):
            temp_spec = (
                self.map_genotype_to_body(
                    ind.genotype[
                        "morph"
                    ]
                )
            )


            if temp_spec:
                try:
                    temp_model = (
                        temp_spec.compile()
                    )


                    if (
                        mode
                        == "simple"
                        or temp_model.nu
                        == expected_joints
                    ):
                        spec = (
                            temp_spec
                        )

                        model = (
                            temp_model
                        )

                        break


                except Exception:
                    pass


        if model is None:
            spec = (
                self.map_genotype_to_body(
                    ind.genotype[
                        "morph"
                    ]
                )
            )


            if spec:
                try:
                    model = (
                        spec.compile()
                    )

                except Exception:
                    return float(
                        "inf"
                    )

            else:
                return float(
                    "inf"
                )


        # -------------------------------------------------------------------- #
        # 2. PRE-WORLD ACTUATOR CHECK
        # -------------------------------------------------------------------- #

        if (
            model.nu
            <= 0
        ):
            ind.tags[
                "debug_joints"
            ] = 0

            return float(
                "inf"
            )


        if (
            mode
            == "simple"
        ):
            ind.tags[
                "debug_joints"
            ] = (
                model.nu
            )


        # -------------------------------------------------------------------- #
        # 3. SETUP ENVIRONMENT
        # -------------------------------------------------------------------- #

        world = (
            SimpleFlatWorldWithTarget()
        )


        world.spawn(
            spec,
            position=SPAWN_POSITION,
        )


        try:
            model = (
                world.spec.compile()
            )

        except Exception:
            return float(
                "inf"
            )


        # Check again after adding the world.
        if (
            model.nu
            <= 0
        ):
            return float(
                "inf"
            )


        data = mujoco.MjData(
            model
        )


        # -------------------------------------------------------------------- #
        # 4. SETUP BRAIN
        # -------------------------------------------------------------------- #

        adj_dict = (
            create_fully_connected_adjacency(
                model.nu
            )
        )


        cpg = SimpleCPG(
            adj_dict
        )


        self.map_genotype_to_brain(
            cpg,
            ind.genotype[
                "ctrl"
            ],
        )


        if (
            mode
            != "simple"
        ):
            console.log(
                f"[green]"
                f"Simulating with "
                f"{model.nu} joints "
                f"(Target: "
                f"{expected_joints})"
                f"[/green]"
            )


        # -------------------------------------------------------------------- #
        # 5. SETUP CONTROLLER
        # -------------------------------------------------------------------- #

        tracker = Tracker(
            mujoco.mjtObj.mjOBJ_BODY,
            "core",
            [
                "xpos"
            ],
        )


        ctrl = Controller(
            controller_callback_function=(
                lambda m, d, *a, **k:
                cpg.forward(
                    d.time
                )
            ),
            tracker=tracker,
        )


        ctrl.tracker.setup(
            world.spec,
            data,
        )


        mujoco.set_mjcb_control(
            lambda m, d:
            ctrl.set_control(
                m,
                d,
                duration=DURATION,
            )
        )


        mujoco.mj_resetData(
            model,
            data,
        )


        # -------------------------------------------------------------------- #
        # 6. EXECUTE
        # -------------------------------------------------------------------- #

        match mode:
            case "simple":
                self.fast_physics_runner(
                    model,
                    data,
                    duration=DURATION,
                )


            case "video":
                recorder = VideoRecorder(
                    output_folder=str(
                        DATA
                        / "videos"
                    ),
                    file_name=(
                        f"dual_{getattr(ind, 'id', 'best')}"
                    ),
                )


                video_renderer(
                    model,
                    data,
                    duration=DURATION,
                    video_recorder=(
                        recorder
                    ),
                )


            case "launcher":
                viewer.launch(
                    model=model,
                    data=data,
                )


        # -------------------------------------------------------------------- #
        # 7. CALCULATE FITNESS
        # -------------------------------------------------------------------- #

        # Ignore first second to reduce
        # benefit from falling immediately
        # after spawning.
        delay_time = min(
            1.0,
            DURATION,
        )


        delay_fraction = (
            delay_time
            / DURATION
            if (
                DURATION
                > 0
            )
            else 0
        )


        dist = float(
            "inf"
        )


        if tracker.history[
            "xpos"
        ]:
            first_key = next(
                iter(
                    tracker.history[
                        "xpos"
                    ].keys()
                )
            )


            traj = (
                tracker.history[
                    "xpos"
                ][
                    first_key
                ]
            )


            if traj:
                start_idx = max(
                    0,
                    int(
                        len(
                            traj
                        )
                        * delay_fraction
                    ),
                )


                pos_after_delay = (
                    np.array(
                        traj[
                            start_idx
                        ]
                    )
                )


                pos_final = (
                    np.array(
                        traj[
                            -1
                        ]
                    )
                )


                valid_movement_vector = (
                    pos_final
                    - pos_after_delay
                )


                effective_pos = (
                    np.array(
                        SPAWN_POSITION
                    )
                    + valid_movement_vector
                )


                dist = np.sqrt(
                    np.sum(
                        (
                            effective_pos[
                                :2
                            ]
                            - TARGET_POSITION[
                                :2
                            ]
                        )
                        ** 2
                    )
                )


                if (
                    mode
                    != "simple"
                ):
                    console.log(
                        f"[blue]"
                        f"Traj len: "
                        f"{len(traj)}, "
                        f"start_idx: "
                        f"{start_idx}, "
                        f"movement: "
                        f"{np.linalg.norm(valid_movement_vector):.3f}"
                        f"[/blue]"
                    )


        return float(
            dist
        )


    # ======================================================================== #
    #                              MAIN LOOP                                    #
    # ======================================================================== #

    def evolve(
        self,
    ) -> Individual | None:
        """Run joint body/controller evolution."""
        console.log(
            "Initializing population..."
        )


        population = Population([
            self.create_individual()
            for _ in range(
                POP_SIZE
            )
        ])


        self.sync_ids(
            population
        )


        # Initial evaluation
        population = (
            self.evaluate(
                population
            )
        )


        ops = [
            EAOperation(
                self.parent_selection
            ),
            EAOperation(
                self.reproduction
            ),
            EAOperation(
                self.evaluate
            ),
            EAOperation(
                self.survivor_selection
            ),
        ]


        ea = EA(
            population,
            operations=ops,
            num_steps=BUDGET,
            is_maximisation=(
                self.config.is_maximisation
            ),
            db_file_path=(
                self.config.db_file_path
            ),
            db_handling=(
                self.config.db_handling
            ),
            quiet=(
                self.config.quiet
            ),
        )


        ea.run()


        return ea.get_solution(
            "best",
            only_alive=False,
        )


# ============================================================================ #
#                                  MAIN                                        #
# ============================================================================ #


def main(
) -> None:
    console.rule(
        "[bold purple]"
        "Starting Joint Evolution "
        "(Morph + Ctrl)"
        "[/bold purple]"
    )


    console.log(
        f"Population: "
        f"{POP_SIZE}"
    )


    console.log(
        f"Budget: "
        f"{BUDGET}"
    )


    console.log(
        f"Duration: "
        f"{DURATION}s"
    )


    console.log(
        f"Seed: "
        f"{SEED}"
    )


    console.log(
        f"Max Modules: "
        f"{NUM_MODULES}"
    )


    console.log(
        f"Bone Mode: "
        f"{BONE_MODE}"
    )


    if (
        BONE_MODE
        == "fixed"
    ):
        console.log(
            f"Fixed Brick Length: "
            f"{FIXED_BRICK_LENGTH * 1000:.2f} mm"
        )


    evo = Evolution()


    best = (
        evo.evolve()
    )


    if best:
        console.rule(
            "[bold green]"
            "Final Best Result"
            "[/bold green]"
        )


        console.log(
            f"Experiment Condition: "
            f"{BONE_MODE}"
        )


        console.log(
            f"Seed: "
            f"{SEED}"
        )


        console.log(
            f"Best Fitness "
            f"(Dist to Target): "
            f"{best.fitness:.4f}"
        )


        graph = (
            evo.decode_morphology_graph(
                best.genotype[
                    "morph"
                ]
            )
        )


        if graph is not None:
            stats = (
                evo.get_morphology_statistics(
                    graph
                )
            )


            console.log(
                f"Modules: "
                f"{stats['num_modules']}"
            )


            console.log(
                f"Bricks: "
                f"{stats['num_bricks']}"
            )


            console.log(
                f"Hinges: "
                f"{stats['num_hinges']}"
            )


            if (
                stats[
                    "num_bricks"
                ]
                > 0
            ):
                console.log(
                    f"Mean Brick Length: "
                    f"{stats['mean_length'] * 1000:.2f} mm"
                )


                console.log(
                    f"Minimum Brick Length: "
                    f"{stats['min_length'] * 1000:.2f} mm"
                )


                console.log(
                    f"Maximum Brick Length: "
                    f"{stats['max_length'] * 1000:.2f} mm"
                )


                console.log(
                    f"Brick Length SD: "
                    f"{stats['std_length'] * 1000:.2f} mm"
                )


            console.log(
                f"Physically Valid: "
                f"{is_physically_valid(graph)}"
            )


        console.log(
            f"Results Log: "
            f"{LOG_FILE}"
        )


        if args.visualize:
            evo.run_simulation(
                "launcher",
                best,
            )


    else:
        console.log(
            "[red]"
            "No solution found"
            "[/red]"
        )


if __name__ == "__main__":
    main()