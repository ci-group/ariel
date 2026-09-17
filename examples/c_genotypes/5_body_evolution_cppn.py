"""
Morphology-only evolution for CPPN genomes using the same morphological fitness
as the tree-based example. Decodes CPPNs into module graphs using the
best-first decoder and evaluates with MorphologicalMeasures.
"""

# Standard library
import argparse
import random
import time
from pathlib import Path

# Third-party libraries
import mujoco
import numpy as np
from mujoco import viewer
from rich.console import Console
from rich.progress import track
from rich.traceback import install

# Local libraries
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
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import (
    Genome,
)
from ariel.body_phenotypes.robogen_lite.cppn_neat.id_manager import (
    IdManager,
)
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
from ariel.simulation.environments._simple_flat_with_target import (
    SimpleFlatWorldWithTarget,
)
from ariel.utils.morphological_descriptor import (
    MorphologicalMeasures,
)


# Initialize rich console
install()
console = Console()


# ============================================================================
# CONFIGURATION
# ============================================================================

parser = argparse.ArgumentParser(
    description="Morphology-only CPPN evolution"
)

parser.add_argument(
    "--budget",
    type=int,
    default=50,
)

parser.add_argument(
    "--pop",
    type=int,
    default=100,
)

parser.add_argument(
    "--max-modules",
    type=int,
    default=15,
)

parser.add_argument(
    "--visualize",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Launch MuJoCo viewer for best individual",
)

args = parser.parse_args()


POP_SIZE = args.pop
BUDGET = args.budget
NUM_MODULES = args.max_modules


SEED = 42
RNG = np.random.default_rng(
    SEED
)

random.seed(
    SEED
)


SCRIPT_NAME = Path(
    __file__
).stem

CWD = Path.cwd()

DATA = (
    CWD
    / "__data__"
    / SCRIPT_NAME
)

DATA.mkdir(
    exist_ok=True,
    parents=True,
)


SPAWN_POSITION = (
    -0.8,
    0.0,
    0.1,
)


# CPPN configuration
T = NUM_OF_TYPES_OF_MODULES
R = NUM_OF_ROTATIONS

NUM_CPPN_INPUTS = 6

# Outputs:
# 1 connection score
# T module type scores
# R rotation scores
# 1 variable brick length output
NUM_CPPN_OUTPUTS = (
    1
    + T
    + R
    + 1
)


id_manager = IdManager(
    node_start=(
        NUM_CPPN_INPUTS
        + NUM_CPPN_OUTPUTS
        - 1
    ),
    innov_start=(
        NUM_CPPN_INPUTS
        * NUM_CPPN_OUTPUTS
    ) - 1,
)


# ============================================================================
# FITNESS
# ============================================================================


def morpho_score_from_graph(
    graph,
) -> float:
    """Calculate morphology-only fitness score."""
    try:
        measures = MorphologicalMeasures(
            graph
        )

        score = (
            measures.symmetry * 0.20
            + measures.joints * 0.20
            + measures.branching * 0.20
            + measures.length_of_limbs * 0.20
            + measures.module_diversity * 0.20
        )

        return score

    except Exception:
        return float(
            "nan"
        )


def visualize_genome(
    cppn_genome: Genome,
) -> None:
    """Decode a CPPN genome and launch the MuJoCo viewer."""
    try:
        decoder = (
            MorphologyDecoderBestFirst(
                cppn_genome=cppn_genome,
                max_modules=NUM_MODULES,
            )
        )

        graph = decoder.decode()

        if (
            graph.number_of_nodes()
            == 0
        ):
            console.log(
                "[red]Cannot visualize "
                "empty decoded graph[/red]"
            )
            return

        spec = (
            construct_mjspec_from_graph(
                graph
            ).spec
        )

        world = (
            SimpleFlatWorldWithTarget()
        )

        world.spawn(
            spec,
            position=SPAWN_POSITION,
        )

        model = (
            world.spec.compile()
        )

        data = mujoco.MjData(
            model
        )

        viewer.launch(
            model=model,
            data=data,
        )

    except Exception as e:
        console.log(
            f"[red]Visualization failed: "
            f"{e}[/red]"
        )


# ============================================================================
# EVOLUTION CLASS
# ============================================================================


class CPPNEvolution:
    def __init__(
        self,
    ) -> None:
        self.config = EASettings(
            is_maximisation=False,
            num_steps=BUDGET,
            target_population_size=POP_SIZE,
            output_folder=DATA,
            db_file_name="database.db",
        )

    def create_random_genome(
        self,
    ) -> Genome:
        """Create a random CPPN genome."""
        genome = Genome.random(
            num_inputs=NUM_CPPN_INPUTS,
            num_outputs=NUM_CPPN_OUTPUTS,
            next_node_id=(
                NUM_CPPN_INPUTS
                + NUM_CPPN_OUTPUTS
            ),
            next_innov_id=0,
        )

        # Apply initial structural mutations
        for _ in range(
            3
        ):
            genome.mutate(
                0.6,
                0.6,
                id_manager.get_next_innov_id,
                id_manager.get_next_node_id,
            )

        return genome

    def create_individual(
        self,
    ) -> Individual:
        """Create one random individual."""
        genome = (
            self.create_random_genome()
        )

        ind = Individual()

        ind.genotype = {
            "cppn": genome.to_dict()
        }

        ind.tags["ps"] = False
        ind.tags["valid"] = True

        return ind

    def decode_to_graph(
        self,
        genome: Genome,
    ):
        """Decode a CPPN genome into a morphology graph."""
        decoder = (
            MorphologyDecoderBestFirst(
                cppn_genome=genome,
                max_modules=NUM_MODULES,
            )
        )

        return decoder.decode()

    def evaluate(
        self,
        population: Population,
    ) -> Population:
        """Evaluate population using morphology descriptors."""
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
            cppn = Genome.from_dict(
                ind.genotype[
                    "cppn"
                ]
            )

            graph = (
                self.decode_to_graph(
                    cppn
                )
            )

            # Reject physically invalid morphologies
            if not is_physically_valid(
                graph
            ):
                ind.fitness = float(
                    "inf"
                )

                ind.requires_eval = False

                continue

            score = (
                morpho_score_from_graph(
                    graph
                )
            )

            # EA expects minimization
            ind.fitness = (
                -score
                if not np.isnan(
                    score
                )
                else float("inf")
            )

            ind.requires_eval = False

        return population

    def mutate(
        self,
        genome: Genome,
    ) -> Genome:
        """Mutate a CPPN genome."""
        child = genome.copy()

        child.mutate(
            0.2,
            0.3,
            id_manager.get_next_innov_id,
            id_manager.get_next_node_id,
        )

        return child

    def crossover(
        self,
        a: Genome,
        b: Genome,
    ) -> Genome:
        """Crossover two CPPN genomes."""
        return a.crossover(
            b,
            is_maximisation=False,
        )

    def reproduction(
        self,
        population: Population,
    ) -> Population:
        """Create offspring through crossover and mutation."""
        parents = [
            ind
            for ind in population
            if ind.tags.get(
                "ps",
                False,
            )
        ]

        if not parents:
            parents = population

        new_offspring: list[
            Individual
        ] = []

        target_pool = (
            self.config.target_population_size
            * 2
        )

        while (
            len(population)
            + len(new_offspring)
            < target_pool
        ):
            if (
                len(parents) >= 2
                and RNG.random() < 0.5
            ):
                p1, p2 = random.sample(
                    parents,
                    2,
                )

                g1 = (
                    Genome.from_dict(
                        p1.genotype["cppn"]
                    )
                    if isinstance(
                        p1.genotype["cppn"],
                        dict,
                    )
                    else p1.genotype["cppn"]
                )

                g2 = (
                    Genome.from_dict(
                        p2.genotype["cppn"]
                    )
                    if isinstance(
                        p2.genotype["cppn"],
                        dict,
                    )
                    else p2.genotype["cppn"]
                )

                child = self.crossover(
                    g1,
                    g2,
                )

            else:
                parent = random.choice(
                    parents
                )

                child = (
                    Genome.from_dict(
                        parent.genotype[
                            "cppn"
                        ]
                    )
                    if isinstance(
                        parent.genotype[
                            "cppn"
                        ],
                        dict,
                    )
                    else parent.genotype[
                        "cppn"
                    ]
                )

            child = self.mutate(
                child
            )

            ind = Individual()

            ind.genotype = {
                "cppn": child.to_dict()
            }

            ind.tags["ps"] = False
            ind.tags["valid"] = True

            new_offspring.append(
                ind
            )

        population.extend(
            new_offspring
        )

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
            ind.tags["ps"] = (
                i < cutoff
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
            f"[cyan]Parent Selection: "
            f"{ps_count}/{len(population)} "
            f"marked[/cyan]"
        )

        return population

    def survivor_selection(
        self,
        population: Population,
    ) -> Population:
        """Keep best population-size individuals."""
        population = population.sort(
            sort="min",
            attribute="fitness_",
        )

        survivors = population[
            : self.config.target_population_size
        ]

        for ind in population:
            if ind not in survivors:
                ind.alive = False

        valid_fitnesses = [
            ind.fitness_
            for ind in survivors
            if (
                ind.fitness_ is not None
                and ind.fitness_
                != float("inf")
            )
        ]

        if valid_fitnesses:
            console.log(
                f"[green]Survivor Selection:[/green] "
                f"Avg={np.mean(valid_fitnesses):.4f}, "
                f"Min={min(valid_fitnesses):.4f}, "
                f"Max={max(valid_fitnesses):.4f}"
            )

        else:
            console.log(
                "[yellow]Survivor Selection: "
                "No physically valid survivors[/yellow]"
            )

        return population

    def evolve(
        self,
    ) -> Individual | None:
        """Run CPPN evolution."""
        console.log(
            "Initializing CPPN population..."
        )

        population = Population([
            self.create_individual()
            for _ in range(
                POP_SIZE
            )
        ])

        population = self.evaluate(
            population
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
            quiet=self.config.quiet,
        )

        ea.run()

        return ea.get_solution(
            "best",
            only_alive=False,
        )


def main(
) -> None:
    console.rule(
        "[bold purple]"
        "Starting Morphology-Only Evolution "
        "(CPPN Genomes)"
        "[/bold purple]"
    )

    console.log(
        f"Population: {POP_SIZE}, "
        f"Budget: {BUDGET}, "
        f"Max Modules: {NUM_MODULES}"
    )

    evo = CPPNEvolution()

    start = time.time()

    best = evo.evolve()

    elapsed = (
        time.time()
        - start
    )

    if best:
        cppn = Genome.from_dict(
            best.genotype[
                "cppn"
            ]
        )

        graph = (
            evo.decode_to_graph(
                cppn
            )
        )

        score = (
            morpho_score_from_graph(
                graph
            )
        )

        try:
            measures = (
                MorphologicalMeasures(
                    graph
                )
            )

            console.rule(
                "[bold green]"
                "Final Best Result"
                "[/bold green]"
            )

            console.log(
                f"Best morphological score: "
                f"{score:.4f}"
            )

            console.log(
                f"Modules: "
                f"{measures.num_modules}"
            )

            console.log(
                f"Joints: "
                f"{measures.num_active_hinges}"
            )

            console.log(
                f"Symmetry: "
                f"{measures.symmetry:.4f}"
            )

            console.log(
                f"Branching: "
                f"{measures.branching:.4f}"
            )

            console.log(
                f"Diversity: "
                f"{measures.module_diversity:.4f}"
            )

            brick_lengths = [
                float(
                    data["length"]
                )
                for _, data
                in graph.nodes(
                    data=True
                )
                if (
                    data["type"]
                    == ModuleType.BRICK.name
                    and "length" in data
                )
            ]

            if brick_lengths:
                console.log(
                    "Brick Lengths: "
                    + ", ".join(
                        f"{length * 1000:.2f} mm"
                        for length
                        in brick_lengths
                    )
                )

                console.log(
                    f"Mean Brick Length: "
                    f"{np.mean(brick_lengths) * 1000:.2f} mm"
                )

                console.log(
                    f"Min Brick Length: "
                    f"{min(brick_lengths) * 1000:.2f} mm"
                )

                console.log(
                    f"Max Brick Length: "
                    f"{max(brick_lengths) * 1000:.2f} mm"
                )

            console.log(
                f"Physically Valid: "
                f"{is_physically_valid(graph)}"
            )

            console.log(
                f"Elapsed: "
                f"{elapsed:.2f}s"
            )

        except Exception as e:
            console.log(
                f"[red]Error analyzing "
                f"best individual: "
                f"{e}[/red]"
            )

        if args.visualize:
            try:
                visualize_genome(
                    cppn
                )

            except Exception as e:
                console.log(
                    f"[red]Visualization error: "
                    f"{e}[/red]"
                )

    else:
        console.log(
            "[red]No solution found[/red]"
        )


if __name__ == "__main__":
    main()