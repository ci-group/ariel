"""Test CPPN evolution with variable-length brick modules."""

# Standard library
import copy
import random

# Third-party libraries
import numpy as np

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome
from ariel.body_phenotypes.robogen_lite.cppn_neat.id_manager import IdManager
from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import (
    MorphologyDecoderBestFirst,
)
from ariel.parameters.ariel_modules import ArielModulesConfig


SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)

T = NUM_OF_TYPES_OF_MODULES
R = NUM_OF_ROTATIONS

NUM_CPPN_INPUTS = 6
NUM_CPPN_OUTPUTS = 1 + T + R + 1

MAX_MODULES = 15

ariel_modules_config = ArielModulesConfig()

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


def create_random_genome() -> Genome:
    """Create a random CPPN genome with a brick-length output."""
    genome = Genome.random(
        num_inputs=NUM_CPPN_INPUTS,
        num_outputs=NUM_CPPN_OUTPUTS,
        next_node_id=(
            NUM_CPPN_INPUTS
            + NUM_CPPN_OUTPUTS
        ),
        next_innov_id=0,
    )

    # Apply initial mutations
    for _ in range(3):
        genome.mutate(
            0.6,
            0.6,
            id_manager.get_next_innov_id,
            id_manager.get_next_node_id,
        )

    return genome


def decode_genome(
    genome: Genome,
):
    """Decode a CPPN genome into a robot graph."""
    decoder = MorphologyDecoderBestFirst(
        cppn_genome=genome,
        max_modules=MAX_MODULES,
    )

    return decoder.decode()


def get_brick_lengths(
    graph,
) -> list[float]:
    """Return all brick lengths from a decoded graph."""
    lengths = []

    for _, node_data in graph.nodes(
        data=True
    ):
        if (
            node_data["type"]
            == ModuleType.BRICK.name
        ):
            assert "length" in node_data, (
                "Decoded BRICK node does not "
                "contain a length attribute."
            )

            lengths.append(
                float(
                    node_data["length"]
                )
            )

    return lengths


def validate_lengths(
    lengths: list[float],
) -> None:
    """Check that all brick lengths are valid."""
    for length in lengths:
        assert (
            ariel_modules_config.BRICK_LENGTH_MIN
            <= length
            <= ariel_modules_config.BRICK_LENGTH_MAX
        ), (
            f"Invalid brick length: {length}"
        )


def test_creation_and_decoding() -> None:
    """Test initial CPPN creation and decoding."""
    genome = create_random_genome()

    graph = decode_genome(
        genome
    )

    lengths = get_brick_lengths(
        graph
    )

    validate_lengths(
        lengths
    )

    print()
    print("=" * 70)
    print(
        "INITIAL CPPN DECODING"
    )
    print("=" * 70)

    print(
        f"Modules decoded: "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Brick modules: "
        f"{len(lengths)}"
    )

    if lengths:
        print(
            "Brick lengths:"
        )

        for length in lengths:
            print(
                f"  {length:.6f} m "
                f"({length * 1000:.2f} mm)"
            )

    print(
        "Initial genome decoding passed."
    )


def test_mutation() -> None:
    """Test that mutated CPPNs still decode valid brick lengths."""
    parent = create_random_genome()

    child = parent.copy()

    child.mutate(
        0.2,
        0.3,
        id_manager.get_next_innov_id,
        id_manager.get_next_node_id,
    )

    parent_graph = decode_genome(
        parent
    )

    child_graph = decode_genome(
        child
    )

    parent_lengths = get_brick_lengths(
        parent_graph
    )

    child_lengths = get_brick_lengths(
        child_graph
    )

    validate_lengths(
        parent_lengths
    )

    validate_lengths(
        child_lengths
    )

    print()
    print("=" * 70)
    print(
        "CPPN MUTATION TEST"
    )
    print("=" * 70)

    print(
        "Parent brick lengths:",
        parent_lengths,
    )

    print(
        "Child brick lengths:",
        child_lengths,
    )

    print(
        "Mutation produced a valid "
        "decoded morphology."
    )


def test_crossover() -> None:
    """Test that CPPN crossover still produces valid brick lengths."""
    parent_a = create_random_genome()
    parent_b = create_random_genome()

    child = parent_a.crossover(
        parent_b,
        is_maximisation=False,
    )

    graph = decode_genome(
        child
    )

    lengths = get_brick_lengths(
        graph
    )

    validate_lengths(
        lengths
    )

    print()
    print("=" * 70)
    print(
        "CPPN CROSSOVER TEST"
    )
    print("=" * 70)

    print(
        f"Child modules: "
        f"{graph.number_of_nodes()}"
    )

    print(
        "Child brick lengths:",
        lengths,
    )

    print(
        "Crossover produced a valid "
        "decoded morphology."
    )


def test_multiple_generations() -> None:
    """Simulate multiple generations of CPPN mutation and crossover."""
    population_size = 20
    generations = 10

    population = [
        create_random_genome()
        for _ in range(
            population_size
        )
    ]

    all_lengths = []

    print()
    print("=" * 70)
    print(
        "MULTI-GENERATION CPPN TEST"
    )
    print("=" * 70)

    for generation in range(
        generations
    ):
        next_population = []

        generation_lengths = []

        while (
            len(next_population)
            < population_size
        ):
            if (
                len(population) >= 2
                and RNG.random() < 0.5
            ):
                parent_a, parent_b = (
                    random.sample(
                        population,
                        2,
                    )
                )

                child = parent_a.crossover(
                    parent_b,
                    is_maximisation=False,
                )

            else:
                parent = random.choice(
                    population
                )

                child = parent.copy()

            child.mutate(
                0.2,
                0.3,
                id_manager.get_next_innov_id,
                id_manager.get_next_node_id,
            )

            graph = decode_genome(
                child
            )

            lengths = get_brick_lengths(
                graph
            )

            validate_lengths(
                lengths
            )

            generation_lengths.extend(
                lengths
            )

            next_population.append(
                child
            )

        population = next_population

        all_lengths.extend(
            generation_lengths
        )

        print(
            f"Generation "
            f"{generation + 1}: "
            f"{len(generation_lengths)} bricks"
        )

        if generation_lengths:
            print(
                f"  min = "
                f"{min(generation_lengths) * 1000:.2f} mm"
            )

            print(
                f"  max = "
                f"{max(generation_lengths) * 1000:.2f} mm"
            )

            print(
                f"  mean = "
                f"{np.mean(generation_lengths) * 1000:.2f} mm"
            )

    assert len(all_lengths) > 0, (
        "No brick modules were produced "
        "during the multi-generation test."
    )

    print()
    print(
        "All generations produced "
        "valid variable-length bricks."
    )


if __name__ == "__main__":
    test_creation_and_decoding()
    test_mutation()
    test_crossover()
    test_multiple_generations()

    print()
    print("=" * 70)
    print(
        "PASSED: FULL CPPN VARIABLE-LENGTH "
        "EVOLUTION PIPELINE WORKS."
    )
    print("=" * 70)