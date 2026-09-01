"""Test: HyperNEAT genetic operators for the spatial EA."""

# Standard library
import random

# Third-party libraries
import numpy as np
import pytest

# Local libraries
from ariel.spatial_ea.genetics import (
    clone_individual,
    create_initial_hyperneat_genome,
    crossover_genomes,
    crossover_hyperneat,
    genome_summary,
    mutate_genome,
    mutate_hyperneat,
)
from ariel.spatial_ea.hyperneat import CPPN, CPPNConnection, CPPNNode
from ariel.spatial_ea.individual import SpatialIndividual


def _node_ids(genome: dict) -> set[int]:
    """Collect the node identifiers of a genome."""
    return {node.node_id for node in genome["nodes"]}


def test_initial_genome_is_valid() -> None:
    """A fresh genome should be a connected, activatable feed-forward net."""
    for _ in range(25):
        genome = create_initial_hyperneat_genome()

        assert len(genome["nodes"]) >= 5
        assert len(genome["connections"]) >= 1

        # Every connection points forward and lands on a declared node.
        layers = {node.node_id: node.layer for node in genome["nodes"]}
        for conn in genome["connections"]:
            assert conn.from_node in layers
            assert conn.to_node in layers
            assert layers[conn.from_node] < layers[conn.to_node]

        output = CPPN(genome).activate(np.array([0.1, 0.2, 0.3, 0.4]))
        assert output.shape == (1,)
        assert np.isfinite(output).all()


def test_crossover_keeps_every_connection_endpoint() -> None:
    """Offspring must contain the nodes their inherited connections need.

    The research prototype's spatial crossover kept only one parent's nodes
    while inheriting both parents' connections, leaving dangling references.
    """
    random.seed(3)
    np.random.seed(3)
    for _ in range(25):
        parent1 = create_initial_hyperneat_genome()
        parent2 = create_initial_hyperneat_genome()
        child = crossover_genomes(parent1, parent2)

        available = _node_ids(child)
        for conn in child["connections"]:
            assert conn.from_node in available
            assert conn.to_node in available


def test_add_node_keeps_output_layer_deepest() -> None:
    """Splitting a connection must not promote a hidden node to an output."""
    random.seed(11)
    np.random.seed(11)
    genome = {
        "nodes": [
            CPPNNode(node_id=0, activation="linear", layer=0),
            CPPNNode(node_id=1, activation="linear", layer=0),
            CPPNNode(node_id=2, activation="sine", layer=1),
        ],
        "connections": [
            CPPNConnection(from_node=0, to_node=2, weight=0.5),
            CPPNConnection(from_node=1, to_node=2, weight=-0.5),
        ],
    }

    mutate_genome(
        genome,
        weight_mutation_rate=0.0,
        add_connection_rate=0.0,
        add_node_rate=1.0,
    )

    layers = {node.node_id: node.layer for node in genome["nodes"]}
    max_layer = max(layers.values())

    # The original output stays alone on the deepest layer.
    assert layers[2] == max_layer
    assert [n for n, layer in layers.items() if layer == max_layer] == [2]

    # The inserted node sits strictly between its endpoints.
    new_ids = set(layers) - {0, 1, 2}
    assert len(new_ids) == 1
    new_id = new_ids.pop()
    assert 0 < layers[new_id] < max_layer

    assert CPPN(genome).activate(np.array([0.5, 0.5])).shape == (1,)


def test_perturbation_is_clipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated perturbation must not let weights run away.

    Forcing ``random.random`` to zero takes the perturb branch every time,
    which is the branch ``WEIGHT_CLIP`` bounds.
    """
    monkeypatch.setattr(
        "ariel.spatial_ea.genetics.random.random",
        lambda: 0.0,
    )

    genome = create_initial_hyperneat_genome()
    for _ in range(50):
        mutate_genome(
            genome,
            weight_mutation_rate=1.0,
            weight_mutation_power=5.0,
            add_connection_rate=0.0,
            add_node_rate=0.0,
        )

    for conn in genome["connections"]:
        assert abs(conn.weight) <= 3.0 + 1e-9


def test_clipping_can_be_turned_off() -> None:
    """Unclipped perturbation is allowed to leave the range."""
    # Both generators matter: branch choice comes from ``random``, the
    # perturbation itself from ``numpy.random``.
    random.seed(5)
    np.random.seed(5)
    genome = {
        "nodes": [
            CPPNNode(node_id=0, activation="linear", layer=0),
            CPPNNode(node_id=1, activation="sine", layer=1),
        ],
        "connections": [CPPNConnection(from_node=0, to_node=1, weight=0.0)],
    }

    for _ in range(200):
        mutate_genome(
            genome,
            weight_mutation_rate=1.0,
            weight_mutation_power=5.0,
            add_connection_rate=0.0,
            add_node_rate=0.0,
            clip_weights=False,
        )

    assert abs(genome["connections"][0].weight) > 3.0


def test_crossover_hyperneat_assigns_fresh_identities() -> None:
    """Two offspring should get consecutive ids and record both parents."""
    parent1 = SpatialIndividual(
        unique_id=1,
        genotype=create_initial_hyperneat_genome(),
    )
    parent2 = SpatialIndividual(
        unique_id=2,
        genotype=create_initial_hyperneat_genome(),
    )

    child1, child2, next_id = crossover_hyperneat(
        parent1,
        parent2,
        next_unique_id=10,
        generation=4,
    )

    assert (child1.unique_id, child2.unique_id) == (10, 11)
    assert next_id == 12
    assert child1.parent_ids == [1, 2]
    assert child2.generation == 4
    assert child1.genotype is not child2.genotype


def test_mutate_hyperneat_does_not_touch_the_source() -> None:
    """Mutation must copy the genome rather than edit the parent's."""
    parent = SpatialIndividual(
        unique_id=1,
        genotype=create_initial_hyperneat_genome(),
    )
    before = [conn.weight for conn in parent.genotype["connections"]]

    mutant, next_id = mutate_hyperneat(
        parent,
        next_unique_id=7,
        weight_mutation_rate=1.0,
        weight_mutation_power=1.0,
    )

    after = [conn.weight for conn in parent.genotype["connections"]]
    assert before == after
    assert mutant.unique_id == 7
    assert next_id == 8
    assert mutant.parent_ids == [1]


def test_clone_individual_deep_copies_the_genome() -> None:
    """A clone's genome must not alias its source's."""
    source = SpatialIndividual(
        unique_id=3,
        genotype=create_initial_hyperneat_genome(),
    )
    clone, next_id = clone_individual(source, next_unique_id=20, generation=2)

    clone.genotype["connections"][0].weight = 99.0

    assert next_id == 21
    assert clone.parent_ids == [3]
    assert source.genotype["connections"][0].weight != 99.0


def test_genome_summary_counts_enabled_connections() -> None:
    """The summary should separate total from enabled connections."""
    genome = {
        "nodes": [CPPNNode(node_id=0, activation="linear", layer=0)],
        "connections": [
            CPPNConnection(from_node=0, to_node=1, weight=1.0, enabled=True),
            CPPNConnection(from_node=0, to_node=2, weight=1.0, enabled=False),
        ],
    }

    assert genome_summary(genome) == {
        "num_nodes": 1,
        "num_connections": 2,
        "num_enabled": 1,
    }
