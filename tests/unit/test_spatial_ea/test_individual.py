"""Test: the spatial individual record."""

# Third-party libraries
import numpy as np

# Local libraries
from ariel.spatial_ea.hyperneat import CPPNConnection, CPPNNode
from ariel.spatial_ea.individual import SpatialIndividual


def _genotype() -> dict:
    """Build a tiny CPPN genome."""
    return {
        "nodes": [CPPNNode(node_id=0, activation="linear", layer=0)],
        "connections": [CPPNConnection(from_node=0, to_node=1, weight=1.0)],
    }


def test_defaults() -> None:
    """A fresh individual starts unevaluated, unplaced and at full energy."""
    individual = SpatialIndividual()

    assert individual.unique_id is None
    assert individual.generation == 0
    assert individual.genotype == {}
    assert individual.fitness == 0.0
    assert individual.evaluated is False
    assert individual.spawn_position is None
    assert individual.target_position is None
    assert individual.assigned_zone is None
    assert individual.orientation == 0.0
    assert individual.parent_ids == []
    assert individual.energy == 100.0


def test_age_is_measured_from_the_generation_of_birth() -> None:
    """Age counts generations survived and never goes negative."""
    individual = SpatialIndividual(generation=3)

    assert individual.age_at(3) == 0
    assert individual.age_at(7) == 4
    # A generation before birth is still age zero, not a negative age.
    assert individual.age_at(1) == 0


def test_copy_is_independent() -> None:
    """Copying must not share arrays, lists or the genome with the source."""
    individual = SpatialIndividual(
        unique_id=7,
        generation=2,
        genotype=_genotype(),
        fitness=1.5,
        evaluated=True,
        start_position=np.array([1.0, 2.0, 3.0]),
        end_position=np.array([4.0, 5.0, 6.0]),
        spawn_position=np.array([7.0, 8.0, 9.0]),
        orientation=1.25,
        target_position=np.array([0.0, 1.0, 0.0]),
        progress_toward_target=0.5,
        total_distance=2.0,
        robot_index=1,
        assigned_zone=3,
        parent_ids=[2, 5],
        energy=42.0,
    )

    copied = individual.copy()

    assert copied is not individual
    assert copied.unique_id == 7
    assert copied.generation == 2
    assert copied.fitness == 1.5
    assert copied.evaluated is True
    assert copied.orientation == 1.25
    assert copied.assigned_zone == 3
    assert copied.progress_toward_target == 0.5
    assert copied.total_distance == 2.0
    assert copied.robot_index == 1
    assert copied.energy == 42.0

    for name in (
        "start_position",
        "end_position",
        "spawn_position",
        "target_position",
    ):
        original = getattr(individual, name)
        duplicate = getattr(copied, name)
        assert duplicate is not original
        assert np.array_equal(duplicate, original)

    assert copied.parent_ids == [2, 5]
    assert copied.parent_ids is not individual.parent_ids

    # The genome is deep-copied, so mutating the copy is safe.
    copied.genotype["connections"][0].weight = 99.0
    assert individual.genotype["connections"][0].weight == 1.0


def test_copy_handles_unset_positions() -> None:
    """Copying an unplaced individual should not fail on missing arrays."""
    copied = SpatialIndividual(unique_id=1).copy()

    assert copied.start_position is None
    assert copied.end_position is None
    assert copied.spawn_position is None
    assert copied.target_position is None
