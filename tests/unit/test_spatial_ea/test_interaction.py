"""Test: spatial geometry, pairing and offspring placement."""

# Third-party libraries
import numpy as np
import pytest

# Local libraries
from ariel.spatial_ea.individual import SpatialIndividual
from ariel.spatial_ea.interaction import (
    apply_movement_bias,
    apply_world_boundaries,
    calculate_offspring_positions,
    calculate_periodic_displacement,
    check_periodic_spawn_overlap,
    find_nearest_periodic,
    find_pairs,
    find_pairs_with_strategy,
    generate_random_zone_centers,
    is_in_mating_zone,
    split_trajectory_at_wraps,
    wrap_offspring_position,
)


def test_generate_random_zone_centers() -> None:
    """Zone centers should be generated inside the world bounds."""
    centers = generate_random_zone_centers(2, (10.0, 10.0), 1.0, 2.0)
    assert len(centers) == 2


def test_find_pairs_and_offspring_positions() -> None:
    """Pairing should detect nearby individuals and place offspring nearby."""
    population = [
        SpatialIndividual(unique_id=1),
        SpatialIndividual(unique_id=2),
        SpatialIndividual(unique_id=3),
        SpatialIndividual(unique_id=4),
    ]
    positions = [
        np.array([0.0, 0.0, 0.5]),
        np.array([0.2, 0.0, 0.5]),
        np.array([5.0, 5.0, 0.5]),
        np.array([5.1, 5.0, 0.5]),
    ]

    pairs, paired_indices = find_pairs(
        population,
        positions,
        pairing_radius=0.5,
        world_size=(10.0, 10.0),
    )

    offspring_positions = calculate_offspring_positions(
        pairs,
        positions,
        offspring_radius=0.3,
        world_size=(10.0, 10.0),
    )

    assert len(pairs) == 2
    assert paired_indices == {0, 1, 2, 3}
    assert len(offspring_positions) == 2


def test_apply_world_boundaries_periodic_wraps_position() -> None:
    """Periodic boundaries should wrap positions to the world extents."""
    wrapped = apply_world_boundaries(
        np.array([10.4, -0.2, 0.5]),
        (10.0, 10.0),
        use_periodic_boundaries=True,
    )
    assert np.isclose(wrapped[0], 0.4)
    assert np.isclose(wrapped[1], 9.8)


def test_calculate_periodic_displacement_prefers_short_path() -> None:
    """Periodic displacement should return the shortest wrap-around vector."""
    displacement = calculate_periodic_displacement(
        np.array([9.0, 9.0, 0.5]),
        np.array([1.0, 1.0, 0.5]),
        (10.0, 10.0),
    )
    assert np.allclose(displacement[:2], np.array([2.0, 2.0]))


def test_apply_movement_bias_nearest_neighbor_moves_closer() -> None:
    """Nearest-neighbor bias should move each robot toward its closest neighbor."""
    positions = [
        np.array([0.0, 0.0, 0.5]),
        np.array([2.0, 0.0, 0.5]),
    ]
    moved = apply_movement_bias(
        positions,
        movement_bias="nearest_neighbor",
        movement_step_size=0.5,
        world_size=(10.0, 10.0),
        use_periodic_boundaries=False,
    )
    assert moved[0][0] > positions[0][0]
    assert moved[1][0] < positions[1][0]


def test_find_pairs_with_mating_zone_strategy_tracks_zone_indices() -> None:
    """Mating-zone strategy should pair inside zones and report active zones."""
    population = [SpatialIndividual(unique_id=i) for i in range(4)]
    positions = [
        np.array([0.1, 0.0, 0.5]),
        np.array([0.3, 0.0, 0.5]),
        np.array([4.9, 5.0, 0.5]),
        np.array([5.1, 5.0, 0.5]),
    ]

    pairs, paired_indices, zones_with_matings = find_pairs_with_strategy(
        population,
        positions,
        pairing_radius=0.5,
        world_size=(10.0, 10.0),
        method="mating_zone",
        mating_zone_centers=[(0.0, 0.0), (5.0, 5.0)],
        mating_zone_radius=1.0,
    )

    assert len(pairs) == 2
    assert paired_indices == {0, 1, 2, 3}
    assert zones_with_matings == {0, 1}


def test_apply_movement_bias_assigned_zone_targets_configured_zone() -> None:
    """Assigned-zone movement should move robots toward their own centers."""
    positions = [
        np.array([5.0, 5.0, 0.5]),
        np.array([5.0, 5.0, 0.5]),
    ]
    moved = apply_movement_bias(
        positions,
        movement_bias="assigned_zone",
        movement_step_size=0.5,
        world_size=(10.0, 10.0),
        use_periodic_boundaries=False,
        mating_zone_centers=[(9.0, 5.0), (1.0, 5.0)],
        assigned_zone_indices=[0, 1],
    )

    # Same start, opposite destinations.
    assert moved[0][0] > 5.0
    assert moved[1][0] < 5.0


def test_apply_world_boundaries_clips_into_the_world() -> None:
    """Without wrapping, positions are clipped into ``[0, world_size]``.

    Both boundary modes use the same ``[0, W]`` framing, so a configuration
    reads the same whether or not boundaries are periodic.
    """
    clipped = apply_world_boundaries(
        np.array([12.0, -3.0, 0.5]),
        (10.0, 10.0),
        use_periodic_boundaries=False,
    )

    assert clipped[0] == 10.0
    assert clipped[1] == 0.0
    # The height is untouched.
    assert clipped[2] == 0.5


def test_is_in_mating_zone() -> None:
    """Zone membership should respect the radius and wrap when periodic."""
    assert is_in_mating_zone(
        np.array([5.5, 5.0, 0.1]),
        (5.0, 5.0),
        1.0,
        (10.0, 10.0),
    )
    assert not is_in_mating_zone(
        np.array([7.0, 5.0, 0.1]),
        (5.0, 5.0),
        1.0,
        (10.0, 10.0),
    )
    # Across the seam: 0.5 and 9.8 are 0.7 apart on a torus.
    assert is_in_mating_zone(
        np.array([0.5, 5.0, 0.1]),
        (9.8, 5.0),
        1.0,
        (10.0, 10.0),
        use_periodic_boundaries=True,
    )


def test_find_nearest_periodic_looks_through_the_seam() -> None:
    """The closest neighbour may be the one across the boundary."""
    current = np.array([0.5, 5.0, 0.1])
    others = [
        np.array([4.0, 5.0, 0.1]),
        np.array([9.6, 5.0, 0.1]),
    ]

    index, distance = find_nearest_periodic(current, others, (10.0, 10.0))

    assert index == 1
    assert distance == pytest.approx(0.9)


def test_find_nearest_periodic_honours_exclusions() -> None:
    """Excluded candidates should be skipped, including the searcher."""
    current = np.array([0.5, 5.0, 0.1])
    others = [current.copy(), np.array([4.0, 5.0, 0.1])]

    index, _ = find_nearest_periodic(
        current,
        others,
        (10.0, 10.0),
        exclude_indices={0},
    )

    assert index == 1


def test_check_periodic_spawn_overlap() -> None:
    """Overlap detection should account for wrap-around proximity."""
    existing = [np.array([9.8, 5.0, 0.1])]

    assert not check_periodic_spawn_overlap(
        np.array([0.2, 5.0, 0.1]),
        existing,
        (10.0, 10.0),
        min_distance=1.0,
    )
    assert check_periodic_spawn_overlap(
        np.array([5.0, 5.0, 0.1]),
        existing,
        (10.0, 10.0),
        min_distance=1.0,
    )


def test_wrap_offspring_position() -> None:
    """An offspring pushed past the edge reappears on the other side."""
    child = wrap_offspring_position(
        np.array([9.5, 0.5, 0.1]),
        np.array([1.0, -1.0, 0.0]),
        (10.0, 10.0),
    )

    assert child[0] == pytest.approx(0.5)
    assert child[1] == pytest.approx(9.5)


def test_split_trajectory_at_wraps() -> None:
    """A path that wraps should be cut so plots do not cross the world."""
    trajectory = [
        np.array([8.0, 5.0]),
        np.array([9.5, 5.0]),
        np.array([0.5, 5.0]),
        np.array([1.5, 5.0]),
    ]

    segments = split_trajectory_at_wraps(trajectory, (10.0, 10.0))

    assert len(segments) == 2
    assert len(segments[0]) == 2
    assert len(segments[1]) == 2


def test_split_trajectory_leaves_continuous_paths_alone() -> None:
    """A path that never wraps stays in one piece."""
    trajectory = [np.array([1.0, 1.0]), np.array([2.0, 2.0])]
    segments = split_trajectory_at_wraps(trajectory, (10.0, 10.0))

    assert len(segments) == 1
    assert len(segments[0]) == 2


def test_find_pairs_with_random_strategy() -> None:
    """Random pairing should ignore distance and pair everyone it can."""
    population = [SpatialIndividual(unique_id=i) for i in range(5)]
    positions = [np.array([float(i) * 9.0, 0.0, 0.1]) for i in range(5)]

    pairs, paired_indices, zones = find_pairs_with_strategy(
        population,
        positions,
        pairing_radius=0.001,
        world_size=(100.0, 100.0),
        method="random",
    )

    assert len(pairs) == 2
    assert len(paired_indices) == 4
    assert zones == set()
