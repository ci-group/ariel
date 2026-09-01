"""Test: mating-zone helpers for the spatial EA."""

from ariel.spatial_ea.interaction import relocate_zone_centers


def test_relocate_zone_centers() -> None:
    """Selected zones should be relocated while preserving list length."""
    centers = [(0.0, 0.0), (2.0, 2.0), (4.0, 4.0)]
    updated = relocate_zone_centers(
        centers,
        {1},
        world_size=(10.0, 10.0),
        zone_radius=1.0,
        min_zone_distance=2.0,
    )

    assert len(updated) == len(centers)
    assert updated[0] == centers[0]
    assert updated[2] == centers[2]
