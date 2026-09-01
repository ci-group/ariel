"""Spatial geometry, pairing and offspring placement for the spatial EA.

The world is a rectangle spanning ``[0, world_size]`` on both axes. With
periodic boundaries enabled it is a torus: positions wrap, and distances and
displacements take the shortest path around the edges.

Notes
-----
    * Every coordinate in this module uses the ``[0, W]`` framing, including
      the non-periodic clipping branch and the mating zone centres, so a
      configuration reads the same either way.

"""

# Standard library
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

# Third-party libraries
import mujoco
import numpy as np

# Evaluate type annotations in a deferred manner (ruff: UP037)
if TYPE_CHECKING:
    from ariel.parameters.ariel_types import FloatArray
    from ariel.spatial_ea.individual import SpatialIndividual

# Global constants
WRAP_EPSILON = 1e-3


def calculate_periodic_distance(
    pos1: np.ndarray,
    pos2: np.ndarray,
    world_size: tuple[float, float],
) -> float:
    """Compute distance on a toroidal world."""
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])

    if dx > world_size[0] / 2:
        dx = world_size[0] - dx
    if dy > world_size[1] / 2:
        dy = world_size[1] - dy

    return float(np.sqrt(dx**2 + dy**2))


def calculate_periodic_displacement(
    pos1: np.ndarray,
    pos2: np.ndarray,
    world_size: tuple[float, float],
) -> np.ndarray:
    """Return the shortest toroidal displacement vector from pos1 to pos2."""
    dx = float(pos2[0] - pos1[0])
    dy = float(pos2[1] - pos1[1])

    if dx > world_size[0] / 2:
        dx -= world_size[0]
    elif dx < -world_size[0] / 2:
        dx += world_size[0]

    if dy > world_size[1] / 2:
        dy -= world_size[1]
    elif dy < -world_size[1] / 2:
        dy += world_size[1]

    return np.array([dx, dy, 0.0], dtype=float)


def apply_world_boundaries(
    position: np.ndarray,
    world_size: tuple[float, float],
    *,
    use_periodic_boundaries: bool,
) -> np.ndarray:
    """Keep a position inside the world.

    Parameters
    ----------
    position
        Position to constrain. Only x and y are touched.
    world_size
        World dimensions ``(width, height)``.
    use_periodic_boundaries
        Wrap around the edges when true, clip to them when false.

    Returns
    -------
        A new position array inside ``[0, world_size]`` on both axes.
    """
    bounded = position.copy()
    if use_periodic_boundaries:
        bounded[0] %= world_size[0]
        bounded[1] %= world_size[1]
    else:
        bounded[0] = np.clip(bounded[0], 0.0, world_size[0])
        bounded[1] = np.clip(bounded[1], 0.0, world_size[1])
    return bounded


def apply_movement_bias(
    positions: list[np.ndarray],
    *,
    movement_bias: str,
    movement_step_size: float,
    world_size: tuple[float, float],
    use_periodic_boundaries: bool,
    mating_zone_centers: list[tuple[float, float]] | None = None,
    assigned_zone_indices: list[int] | None = None,
) -> list[np.ndarray]:
    """Nudge positions according to movement-bias strategy."""
    if movement_step_size <= 0.0 or movement_bias == "none" or not positions:
        return [position.copy() for position in positions]

    updated: list[np.ndarray] = []
    for idx, current_pos in enumerate(positions):
        target_vector = np.zeros(3, dtype=float)

        if movement_bias == "nearest_neighbor" and len(positions) > 1:
            nearest_idx: int | None = None
            nearest_distance = float("inf")
            for other_idx, other_pos in enumerate(positions):
                if other_idx == idx:
                    continue
                if use_periodic_boundaries:
                    distance = calculate_periodic_distance(
                        current_pos,
                        other_pos,
                        world_size,
                    )
                else:
                    distance = float(
                        np.linalg.norm(current_pos[:2] - other_pos[:2]),
                    )
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_idx = other_idx

            if nearest_idx is not None:
                neighbor_pos = positions[nearest_idx]
                if use_periodic_boundaries:
                    target_vector = calculate_periodic_displacement(
                        current_pos,
                        neighbor_pos,
                        world_size,
                    )
                else:
                    target_vector = neighbor_pos - current_pos

        elif movement_bias == "nearest_zone" and mating_zone_centers:
            nearest_center: tuple[float, float] | None = None
            nearest_distance = float("inf")
            for cx, cy in mating_zone_centers:
                zone_pos = np.array([cx, cy, current_pos[2]], dtype=float)
                if use_periodic_boundaries:
                    distance = calculate_periodic_distance(
                        current_pos,
                        zone_pos,
                        world_size,
                    )
                else:
                    distance = float(
                        np.linalg.norm(current_pos[:2] - zone_pos[:2]),
                    )
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_center = (cx, cy)

            if nearest_center is not None:
                zone_pos = np.array(
                    [nearest_center[0], nearest_center[1], current_pos[2]],
                    dtype=float,
                )
                if use_periodic_boundaries:
                    target_vector = calculate_periodic_displacement(
                        current_pos,
                        zone_pos,
                        world_size,
                    )
                else:
                    target_vector = zone_pos - current_pos

        elif (
            movement_bias == "assigned_zone"
            and mating_zone_centers
            and assigned_zone_indices is not None
        ):
            if idx < len(assigned_zone_indices):
                zone_index = assigned_zone_indices[idx]
                if 0 <= zone_index < len(mating_zone_centers):
                    cx, cy = mating_zone_centers[zone_index]
                    zone_pos = np.array([cx, cy, current_pos[2]], dtype=float)
                    if use_periodic_boundaries:
                        target_vector = calculate_periodic_displacement(
                            current_pos,
                            zone_pos,
                            world_size,
                        )
                    else:
                        target_vector = zone_pos - current_pos

        norm = float(np.linalg.norm(target_vector[:2]))
        if norm > 0.0:
            target_vector /= norm
            candidate = current_pos + movement_step_size * target_vector
        else:
            candidate = current_pos.copy()

        updated.append(
            apply_world_boundaries(
                candidate,
                world_size,
                use_periodic_boundaries=use_periodic_boundaries,
            ),
        )

    return updated


def _find_nearest_partner(
    idx: int,
    candidate_indices: list[int],
    positions: list[np.ndarray],
    *,
    pairing_radius: float,
    world_size: tuple[float, float],
    use_periodic_boundaries: bool,
    paired_indices: set[int],
) -> int | None:
    """Return nearest viable partner index for one individual."""
    position = positions[idx]
    nearest_idx: int | None = None
    nearest_distance = float("inf")

    for other_idx in candidate_indices:
        if other_idx == idx or other_idx in paired_indices:
            continue

        other_position = positions[other_idx]
        if use_periodic_boundaries:
            distance = calculate_periodic_distance(
                position,
                other_position,
                world_size,
            )
        else:
            distance = float(np.linalg.norm(position[:2] - other_position[:2]))

        if distance <= pairing_radius and distance < nearest_distance:
            nearest_distance = distance
            nearest_idx = other_idx

    return nearest_idx


def find_pairs_with_strategy(
    population: list[SpatialIndividual],
    positions: list[np.ndarray],
    *,
    pairing_radius: float,
    world_size: tuple[float, float],
    use_periodic_boundaries: bool = False,
    method: str = "proximity_pairing",
    mating_zone_centers: list[tuple[float, float]] | None = None,
    mating_zone_radius: float = 3.0,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Pair individuals using proximity, random, or mating-zone strategy.

    Parameters
    ----------
    population
        The individuals being paired, indexed in step with ``positions``.
    positions
        Current position of each individual.
    pairing_radius
        Maximum separation at which two individuals may pair.
    world_size
        World dimensions ``(width, height)``.
    use_periodic_boundaries
        Whether distances wrap around the world edges.
    method
        One of ``proximity_pairing``, ``random`` or ``mating_zone``.
    mating_zone_centers
        Zone centres, required by the ``mating_zone`` method.
    mating_zone_radius
        Radius of each mating zone.

    Returns
    -------
    pairs
        Index pairs that will reproduce.
    paired_indices
        Every index that appears in ``pairs``.
    zones_with_matings
        Indices of zones in which at least one pairing happened, used to drive
        event-driven zone relocation.
    """
    del population  # Pairing is positional; kept for signature stability.

    pairs: list[tuple[int, int]] = []
    paired_indices: set[int] = set()
    zones_with_matings: set[int] = set()

    all_indices = list(range(len(positions)))

    if method == "random":
        shuffled = list(all_indices)
        np.random.shuffle(shuffled)
        for i in range(0, len(shuffled) - 1, 2):
            first = shuffled[i]
            second = shuffled[i + 1]
            pairs.append((first, second))
            paired_indices.add(first)
            paired_indices.add(second)
        return pairs, paired_indices, zones_with_matings

    if method == "mating_zone" and mating_zone_centers:
        for zone_index, (cx, cy) in enumerate(mating_zone_centers):
            zone_position = np.array([cx, cy, 0.0], dtype=float)
            zone_members: list[int] = []
            for idx, position in enumerate(positions):
                if idx in paired_indices:
                    continue

                if use_periodic_boundaries:
                    distance_to_zone = calculate_periodic_distance(
                        position,
                        zone_position,
                        world_size,
                    )
                else:
                    distance_to_zone = float(
                        np.linalg.norm(position[:2] - zone_position[:2]),
                    )

                if distance_to_zone <= mating_zone_radius:
                    zone_members.append(idx)

            zone_had_mating = False
            for idx in zone_members:
                if idx in paired_indices:
                    continue
                partner_idx = _find_nearest_partner(
                    idx,
                    zone_members,
                    positions,
                    pairing_radius=pairing_radius,
                    world_size=world_size,
                    use_periodic_boundaries=use_periodic_boundaries,
                    paired_indices=paired_indices,
                )
                if partner_idx is not None:
                    pairs.append((idx, partner_idx))
                    paired_indices.add(idx)
                    paired_indices.add(partner_idx)
                    zone_had_mating = True

            if zone_had_mating:
                zones_with_matings.add(zone_index)

        return pairs, paired_indices, zones_with_matings

    # Default: nearest-neighbor proximity pairing.
    for idx in all_indices:
        if idx in paired_indices:
            continue

        partner_idx = _find_nearest_partner(
            idx,
            all_indices,
            positions,
            pairing_radius=pairing_radius,
            world_size=world_size,
            use_periodic_boundaries=use_periodic_boundaries,
            paired_indices=paired_indices,
        )

        if partner_idx is not None:
            pairs.append((idx, partner_idx))
            paired_indices.add(idx)
            paired_indices.add(partner_idx)

    return pairs, paired_indices, zones_with_matings


def generate_random_zone_centers(
    num_zones: int,
    world_size: tuple[float, float],
    zone_radius: float,
    min_zone_distance: float,
    *,
    margin: float = 1.0,
) -> list[tuple[float, float]]:
    """Generate non-overlapping mating zone centers."""
    if num_zones <= 0:
        return []

    centers: list[tuple[float, float]] = []
    min_distance = zone_radius * min_zone_distance

    x_min = margin + zone_radius
    x_max = world_size[0] - margin - zone_radius
    y_min = margin + zone_radius
    y_max = world_size[1] - margin - zone_radius

    if x_min >= x_max:
        x_min, x_max = 0.0, world_size[0]
    if y_min >= y_max:
        y_min, y_max = 0.0, world_size[1]

    attempts = 0
    max_attempts = 1000 * num_zones
    while len(centers) < num_zones and attempts < max_attempts:
        attempts += 1
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        candidate = (x, y)

        if all(np.hypot(x - cx, y - cy) >= min_distance for cx, cy in centers):
            centers.append(candidate)

    # If strict spacing is impossible, fill remaining zones with bounded random centers.
    while len(centers) < num_zones:
        centers.append((
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
        ))

    return centers


def relocate_zone_centers(
    zone_centers: list[tuple[float, float]],
    zone_indices: set[int],
    world_size: tuple[float, float],
    zone_radius: float,
    min_zone_distance: float,
    *,
    margin: float = 1.0,
) -> list[tuple[float, float]]:
    """Relocate selected zones while keeping the rest fixed."""
    if not zone_centers or not zone_indices:
        return zone_centers

    relocated = list(zone_centers)
    available = generate_random_zone_centers(
        num_zones=len(zone_indices),
        world_size=world_size,
        zone_radius=zone_radius,
        min_zone_distance=min_zone_distance,
        margin=margin,
    )

    for zone_idx, new_center in zip(
        sorted(zone_indices),
        available,
        strict=False,
    ):
        if zone_idx < len(relocated):
            relocated[zone_idx] = new_center

    return relocated


def find_pairs(
    population: list[SpatialIndividual],
    positions: list[np.ndarray],
    *,
    pairing_radius: float,
    world_size: tuple[float, float],
    use_periodic_boundaries: bool = False,
) -> tuple[list[tuple[int, int]], set[int]]:
    """Backward-compatible proximity pairing helper."""
    pairs, paired_indices, _ = find_pairs_with_strategy(
        population,
        positions,
        pairing_radius=pairing_radius,
        world_size=world_size,
        use_periodic_boundaries=use_periodic_boundaries,
        method="proximity_pairing",
    )
    return pairs, paired_indices


def calculate_offspring_positions(
    pairs: list[tuple[int, int]],
    positions: list[np.ndarray],
    *,
    offspring_radius: float,
    world_size: tuple[float, float],
    use_periodic_boundaries: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute spawn locations for offspring produced by the selected pairs."""
    offspring_positions: list[tuple[np.ndarray, np.ndarray]] = []

    for parent1_idx, parent2_idx in pairs:
        parent1 = positions[parent1_idx]
        parent2 = positions[parent2_idx]

        offset1 = np.array([
            offspring_radius * np.cos(np.random.uniform(0.0, 2.0 * np.pi)),
            offspring_radius * np.sin(np.random.uniform(0.0, 2.0 * np.pi)),
            0.0,
        ])
        offset2 = np.array([
            offspring_radius * np.cos(np.random.uniform(0.0, 2.0 * np.pi)),
            offspring_radius * np.sin(np.random.uniform(0.0, 2.0 * np.pi)),
            0.0,
        ])

        child1 = parent1 + offset1
        child2 = parent2 + offset2

        child1 = apply_world_boundaries(
            child1,
            world_size,
            use_periodic_boundaries=use_periodic_boundaries,
        )
        child2 = apply_world_boundaries(
            child2,
            world_size,
            use_periodic_boundaries=use_periodic_boundaries,
        )

        offspring_positions.append((child1, child2))

    return offspring_positions


# -- Toroidal helpers ----------------------------------------------------------
def is_in_mating_zone(
    position: FloatArray,
    zone_center: tuple[float, float],
    zone_radius: float,
    world_size: tuple[float, float],
    *,
    use_periodic_boundaries: bool = False,
) -> bool:
    """Test whether a position falls inside a mating zone.

    Parameters
    ----------
    position
        Position to test.
    zone_center
        Centre of the zone.
    zone_radius
        Radius of the zone.
    world_size
        World dimensions ``(width, height)``.
    use_periodic_boundaries
        Whether the distance to the centre wraps around the world edges.

    Returns
    -------
        ``True`` when the position lies within ``zone_radius`` of the centre.
    """
    center = np.array([zone_center[0], zone_center[1], 0.0], dtype=float)
    if use_periodic_boundaries:
        distance = calculate_periodic_distance(position, center, world_size)
    else:
        distance = float(np.linalg.norm(position[:2] - center[:2]))
    return distance <= zone_radius


def find_nearest_periodic(
    current_pos: FloatArray,
    other_positions: list[FloatArray],
    world_size: tuple[float, float],
    exclude_indices: set[int] | None = None,
) -> tuple[int | None, float]:
    """Find the nearest position on a toroidal world.

    Parameters
    ----------
    current_pos
        Position to search from.
    other_positions
        Candidate positions.
    world_size
        World dimensions ``(width, height)``.
    exclude_indices
        Candidate indices to skip.

    Returns
    -------
    nearest_index
        Index of the closest candidate, or ``None`` when there is none.
    nearest_distance
        Distance to that candidate, or infinity.
    """
    excluded = exclude_indices if exclude_indices is not None else set()

    nearest_idx: int | None = None
    nearest_dist = float("inf")
    for i, other_pos in enumerate(other_positions):
        if i in excluded:
            continue
        distance = calculate_periodic_distance(
            current_pos,
            other_pos,
            world_size,
        )
        if distance < nearest_dist:
            nearest_dist = distance
            nearest_idx = i

    return nearest_idx, nearest_dist


def check_periodic_spawn_overlap(
    new_pos: FloatArray,
    existing_positions: list[FloatArray],
    world_size: tuple[float, float],
    min_distance: float,
) -> bool:
    """Test whether a candidate spawn position is far enough from the rest.

    Parameters
    ----------
    new_pos
        Candidate position.
    existing_positions
        Positions already taken.
    world_size
        World dimensions ``(width, height)``.
    min_distance
        Minimum acceptable toroidal separation.

    Returns
    -------
        ``True`` when the candidate clears every existing position.
    """
    return all(
        calculate_periodic_distance(new_pos, existing, world_size)
        >= min_distance
        for existing in existing_positions
    )


def wrap_offspring_position(
    parent_pos: FloatArray,
    offset: FloatArray,
    world_size: tuple[float, float],
) -> FloatArray:
    """Offset a parent position and wrap the result into the world.

    Parameters
    ----------
    parent_pos
        Position of the parent.
    offset
        Displacement applied to the parent position.
    world_size
        World dimensions ``(width, height)``.

    Returns
    -------
        The wrapped offspring position.
    """
    return apply_world_boundaries(
        parent_pos + offset,
        world_size,
        use_periodic_boundaries=True,
    )


def split_trajectory_at_wraps(
    trajectory: list[FloatArray],
    world_size: tuple[float, float],
    wrap_threshold: float = 0.5,
) -> list[list[FloatArray]]:
    """Cut a trajectory wherever it wraps around a world edge.

    Plotting a wrapped trajectory as one polyline draws a spurious line right
    across the world; splitting it into segments avoids that.

    Parameters
    ----------
    trajectory
        Successive ``(x, y)`` positions.
    world_size
        World dimensions ``(width, height)``.
    wrap_threshold
        Fraction of the world size that a single step must exceed to count as
        a wrap rather than genuine movement.

    Returns
    -------
        The trajectory split into contiguous segments.
    """
    if len(trajectory) < 2:
        return [trajectory]

    segments: list[list[FloatArray]] = []
    current: list[FloatArray] = [trajectory[0]]

    for previous, position in itertools.pairwise(trajectory):
        wrapped_x = (
            abs(position[0] - previous[0]) > world_size[0] * wrap_threshold
        )
        wrapped_y = (
            abs(position[1] - previous[1]) > world_size[1] * wrap_threshold
        )
        if wrapped_x or wrapped_y:
            segments.append(current)
            current = [position]
        else:
            current.append(position)

    if current:
        segments.append(current)

    return segments


def apply_periodic_boundaries_to_simulation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    free_joint_qpos_adr: list[int],
    world_size: tuple[float, float],
) -> None:
    """Wrap every robot that has left the world back through the far edge.

    Wrapping is written to each robot's free-joint ``qpos``, which is the state
    MuJoCo integrates. Writing to ``data.geom_xpos`` instead has no effect:
    that array is recomputed from ``qpos`` on the next forward pass.

    Parameters
    ----------
    model
        The compiled model. Needed to re-derive positions after wrapping.
    data
        Simulation state, modified in place.
    free_joint_qpos_adr
        ``qpos`` address of each robot's free joint. Negative entries are
        skipped.
    world_size
        World dimensions ``(width, height)``.
    """
    wrapped_any = False

    for qpos_adr in free_joint_qpos_adr:
        if qpos_adr < 0:
            continue

        x = float(data.qpos[qpos_adr])
        y = float(data.qpos[qpos_adr + 1])
        wrapped_x = x % world_size[0]
        wrapped_y = y % world_size[1]

        if (
            abs(wrapped_x - x) > WRAP_EPSILON
            or abs(wrapped_y - y) > WRAP_EPSILON
        ):
            data.qpos[qpos_adr] = wrapped_x
            data.qpos[qpos_adr + 1] = wrapped_y
            wrapped_any = True

    if wrapped_any:
        mujoco.mj_forward(model, data)
