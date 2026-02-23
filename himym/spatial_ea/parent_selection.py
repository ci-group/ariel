import numpy as np
from typing import Any

try:
    from spatial_individual import SpatialIndividual
    from periodic_boundary_utils import periodic_distance
except ImportError:
    from himym.spatial_ea.spatial_individual import SpatialIndividual
    from himym.spatial_ea.periodic_boundary_utils import periodic_distance


def find_pairs(
    population: list[SpatialIndividual],
    tracked_geoms: list,
    method: str,
    pairing_radius: float,
    world_size: tuple[float, float],
    use_periodic_boundaries: bool = False,
    **kwargs
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Find mating pairs in the population."""
    if method == "proximity_pairing":
        pairs, paired_indices = _proximity_pairing(
            population, tracked_geoms, pairing_radius, 
            world_size, use_periodic_boundaries
        )
        return pairs, paired_indices, set()
    elif method == "random":
        pairs, paired_indices = _random_pairing(population)
        return pairs, paired_indices, set()
    elif method == "mating_zone":
        mating_zone_centers = kwargs.get('mating_zone_centers', None)
        if mating_zone_centers is None:
            mating_zone_center = kwargs.get('mating_zone_center', (0.0, 0.0))
            mating_zone_centers = [mating_zone_center]
        
        mating_zone_radius = kwargs.get('mating_zone_radius', 3.0)
        return _mating_zone_pairing(
            population, tracked_geoms, pairing_radius,
            world_size, use_periodic_boundaries,
            mating_zone_centers, mating_zone_radius
        )
    else:
        print(f"  Warning: Unknown pairing method '{method}', using proximity_pairing")
        pairs, paired_indices = _proximity_pairing(
            population, tracked_geoms, pairing_radius,
            world_size, use_periodic_boundaries
        )
        return pairs, paired_indices, set()


def _proximity_pairing(
    population: list[SpatialIndividual],
    tracked_geoms: list,
    pairing_radius: float,
    world_size: tuple[float, float],
    use_periodic_boundaries: bool
) -> tuple[list[tuple[int, int]], set[int]]:
    """Pair individuals based on nearest neighbor within pairing radius."""
    pairs = []
    paired_indices = set()
    pair_distances = []
    
    for idx in range(len(population)):
        if idx in paired_indices:
            continue
            
        current_pos = tracked_geoms[idx].xpos.copy()
        nearest_partner_idx = None
        nearest_distance = float('inf')
        
        for other_idx in range(len(population)):
            if other_idx == idx or other_idx in paired_indices:
                continue
            
            other_pos = tracked_geoms[other_idx].xpos.copy()
            
            if use_periodic_boundaries:
                distance = periodic_distance(current_pos, other_pos, world_size)
            else:
                distance = np.linalg.norm(current_pos - other_pos)
            
            if distance <= pairing_radius and distance < nearest_distance:
                nearest_distance = distance
                nearest_partner_idx = other_idx
        
        if nearest_partner_idx is not None:
            pairs.append((idx, nearest_partner_idx))
            paired_indices.add(idx)
            paired_indices.add(nearest_partner_idx)
            pair_distances.append(nearest_distance)
    
    if pair_distances:
        print(f"    Proximity pairing distances: min={min(pair_distances):.2f}m, "
              f"max={max(pair_distances):.2f}m, avg={np.mean(pair_distances):.2f}m")
    
    return pairs, paired_indices


def _random_pairing(
    population: list[SpatialIndividual]
) -> tuple[list[tuple[int, int]], set[int]]:
    """Randomly pair individuals from the population."""
    pairs = []
    paired_indices = set()
    
    indices = list(range(len(population)))
    np.random.shuffle(indices)
    
    for i in range(0, len(indices) - 1, 2):
        idx1 = indices[i]
        idx2 = indices[i + 1]
        pairs.append((idx1, idx2))
        paired_indices.add(idx1)
        paired_indices.add(idx2)
    
    return pairs, paired_indices

def _mating_zone_pairing(
    population: list[SpatialIndividual],
    tracked_geoms: list,
    pairing_radius: float,
    world_size: tuple[float, float],
    use_periodic_boundaries: bool,
    mating_zone_centers: list[tuple[float, float]],
    mating_zone_radius: float
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Pair individuals based on nearest neighbor within mating zones."""
    pairs = []
    paired_indices = set()
    zones_with_matings = set()
    
    robots_per_zone: dict[int, list[int]] = {i: [] for i in range(len(mating_zone_centers))}
    
    for idx in range(len(population)):
        robot_pos = tracked_geoms[idx].xpos.copy()
        
        for zone_idx, zone_center in enumerate(mating_zone_centers):
            zone_center_3d = np.array([zone_center[0], zone_center[1], 0.0])
            
            if use_periodic_boundaries:
                distance_to_center = periodic_distance(robot_pos, zone_center_3d, world_size)
            else:
                distance_to_center = np.linalg.norm(robot_pos - zone_center_3d)
            
            if distance_to_center <= mating_zone_radius:
                robots_per_zone[zone_idx].append(idx)
    
    for zone_idx, robots_in_zone in robots_per_zone.items():
        zone_paired_in_this_zone = False
        
        for idx in robots_in_zone:
            if idx in paired_indices:
                continue
            
            current_pos = tracked_geoms[idx].xpos.copy()
            nearest_partner_idx = None
            nearest_distance = float('inf')
            
            for other_idx in robots_in_zone:
                if other_idx == idx or other_idx in paired_indices:
                    continue
                
                other_pos = tracked_geoms[other_idx].xpos.copy()
                
                if use_periodic_boundaries:
                    distance = periodic_distance(current_pos, other_pos, world_size)
                else:
                    distance = np.linalg.norm(current_pos - other_pos)
                
                if distance <= pairing_radius and distance < nearest_distance:
                    nearest_distance = distance
                    nearest_partner_idx = other_idx
            
            if nearest_partner_idx is not None:
                pairs.append((idx, nearest_partner_idx))
                paired_indices.add(idx)
                paired_indices.add(nearest_partner_idx)
                zone_paired_in_this_zone = True
        
        if zone_paired_in_this_zone:
            zones_with_matings.add(zone_idx)
    
    return pairs, paired_indices, zones_with_matings

def generate_random_zone_centers(
    num_zones: int,
    world_size: tuple[float, float],
    zone_radius: float,
    min_zone_distance: float,
    margin: float = 1.0
) -> list[tuple[float, float]]:
    """Generate random positions for multiple mating zones."""
    if num_zones <= 0:
        return []
    
    zone_centers = []
    min_distance = zone_radius * min_zone_distance
    
    x_min = margin + zone_radius
    x_max = world_size[0] - margin - zone_radius
    y_min = margin + zone_radius
    y_max = world_size[1] - margin - zone_radius
    
    max_attempts = 1000 * num_zones
    attempts = 0
    
    while len(zone_centers) < num_zones and attempts < max_attempts:
        attempts += 1
        
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        new_center = (x, y)
        
        too_close = False
        for existing_center in zone_centers:
            distance = np.sqrt((x - existing_center[0])**2 + (y - existing_center[1])**2)
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            zone_centers.append(new_center)
    
    if len(zone_centers) < num_zones:
        print(f"  Warning: Could only place {len(zone_centers)} of {num_zones} zones")
        print(f"  Consider reducing num_mating_zones or min_zone_distance")
    
    return zone_centers

def get_mating_zone_info(config: Any) -> dict[str, Any]:
    """Get information about the mating zone for visualization or robot behavior."""
    return {
        'center': tuple(config.mating_zone_center),
        'radius': config.mating_zone_radius,
        'enabled': config.pairing_method == 'mating_zone'
    }

def is_in_mating_zone(
    position: np.ndarray,
    mating_zone_centers: list[tuple[float, float]],
    mating_zone_radius: float,
    world_size: tuple[float, float],
    use_periodic_boundaries: bool = False
) -> bool:
    """Check if a robot position is within any mating zone."""
    for zone_center in mating_zone_centers:
        zone_center_3d = np.array([zone_center[0], zone_center[1], 0.0])
        
        if use_periodic_boundaries:
            distance = periodic_distance(position, zone_center_3d, world_size)
        else:
            distance = np.linalg.norm(position - zone_center_3d)
        
        if distance <= mating_zone_radius:
            return True
    
    return False

def calculate_offspring_positions(
    pairs: list[tuple[int, int]],
    current_positions: list[np.ndarray],
    offspring_radius: float,
    world_size: tuple[float, float],
    use_periodic_boundaries: bool = False
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Calculate spawn positions for offspring from parent pairs."""
    pair_positions = []
    
    for parent1_idx, parent2_idx in pairs:
        parent1_pos = current_positions[parent1_idx]
        parent2_pos = current_positions[parent2_idx]
        
        angle1 = np.random.uniform(0, 2 * np.pi)
        child1_offset = np.array([
            offspring_radius * np.cos(angle1),
            offspring_radius * np.sin(angle1),
            0.0
        ])
        
        angle2 = np.random.uniform(0, 2 * np.pi)
        child2_offset = np.array([
            offspring_radius * np.cos(angle2),
            offspring_radius * np.sin(angle2),
            0.0
        ])
        
        if use_periodic_boundaries:
            try:
                from periodic_boundary_utils import wrap_offspring_position
            except ImportError:
                from himym.spatial_ea.periodic_boundary_utils import wrap_offspring_position
            
            child1_pos = wrap_offspring_position(
                parent1_pos, child1_offset, world_size
            )
            child2_pos = wrap_offspring_position(
                parent2_pos, child2_offset, world_size
            )
        else:
            child1_pos = parent1_pos + child1_offset
            child2_pos = parent2_pos + child2_offset
        
        pair_positions.append((child1_pos, child2_pos))
    
    return pair_positions
