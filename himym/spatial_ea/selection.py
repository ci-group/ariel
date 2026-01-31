"""Selection strategies for evolutionary algorithm."""

import numpy as np
import random
from spatial_individual import SpatialIndividual


def _calculate_periodic_distance(
    pos1: np.ndarray,
    pos2: np.ndarray,
    world_size: tuple[float, float]
) -> float:
    """Calculate distance with periodic boundary conditions (toroidal world)."""
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    
    # Wrap distances to account for periodic boundaries
    if dx > world_size[0] / 2:
        dx = world_size[0] - dx
    if dy > world_size[1] / 2:
        dy = world_size[1] - dy
    
    return np.sqrt(dx**2 + dy**2)


def _calculate_local_densities(
    positions: list[np.ndarray],
    world_size: tuple[float, float],
    locality_radius: float,
    use_periodic_boundaries: bool = True
) -> np.ndarray:
    """
    Calculate local density for each individual using a Gaussian kernel.
    
    The local density ρ_i for individual i is:
        ρ_i = Σ_j exp(-d_ij² / (2σ²))
    
    where d_ij is the distance between individuals i and j,
    and σ is the locality_radius.
    
    Args:
        positions: List of position arrays for each individual
        world_size: Tuple of (width, height) for the world
        locality_radius: Standard deviation σ for the Gaussian kernel
        use_periodic_boundaries: Whether to use toroidal distance calculation
    
    Returns:
        Array of local density values for each individual
    """
    n = len(positions)
    densities = np.zeros(n)
    
    if n == 0:
        return densities
    
    sigma_sq_2 = 2 * locality_radius ** 2
    
    for i in range(n):
        # Sum contributions from all other individuals (not self)
        for j in range(n):
            if i == j:
                continue
            
            if use_periodic_boundaries:
                dist = _calculate_periodic_distance(positions[i], positions[j], world_size)
            else:
                dist = np.linalg.norm(positions[i] - positions[j])
            
            # Gaussian kernel contribution
            densities[i] += np.exp(-(dist ** 2) / sigma_sq_2)
    
    return densities


def _selection_density_based(
    population: list[SpatialIndividual],
    positions: list[np.ndarray],
    world_size: tuple[float, float],
    use_periodic_boundaries: bool = True,
    locality_radius: float = 3.0,
    critical_density: float = 5.0,
    base_death_prob: float = 0.05,
    max_density_death_prob: float = 0.8,
    fitness_protection: float = 0.0
) -> list[int]:
    """
    Density-dependent death selection based on local crowding.
    
    Creates negative feedback between spatial density and mortality:
    individuals in crowded areas face higher death probability.
    
    Death probability formula:
        P(death) = P_base + P_max × (1 - exp(-ρ/ρ_c))
    
    where:
        - P_base: baseline death probability for isolated individuals
        - P_max: maximum additional death probability at high density
        - ρ: local density (Gaussian kernel sum)
        - ρ_c: critical density threshold
    
    Args:
        population: List of individuals
        positions: Current positions of individuals
        world_size: World dimensions (width, height)
        use_periodic_boundaries: Whether world wraps around
        locality_radius: Gaussian kernel σ for density calculation
        critical_density: ρ_c threshold where P_density ≈ 0.63 × P_max
        base_death_prob: P_base - baseline death probability
        max_density_death_prob: P_max - max additional death prob from density
        fitness_protection: Reduction in death probability for high-fitness individuals (0-1)
    
    Returns:
        List of indices of surviving individuals
    """
    if len(population) == 0:
        return []
    
    # Calculate local densities for all individuals
    densities = _calculate_local_densities(
        positions, world_size, locality_radius, use_periodic_boundaries
    )
    
    survivors = []
    deaths = []
    death_causes = {"crowding": 0, "baseline": 0}
    
    # Normalize fitness for protection calculation (if using fitness protection)
    if fitness_protection > 0 and len(population) > 0:
        fitnesses = np.array([ind.fitness for ind in population])
        min_fit, max_fit = fitnesses.min(), fitnesses.max()
        if max_fit > min_fit:
            normalized_fitness = (fitnesses - min_fit) / (max_fit - min_fit)
        else:
            normalized_fitness = np.ones(len(population))
    else:
        normalized_fitness = np.zeros(len(population))
    
    for i, _ind in enumerate(population):
        local_density = densities[i]
        
        # Calculate density-dependent death probability
        # P(death) = P_base + P_max × (1 - exp(-ρ/ρ_c))
        density_factor = 1.0 - np.exp(-local_density / critical_density) if critical_density > 0 else 0.0
        death_prob = base_death_prob + max_density_death_prob * density_factor
        
        # Apply fitness protection (reduces death probability for fitter individuals)
        if fitness_protection > 0:
            protection = normalized_fitness[i] * fitness_protection
            death_prob = death_prob * (1.0 - protection)
        
        # Clamp probability to [0, 1]
        death_prob = max(0.0, min(1.0, death_prob))
        
        # Stochastic survival
        if random.random() > death_prob:
            survivors.append(i)
        else:
            deaths.append(i)
            if density_factor > 0.5:
                death_causes["crowding"] += 1
            else:
                death_causes["baseline"] += 1
    
    # Print statistics
    if len(densities) > 0:
        print(f"    Density-based selection (σ={locality_radius}, ρ_c={critical_density}):")
        print(f"      Local density range: {densities.min():.2f} to {densities.max():.2f} (mean: {densities.mean():.2f})")
        print(f"      Deaths: {len(deaths)} ({death_causes['crowding']} from crowding, {death_causes['baseline']} from baseline)")
        print(f"      Survivors: {len(survivors)}")
    else:
        print(f"    Density-based selection: No population")
    
    return survivors


def apply_selection(
    population: list[SpatialIndividual],
    current_positions: list[np.ndarray],
    method: str,
    target_size: int,
    current_generation: int,
    current_orientations: list[float] | None = None,
    paired_indices: set[int] | None = None,
    max_age: int = 10,
    # Density-based selection parameters
    world_size: tuple[float, float] | None = None,
    use_periodic_boundaries: bool = True,
    locality_radius: float = 3.0,
    critical_density: float = 5.0,
    base_death_prob: float = 0.05,
    max_density_death_prob: float = 0.8,
    density_fitness_protection: float = 0.0
) -> tuple[list[SpatialIndividual], list[np.ndarray], int, list[float]]:
    """Apply selection to reduce population to target size."""
    initial_size = len(population)
    
    if method == "probabilistic_age":
        print(f"  Applying {method} selection: {initial_size} → natural dynamics (no target)")
        indices_to_keep = _selection_probabilistic_age(
            population, target_size, current_generation, max_age
        )
    elif method == "energy_based":
        print(f"  Applying {method} selection: {initial_size} → natural dynamics (no target)")
        indices_to_keep = _selection_energy_based(population, target_size)
    elif method == "density_based":
        print(f"  Applying {method} selection: {initial_size} → natural dynamics (density-dependent)")
        if world_size is None:
            raise ValueError("density_based selection requires world_size parameter")
        indices_to_keep = _selection_density_based(
            population=population,
            positions=current_positions,
            world_size=world_size,
            use_periodic_boundaries=use_periodic_boundaries,
            locality_radius=locality_radius,
            critical_density=critical_density,
            base_death_prob=base_death_prob,
            max_density_death_prob=max_density_death_prob,
            fitness_protection=density_fitness_protection
        )
    else:
        if initial_size < target_size:
            print(f"  Selection: Population size ({initial_size}) < target ({target_size}), no selection needed")
            orientations = current_orientations if current_orientations else []
            return population, current_positions, initial_size, orientations
        
        print(f"  Applying {method} selection: {initial_size} → {target_size}")
        
        if method == "parents_die":
            indices_to_keep = _selection_parents_die(
                population, target_size, current_generation, paired_indices
            )
        elif method == "fitness_based":
            indices_to_keep = _selection_fitness_based(population, target_size)
        elif method == "age_based":
            indices_to_keep = _selection_age_based(population, target_size)
        else:
            print(f"    Warning: Unknown selection method '{method}', using parents_die")
            indices_to_keep = _selection_parents_die(
                population, target_size, current_generation, paired_indices
            )
    
    indices_to_keep.sort()
    
    if len(population) > 0 and hasattr(population[0], 'robot_index'):
        old_robot_indices = [population[i].robot_index for i in indices_to_keep]
        print(f"    Keeping population indices: {indices_to_keep[:10]}{'...' if len(indices_to_keep) > 10 else ''}")
        print(f"    Their robot_index values: {old_robot_indices[:10]}{'...' if len(old_robot_indices) > 10 else ''}")
    
    new_population = [population[i] for i in indices_to_keep]
    new_positions = [current_positions[i] for i in indices_to_keep]
    new_orientations = [current_orientations[i] for i in indices_to_keep] if current_orientations else []
    new_size = len(new_population)

    print(f"  SELECTION COMPLETE: {initial_size} -> {new_size}")

    if method == "parents_die":
        current_gen_count = sum(1 for ind in new_population if ind.generation == current_generation)
        offspring_count = sum(1 for ind in new_population if ind.generation > current_generation)
        older_gen_count = sum(1 for ind in new_population if ind.generation < current_generation)
        print(f"    Final survivors: {offspring_count} offspring, {current_gen_count} current gen, {older_gen_count} older gen")
    elif method == "fitness_based":
        fitnesses = [ind.fitness for ind in new_population]
        if fitnesses:
            print(f"    Fitness range: {min(fitnesses):.4f} to {max(fitnesses):.4f} (range: {max(fitnesses) - min(fitnesses):.4f})")
        else:
            print(f"    No survivors (population extinct)")
    elif method == "probabilistic_age":
        ages = [current_generation - ind.generation for ind in new_population]
        if ages:
            avg_age = sum(ages) / len(ages)
            print(f"    Survivor age range: {min(ages)} to {max(ages)} generations (avg: {avg_age:.1f})")
            gen_counts = {}
            for ind in new_population:
                gen_counts[ind.generation] = gen_counts.get(ind.generation, 0) + 1
            print(f"    Generation distribution: {dict(sorted(gen_counts.items()))}")
        else:
            print(f"    No survivors (population extinct)")
    elif method == "energy_based":
        if new_population:
            energies = [ind.energy for ind in new_population]
            avg_energy = sum(energies) / len(energies)
            print(f"    Survivor energy: min={min(energies):.1f}, max={max(energies):.1f}, avg={avg_energy:.1f}")
        else:
            print(f"    No survivors (population extinct)")
    elif method == "density_based":
        if new_population:
            ages = [current_generation - ind.generation for ind in new_population]
            avg_age = sum(ages) / len(ages) if ages else 0
            print(f"    Survivor age range: {min(ages)} to {max(ages)} generations (avg: {avg_age:.1f})")
        else:
            print(f"    No survivors (population extinct)")
    
    return new_population, new_positions, new_size, new_orientations


def _selection_parents_die(
    population: list[SpatialIndividual],
    target_size: int,
    current_generation: int,
    paired_indices: set[int] | None = None
) -> list[int]:
    """Selection where parents that mated this generation die and offspring survive."""
    offspring_indices = []
    parents_indices = []
    survivors_indices = []
    
    for i, ind in enumerate(population):
        if ind.generation > current_generation:
            offspring_indices.append(i)
        elif paired_indices is not None and i in paired_indices and ind.generation == current_generation:
            parents_indices.append(i)
        else:
            survivors_indices.append(i)
    
    print(f"    Population breakdown: {len(offspring_indices)} offspring, "
          f"{len(parents_indices)} parents that mated, {len(survivors_indices)} other survivors")
    
    indices_to_keep = offspring_indices + survivors_indices
    
    if len(indices_to_keep) < target_size:
        needed = target_size - len(indices_to_keep)
        print(f"    Not enough offspring+survivors ({len(indices_to_keep)}), keeping {needed} parents")
        
        parents_fitness = [(i, population[i].fitness) for i in parents_indices]
        parents_fitness.sort(key=lambda x: x[1], reverse=True)
        indices_to_keep.extend([i for i, _ in parents_fitness[:needed]])
    
    elif len(indices_to_keep) > target_size:
        print(f"    Too many individuals ({len(indices_to_keep)}), applying fitness selection")
        fitness_ranking = [(i, population[i].fitness) for i in indices_to_keep]
        fitness_ranking.sort(key=lambda x: x[1], reverse=True)
        indices_to_keep = [i for i, _ in fitness_ranking[:target_size]]
    else:
        print(f"    Removing all {len(parents_indices)} parents that mated")
    
    return indices_to_keep


def _selection_fitness_based(
    population: list[SpatialIndividual],
    target_size: int
) -> list[int]:
    """Selection based purely on fitness (keep best individuals)."""
    fitness_ranking = [(i, ind.fitness) for i, ind in enumerate(population)]
    fitness_ranking.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in fitness_ranking[:target_size]]


def _selection_age_based(
    population: list[SpatialIndividual],
    target_size: int
) -> list[int]:
    """Selection based on age (keep youngest individuals)."""
    age_ranking = [(i, ind.generation) for i, ind in enumerate(population)]
    age_ranking.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in age_ranking[:target_size]]


def _selection_probabilistic_age(
    population: list[SpatialIndividual],
    target_size: int,
    current_generation: int,
    max_age: int = 10
) -> list[int]:
    """Selection with age-dependent death probability (no population size enforcement)."""
    survivors = []
    deaths = []
    
    for i, ind in enumerate(population):
        age = current_generation - ind.generation
        p_death = min(1.0, age / max_age) if max_age > 0 else 0.0
        
        if random.random() > p_death:
            survivors.append(i)
        else:
            deaths.append(i)
    
    print(f"    Probabilistic age-based death (max_age={max_age}): {len(deaths)} died, {len(survivors)} survived naturally")
    print(f"    Population change: {len(population)} → {len(survivors)} (no size enforcement)")
    
    return survivors


def _selection_energy_based(
    population: list[SpatialIndividual],
    target_size: int
) -> list[int]:
    """Selection based on energy levels (individuals with energy <= 0 die)."""
    survivors = []
    deaths = []
    
    for i, ind in enumerate(population):
        if ind.energy > 0:
            survivors.append(i)
        else:
            deaths.append(i)
    
    print(f"    Energy-based death: {len(deaths)} died (energy depleted), {len(survivors)} survived")
    print(f"    Population change: {len(population)} → {len(survivors)} (no size enforcement)")
    
    if survivors:
        energy_values = [population[i].energy for i in survivors]
        print(f"    Survivor energy: min={min(energy_values):.1f}, max={max(energy_values):.1f}, avg={np.mean(energy_values):.1f}")
    
    return survivors
