"""Survivor selection policies for the spatial EA.

Selection methods fall into two families. Truncating methods
(``fitness_based``, ``age_based``, ``parents_die``, ``zone_capacity``) hold the
population at a target size. Stochastic methods (``probabilistic_age``,
``energy_based``, ``density_based``) let each individual die independently, so
the population is free to grow or collapse — extinction and explosion are
legitimate outcomes of a run rather than errors.
"""

# Standard library
from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

# Third-party libraries
import numpy as np

# Local libraries
from ariel.spatial_ea.interaction import calculate_periodic_distance

# Evaluate type annotations in a deferred manner (ruff: UP037)
if TYPE_CHECKING:
    from ariel.parameters.ariel_types import FloatArray
    from ariel.spatial_ea.individual import SpatialIndividual

# Global constants
SIZE_ENFORCING_METHODS = frozenset({
    "fitness_based",
    "age_based",
    "parents_die",
    "zone_capacity",
})


def select_top_individuals(
    population: list[SpatialIndividual],
    target_size: int,
) -> list[SpatialIndividual]:
    """Keep the highest-fitness individuals.

    Parameters
    ----------
    population
        Candidates to rank.
    target_size
        Number of individuals to keep.

    Returns
    -------
        The ``target_size`` fittest individuals, best first.
    """
    if target_size <= 0:
        return []
    return sorted(
        population,
        key=lambda individual: individual.fitness,
        reverse=True,
    )[:target_size]


def _calculate_local_densities(
    positions: list[FloatArray],
    world_size: tuple[float, float],
    locality_radius: float,
    *,
    use_periodic_boundaries: bool,
) -> FloatArray:
    """Compute a Gaussian-kernel crowding measure for each position.

    Parameters
    ----------
    positions
        Positions to measure.
    world_size
        World dimensions ``(width, height)``.
    locality_radius
        Kernel width, the sigma of the Gaussian.
    use_periodic_boundaries
        Whether distances wrap around the world edges.

    Returns
    -------
        One density per position; higher means more crowded.
    """
    count = len(positions)
    densities = np.zeros(count, dtype=float)
    if count == 0 or locality_radius <= 0:
        return densities

    two_sigma_sq = 2.0 * locality_radius * locality_radius
    for i in range(count):
        for j in range(i + 1, count):
            if use_periodic_boundaries:
                distance = calculate_periodic_distance(
                    positions[i],
                    positions[j],
                    world_size,
                )
            else:
                distance = float(
                    np.linalg.norm(positions[i][:2] - positions[j][:2]),
                )

            contribution = float(np.exp(-(distance * distance) / two_sigma_sq))
            densities[i] += contribution
            densities[j] += contribution

    return densities


def _normalised_fitness(
    population: list[SpatialIndividual],
) -> FloatArray:
    """Scale fitness onto ``[0, 1]`` across the population.

    Parameters
    ----------
    population
        Individuals whose fitness should be scaled.

    Returns
    -------
        The scaled fitness values, all ones when every fitness is equal.
    """
    values = np.asarray(
        [individual.fitness for individual in population],
        dtype=float,
    )
    lowest = float(np.min(values))
    highest = float(np.max(values))
    if highest <= lowest:
        return np.ones(len(population), dtype=float)
    return (values - lowest) / (highest - lowest)


def _select_parents_die(
    population: list[SpatialIndividual],
    target_size: int,
    current_generation: int,
    paired_indices: set[int] | None,
) -> list[SpatialIndividual]:
    """Retire the individuals that reproduced, keeping their offspring.

    Parameters
    ----------
    population
        Parents followed by this generation's offspring.
    target_size
        Population size to hold.
    current_generation
        The generation the parents belong to.
    paired_indices
        Indices of parents that reproduced this generation.

    Returns
    -------
        The surviving individuals.
    """
    paired = paired_indices if paired_indices is not None else set()

    offspring: list[SpatialIndividual] = []
    mated_parents: list[SpatialIndividual] = []
    survivors: list[SpatialIndividual] = []

    for index, individual in enumerate(population):
        if individual.generation > current_generation:
            offspring.append(individual)
        elif index in paired and individual.generation == current_generation:
            mated_parents.append(individual)
        else:
            survivors.append(individual)

    candidates = offspring + survivors

    if len(candidates) < target_size and mated_parents:
        needed = target_size - len(candidates)
        candidates.extend(select_top_individuals(mated_parents, needed))
    elif len(candidates) > target_size:
        candidates = select_top_individuals(candidates, target_size)

    return candidates


def _select_zone_capacity(
    population: list[SpatialIndividual],
    target_size: int,
    zone_indices: list[int] | None,
    num_zones: int | None,
    zone_capacity_softness: int,
) -> list[SpatialIndividual]:
    """Cap how many individuals each mating zone may carry.

    Parameters
    ----------
    population
        Candidates to thin.
    target_size
        Population size to hold.
    zone_indices
        Zone each candidate belongs to, in population order.
    num_zones
        Number of zones in play.
    zone_capacity_softness
        Extra headroom per zone above the even share.

    Returns
    -------
        The surviving individuals.
    """
    if zone_indices is None or num_zones is None or num_zones <= 0:
        return select_top_individuals(population, target_size)

    grouped: dict[int, list[int]] = {zone: [] for zone in range(num_zones)}
    for index, zone_index in enumerate(zone_indices):
        if 0 <= zone_index < num_zones:
            grouped[zone_index].append(index)

    capacity = max(1, math.ceil(target_size / num_zones)) + max(
        0,
        zone_capacity_softness - 1,
    )

    selected_indices: list[int] = []
    for zone in range(num_zones):
        ranked = sorted(
            grouped[zone],
            key=lambda idx: population[idx].fitness,
            reverse=True,
        )
        selected_indices.extend(ranked[:capacity])

    selected = [population[index] for index in dict.fromkeys(selected_indices)]

    if len(selected) < target_size:
        chosen = {id(individual) for individual in selected}
        remaining = [
            individual
            for individual in select_top_individuals(
                population,
                len(population),
            )
            if id(individual) not in chosen
        ]
        selected.extend(remaining[: target_size - len(selected)])
    elif len(selected) > target_size:
        selected = select_top_individuals(selected, target_size)

    return selected


def _select_probabilistic_age(
    population: list[SpatialIndividual],
    current_generation: int,
    max_age: int,
) -> list[SpatialIndividual]:
    """Kill individuals with a probability that grows with their age.

    The population size is not enforced, so an ageing population can shrink
    toward extinction.

    Parameters
    ----------
    population
        Candidates.
    current_generation
        Generation used to measure age.
    max_age
        Age at which death becomes certain.

    Returns
    -------
        The surviving individuals.
    """
    survivors: list[SpatialIndividual] = []
    for individual in population:
        age = individual.age_at(current_generation)
        death_probability = min(1.0, age / max_age) if max_age > 0 else 0.0
        if random.random() > death_probability:
            survivors.append(individual)
    return survivors


def _select_energy_based(
    population: list[SpatialIndividual],
) -> list[SpatialIndividual]:
    """Kill individuals whose energy has run out.

    The population size is not enforced.

    Parameters
    ----------
    population
        Candidates.

    Returns
    -------
        The individuals with energy left.
    """
    return [individual for individual in population if individual.energy > 0]


def _select_density_based(
    population: list[SpatialIndividual],
    positions: list[FloatArray] | None,
    world_size: tuple[float, float] | None,
    *,
    use_periodic_boundaries: bool,
    locality_radius: float,
    critical_density: float,
    base_death_prob: float,
    max_death_prob: float,
    fitness_protection: float,
) -> list[SpatialIndividual]:
    """Kill individuals with a probability that grows with local crowding.

    Death probability is ``P_base + P_max * (1 - exp(-rho / rho_c))``, which
    creates negative feedback between density and survival. The population size
    is not enforced.

    Parameters
    ----------
    population
        Candidates.
    positions
        Position of each candidate. Falls back to fitness truncation when
        missing or mismatched.
    world_size
        World dimensions ``(width, height)``.
    use_periodic_boundaries
        Whether distances wrap around the world edges.
    locality_radius
        Gaussian kernel width used for the density estimate.
    critical_density
        Density at which the crowding term reaches about 63 percent of its
        maximum.
    base_death_prob
        Death probability for a completely isolated individual.
    max_death_prob
        Largest additional death probability crowding can contribute.
    fitness_protection
        How much high fitness reduces death probability, from zero to one.

    Returns
    -------
        The surviving individuals.
    """
    if (
        positions is None
        or world_size is None
        or len(positions) != len(population)
    ):
        return select_top_individuals(population, len(population))

    densities = _calculate_local_densities(
        positions,
        world_size,
        locality_radius,
        use_periodic_boundaries=use_periodic_boundaries,
    )

    if fitness_protection > 0.0:
        normalised = _normalised_fitness(population)
    else:
        normalised = np.zeros(len(population), dtype=float)

    survivors: list[SpatialIndividual] = []
    for index, individual in enumerate(population):
        if critical_density > 0.0:
            density_factor = 1.0 - float(
                np.exp(-densities[index] / critical_density),
            )
        else:
            density_factor = 0.0

        death_prob = base_death_prob + max_death_prob * density_factor
        if fitness_protection > 0.0:
            death_prob *= 1.0 - float(normalised[index] * fitness_protection)
        death_prob = min(1.0, max(0.0, death_prob))

        if random.random() > death_prob:
            survivors.append(individual)

    return survivors


def select_individuals(
    population: list[SpatialIndividual],
    target_size: int,
    *,
    method: str,
    current_generation: int,
    paired_indices: set[int] | None = None,
    max_age: int = 10,
    zone_indices: list[int] | None = None,
    num_zones: int | None = None,
    zone_capacity_softness: int = 1,
    positions: list[FloatArray] | None = None,
    world_size: tuple[float, float] | None = None,
    use_periodic_boundaries: bool = False,
    density_locality_radius: float = 3.0,
    density_critical_density: float = 5.0,
    density_base_death_prob: float = 0.05,
    density_max_death_prob: float = 0.8,
    density_fitness_protection: float = 0.0,
) -> list[SpatialIndividual]:
    """Choose which individuals survive into the next generation.

    Parameters
    ----------
    population
        This generation's parents followed by their offspring.
    target_size
        Population size the size-enforcing methods aim for. Ignored by the
        stochastic methods.
    method
        Selection policy to apply.
    current_generation
        The generation the parents belong to.
    paired_indices
        Indices of parents that reproduced, used by ``parents_die``.
    max_age
        Age of certain death, used by ``probabilistic_age``.
    zone_indices
        Zone of each candidate, used by ``zone_capacity``.
    num_zones
        Number of zones in play, used by ``zone_capacity``.
    zone_capacity_softness
        Extra headroom per zone, used by ``zone_capacity``.
    positions
        Position of each candidate, used by ``density_based``.
    world_size
        World dimensions ``(width, height)``.
    use_periodic_boundaries
        Whether distances wrap around the world edges.
    density_locality_radius
        Gaussian kernel width for the density estimate.
    density_critical_density
        Density at which crowding mortality saturates.
    density_base_death_prob
        Baseline death probability.
    density_max_death_prob
        Maximum crowding contribution to death probability.
    density_fitness_protection
        How much high fitness reduces death probability.

    Returns
    -------
        The surviving individuals.
    """
    if not population:
        return []

    if method == "probabilistic_age":
        return _select_probabilistic_age(
            population,
            current_generation,
            max_age,
        )

    if method == "energy_based":
        return _select_energy_based(population)

    if method == "density_based":
        return _select_density_based(
            population,
            positions,
            world_size,
            use_periodic_boundaries=use_periodic_boundaries,
            locality_radius=density_locality_radius,
            critical_density=density_critical_density,
            base_death_prob=density_base_death_prob,
            max_death_prob=density_max_death_prob,
            fitness_protection=density_fitness_protection,
        )

    if target_size <= 0:
        return []

    if method == "age_based":
        return sorted(
            population,
            key=lambda individual: individual.generation,
            reverse=True,
        )[:target_size]

    if method == "parents_die":
        return _select_parents_die(
            population,
            target_size,
            current_generation,
            paired_indices,
        )

    if method == "zone_capacity":
        return _select_zone_capacity(
            population,
            target_size,
            zone_indices,
            num_zones,
            zone_capacity_softness,
        )

    # ``fitness_based`` and any unrecognised method.
    return select_top_individuals(population, target_size)
