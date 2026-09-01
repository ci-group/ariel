"""The incubation phase that precedes spatial evolution.

A conventional non-spatial generational GA with tournament selection, elitism,
crossover and mutation. Its purpose is to hand the spatial phase a population
that can already locomote: robots dropped into the shared world without it
mostly cannot reach a mate, so nothing ever reproduces and the run dies out
before selection has anything to act on.
"""

# Standard library
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# Third-party libraries
import numpy as np

# Local libraries
from ariel import log
from ariel.spatial_ea.evaluation import evaluate_population
from ariel.spatial_ea.genetics import (
    create_initial_hyperneat_genome,
    crossover_genomes,
    mutate_genome,
)
from ariel.spatial_ea.individual import SpatialIndividual

# Evaluate type annotations in a deferred manner (ruff: UP037)
if TYPE_CHECKING:
    from ariel.spatial_ea.config import SpatialEAConfig


@dataclass
class IncubationEvolution:
    """A non-spatial GA used to pre-adapt controllers.

    Parameters
    ----------
    config
        Run configuration; supplies the incubation parameters and the
        simulation settings used for evaluation.
    next_unique_id
        Identifier counter, continued from the caller.
    population
        Current population; filled by :meth:`run`.
    generation
        Current incubation generation.
    best_fitness_history
        Best fitness per generation.
    avg_fitness_history
        Mean fitness per generation.
    worst_fitness_history
        Worst fitness per generation.
    """

    config: SpatialEAConfig
    next_unique_id: int = 0
    population: list[SpatialIndividual] = field(default_factory=list)
    generation: int = 0
    best_fitness_history: list[float] = field(default_factory=list)
    avg_fitness_history: list[float] = field(default_factory=list)
    worst_fitness_history: list[float] = field(default_factory=list)

    def create_individual(self) -> SpatialIndividual:
        """Create one individual with a fresh random genome.

        Returns
        -------
            The new individual.
        """
        individual = SpatialIndividual(
            unique_id=self.next_unique_id,
            generation=self.generation,
            genotype=create_initial_hyperneat_genome(),
            energy=self.config.initial_energy,
        )
        self.next_unique_id += 1
        return individual

    def initialize_population(self) -> None:
        """Fill the population with fresh random individuals."""
        self.population = [
            self.create_individual()
            for _ in range(self.config.incubation_population_size)
        ]

    def tournament_selection(self) -> SpatialIndividual:
        """Pick one individual by fitness tournament.

        Returns
        -------
            The fittest of a small random sample.
        """
        size = min(self.config.incubation_tournament_size, len(self.population))
        tournament = random.sample(self.population, max(1, size))
        return max(tournament, key=lambda individual: individual.fitness)

    def create_offspring(
        self,
        parent1: SpatialIndividual,
        parent2: SpatialIndividual,
    ) -> SpatialIndividual:
        """Produce one offspring from two parents.

        Parameters
        ----------
        parent1
            First parent, and the one cloned when crossover does not fire.
        parent2
            Second parent.

        Returns
        -------
            The new offspring, unevaluated.
        """
        if random.random() < self.config.incubation_crossover_rate:
            genotype = crossover_genomes(parent1.genotype, parent2.genotype)
        else:
            genotype = copy.deepcopy(parent1.genotype)

        genotype = mutate_genome(
            genotype,
            weight_mutation_rate=self.config.incubation_mutation_probability,
            weight_mutation_power=self.config.incubation_mutation_strength,
            add_connection_rate=self.config.incubation_add_connection_rate,
            add_node_rate=self.config.incubation_add_node_rate,
        )

        offspring = SpatialIndividual(
            unique_id=self.next_unique_id,
            generation=self.generation + 1,
            genotype=genotype,
            energy=self.config.initial_energy,
            parent_ids=[
                parent_id
                for parent_id in (parent1.unique_id, parent2.unique_id)
                if parent_id is not None
            ],
        )
        self.next_unique_id += 1
        return offspring

    def create_next_generation(self) -> None:
        """Replace the population with elites plus fresh offspring."""
        self.population.sort(
            key=lambda individual: individual.fitness,
            reverse=True,
        )

        next_generation: list[SpatialIndividual] = []

        elitism = min(
            self.config.incubation_elitism_count,
            len(self.population),
        )
        for elite_source in self.population[:elitism]:
            elite = SpatialIndividual(
                unique_id=self.next_unique_id,
                generation=self.generation + 1,
                genotype=copy.deepcopy(elite_source.genotype),
                energy=self.config.initial_energy,
                parent_ids=(
                    [elite_source.unique_id]
                    if elite_source.unique_id is not None
                    else []
                ),
            )
            self.next_unique_id += 1
            next_generation.append(elite)

        while len(next_generation) < self.config.incubation_population_size:
            next_generation.append(
                self.create_offspring(
                    self.tournament_selection(),
                    self.tournament_selection(),
                ),
            )

        self.population = next_generation

    def _record_generation(self) -> None:
        """Append this generation's fitness spread to the history."""
        values = [individual.fitness for individual in self.population]
        if not values:
            values = [0.0]

        best = float(max(values))
        average = float(np.mean(values))
        worst = float(min(values))

        self.best_fitness_history.append(best)
        self.avg_fitness_history.append(average)
        self.worst_fitness_history.append(worst)

        if self.config.print_generation_stats:
            msg = (
                f"Incubation generation {self.generation}: "
                f"best={best:.4f} avg={average:.4f} worst={worst:.4f}"
            )
            log.info(msg)

    def run(self) -> tuple[list[SpatialIndividual], int]:
        """Run the incubation GA to completion.

        Returns
        -------
        population
            The final incubated population, evaluated.
        next_unique_id
            The identifier counter, advanced past every individual created.
        """
        msg = (
            f"Incubation phase: {self.config.incubation_population_size} "
            f"individuals for {self.config.incubation_num_generations} "
            f"generations"
        )
        log.info(msg)

        if not self.population:
            self.initialize_population()

        for generation in range(self.config.incubation_num_generations):
            self.generation = generation
            evaluate_population(self.population, self.config)
            self._record_generation()

            if generation < self.config.incubation_num_generations - 1:
                self.create_next_generation()

        return self.population, self.next_unique_id


def seed_spatial_population_from_incubation(
    incubation_population: list[SpatialIndividual],
    target_population_size: int,
    generation: int,
    next_unique_id: int,
    initial_energy: float = 100.0,
) -> tuple[list[SpatialIndividual], int]:
    """Turn an incubated population into a spatial starting population.

    The best incubated individuals are taken first; if incubation produced
    fewer individuals than the spatial phase needs, the best are cloned to make
    up the difference.

    Parameters
    ----------
    incubation_population
        The evaluated incubation population.
    target_population_size
        Size of the spatial population to build.
    generation
        Generation the spatial individuals are born into.
    next_unique_id
        Identifier counter, continued from the caller.
    initial_energy
        Starting energy of each spatial individual.

    Returns
    -------
    population
        The spatial starting population, carrying its incubated fitness.
    next_unique_id
        The identifier counter, advanced past every individual created.
    """
    if not incubation_population:
        return [], next_unique_id

    ranked = sorted(
        incubation_population,
        key=lambda individual: individual.fitness,
        reverse=True,
    )

    population: list[SpatialIndividual] = []
    for i in range(target_population_size):
        source = ranked[i % len(ranked)]
        population.append(
            SpatialIndividual(
                unique_id=next_unique_id,
                generation=generation,
                genotype=copy.deepcopy(source.genotype),
                fitness=source.fitness,
                evaluated=source.evaluated,
                energy=initial_energy,
                parent_ids=(
                    [source.unique_id] if source.unique_id is not None else []
                ),
            ),
        )
        next_unique_id += 1

    return population, next_unique_id
