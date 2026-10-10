"""Speciation state management for NEAT evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class Species:
    """A NEAT species: a group of genetically similar individuals."""
    id: int
    representative_genome: Any
    member_ids: List[str]
    best_fitness: float
    generations_since_improvement: int = 0


@dataclass
class SpeciationState:
    """Manages species assignments across generations."""
    species: Dict[int, Species] = field(default_factory=dict)
    next_species_id: int = 0
    compatibility_threshold: float = 3.0

    def speciate(
        self,
        population_df: pd.DataFrame,
        genome_handler_class: type,
        handler_kwargs: dict,
    ) -> Dict[str, int]:
        """Assign each individual to a species.

        Returns a mapping from individual ID to species ID.
        """
        assignments: Dict[str, int] = {}

        # Clear existing member lists
        for sp in self.species.values():
            sp.member_ids = []

        for _, row in population_df.iterrows():
            ind_id = row["id"]
            genome = row["genome"]

            # Wrap genome in a handler for distance computation
            ind_handler = genome_handler_class(genome=genome, **handler_kwargs)

            assigned = False
            for sp in self.species.values():
                rep_handler = genome_handler_class(
                    genome=sp.representative_genome, **handler_kwargs
                )
                dist = ind_handler.compatibility_distance(rep_handler)
                if dist < self.compatibility_threshold:
                    sp.member_ids.append(ind_id)
                    assignments[ind_id] = sp.id
                    assigned = True
                    break

            if not assigned:
                new_id = self.next_species_id
                self.next_species_id += 1
                self.species[new_id] = Species(
                    id=new_id,
                    representative_genome=genome,
                    member_ids=[ind_id],
                    best_fitness=float("-inf"),
                )
                assignments[ind_id] = new_id

        # Remove empty species
        empty = [sid for sid, sp in self.species.items() if not sp.member_ids]
        for sid in empty:
            del self.species[sid]

        return assignments

    def update_representatives(self, population_df: pd.DataFrame) -> None:
        """Update species representatives after selection.

        Per the original NEAT paper, the representative is always a random
        member of the species from the current generation.
        """
        import random

        id_to_genome = dict(zip(population_df["id"], population_df["genome"]))

        for sp in self.species.values():
            if not sp.member_ids:
                continue
            chosen_id = random.choice(sp.member_ids)
            if chosen_id in id_to_genome:
                sp.representative_genome = id_to_genome[chosen_id]

    def update_best_fitness(self, population_df: pd.DataFrame) -> None:
        """Track best fitness per species and update stagnation counters."""
        id_to_fitness = dict(zip(population_df["id"], population_df["fitness"]))

        for sp in self.species.values():
            if not sp.member_ids:
                continue
            member_fitnesses = [
                id_to_fitness[mid]
                for mid in sp.member_ids
                if mid in id_to_fitness
            ]
            if not member_fitnesses:
                continue
            current_best = max(member_fitnesses)
            if current_best > sp.best_fitness:
                sp.best_fitness = current_best
                sp.generations_since_improvement = 0
            else:
                sp.generations_since_improvement += 1

    def adjust_threshold(self, target_species_count: int, delta: float = 0.3) -> None:
        """Adjust compatibility threshold toward target species count."""
        current = len(self.species)
        if current > target_species_count:
            self.compatibility_threshold += delta
        elif current < target_species_count:
            self.compatibility_threshold = max(0.1, self.compatibility_threshold - delta)

    def remove_stagnant_species(
        self, stagnation_limit: int, protect_top_n: int = 2
    ) -> None:
        """Remove species that haven't improved for stagnation_limit generations.

        Always protect the top N species by best_fitness.
        """
        if len(self.species) <= protect_top_n:
            return

        # Sort species by best fitness descending
        sorted_species = sorted(
            self.species.values(), key=lambda s: s.best_fitness, reverse=True
        )
        protected_ids = {s.id for s in sorted_species[:protect_top_n]}

        to_remove = [
            sid
            for sid, sp in self.species.items()
            if sp.generations_since_improvement >= stagnation_limit
            and sid not in protected_ids
        ]
        for sid in to_remove:
            del self.species[sid]
