"""Spatial individual model for the spatial EA."""

# Standard library
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Evaluate type annotations in a deferred manner (ruff: UP037)
if TYPE_CHECKING:
    from ariel.parameters.ariel_types import FloatArray


@dataclass(slots=True)
class SpatialIndividual:
    """State carried by one robot during spatial evolution.

    Kept deliberately separate from :class:`ariel.ec.Individual` and its
    SQLModel persistence layer: the genotype here is a nested CPPN genome
    rather than a JSON-friendly vector, and the spatial phase needs mutable
    per-generation position and energy state that does not belong in a
    database row.

    Parameters
    ----------
    unique_id
        Identifier, unique across a whole run.
    generation
        Generation in which this individual was created.
    genotype
        CPPN genome with ``nodes`` and ``connections`` entries.
    fitness
        Most recent fitness score. Higher is better.
    evaluated
        Whether ``fitness`` reflects a completed simulation. The EA evaluates
        each individual once and inherits the score thereafter.
    start_position
        Core position at the start of the fitness evaluation.
    end_position
        Core position at the end of the fitness evaluation.
    spawn_position
        Position this individual occupies in the shared world.
    orientation
        Heading in radians this individual is spawned with, about the world
        z axis.
    target_position
        Randomly placed target used by directional fitness.
    progress_toward_target
        Distance closed toward ``target_position`` during evaluation.
    total_distance
        Straight-line distance travelled during evaluation.
    robot_index
        Index of this individual among the robots spawned in the shared world.
    assigned_zone
        Index of the mating zone this individual is bound to, under the
        ``assigned_zone`` movement bias.
    parent_ids
        Identifiers of the individuals this one descends from.
    energy
        Current energy level, consumed by energy-based selection.
    """

    unique_id: int | None = None
    generation: int = 0
    genotype: dict[str, Any] = field(default_factory=dict)
    fitness: float = 0.0
    evaluated: bool = False
    start_position: FloatArray | None = None
    end_position: FloatArray | None = None
    spawn_position: FloatArray | None = None
    orientation: float = 0.0
    target_position: FloatArray | None = None
    progress_toward_target: float = 0.0
    total_distance: float = 0.0
    robot_index: int | None = None
    assigned_zone: int | None = None
    parent_ids: list[int] = field(default_factory=list)
    energy: float = 100.0

    def age_at(self, current_generation: int) -> int:
        """Age in generations at a given point in the run.

        Parameters
        ----------
        current_generation
            The generation to measure against.

        Returns
        -------
            Number of generations survived, never negative.
        """
        return max(0, current_generation - self.generation)

    def copy(self) -> SpatialIndividual:
        """Return a deep copy of this individual's state.

        The copy keeps ``unique_id``, ``evaluated`` and ``parent_ids``
        untouched; assigning a fresh identity is the caller's job, and
        :func:`ariel.spatial_ea.genetics.clone_individual` does it.

        Returns
        -------
            An independent copy sharing no mutable state with the original.
        """
        return SpatialIndividual(
            unique_id=self.unique_id,
            generation=self.generation,
            genotype=copy.deepcopy(self.genotype),
            fitness=self.fitness,
            evaluated=self.evaluated,
            start_position=(
                self.start_position.copy()
                if self.start_position is not None
                else None
            ),
            end_position=(
                self.end_position.copy()
                if self.end_position is not None
                else None
            ),
            spawn_position=(
                self.spawn_position.copy()
                if self.spawn_position is not None
                else None
            ),
            orientation=self.orientation,
            target_position=(
                self.target_position.copy()
                if self.target_position is not None
                else None
            ),
            progress_toward_target=self.progress_toward_target,
            total_distance=self.total_distance,
            robot_index=self.robot_index,
            assigned_zone=self.assigned_zone,
            parent_ids=list(self.parent_ids),
            energy=self.energy,
        )
