"""Innovation counter for NEAT-style structural mutations."""

from __future__ import annotations

from typing import Dict, Tuple


class InnovationCounter:
    """Tracks structural innovation numbers across a population within a generation.

    Same structural mutation (same source → target pair) within a generation
    receives the same innovation number so that genomes can be aligned for
    crossover.
    """

    def __init__(self) -> None:
        self._global_counter: int = 0
        self._generation_cache: Dict[Tuple[int, int], int] = {}

    @property
    def current(self) -> int:
        """The next innovation number that will be assigned."""
        return self._global_counter

    def get_innovation(self, source_id: int, target_id: int) -> int:
        """Return innovation number for a connection, creating one if new."""
        key = (source_id, target_id)
        if key not in self._generation_cache:
            self._generation_cache[key] = self._global_counter
            self._global_counter += 1
        return self._generation_cache[key]

    def next_innovation(self) -> int:
        """Return the next innovation number without caching.

        Use this for structural mutations that have no natural cache key
        (e.g. arm additions in the spherical encoding).
        """
        inno = self._global_counter
        self._global_counter += 1
        return inno

    def reset_generation(self) -> None:
        """Clear the per-generation cache. Call at the start of each generation."""
        self._generation_cache.clear()
