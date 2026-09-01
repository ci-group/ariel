"""Per-generation statistics collection for the spatial EA.

The collector accumulates parallel series indexed by generation and writes them
to CSV and NPZ. The column names match the research prototype's exactly, so
existing analysis and plotting tooling reads either implementation's output
without modification.
"""

# Standard library
from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Third-party libraries
import numpy as np

# Local libraries
from ariel import log

# Evaluate type annotations in a deferred manner (ruff: UP037)
if TYPE_CHECKING:
    from ariel.spatial_ea.individual import SpatialIndividual

# Global constants
BASE_CSV_FIELDS = (
    "generation",
    "population_size",
    "births",
    "deaths",
    "fitness_best",
    "fitness_avg",
    "fitness_worst",
    "fitness_std",
    "age_min",
    "age_max",
    "age_avg",
    "age_std",
    "mating_pairs",
    "unpaired",
    "mating_success_rate",
    "genotype_diversity",
)
ENERGY_CSV_FIELDS = (
    "energy_min",
    "energy_max",
    "energy_avg",
    "energy_std",
    "energy_depleted_count",
)


class EvolutionDataCollector:
    """Accumulates the per-generation record of one run.

    Population size is not constant, and a run may end early through
    extinction or explosion, so every series is appended independently and
    export tolerates series of differing length.
    """

    def __init__(self, config: Any = None) -> None:
        """Start an empty record.

        Parameters
        ----------
        config
            Run configuration, retained for metadata only.
        """
        self.config = config
        self.start_time = datetime.now(UTC)

        self.generations: list[int] = []
        self.population_size: list[int] = []
        self.births: list[int] = []
        self.deaths: list[int] = []

        self.fitness_best: list[float] = []
        self.fitness_avg: list[float] = []
        self.fitness_worst: list[float] = []
        self.fitness_std: list[float] = []

        self.energy_min: list[float] = []
        self.energy_max: list[float] = []
        self.energy_avg: list[float] = []
        self.energy_std: list[float] = []
        self.energy_depleted_count: list[int] = []

        self.age_min: list[int] = []
        self.age_max: list[int] = []
        self.age_avg: list[float] = []
        self.age_std: list[float] = []

        self.mating_pairs: list[int] = []
        self.unpaired_individuals: list[int] = []
        self.mating_success_rate: list[float] = []

        self.genotype_diversity: list[float] = []

        self.events: dict[int, list[str]] = {}

        self.stopped_early: bool = False
        self.stop_reason: str = ""
        self.planned_generations: int = 0

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> EvolutionDataCollector:
        """Rebuild a collector from a run's saved CSV.

        Lets every figure that works on a live run work on a finished one, so
        analysing an old result needs no re-run.

        Series are stored to different lengths — births and deaths lag the
        population by a generation, and a run that stopped early leaves
        trailing blanks — so blank cells are skipped rather than read as zero.

        Parameters
        ----------
        csv_path
            Path to an ``evolution_data_*.csv`` file.

        Returns
        -------
            A collector holding the saved series.
        """
        collector = cls()
        columns: dict[str, list[float]] = {}

        with Path(csv_path).open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for name, raw in row.items():
                    if name is None or raw is None or raw == "":
                        continue
                    columns.setdefault(name, []).append(float(raw))

        collector.generations = [int(v) for v in columns.get("generation", [])]
        collector.population_size = [
            int(v) for v in columns.get("population_size", [])
        ]
        collector.births = [int(v) for v in columns.get("births", [])]
        collector.deaths = [int(v) for v in columns.get("deaths", [])]
        collector.age_min = [int(v) for v in columns.get("age_min", [])]
        collector.age_max = [int(v) for v in columns.get("age_max", [])]
        collector.mating_pairs = [
            int(v) for v in columns.get("mating_pairs", [])
        ]
        collector.unpaired_individuals = [
            int(v) for v in columns.get("unpaired", [])
        ]
        collector.energy_depleted_count = [
            int(v) for v in columns.get("energy_depleted_count", [])
        ]

        for attribute, column in (
            ("fitness_best", "fitness_best"),
            ("fitness_avg", "fitness_avg"),
            ("fitness_worst", "fitness_worst"),
            ("fitness_std", "fitness_std"),
            ("age_avg", "age_avg"),
            ("age_std", "age_std"),
            ("mating_success_rate", "mating_success_rate"),
            ("genotype_diversity", "genotype_diversity"),
            ("energy_min", "energy_min"),
            ("energy_max", "energy_max"),
            ("energy_avg", "energy_avg"),
            ("energy_std", "energy_std"),
        ):
            setattr(collector, attribute, list(columns.get(column, [])))

        for generation in collector.generations:
            collector.events.setdefault(generation, [])

        return collector

    @staticmethod
    def latest_csv(results_folder: str | Path) -> Path | None:
        """Find the most recent run in a results folder.

        Parameters
        ----------
        results_folder
            Directory to search.

        Returns
        -------
            The newest ``evolution_data_*.csv``, or ``None``.
        """
        found = sorted(Path(results_folder).glob("evolution_data_*.csv"))
        return found[-1] if found else None

    def record_generation_start(
        self,
        generation: int,
        population_size: int,
    ) -> None:
        """Open the record for a generation.

        Parameters
        ----------
        generation
            Generation number.
        population_size
            Population size at the start of the generation.
        """
        self.generations.append(generation)
        self.population_size.append(population_size)
        self.events[generation] = []

    def record_extinct_generation(self, generation: int) -> None:
        """Record a generation in which the population died out.

        Births, deaths and mating series are offset by one generation and are
        deliberately left untouched.

        Parameters
        ----------
        generation
            Generation number.
        """
        self.generations.append(generation)
        self.population_size.append(0)
        self.events[generation] = []

        self.fitness_best.append(0.0)
        self.fitness_avg.append(0.0)
        self.fitness_worst.append(0.0)
        self.fitness_std.append(0.0)

        self.age_min.append(0)
        self.age_max.append(0)
        self.age_avg.append(0.0)
        self.age_std.append(0.0)

        self.genotype_diversity.append(0.0)

    def record_fitness_stats(
        self,
        population: list[SpatialIndividual],
        generation: int,
    ) -> None:
        """Record the fitness distribution of a population.

        Parameters
        ----------
        population
            The population to summarise.
        generation
            Generation number, accepted for call-site symmetry.
        """
        del generation

        values = [individual.fitness for individual in population]
        if not values:
            values = [0.0]

        self.fitness_best.append(float(max(values)))
        self.fitness_avg.append(float(np.mean(values)))
        self.fitness_worst.append(float(min(values)))
        self.fitness_std.append(float(np.std(values)))

    def record_energy_stats(
        self,
        population: list[SpatialIndividual],
        stage: str = "",
    ) -> None:
        """Record the energy distribution of a population.

        Parameters
        ----------
        population
            The population to summarise.
        stage
            Optional description of when in the generation this was measured.
        """
        del stage

        values = [individual.energy for individual in population]
        if not values:
            values = [0.0]

        self.energy_min.append(float(min(values)))
        self.energy_max.append(float(max(values)))
        self.energy_avg.append(float(np.mean(values)))
        self.energy_std.append(float(np.std(values)))
        self.energy_depleted_count.append(
            sum(1 for value in values if value <= 0),
        )

    def record_age_stats(
        self,
        population: list[SpatialIndividual],
        current_generation: int,
    ) -> None:
        """Record the age distribution of a population.

        Parameters
        ----------
        population
            The population to summarise.
        current_generation
            Generation used to measure age.
        """
        ages = [
            individual.age_at(current_generation) for individual in population
        ]

        self.age_min.append(int(min(ages)) if ages else 0)
        self.age_max.append(int(max(ages)) if ages else 0)
        self.age_avg.append(float(np.mean(ages)) if ages else 0.0)
        self.age_std.append(float(np.std(ages)) if ages else 0.0)

    def record_mating_stats(
        self,
        num_pairs: int,
        num_unpaired: int,
        population_size: int,
    ) -> None:
        """Record how much of the population managed to pair.

        Parameters
        ----------
        num_pairs
            Number of pairs formed.
        num_unpaired
            Number of individuals left unpaired.
        population_size
            Population size before mating.
        """
        self.mating_pairs.append(num_pairs)
        self.unpaired_individuals.append(num_unpaired)

        max_possible_pairs = population_size // 2
        rate = (
            (num_pairs / max_possible_pairs * 100.0)
            if max_possible_pairs > 0
            else 0.0
        )
        self.mating_success_rate.append(rate)

    def record_reproduction(
        self,
        num_offspring: int,
        population_before: int,
    ) -> None:
        """Record how many offspring were produced.

        Parameters
        ----------
        num_offspring
            Number of offspring created.
        population_before
            Population size before the offspring were added.
        """
        del population_before
        self.births.append(num_offspring)

    def record_selection(
        self,
        population_before: int,
        population_after: int,
    ) -> None:
        """Record how many individuals selection removed.

        Parameters
        ----------
        population_before
            Population size before selection.
        population_after
            Population size after selection.
        """
        self.deaths.append(population_before - population_after)

    def record_genotype_diversity(
        self,
        population: list[SpatialIndividual],
    ) -> None:
        """Record genome diversity as the spread of connection weights.

        Parameters
        ----------
        population
            The population to summarise.
        """
        if not population:
            self.genotype_diversity.append(0.0)
            return

        weights: list[float] = []
        for individual in population:
            weights.extend(
                conn.weight
                for conn in individual.genotype.get("connections", [])
                if conn.enabled
            )

        self.genotype_diversity.append(
            float(np.std(weights)) if weights else 0.0,
        )

    def add_event(self, generation: int, event_description: str) -> None:
        """Attach a free-form note to a generation.

        Parameters
        ----------
        generation
            Generation the note belongs to.
        event_description
            The note.
        """
        self.events.setdefault(generation, []).append(event_description)

    def record_early_stop(
        self,
        generation: int,
        reason: str,
        planned_generations: int,
    ) -> None:
        """Record that the run ended before its planned last generation.

        Parameters
        ----------
        generation
            Generation at which the run stopped.
        reason
            Why it stopped.
        planned_generations
            How many generations were planned.
        """
        self.stopped_early = True
        self.stop_reason = reason
        self.planned_generations = planned_generations
        self.add_event(generation, f"EARLY STOP: {reason}")

    def get_summary_stats(self) -> dict[str, Any]:
        """Summarise the whole run.

        Returns
        -------
            Generation counts, stop reason, run duration, and population,
            fitness, birth, death, mating and energy summaries.
        """
        summary: dict[str, Any] = {
            "total_generations": len(self.generations),
            "planned_generations": (
                self.planned_generations
                if self.stopped_early
                else len(self.generations)
            ),
            "stopped_early": self.stopped_early,
            "stop_reason": (
                self.stop_reason
                if self.stopped_early
                else "Completed all generations"
            ),
            "run_duration": str(datetime.now(UTC) - self.start_time),
            "population": {
                "initial": self.population_size[0]
                if self.population_size
                else 0,
                "final": self.population_size[-1]
                if self.population_size
                else 0,
                "max": max(self.population_size) if self.population_size else 0,
                "min": min(self.population_size) if self.population_size else 0,
                "avg": (
                    float(np.mean(self.population_size))
                    if self.population_size
                    else 0.0
                ),
            },
            "fitness": {
                "best_ever": (
                    float(max(self.fitness_best)) if self.fitness_best else 0.0
                ),
                "final_best": (
                    float(self.fitness_best[-1]) if self.fitness_best else 0.0
                ),
                "avg_improvement": (
                    float(self.fitness_avg[-1] - self.fitness_avg[0])
                    if len(self.fitness_avg) > 1
                    else 0.0
                ),
            },
            "total_births": sum(self.births) if self.births else 0,
            "total_deaths": sum(self.deaths) if self.deaths else 0,
            "avg_mating_success_rate": (
                float(np.mean(self.mating_success_rate))
                if self.mating_success_rate
                else 0.0
            ),
        }

        if self.energy_avg:
            summary["energy"] = {
                "final_avg": self.energy_avg[-1],
                "final_min": self.energy_min[-1],
                "total_depleted": sum(self.energy_depleted_count),
            }

        return summary

    def _series_by_field(self) -> dict[str, list[Any]]:
        """Map every export column onto its backing series.

        Returns
        -------
            Column name to series, energy columns included only when energy
            was recorded.
        """
        series: dict[str, list[Any]] = {
            "generation": self.generations,
            "population_size": self.population_size,
            "births": self.births,
            "deaths": self.deaths,
            "fitness_best": self.fitness_best,
            "fitness_avg": self.fitness_avg,
            "fitness_worst": self.fitness_worst,
            "fitness_std": self.fitness_std,
            "age_min": self.age_min,
            "age_max": self.age_max,
            "age_avg": self.age_avg,
            "age_std": self.age_std,
            "mating_pairs": self.mating_pairs,
            "unpaired": self.unpaired_individuals,
            "mating_success_rate": self.mating_success_rate,
            "genotype_diversity": self.genotype_diversity,
        }
        if self.energy_avg:
            series.update({
                "energy_min": self.energy_min,
                "energy_max": self.energy_max,
                "energy_avg": self.energy_avg,
                "energy_std": self.energy_std,
                "energy_depleted_count": self.energy_depleted_count,
            })
        return series

    def save_to_csv(
        self,
        output_folder: str | Path = "./__results__",
        timestamp: str | None = None,
    ) -> Path:
        """Write the per-generation record to CSV.

        Parameters
        ----------
        output_folder
            Directory to write into, created if absent.
        timestamp
            Filename timestamp. Defaults to the current local time.

        Returns
        -------
            Path of the written file.
        """
        folder = Path(output_folder)
        folder.mkdir(parents=True, exist_ok=True)

        stamp = timestamp or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        csv_path = folder / f"evolution_data_{stamp}.csv"

        series = self._series_by_field()
        fieldnames = list(BASE_CSV_FIELDS)
        if self.energy_avg:
            fieldnames.extend(ENERGY_CSV_FIELDS)

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(self.generations)):
                writer.writerow({
                    name: (series[name][i] if i < len(series[name]) else "")
                    for name in fieldnames
                })

        msg = f"Evolution data saved to CSV: {csv_path}"
        log.info(msg)
        return csv_path

    def save_to_npz(
        self,
        output_folder: str | Path = "./__results__",
        timestamp: str | None = None,
    ) -> Path:
        """Write the per-generation record to a compressed NumPy archive.

        Parameters
        ----------
        output_folder
            Directory to write into, created if absent.
        timestamp
            Filename timestamp. Defaults to the current local time.

        Returns
        -------
            Path of the written file.
        """
        folder = Path(output_folder)
        folder.mkdir(parents=True, exist_ok=True)

        stamp = timestamp or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        npz_path = folder / f"evolution_data_{stamp}.npz"

        series = self._series_by_field()
        data_dict: dict[str, Any] = {
            ("generations" if name == "generation" else name): np.array(values)
            for name, values in series.items()
        }
        data_dict["summary_json"] = json.dumps(self.get_summary_stats())

        np.savez(npz_path, **data_dict)

        msg = f"Evolution data saved to NPZ: {npz_path}"
        log.info(msg)
        return npz_path
