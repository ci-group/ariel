"""Test: per-generation statistics collection and export."""

# Standard library
import csv
import json
from pathlib import Path

# Third-party libraries
import numpy as np
import pytest

# Local libraries
from ariel.spatial_ea.data import BASE_CSV_FIELDS, EvolutionDataCollector
from ariel.spatial_ea.hyperneat import CPPNConnection
from ariel.spatial_ea.individual import SpatialIndividual


def _population() -> list[SpatialIndividual]:
    """Build a small population with known fitness, energy and ages."""
    return [
        SpatialIndividual(unique_id=0, generation=0, fitness=1.0, energy=50.0),
        SpatialIndividual(unique_id=1, generation=1, fitness=3.0, energy=0.0),
        SpatialIndividual(unique_id=2, generation=2, fitness=2.0, energy=-5.0),
    ]


def test_fitness_stats() -> None:
    """Best, worst and mean should reflect the population."""
    collector = EvolutionDataCollector()
    collector.record_fitness_stats(_population(), 2)

    assert collector.fitness_best == [3.0]
    assert collector.fitness_worst == [1.0]
    assert collector.fitness_avg == [2.0]


def test_age_stats_measure_from_birth_generation() -> None:
    """Age is the distance from the generation of birth."""
    collector = EvolutionDataCollector()
    collector.record_age_stats(_population(), current_generation=2)

    assert collector.age_min == [0]
    assert collector.age_max == [2]
    assert collector.age_avg == [1.0]


def test_energy_stats_count_the_depleted() -> None:
    """Zero counts as depleted, matching energy-based selection."""
    collector = EvolutionDataCollector()
    collector.record_energy_stats(_population())

    assert collector.energy_min == [-5.0]
    assert collector.energy_max == [50.0]
    assert collector.energy_depleted_count == [2]


def test_mating_success_rate_is_a_percentage_of_possible_pairs() -> None:
    """Six individuals allow three pairs, so two pairs is two thirds."""
    collector = EvolutionDataCollector()
    collector.record_mating_stats(
        num_pairs=2,
        num_unpaired=2,
        population_size=6,
    )

    assert collector.mating_pairs == [2]
    assert collector.mating_success_rate[0] == pytest.approx(200.0 / 3.0)

    collector.record_mating_stats(0, 1, 1)
    assert collector.mating_success_rate[1] == 0.0


def test_genotype_diversity_uses_enabled_weights_only() -> None:
    """A disabled connection must not affect the diversity measure."""
    collector = EvolutionDataCollector()
    population = [
        SpatialIndividual(
            unique_id=0,
            genotype={
                "nodes": [],
                "connections": [
                    CPPNConnection(0, 1, 1.0, enabled=True),
                    CPPNConnection(0, 2, 999.0, enabled=False),
                ],
            },
        ),
        SpatialIndividual(
            unique_id=1,
            genotype={
                "nodes": [],
                "connections": [CPPNConnection(0, 1, 3.0, enabled=True)],
            },
        ),
    ]

    collector.record_genotype_diversity(population)

    assert collector.genotype_diversity == [1.0]


def test_genotype_diversity_of_empty_population() -> None:
    """An extinct population has no diversity rather than an error."""
    collector = EvolutionDataCollector()
    collector.record_genotype_diversity([])

    assert collector.genotype_diversity == [0.0]


def test_extinct_generation_records_placeholders() -> None:
    """Extinction should still produce a plottable row."""
    collector = EvolutionDataCollector()
    collector.record_extinct_generation(4)

    assert collector.generations == [4]
    assert collector.population_size == [0]
    assert collector.fitness_best == [0.0]
    assert collector.age_max == [0]


def test_early_stop_is_summarised_and_logged_as_an_event() -> None:
    """Stopping early should be visible in both the events and the summary."""
    collector = EvolutionDataCollector()
    collector.record_generation_start(0, 5)
    collector.record_early_stop(0, "Population extinction", 50)

    summary = collector.get_summary_stats()
    assert summary["stopped_early"] is True
    assert summary["stop_reason"] == "Population extinction"
    assert summary["planned_generations"] == 50
    assert "EARLY STOP: Population extinction" in collector.events[0]


def test_summary_of_a_completed_run() -> None:
    """A run that finished should say so."""
    collector = EvolutionDataCollector()
    for generation in range(3):
        collector.record_generation_start(generation, 10)
        collector.record_fitness_stats(_population(), generation)

    summary = collector.get_summary_stats()
    assert summary["stopped_early"] is False
    assert summary["stop_reason"] == "Completed all generations"
    assert summary["total_generations"] == 3
    assert summary["population"]["max"] == 10


def _fill(collector: EvolutionDataCollector, generations: int = 3) -> None:
    """Record a short synthetic run."""
    for generation in range(generations):
        collector.record_generation_start(generation, 10 + generation)
        collector.record_fitness_stats(_population(), generation)
        collector.record_age_stats(_population(), generation)
        collector.record_energy_stats(_population())
        collector.record_genotype_diversity([])
        collector.record_mating_stats(2, 1, 10)
        collector.record_reproduction(4, 10)
        collector.record_selection(14, 10)


def test_csv_export_matches_the_documented_schema(tmp_path: Path) -> None:
    """Column names must match the research prototype's exactly."""
    collector = EvolutionDataCollector()
    _fill(collector)

    csv_path = collector.save_to_csv(tmp_path, timestamp="20260101_000000")

    assert csv_path.name == "evolution_data_20260101_000000.csv"
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    for field in BASE_CSV_FIELDS:
        assert field in rows[0]
    # Energy columns appear because energy was recorded.
    assert "energy_avg" in rows[0]
    assert rows[0]["generation"] == "0"
    assert rows[2]["population_size"] == "12"


def test_csv_export_omits_energy_when_never_recorded(tmp_path: Path) -> None:
    """A run without energy should not emit empty energy columns."""
    collector = EvolutionDataCollector()
    collector.record_generation_start(0, 5)
    collector.record_fitness_stats(_population(), 0)

    csv_path = collector.save_to_csv(tmp_path, timestamp="20260101_000001")
    with csv_path.open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))

    assert "energy_avg" not in header


def test_npz_export_round_trips(tmp_path: Path) -> None:
    """The archive should carry every series plus the JSON summary."""
    collector = EvolutionDataCollector()
    _fill(collector)

    npz_path = collector.save_to_npz(tmp_path, timestamp="20260101_000002")
    data = np.load(npz_path, allow_pickle=True)

    assert npz_path.name == "evolution_data_20260101_000002.npz"
    # The generation series is named ``generations`` in the archive.
    assert "generations" in data.files
    assert list(data["generations"]) == [0, 1, 2]
    assert list(data["population_size"]) == [10, 11, 12]
    assert "energy_avg" in data.files

    summary = json.loads(str(data["summary_json"]))
    assert summary["total_generations"] == 3
    assert summary["total_births"] == 12


def test_csv_round_trips_back_into_a_collector(tmp_path: Path) -> None:
    """A saved run reloads into a collector the figures can use."""
    original = EvolutionDataCollector()
    _fill(original, generations=4)
    csv_path = original.save_to_csv(tmp_path, timestamp="20260101_000100")

    restored = EvolutionDataCollector.from_csv(csv_path)

    assert restored.generations == original.generations
    assert restored.population_size == original.population_size
    assert restored.fitness_best == pytest.approx(original.fitness_best)
    assert restored.age_max == original.age_max
    assert restored.mating_pairs == original.mating_pairs
    assert restored.energy_avg == pytest.approx(original.energy_avg)


def test_csv_loader_keeps_lagging_series_short(tmp_path: Path) -> None:
    """Births and deaths lag the population; blanks must not become zeros."""
    collector = EvolutionDataCollector()
    _fill(collector, generations=4)
    # The last generation has no deaths recorded yet.
    collector.deaths = collector.deaths[:-1]
    csv_path = collector.save_to_csv(tmp_path, timestamp="20260101_000200")

    restored = EvolutionDataCollector.from_csv(csv_path)

    assert len(restored.generations) == 4
    assert len(restored.deaths) == len(collector.deaths)
    assert 0 not in restored.deaths[len(collector.deaths) :]


def test_csv_loader_without_energy(tmp_path: Path) -> None:
    """A run that recorded no energy reloads with none."""
    collector = EvolutionDataCollector()
    collector.record_generation_start(0, 5)
    collector.record_fitness_stats(_population(), 0)
    csv_path = collector.save_to_csv(tmp_path, timestamp="20260101_000300")

    restored = EvolutionDataCollector.from_csv(csv_path)

    assert restored.energy_avg == []
    assert restored.generations == [0]


def test_latest_csv_picks_the_newest(tmp_path: Path) -> None:
    """The loader finds the most recent run in a folder."""
    collector = EvolutionDataCollector()
    _fill(collector, generations=2)
    collector.save_to_csv(tmp_path, timestamp="20260101_000000")
    newest = collector.save_to_csv(tmp_path, timestamp="20260101_235959")

    assert EvolutionDataCollector.latest_csv(tmp_path) == newest


def test_latest_csv_of_an_empty_folder(tmp_path: Path) -> None:
    """No runs means nothing to load."""
    assert EvolutionDataCollector.latest_csv(tmp_path) is None
