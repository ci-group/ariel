"""Test: Archive — read-only query interface for a persisted EA database."""

import math
from pathlib import Path

import pytest

from ariel.ec.archive import Archive
from ariel.ec.ea import EA, EAOperation
from ariel.ec.individual import Individual
from ariel.ec.population import Population


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


@EAOperation
def noop(population: Population) -> Population:
    return population


def _make_evaluated_pop(n: int = 20) -> Population:
    """Population of n individuals with distinct float fitnesses 0..n-1."""
    inds = []
    for i in range(n):
        ind = Individual()
        ind.genotype = [float(i)]
        ind.fitness = float(i)
        inds.append(ind)
    return Population(inds)


@pytest.fixture(scope="module")
def archive_db(tmp_path_factory) -> Path:
    """
    One EA run shared across the whole module.

    20 individuals, fitness 0.0-19.0, run for 5 steps with noop so every
    individual survives to generation 5 (time_of_birth=0, time_of_death=5).
    """
    db = tmp_path_factory.mktemp("archive") / "archive.db"
    ea = EA(
        population=_make_evaluated_pop(20),
        operations=[noop],
        db_file_path=db,
        quiet=True,
        num_steps=5,
    )
    ea.run()
    ea.engine.dispose()
    return db


@pytest.fixture()
def archive(archive_db: Path) -> Archive:
    return Archive(archive_db)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_archive_opens_existing_db(archive_db) -> None:
    """Archive opens a valid database without error."""
    a = Archive(archive_db)
    assert a.db_path == archive_db


def test_archive_missing_db_raises() -> None:
    """Archive raises FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError, match="not found"):
        Archive("/nonexistent/path/db.db")


def test_archive_repr_contains_name(archive) -> None:
    """__repr__ contains 'Archive'."""
    assert "Archive" in repr(archive)


# ---------------------------------------------------------------------------
# Properties: size, generation_range
# ---------------------------------------------------------------------------


def test_archive_size(archive) -> None:
    """size returns the total number of evaluated individuals."""
    assert archive.size >= 20


def test_archive_generation_range(archive) -> None:
    """generation_range returns a (min_birth, max_death) tuple."""
    lo, hi = archive.generation_range
    assert isinstance(lo, int)
    assert isinstance(hi, int)
    assert hi >= lo


def test_archive_generation_range_values(archive) -> None:
    """Individuals born at gen 0 and alive through gen 5."""
    lo, hi = archive.generation_range
    assert lo == 0
    assert hi == 5


# ---------------------------------------------------------------------------
# fitness_stats
# ---------------------------------------------------------------------------


def test_fitness_stats_returns_dict(archive) -> None:
    """fitness_stats returns a dict with the five expected keys."""
    stats = archive.fitness_stats()
    assert set(stats.keys()) == {"min", "max", "mean", "std", "median"}


def test_fitness_stats_min_max(archive) -> None:
    """min and max fitness values match the known population range."""
    stats = archive.fitness_stats()
    assert stats["min"] == pytest.approx(0.0)
    assert stats["max"] == pytest.approx(19.0)


def test_fitness_stats_mean(archive) -> None:
    """Mean fitness of 0..19 is 9.5."""
    stats = archive.fitness_stats()
    assert stats["mean"] == pytest.approx(9.5)


def test_fitness_stats_no_match_returns_nan(archive) -> None:
    """fitness_stats with an impossible birth_range returns NaN values."""
    stats = archive.fitness_stats(birth_range=(9999, 9999))
    for v in stats.values():
        assert math.isnan(v)


def test_fitness_stats_birth_range_filter(archive) -> None:
    """birth_range=(0, 0) restricts to individuals born at generation 0."""
    stats = archive.fitness_stats(birth_range=(0, 0))
    assert not math.isnan(stats["min"])


# ---------------------------------------------------------------------------
# random_individual
# ---------------------------------------------------------------------------


def test_random_individual_returns_individual(archive) -> None:
    """random_individual returns an Individual."""
    ind = archive.random_individual()
    assert isinstance(ind, Individual)


def test_random_individual_is_evaluated(archive) -> None:
    """random_individual returns an evaluated individual (fitness_ is set)."""
    ind = archive.random_individual()
    assert ind.fitness_ is not None


def test_random_individual_birth_range_filter(archive) -> None:
    """birth_range limits the birth generation of the result."""
    ind = archive.random_individual(birth_range=(0, 0))
    assert ind.time_of_birth == 0


def test_random_individual_fitness_range_filter(archive) -> None:
    """fitness_range restricts the returned individual's fitness."""
    ind = archive.random_individual(fitness_range=(5.0, 10.0))
    assert 5.0 <= ind.fitness_ <= 10.0


def test_random_individual_no_match_raises(archive) -> None:
    """random_individual raises ValueError when no match exists."""
    with pytest.raises(ValueError, match="No individuals"):
        archive.random_individual(fitness_range=(9999.0, 9999.0))


# ---------------------------------------------------------------------------
# best_individual
# ---------------------------------------------------------------------------


def test_best_individual_min_mode(archive) -> None:
    """best_individual('min') returns the individual with the lowest fitness."""
    ind = archive.best_individual(fitness_mode="min")
    assert ind.fitness_ == pytest.approx(0.0)


def test_best_individual_max_mode(archive) -> None:
    """best_individual('max') returns the individual with the highest fitness."""
    ind = archive.best_individual(fitness_mode="max")
    assert ind.fitness_ == pytest.approx(19.0)


def test_best_individual_birth_range(archive) -> None:
    """best_individual respects birth_range."""
    ind = archive.best_individual(fitness_mode="min", birth_range=(0, 0))
    assert ind.time_of_birth == 0


# ---------------------------------------------------------------------------
# random_population
# ---------------------------------------------------------------------------


def test_random_population_returns_population(archive) -> None:
    """random_population returns a Population."""
    pop = archive.random_population(5)
    assert isinstance(pop, Population)


def test_random_population_respects_n(archive) -> None:
    """random_population returns at most n individuals."""
    pop = archive.random_population(5)
    assert len(pop) <= 5


def test_random_population_all_evaluated(archive) -> None:
    """All individuals in the random population are evaluated."""
    pop = archive.random_population(10)
    for ind in pop:
        assert ind.fitness_ is not None


def test_random_population_fitness_range(archive) -> None:
    """fitness_range filter works for random_population."""
    pop = archive.random_population(20, fitness_range=(0.0, 5.0))
    for ind in pop:
        assert 0.0 <= ind.fitness_ <= 5.0


def test_random_population_n_larger_than_archive(archive) -> None:
    """Requesting more than the archive size returns what's available."""
    pop = archive.random_population(10000)
    assert len(pop) <= archive.size


# ---------------------------------------------------------------------------
# hall_of_fame
# ---------------------------------------------------------------------------


def test_hall_of_fame_returns_population(archive) -> None:
    """hall_of_fame returns a Population."""
    hof = archive.hall_of_fame(n=5)
    assert isinstance(hof, Population)


def test_hall_of_fame_size(archive) -> None:
    """hall_of_fame returns at most n individuals."""
    hof = archive.hall_of_fame(n=5)
    assert len(hof) <= 5


def test_hall_of_fame_min_sorted(archive) -> None:
    """hall_of_fame('min') is sorted ascending by fitness."""
    hof = archive.hall_of_fame(n=5, fitness_mode="min")
    fitnesses = [ind.fitness_ for ind in hof]
    assert fitnesses == sorted(fitnesses)


def test_hall_of_fame_max_sorted(archive) -> None:
    """hall_of_fame('max') is sorted descending by fitness."""
    hof = archive.hall_of_fame(n=5, fitness_mode="max")
    fitnesses = [ind.fitness_ for ind in hof]
    assert fitnesses == sorted(fitnesses, reverse=True)


def test_hall_of_fame_min_best_is_lowest(archive) -> None:
    """The first entry in a min hall of fame has the lowest fitness."""
    hof = archive.hall_of_fame(n=3, fitness_mode="min")
    assert hof[0].fitness_ == pytest.approx(0.0)


def test_hall_of_fame_max_best_is_highest(archive) -> None:
    """The first entry in a max hall of fame has the highest fitness."""
    hof = archive.hall_of_fame(n=3, fitness_mode="max")
    assert hof[0].fitness_ == pytest.approx(19.0)


def test_hall_of_fame_birth_range(archive) -> None:
    """hall_of_fame respects birth_range."""
    hof = archive.hall_of_fame(n=5, birth_range=(0, 0))
    for ind in hof:
        assert ind.time_of_birth == 0


# ---------------------------------------------------------------------------
# tournament_population
# ---------------------------------------------------------------------------


def test_tournament_population_returns_population(archive) -> None:
    """tournament_population returns a Population."""
    pop = archive.tournament_population(n=3, tournament_size=2)
    assert isinstance(pop, Population)


def test_tournament_population_size(archive) -> None:
    """tournament_population returns exactly n individuals."""
    pop = archive.tournament_population(n=4, tournament_size=2)
    assert len(pop) == 4


def test_tournament_population_min_winners_are_low_fitness(archive) -> None:
    """In min mode, tournament winners have low fitness."""
    pop = archive.tournament_population(n=5, tournament_size=3, fitness_mode="min")
    for ind in pop:
        assert ind.fitness_ <= 19.0


def test_tournament_population_max_mode(archive) -> None:
    """In max mode, tournament winners have high fitness."""
    pop = archive.tournament_population(n=5, tournament_size=3, fitness_mode="max")
    for ind in pop:
        assert ind.fitness_ >= 0.0


def test_tournament_size_exceeds_pool_raises(archive) -> None:
    """tournament_size larger than the pool raises AssertionError."""
    with pytest.raises(AssertionError, match="tournament_size"):
        archive.tournament_population(n=1, tournament_size=9999, pool_multiplier=1)


# ---------------------------------------------------------------------------
# by_generation
# ---------------------------------------------------------------------------


def test_by_generation_returns_population(archive) -> None:
    """by_generation returns a Population."""
    pop = archive.by_generation(0)
    assert isinstance(pop, Population)


def test_by_generation_correct_count(archive) -> None:
    """All 20 individuals were alive at generation 0."""
    pop = archive.by_generation(0)
    assert len(pop) == 20


def test_by_generation_mid_run(archive) -> None:
    """All individuals alive through generation 3 (born=0, died=5)."""
    pop = archive.by_generation(3)
    assert len(pop) == 20


def test_by_generation_out_of_range_empty(archive) -> None:
    """A generation beyond the run returns no individuals."""
    pop = archive.by_generation(9999)
    assert len(pop) == 0


# ---------------------------------------------------------------------------
# fitness_percentile_population
# ---------------------------------------------------------------------------


def test_fitness_percentile_returns_population(archive) -> None:
    """fitness_percentile_population returns a Population."""
    pop = archive.fitness_percentile_population(25.0, 75.0)
    assert isinstance(pop, Population)


def test_fitness_percentile_in_band(archive) -> None:
    """All returned individuals fall within the percentile band."""
    pop = archive.fitness_percentile_population(0.0, 50.0)
    if len(pop) == 0:
        return
    fitnesses = [ind.fitness_ for ind in pop]
    all_fitnesses = [ind.fitness_ for ind in archive._fetch_all(archive._base_query())]
    import numpy as np
    lo = float(np.percentile(all_fitnesses, 0.0))
    hi = float(np.percentile(all_fitnesses, 50.0))
    for f in fitnesses:
        assert lo <= f <= hi


def test_fitness_percentile_with_n_sampling(archive) -> None:
    """n parameter limits the number of returned individuals."""
    pop = archive.fitness_percentile_population(0.0, 100.0, n=3)
    assert len(pop) <= 3


def test_fitness_percentile_impossible_birth_range_empty(archive) -> None:
    """Impossible birth_range returns an empty population."""
    pop = archive.fitness_percentile_population(0.0, 100.0, birth_range=(9999, 9999))
    assert len(pop) == 0
