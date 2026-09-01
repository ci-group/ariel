"""Test: batch experiments and cross-run aggregation."""

# Standard library
import json
from pathlib import Path

# Third-party libraries
import numpy as np
import pytest

# Local libraries
from ariel.spatial_ea.config import SpatialEAConfig
from ariel.spatial_ea.experiment import (
    SERIES_NAMES,
    AggregatedResults,
    ExperimentRunner,
    ExperimentSpec,
    RunResult,
    run_trial,
)

EXTINCT = "Population extinction (below 1)"
EXPLODED = "Population reached maximum limit (40)"


def _base(**overrides: object) -> SpatialEAConfig:
    """Build a fast base configuration."""
    defaults = {
        "population_size": 4,
        "num_generations": 3,
        "simulation_time": 0.15,
        "world_size": (4.0, 4.0),
        "min_spawn_distance": 0.5,
        "pairing_radius": 2.0,
        "save_results": False,
        "save_plots": False,
        "print_generation_stats": False,
    }
    return SpatialEAConfig(**{**defaults, **overrides})


def _result(
    run_id: int,
    populations: list[float],
    reason: str = "",
    final: float | None = None,
) -> RunResult:
    """Build a synthetic trial with a given population trace."""
    count = len(populations)
    return RunResult(
        experiment="x",
        run_id=run_id,
        seed=run_id,
        generations=list(range(count)),
        series={
            name: (
                [float(p) for p in populations]
                if name == "population_size"
                else [2.0] * count
            )
            for name in SERIES_NAMES
        },
        final_population_size=int(
            final if final is not None else populations[-1],
        ),
        stopped_early=bool(reason),
        stop_reason=reason,
    )


# -- Spec ----------------------------------------------------------------------
def test_spec_applies_overrides() -> None:
    """Overrides replace fields on the base configuration."""
    spec = ExperimentSpec(
        name="e",
        overrides={"selection_method": "density_based", "population_size": 12},
    )
    config = spec.config(_base())

    assert config.selection_method == "density_based"
    assert config.population_size == 12
    # Untouched fields survive.
    assert config.simulation_time == 0.15


def test_spec_revalidates_dependent_fields() -> None:
    """Overriding the world must re-run the spawn-area clamp."""
    spec = ExperimentSpec(name="e", overrides={"world_size": (3.0, 3.0)})
    config = spec.config(SpatialEAConfig())

    assert config.spawn_x_max <= 3.0
    assert config.spawn_y_max <= 3.0


def test_spec_rejects_unknown_fields() -> None:
    """A typo in an override should fail loudly, not be ignored."""
    spec = ExperimentSpec(name="e", overrides={"populaton_size": 10})

    with pytest.raises(ValueError, match="unknown config"):
        spec.config(_base())


# -- Trials --------------------------------------------------------------------
def test_run_trial_records_a_run() -> None:
    """A trial returns its series and metadata."""
    result = run_trial(_base(), "e", run_id=2, seed=5)

    assert result.error is None
    assert result.run_id == 2
    assert result.seed == 5
    assert result.completed_generations == 3
    assert set(result.series) == set(SERIES_NAMES)
    assert len(result.series["population_size"]) == 3
    assert result.duration_seconds > 0


def test_the_same_seed_reproduces_a_trial() -> None:
    """Seeding covers both generators the EA draws from."""
    first = run_trial(_base(), "e", run_id=0, seed=99)
    second = run_trial(_base(), "e", run_id=0, seed=99)

    assert first.best_fitness == second.best_fitness
    assert first.series["population_size"] == second.series["population_size"]


def test_different_seeds_diverge() -> None:
    """Trials of one experiment must not all be the same run."""
    traces = {
        run_trial(_base(), "e", run_id=i, seed=i).best_fitness for i in range(3)
    }

    assert len(traces) > 1


def test_a_failing_trial_is_captured_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One bad trial must not abort a batch."""

    def explode(self: object, *args: object, **kwargs: object) -> None:
        msg = "simulated engine failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        "ariel.spatial_ea.experiment.SpatialEA.run",
        explode,
    )

    result = run_trial(_base(), "e", run_id=0, seed=0)

    assert result.error == "simulated engine failure"
    assert result.completed_generations == 0
    assert result.duration_seconds >= 0


def test_an_empty_population_is_extinction_not_a_crash() -> None:
    """A degenerate population size is a run outcome, not an error."""
    result = run_trial(_base(population_size=0), "e", run_id=0, seed=0)

    assert result.error is None
    assert result.stopped_early is True
    assert "extinction" in result.stop_reason.lower()


# -- Running -------------------------------------------------------------------
def test_run_experiment_uses_one_seed_per_trial(tmp_path: Path) -> None:
    """Trial i runs with ``seed + i``."""
    runner = ExperimentRunner(
        base_config=_base(),
        output_folder=tmp_path,
        seed=200,
    )
    results = runner.run_experiment(ExperimentSpec(name="e", num_runs=3))

    assert [r.seed for r in results] == [200, 201, 202]
    assert [r.run_id for r in results] == [0, 1, 2]


def test_trials_write_into_separate_folders(tmp_path: Path) -> None:
    """Concurrent trials must not overwrite one another's output."""
    runner = ExperimentRunner(base_config=_base(), output_folder=tmp_path)
    spec = ExperimentSpec(name="e", num_runs=2)

    folders = {
        runner._trial_config(spec, run_id).result_folder for run_id in range(2)
    }

    assert len(folders) == 2


@pytest.mark.parametrize("parallel", [False, True])
def test_experiment_completes_either_way(
    tmp_path: Path,
    *,
    parallel: bool,
) -> None:
    """Sequential and parallel execution produce the same trials."""
    runner = ExperimentRunner(
        base_config=_base(),
        output_folder=tmp_path,
        seed=11,
    )
    results = runner.run_experiment(
        ExperimentSpec(name="e", num_runs=2),
        parallel=parallel,
        num_workers=2,
    )

    assert len(results) == 2
    assert all(r.error is None for r in results)
    assert [r.run_id for r in results] == [0, 1]


# -- Aggregation ---------------------------------------------------------------
def test_aggregate_spans_the_longest_run() -> None:
    """The generation axis covers every trial."""
    aggregated = ExperimentRunner().aggregate(
        [_result(0, [10, 10, 10, 10]), _result(1, [10, 8])],
    )

    assert len(aggregated.generations) == 4
    assert aggregated.num_runs == 2


def test_runs_active_is_not_inflated_by_padding() -> None:
    """Padding must not make a finished run look like it is still going."""
    aggregated = ExperimentRunner().aggregate(
        [_result(0, [10] * 4), _result(1, [10, 8]), _result(2, [10, 9, 9])],
        "forward_fill",
    )

    assert aggregated.runs_active is not None
    assert aggregated.runs_active.tolist() == [3.0, 3.0, 2.0, 1.0]


def test_nan_padding_averages_only_live_runs() -> None:
    """With ``nan`` padding a finished run contributes nothing."""
    aggregated = ExperimentRunner().aggregate(
        [_result(0, [10, 10, 4]), _result(1, [10, 8])],
        "nan",
    )

    # Generation 2 has only the first run left.
    assert aggregated.mean["population_size"][2] == pytest.approx(4.0)


def test_forward_fill_holds_the_last_value() -> None:
    """A finished run keeps contributing the value it ended on."""
    aggregated = ExperimentRunner().aggregate(
        [_result(0, [10, 10, 10]), _result(1, [10, 8])],
        "forward_fill",
    )

    assert aggregated.mean["population_size"].tolist() == pytest.approx([
        10.0,
        9.0,
        9.0,
    ])


def test_terminal_state_uses_the_outcome_not_the_last_value() -> None:
    """An extinct run counts as empty, whatever it last recorded."""
    runs = [_result(0, [10, 10, 10]), _result(1, [10, 8], EXTINCT)]

    forward = ExperimentRunner().aggregate(runs, "forward_fill")
    terminal = ExperimentRunner().aggregate(runs, "terminal_state")

    # The extinct run last recorded 8, but it is really gone.
    assert forward.mean["population_size"][2] == pytest.approx(9.0)
    assert terminal.mean["population_size"][2] == pytest.approx(5.0)


def test_terminal_state_holds_an_exploded_run_at_its_ceiling() -> None:
    """A run stopped for growth stays at the size it reached."""
    aggregated = ExperimentRunner().aggregate(
        [_result(0, [10, 10, 10]), _result(1, [10, 40], EXPLODED, final=40)],
        "terminal_state",
    )

    assert aggregated.mean["population_size"][2] == pytest.approx(25.0)


def test_outcomes_are_counted() -> None:
    """Extinctions, explosions and completions are tallied separately."""
    aggregated = ExperimentRunner().aggregate([
        _result(0, [10, 10, 10]),
        _result(1, [10, 0], EXTINCT),
        _result(2, [10, 40], EXPLODED, final=40),
    ])

    assert aggregated.num_extinctions == 1
    assert aggregated.num_explosions == 1
    assert aggregated.num_completed == 1
    assert aggregated.completion_rate == pytest.approx(1 / 3)


def test_failed_trials_are_excluded_from_aggregation() -> None:
    """A crashed trial should not distort the statistics."""
    broken = RunResult(experiment="x", run_id=1, seed=1, error="boom")
    aggregated = ExperimentRunner().aggregate([_result(0, [10, 10]), broken])

    assert aggregated.num_runs == 1


def test_aggregating_nothing_is_an_error() -> None:
    """There is no meaningful aggregate of zero successful runs."""
    with pytest.raises(ValueError, match="No successful runs"):
        ExperimentRunner().aggregate([RunResult("x", 0, 0, error="boom")])


def test_empty_aggregate_summary_is_safe() -> None:
    """A summary of an empty aggregate must not divide by zero."""
    empty = AggregatedResults(experiment="x", generations=np.array([]))

    assert empty.completion_rate == 0.0
    assert empty.summary()["best_fitness_ever"] == 0.0


# -- Persistence and sweeps ----------------------------------------------------
def test_save_writes_runs_aggregate_and_summary(tmp_path: Path) -> None:
    """Everything needed to re-analyse an experiment lands on disk."""
    runner = ExperimentRunner(base_config=_base(), output_folder=tmp_path)
    spec = ExperimentSpec(
        name="e",
        overrides={"selection_method": "age_based"},
        num_runs=2,
    )
    results = runner.run_experiment(spec)

    paths = runner.save(spec, results)

    assert set(paths) == {"runs", "aggregated", "summary"}
    assert all(path.exists() for path in paths.values())

    runs = json.loads(paths["runs"].read_text())
    assert runs["overrides"] == {"selection_method": "age_based"}
    assert len(runs["runs"]) == 2

    archive = np.load(paths["aggregated"])
    assert "generations" in archive
    assert "mean_population_size" in archive
    assert "runs_active" in archive

    summary = json.loads(paths["summary"].read_text())
    assert summary["num_runs"] == 2
    assert summary["experiment"] == "e"


def test_grid_search_covers_every_combination(tmp_path: Path) -> None:
    """A two-by-two grid produces four experiments."""
    runner = ExperimentRunner(base_config=_base(), output_folder=tmp_path)

    aggregated = runner.grid_search(
        "sweep",
        {"mutation_rate": [0.2, 0.8], "crossover_rate": [0.5, 1.0]},
        num_runs=1,
    )

    assert len(aggregated) == 4
    # Names encode the grid point, and are filesystem-safe.
    for name in aggregated:
        assert name.startswith("sweep_")
        assert "/" not in name
        assert " " not in name


def test_compare_ranks_by_best_fitness(tmp_path: Path) -> None:
    """The comparison puts the strongest experiment first."""
    weak = AggregatedResults(
        experiment="weak",
        generations=np.arange(2.0),
        maximum={"fitness_best": np.array([0.1, 0.2])},
        num_runs=1,
    )
    strong = AggregatedResults(
        experiment="strong",
        generations=np.arange(2.0),
        maximum={"fitness_best": np.array([0.5, 0.9])},
        num_runs=1,
    )

    path = tmp_path / "comparison.json"
    rows = ExperimentRunner().compare({"weak": weak, "strong": strong}, path)

    assert [row["experiment"] for row in rows] == ["strong", "weak"]
    assert json.loads(path.read_text())[0]["experiment"] == "strong"
