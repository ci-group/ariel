"""Batch experiments over the spatial EA.

One experiment is a named set of configuration overrides run several times with
different seeds. Because population size is not fixed and runs stop early on
extinction or explosion, runs of the same experiment routinely have different
lengths — so aggregation has to say explicitly how it treats a run that has
already ended.

Notes
-----
    * Overrides are a plain mapping applied with
      ``SpatialEAConfig.model_copy(update=...)``. The research prototype
      mirrored every setting in a second dataclass and round-tripped it through
      YAML; a flat settings model makes that unnecessary.
    * Parallel runs go through ``multiprocessing``. The configuration is
      picklable, so workers receive it directly rather than re-importing the
      engine and re-reading a config file.

"""

# Standard library
from __future__ import annotations

import json
import multiprocessing as mp
import operator
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# Third-party libraries
import numpy as np

# Local libraries
from ariel import log
from ariel.spatial_ea.config import SpatialEAConfig
from ariel.spatial_ea.engine import SpatialEA

# Evaluate type annotations in a deferred manner (ruff: UP037)
if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ariel.parameters.ariel_types import FloatArray

# Type Aliases
type PaddingStrategy = Literal["nan", "forward_fill", "terminal_state"]

# Global constants
SERIES_NAMES = (
    "population_size",
    "fitness_best",
    "fitness_avg",
    "fitness_worst",
    "mating_pairs",
    "births",
    "deaths",
)
DEFAULT_SEED = 0


@dataclass
class RunResult:
    """The outcome of one trial.

    Parameters
    ----------
    experiment
        Name of the experiment this trial belongs to.
    run_id
        Index of the trial within the experiment.
    seed
        Random seed the trial was run with.
    generations
        Generation numbers actually reached.
    series
        Per-generation series, keyed by the names in :data:`SERIES_NAMES`.
    best_fitness
        Best fitness seen in the final population.
    final_population_size
        Population size when the run ended.
    stopped_early
        Whether a population limit ended the run.
    stop_reason
        Why the run ended.
    duration_seconds
        Wall-clock time the trial took.
    error
        Message if the trial raised, otherwise ``None``.
    """

    experiment: str
    run_id: int
    seed: int
    generations: list[int] = field(default_factory=list)
    series: dict[str, list[float]] = field(default_factory=dict)
    best_fitness: float = 0.0
    final_population_size: int = 0
    stopped_early: bool = False
    stop_reason: str = ""
    duration_seconds: float = 0.0
    error: str | None = None

    @property
    def completed_generations(self) -> int:
        """Number of generations the trial recorded.

        Returns
        -------
            Length of the generation series.
        """
        return len(self.generations)

    @property
    def went_extinct(self) -> bool:
        """Whether the run ended by running out of individuals.

        Returns
        -------
            ``True`` when the stop reason mentions extinction.
        """
        return "extinction" in self.stop_reason.lower()

    @property
    def exploded(self) -> bool:
        """Whether the run ended by hitting the population ceiling.

        Returns
        -------
            ``True`` when the stop reason mentions the maximum limit.
        """
        return "maximum limit" in self.stop_reason.lower()

    def to_dict(self) -> dict[str, Any]:
        """Render the result as JSON-serialisable data.

        Returns
        -------
            The trial's metadata and series.
        """
        return {
            "experiment": self.experiment,
            "run_id": self.run_id,
            "seed": self.seed,
            "generations": self.generations,
            "series": self.series,
            "best_fitness": self.best_fitness,
            "final_population_size": self.final_population_size,
            "completed_generations": self.completed_generations,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


@dataclass
class AggregatedResults:
    """Per-generation statistics pooled across the trials of one experiment.

    Parameters
    ----------
    experiment
        Name of the experiment.
    generations
        Generation axis, spanning the longest trial.
    mean, std, minimum, maximum
        Per-series statistics across trials, each an array over
        ``generations``.
    runs_active
        How many trials were still running at each generation. Padding never
        inflates this, so it is the honest denominator.
    num_runs
        Number of trials aggregated.
    num_extinctions, num_explosions, num_completed
        How the trials ended.
    padding_strategy
        How trials that ended early were extended.
    """

    experiment: str
    generations: FloatArray
    mean: dict[str, FloatArray] = field(default_factory=dict)
    std: dict[str, FloatArray] = field(default_factory=dict)
    minimum: dict[str, FloatArray] = field(default_factory=dict)
    maximum: dict[str, FloatArray] = field(default_factory=dict)
    runs_active: FloatArray | None = None
    num_runs: int = 0
    num_extinctions: int = 0
    num_explosions: int = 0
    num_completed: int = 0
    padding_strategy: PaddingStrategy = "forward_fill"

    @property
    def completion_rate(self) -> float:
        """Fraction of trials that ran to their planned last generation.

        Returns
        -------
            A value in ``[0, 1]``; zero when there were no trials.
        """
        if self.num_runs == 0:
            return 0.0
        return self.num_completed / self.num_runs

    def summary(self) -> dict[str, Any]:
        """Summarise the experiment.

        Returns
        -------
            Trial counts, completion rate and headline fitness numbers.
        """
        best = self.maximum.get("fitness_best")
        return {
            "experiment": self.experiment,
            "num_runs": self.num_runs,
            "num_completed": self.num_completed,
            "num_extinctions": self.num_extinctions,
            "num_explosions": self.num_explosions,
            "completion_rate": self.completion_rate,
            "generations": len(self.generations),
            "padding_strategy": self.padding_strategy,
            "best_fitness_ever": (
                float(np.nanmax(best))
                if best is not None and best.size
                else 0.0
            ),
        }


@dataclass
class ExperimentSpec:
    """A named configuration to run several times.

    Parameters
    ----------
    name
        Identifier, used for output folders and reports.
    overrides
        Configuration fields to change from the base configuration.
    num_runs
        Number of trials.
    description
        Optional note about what the experiment is testing.
    """

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    num_runs: int = 3
    description: str = ""

    def config(self, base: SpatialEAConfig) -> SpatialEAConfig:
        """Apply this spec's overrides to a base configuration.

        Parameters
        ----------
        base
            The configuration to start from.

        Returns
        -------
            A new configuration with the overrides applied and revalidated, so
            that dependent fields (the spawn area, for one) stay consistent.

        Raises
        ------
        ValueError
            If an override names a field the configuration does not have.
        """
        unknown = set(self.overrides) - set(SpatialEAConfig.model_fields)
        if unknown:
            msg = (
                f"Experiment {self.name!r} overrides unknown config "
                f"field(s): {sorted(unknown)}"
            )
            raise ValueError(msg)

        updated = base.model_copy(update=dict(self.overrides))
        # model_copy skips validators, so rebuild to re-run them.
        return SpatialEAConfig(**updated.model_dump())


def _collect(engine: SpatialEA) -> dict[str, list[float]]:
    """Pull the per-generation series out of a finished engine.

    Parameters
    ----------
    engine
        The engine after a completed run.

    Returns
    -------
        Series keyed by the names in :data:`SERIES_NAMES`.
    """
    data = engine.data_collector
    raw: dict[str, list[float] | list[int]] = {
        "population_size": data.population_size,
        "fitness_best": data.fitness_best,
        "fitness_avg": data.fitness_avg,
        "fitness_worst": data.fitness_worst,
        "mating_pairs": data.mating_pairs,
        "births": data.births,
        "deaths": data.deaths,
    }
    return {name: [float(v) for v in values] for name, values in raw.items()}


def run_trial(
    config: SpatialEAConfig,
    experiment: str,
    run_id: int,
    seed: int,
) -> RunResult:
    """Run one trial of an experiment.

    Defined at module level so that it can be pickled for a worker pool.

    Parameters
    ----------
    config
        Configuration for this trial.
    experiment
        Name of the experiment.
    run_id
        Index of the trial.
    seed
        Random seed, applied to both generators the EA draws from.

    Returns
    -------
        The trial's result. A trial that raises returns a result carrying the
        error rather than taking the whole experiment down.
    """
    np.random.seed(seed)
    random.seed(seed)

    started = time.time()
    try:
        engine = SpatialEA(config=config)
        best = engine.run()
    except Exception as exc:  # noqa: BLE001 - one bad trial must not abort a batch
        msg = f"{experiment} run {run_id} failed: {exc}"
        log.exception(msg)
        return RunResult(
            experiment=experiment,
            run_id=run_id,
            seed=seed,
            duration_seconds=time.time() - started,
            error=str(exc),
        )

    collector = engine.data_collector
    return RunResult(
        experiment=experiment,
        run_id=run_id,
        seed=seed,
        generations=list(collector.generations),
        series=_collect(engine),
        best_fitness=float(best.fitness) if best is not None else 0.0,
        final_population_size=engine.population_size,
        stopped_early=collector.stopped_early,
        stop_reason=collector.stop_reason,
        duration_seconds=time.time() - started,
    )


def _run_trial_args(args: tuple[SpatialEAConfig, str, int, int]) -> RunResult:
    """Unpack a worker's arguments and run the trial.

    Parameters
    ----------
    args
        ``(config, experiment, run_id, seed)``.

    Returns
    -------
        The trial's result.
    """
    return run_trial(*args)


@dataclass
class ExperimentRunner:
    """Runs experiments and aggregates their trials.

    Parameters
    ----------
    base_config
        Configuration every experiment starts from.
    output_folder
        Directory for per-experiment output.
    seed
        Base seed; trial ``i`` of an experiment uses ``seed + i``.
    """

    base_config: SpatialEAConfig = field(default_factory=SpatialEAConfig)
    output_folder: Path = field(
        default_factory=lambda: Path.cwd() / "__experiments__",
    )
    seed: int = DEFAULT_SEED

    def experiment_folder(self, name: str) -> Path:
        """Directory for one experiment's output.

        Parameters
        ----------
        name
            Experiment name.

        Returns
        -------
            The directory, created if absent.
        """
        folder = Path(self.output_folder) / name
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _trial_config(
        self,
        spec: ExperimentSpec,
        run_id: int,
    ) -> SpatialEAConfig:
        """Build the configuration for one trial.

        Each trial writes into its own subfolder so that concurrent trials
        cannot overwrite one another's results.

        Parameters
        ----------
        spec
            The experiment being run.
        run_id
            Index of the trial.

        Returns
        -------
            The trial's configuration.
        """
        config = spec.config(self.base_config)
        run_folder = self.experiment_folder(spec.name) / f"run_{run_id:03d}"
        return config.model_copy(
            update={
                "result_folder": run_folder,
                "figure_folder": run_folder,
                "video_folder": run_folder,
            },
        )

    def run_experiment(
        self,
        spec: ExperimentSpec,
        *,
        parallel: bool = False,
        num_workers: int | None = None,
    ) -> list[RunResult]:
        """Run every trial of one experiment.

        Parameters
        ----------
        spec
            The experiment to run.
        parallel
            Whether to run trials in worker processes.
        num_workers
            Worker count, defaulting to one fewer than the CPU count.

        Returns
        -------
            One result per trial, in trial order.
        """
        jobs = [
            (
                self._trial_config(spec, run_id),
                spec.name,
                run_id,
                self.seed + run_id,
            )
            for run_id in range(spec.num_runs)
        ]

        msg = (
            f"Experiment {spec.name!r}: {spec.num_runs} run(s)"
            f"{' in parallel' if parallel else ''}"
        )
        log.info(msg)

        if not parallel or spec.num_runs == 1:
            return list(starmap(run_trial, jobs))

        workers = num_workers or max(1, (mp.cpu_count() or 2) - 1)
        # "spawn" keeps each worker's MuJoCo and RNG state independent.
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            results = pool.map(_run_trial_args, jobs)

        return sorted(results, key=lambda result: result.run_id)

    def aggregate(
        self,
        results: Sequence[RunResult],
        padding_strategy: PaddingStrategy = "forward_fill",
    ) -> AggregatedResults:
        """Pool trials of one experiment into per-generation statistics.

        Trials end at different generations, so the shorter ones must be
        extended to a common length before averaging. ``nan`` averages only the
        trials still running; ``forward_fill`` holds each trial's last value;
        ``terminal_state`` treats an extinct run as zero population and an
        exploded one as its ceiling.

        Parameters
        ----------
        results
            Trials to aggregate. Failed trials are ignored.
        padding_strategy
            How to extend trials that ended early.

        Returns
        -------
            The pooled statistics.

        Raises
        ------
        ValueError
            If there is nothing to aggregate.
        """
        usable = [r for r in results if r.error is None and r.generations]
        if not usable:
            msg = "No successful runs to aggregate"
            raise ValueError(msg)

        max_gens = max(r.completed_generations for r in usable)
        generations = np.arange(max_gens, dtype=float)

        aggregated = AggregatedResults(
            experiment=usable[0].experiment,
            generations=generations,
            num_runs=len(usable),
            padding_strategy=padding_strategy,
        )

        active = np.zeros((len(usable), max_gens), dtype=bool)
        for i, result in enumerate(usable):
            active[i, : result.completed_generations] = True
        aggregated.runs_active = active.sum(axis=0).astype(float)

        for name in SERIES_NAMES:
            table = np.full((len(usable), max_gens), np.nan)
            for i, result in enumerate(usable):
                values = result.series.get(name, [])
                count = min(len(values), max_gens)
                table[i, :count] = values[:count]
                if count < max_gens:
                    table[i, count:] = self._pad_value(
                        name,
                        result,
                        values,
                        padding_strategy,
                    )

            # A generation where every run is padded-out is all-NaN; the
            # nanmean of that is undefined, and warning about it is noise.
            with np.errstate(invalid="ignore"):
                aggregated.mean[name] = _nan_reduce(np.nanmean, table)
                aggregated.std[name] = _nan_reduce(np.nanstd, table)
                aggregated.minimum[name] = _nan_reduce(np.nanmin, table)
                aggregated.maximum[name] = _nan_reduce(np.nanmax, table)

        aggregated.num_extinctions = sum(1 for r in usable if r.went_extinct)
        aggregated.num_explosions = sum(1 for r in usable if r.exploded)
        aggregated.num_completed = sum(1 for r in usable if not r.stopped_early)

        return aggregated

    @staticmethod
    def _pad_value(
        name: str,
        result: RunResult,
        values: list[float],
        strategy: PaddingStrategy,
    ) -> float:
        """Decide what a finished trial contributes to later generations.

        Parameters
        ----------
        name
            Series being padded.
        result
            The trial.
        values
            The trial's values for this series.
        strategy
            Padding strategy.

        Returns
        -------
            The value to extend the trial with, possibly ``nan``.
        """
        if strategy == "nan":
            return float("nan")

        if strategy == "terminal_state":
            if result.went_extinct:
                return 0.0
            if result.exploded and name == "population_size":
                return float(result.final_population_size)

        return float(values[-1]) if values else float("nan")

    def save(
        self,
        spec: ExperimentSpec,
        results: Sequence[RunResult],
        aggregated: AggregatedResults | None = None,
    ) -> dict[str, Path]:
        """Write an experiment's raw trials and pooled statistics.

        Parameters
        ----------
        spec
            The experiment that was run.
        results
            Its trials.
        aggregated
            Pooled statistics, computed here when not supplied.

        Returns
        -------
            Paths written, keyed ``runs``, ``aggregated`` and ``summary``.
        """
        folder = self.experiment_folder(spec.name)
        if aggregated is None:
            aggregated = self.aggregate(results)

        runs_path = folder / "runs.json"
        with runs_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "experiment": spec.name,
                    "description": spec.description,
                    "overrides": spec.overrides,
                    "runs": [r.to_dict() for r in results],
                },
                handle,
                indent=2,
            )

        aggregated_path = folder / "aggregated.npz"
        arrays: dict[str, Any] = {"generations": aggregated.generations}
        if aggregated.runs_active is not None:
            arrays["runs_active"] = aggregated.runs_active
        for stat, table in (
            ("mean", aggregated.mean),
            ("std", aggregated.std),
            ("min", aggregated.minimum),
            ("max", aggregated.maximum),
        ):
            for name, values in table.items():
                arrays[f"{stat}_{name}"] = values
        np.savez(aggregated_path, **arrays)

        summary_path = folder / "summary.json"
        summary = aggregated.summary()
        summary["saved_at"] = datetime.now(UTC).isoformat()
        summary["overrides"] = spec.overrides
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        msg = f"Experiment {spec.name!r} written to {folder}"
        log.info(msg)

        return {
            "runs": runs_path,
            "aggregated": aggregated_path,
            "summary": summary_path,
        }

    def run_and_save(
        self,
        spec: ExperimentSpec,
        *,
        parallel: bool = False,
        num_workers: int | None = None,
        padding_strategy: PaddingStrategy = "forward_fill",
    ) -> tuple[list[RunResult], AggregatedResults]:
        """Run one experiment, aggregate it and write it to disk.

        Parameters
        ----------
        spec
            The experiment to run.
        parallel
            Whether to run trials in worker processes.
        num_workers
            Worker count.
        padding_strategy
            How to extend trials that ended early.

        Returns
        -------
        results
            The trials.
        aggregated
            The pooled statistics.
        """
        results = self.run_experiment(
            spec,
            parallel=parallel,
            num_workers=num_workers,
        )
        aggregated = self.aggregate(results, padding_strategy)
        self.save(spec, results, aggregated)
        return results, aggregated

    def grid_search(
        self,
        name: str,
        grid: dict[str, Sequence[Any]],
        num_runs: int = 1,
        *,
        parallel: bool = False,
        num_workers: int | None = None,
    ) -> dict[str, AggregatedResults]:
        """Run one experiment per point of a parameter grid.

        Parameters
        ----------
        name
            Prefix for the generated experiment names.
        grid
            Field name to the values it should take.
        num_runs
            Trials per grid point.
        parallel
            Whether to run trials in worker processes.
        num_workers
            Worker count.

        Returns
        -------
            Pooled statistics per grid point, keyed by experiment name.
        """
        aggregated: dict[str, AggregatedResults] = {}
        for overrides in _grid_points(grid):
            label = "_".join(
                f"{field_name}{_slug(value)}"
                for field_name, value in overrides.items()
            )
            spec = ExperimentSpec(
                name=f"{name}_{label}",
                overrides=overrides,
                num_runs=num_runs,
                description=f"grid point {overrides}",
            )
            _, stats = self.run_and_save(
                spec,
                parallel=parallel,
                num_workers=num_workers,
            )
            aggregated[spec.name] = stats

        return aggregated

    def compare(
        self,
        aggregated: dict[str, AggregatedResults],
        save_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Summarise several experiments side by side.

        Parameters
        ----------
        aggregated
            Pooled statistics per experiment.
        save_path
            Optional JSON file to write the comparison to.

        Returns
        -------
            One summary row per experiment, best fitness first.
        """
        rows = [stats.summary() for stats in aggregated.values()]
        rows.sort(key=operator.itemgetter("best_fitness_ever"), reverse=True)

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w", encoding="utf-8") as handle:
                json.dump(rows, handle, indent=2)

        return rows


def _nan_reduce(
    reducer: Any,
    table: FloatArray,
) -> FloatArray:
    """Apply a NaN-aware reduction down the trial axis.

    Parameters
    ----------
    reducer
        A ``numpy`` ``nan*`` reduction.
    table
        Trials by generations.

    Returns
    -------
        One value per generation; ``nan`` where every trial was ``nan``.
    """
    all_nan = np.all(np.isnan(table), axis=0)
    out = np.full(table.shape[1], np.nan)
    if not np.all(all_nan):
        out[~all_nan] = reducer(table[:, ~all_nan], axis=0)
    return out


def _grid_points(grid: dict[str, Sequence[Any]]) -> Iterable[dict[str, Any]]:
    """Expand a parameter grid into individual override mappings.

    Parameters
    ----------
    grid
        Field name to the values it should take.

    Yields
    ------
        One mapping per combination, in field order.
    """
    if not grid:
        return

    names = list(grid)
    combinations: list[list[Any]] = [[]]
    for name in names:
        combinations = [
            [*prefix, value] for prefix in combinations for value in grid[name]
        ]

    for combination in combinations:
        yield dict(zip(names, combination, strict=True))


def _slug(value: Any) -> str:
    """Render a grid value as a filename-safe fragment.

    Parameters
    ----------
    value
        The value to render.

    Returns
    -------
        A short string with no path separators or spaces.
    """
    text = str(value)
    for bad, good in (
        (" ", ""),
        ("/", "-"),
        (".", "p"),
        ("(", ""),
        (")", ""),
        (",", "-"),
    ):
        text = text.replace(bad, good)
    return text
