"""Figures for inspecting a spatial EA run.

Two views answer different questions. The trajectory plot shows one generation
in space: where robots started, where they walked, which zones they reached and
which pairs actually formed. The statistics plot shows the whole run in time:
population, fitness, age, mating success and energy per generation.

Notes
-----
    * This module imports ``matplotlib.pyplot``, which is why it is not
      re-exported from ``ariel.spatial_ea``. Import it by path so that batch
      runs that never plot do not pay for it.
    * No backend is selected here. Selecting one is the application's call;
      scripts that run headless should call ``matplotlib.use("Agg")`` before
      importing this module.

"""

# Standard library
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# Third-party libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D

# Local libraries
from ariel import log
from ariel.spatial_ea.interaction import (
    calculate_periodic_distance,
    split_trajectory_at_wraps,
)

# Evaluate type annotations in a deferred manner (ruff: UP037)
if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ariel.parameters.ariel_types import FloatArray
    from ariel.spatial_ea.data import EvolutionDataCollector
    from ariel.spatial_ea.experiment import AggregatedResults
    from ariel.spatial_ea.individual import SpatialIndividual

# Global constants
# Mirrors the linkage criteria scipy accepts for a condensed distance matrix.
type DendrogramLinkage = Literal["single", "complete", "average", "weighted"]

TAB20 = mpl.colormaps["tab20"]
# tab20 alternates a dark and a light shade of each hue, which is right for
# many individuals but unreadable for a handful of series; tab10 gives each
# experiment its own hue.
TAB10 = mpl.colormaps["tab10"]
EXPERIMENT_COLOR_COUNT = 10
POINTS_PER_INCH = 72
COLOR_CYCLE_LENGTH = 20
WRAP_THRESHOLD = 0.5


def _marker_size_for(
    robot_size: float,
    ax: Axes,
    fig: Figure,
) -> float:
    """Convert a robot diameter in metres into a marker size in points.

    Drawing robots at their true size is what makes a trajectory plot readable:
    it shows at a glance whether two robots are actually close enough to touch.

    Parameters
    ----------
    robot_size
        Robot diameter in metres.
    ax
        Axes the marker will be drawn on, already scaled.
    fig
        Figure the axes belongs to.

    Returns
    -------
        Marker diameter in points.
    """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    fig_width, fig_height = fig.get_size_inches()

    data_per_inch = (
        (x_max - x_min) / fig_width + (y_max - y_min) / fig_height
    ) / 2.0
    if data_per_inch <= 0:
        return 6.0

    return float((robot_size / data_per_inch) * POINTS_PER_INCH)


def _draw_world(
    ax: Axes,
    world_size: tuple[float, float],
) -> None:
    """Draw the world floor and its outline.

    Parameters
    ----------
    ax
        Axes to draw on.
    world_size
        World dimensions ``(width, height)``.
    """
    ax.set_xlim(-0.2, world_size[0] + 0.2)
    ax.set_ylim(-0.2, world_size[1] + 0.2)
    ax.set_aspect("equal")

    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            world_size[0],
            world_size[1],
            linewidth=0,
            facecolor="lightgray",
            alpha=0.25,
        ),
    )
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            world_size[0],
            world_size[1],
            linewidth=1.5,
            edgecolor="black",
            facecolor="none",
        ),
    )


def _draw_zones(
    ax: Axes,
    zone_centers: list[tuple[float, float]],
    zone_radius: float,
    world_size: tuple[float, float] | None = None,
    *,
    use_periodic_boundaries: bool = False,
) -> None:
    """Draw the mating zones.

    On a toroidal world a zone near an edge also covers ground on the opposite
    edge, so the wrapped part of the circle is drawn too. Without it the figure
    would understate which robots can reach the zone.

    Parameters
    ----------
    ax
        Axes to draw on.
    zone_centers
        Centre of each zone.
    zone_radius
        Radius shared by every zone.
    world_size
        World dimensions ``(width, height)``, needed to draw wrapped copies.
    use_periodic_boundaries
        Whether the world wraps.
    """
    if use_periodic_boundaries and world_size is not None:
        offsets = [
            (dx * world_size[0], dy * world_size[1])
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        ]
    else:
        offsets = [(0.0, 0.0)]

    for center in zone_centers:
        for offset_x, offset_y in offsets:
            ax.add_patch(
                patches.Circle(
                    (center[0] + offset_x, center[1] + offset_y),
                    zone_radius,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="lightcoral",
                    alpha=0.18,
                    linestyle="--",
                ),
            )
        ax.plot(
            center[0],
            center[1],
            marker="*",
            color="red",
            markersize=13,
            markeredgecolor="darkred",
            zorder=5,
        )


def plot_mating_trajectories(
    trajectories: list[list[FloatArray]],
    population: list[SpatialIndividual],
    generation: int,
    save_path: str | Path,
    *,
    world_size: tuple[float, float],
    robot_size: float = 0.4,
    simulation_time: float | None = None,
    use_periodic_boundaries: bool = False,
    mating_zone_centers: list[tuple[float, float]] | None = None,
    mating_zone_radius: float | None = None,
    pairs: list[tuple[int, int]] | None = None,
    pairing_method: str | None = None,
) -> Path:
    """Plot one generation's movement phase from above.

    Each robot is drawn from its start (circle) along its path to its end
    (square, labelled with the individual's id). Pairs that formed are joined
    by a heavy green line, so it is immediately visible whether reproduction
    followed from robots actually reaching each other.

    Parameters
    ----------
    trajectories
        One list of ``(x, y)`` samples per robot, in population order.
    population
        The individuals that were simulated, in the same order.
    generation
        Generation number, used in the title.
    save_path
        Where to write the figure.
    world_size
        World dimensions ``(width, height)``.
    robot_size
        Robot diameter in metres, used to size the markers.
    simulation_time
        Duration of the movement phase, used in the title.
    use_periodic_boundaries
        Whether to split trajectories where they wrap around the world.
    mating_zone_centers
        Zone centres to draw, if any.
    mating_zone_radius
        Radius of the zones.
    pairs
        Index pairs that reproduced, drawn as connections.
    pairing_method
        Name of the pairing strategy, used in the title.

    Returns
    -------
        The path the figure was written to.
    """
    fig, ax = plt.subplots(figsize=(11, 11))
    _draw_world(ax, world_size)

    marker_size = _marker_size_for(robot_size, ax, fig)

    if mating_zone_centers and mating_zone_radius is not None:
        _draw_zones(
            ax,
            mating_zone_centers,
            mating_zone_radius,
            world_size,
            use_periodic_boundaries=use_periodic_boundaries,
        )

    fitness_values = [individual.fitness for individual in population]

    for i, raw in enumerate(trajectories):
        if not len(raw):
            continue

        individual = population[i] if i < len(population) else None
        unique_id = individual.unique_id if individual is not None else i
        color = TAB20(
            (unique_id or 0) % COLOR_CYCLE_LENGTH / COLOR_CYCLE_LENGTH,
        )

        trajectory = np.asarray(raw, dtype=float)

        if use_periodic_boundaries:
            segments = split_trajectory_at_wraps(
                list(trajectory),
                world_size,
                WRAP_THRESHOLD,
            )
        else:
            segments = [list(trajectory)]

        # Start: an empty footprint at true robot size, so the body scale is
        # visible without burying the path under it.
        ax.plot(
            trajectory[0, 0],
            trajectory[0, 1],
            "o",
            markerfacecolor="none",
            markeredgecolor=color,
            markersize=marker_size,
            markeredgewidth=1.4,
            alpha=0.8,
            zorder=2,
        )
        # End: filled, and translucent so overlapping robots stay readable.
        ax.plot(
            trajectory[-1, 0],
            trajectory[-1, 1],
            "o",
            color=color,
            markersize=marker_size,
            markeredgecolor="black",
            markeredgewidth=1.2,
            alpha=0.55,
            zorder=3,
        )

        # Paths go on top: they are far shorter than a robot is wide.
        for segment in segments:
            points = np.asarray(segment, dtype=float)
            if len(points) < 2:
                continue
            ax.plot(
                points[:, 0],
                points[:, 1],
                color="black",
                alpha=0.9,
                linewidth=1.6,
                zorder=7,
            )

        ax.text(
            trajectory[-1, 0],
            trajectory[-1, 1],
            str(unique_id),
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="black",
            zorder=8,
        )

    if pairs:
        for first, second in pairs:
            if first >= len(trajectories) or second >= len(trajectories):
                continue
            start = np.asarray(trajectories[first], dtype=float)[-1]
            end = np.asarray(trajectories[second], dtype=float)[-1]
            # A pair that wrapped would streak across the plot; skip that line.
            if use_periodic_boundaries and (
                abs(start[0] - end[0]) > world_size[0] * WRAP_THRESHOLD
                or abs(start[1] - end[1]) > world_size[1] * WRAP_THRESHOLD
            ):
                continue
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                "-",
                color="green",
                linewidth=2.5,
                alpha=0.85,
                zorder=6,
            )

    ax.grid(visible=True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)

    displacements: list[float] = []
    for trajectory_points in trajectories:
        if len(trajectory_points) < 2:
            continue
        start = np.asarray(trajectory_points[0], dtype=float)
        end = np.asarray(trajectory_points[-1], dtype=float)
        if use_periodic_boundaries:
            # A robot that wrapped has not really crossed the whole world.
            displacements.append(
                calculate_periodic_distance(start, end, world_size),
            )
        else:
            displacements.append(float(np.linalg.norm(end - start)))

    title = [f"Mating movement — generation {generation}"]
    subtitle = [f"{len(trajectories)} robots"]
    if simulation_time is not None:
        subtitle.append(f"{simulation_time:g} s")
    if pairing_method:
        subtitle.append(f"pairing: {pairing_method}")
    if pairs is not None:
        subtitle.append(f"{len(pairs)} pair(s) formed")
    if use_periodic_boundaries:
        subtitle.append("periodic boundaries")
    title.append(" | ".join(subtitle))

    scale = []
    if displacements:
        # Say the movement scale outright: a robot is usually wider than the
        # distance it covers, so short paths are the expected result.
        scale.append(
            f"displacement mean {np.mean(displacements):.3f} m, "
            f"max {max(displacements):.3f} m vs robot {robot_size:g} m",
        )
    if fitness_values:
        scale.append(
            f"fitness {min(fitness_values):.3f} – {max(fitness_values):.3f}",
        )
    if scale:
        title.append(" | ".join(scale))

    ax.set_title("\n".join(title), fontsize=11, pad=14)

    legend_elements: list[Artist] = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markersize=min(marker_size, 14),
            markeredgecolor="gray",
            label="start footprint (true robot size)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=min(marker_size, 14),
            markeredgecolor="black",
            alpha=0.55,
            label="end, labelled with individual id",
        ),
        Line2D([0], [0], color="black", linewidth=1.6, label="path walked"),
    ]
    if pairs:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="green",
                linewidth=2.5,
                label="pair that reproduced",
            ),
        )
    if mating_zone_centers and mating_zone_radius is not None:
        legend_elements.append(
            patches.Patch(
                facecolor="lightcoral",
                edgecolor="red",
                linestyle="--",
                alpha=0.4,
                label="mating zone",
            ),
        )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path


def _aligned(
    generations: FloatArray,
    series: list[float] | list[int],
) -> tuple[FloatArray, list[float] | list[int]]:
    """Trim a series and the generation axis to a common length.

    A run that stops on its very first generation records the generation but
    never gets as far as recording fitness, so the series a collector holds are
    not all the same length. Plotting mismatched lengths is a hard error, so
    every panel goes through here.

    Parameters
    ----------
    generations
        The generation axis.
    series
        The values to plot against it.

    Returns
    -------
    x
        The generation axis, trimmed.
    y
        The series, trimmed to match.
    """
    count = min(len(generations), len(series))
    return generations[:count], series[:count]


def plot_evolution_statistics(
    collector: EvolutionDataCollector,
    save_path: str | Path,
) -> Path:
    """Plot a run's per-generation statistics as stacked time series.

    Parameters
    ----------
    collector
        The record accumulated during the run.
    save_path
        Where to write the figure.

    Returns
    -------
        The path the figure was written to.
    """
    generations = np.asarray(collector.generations, dtype=float)
    has_energy = bool(collector.energy_avg)
    has_diversity = any(v > 0 for v in collector.genotype_diversity)
    num_panels = 4 + int(has_energy) + int(has_diversity)

    fig, axes = plt.subplots(
        num_panels,
        1,
        figsize=(11, 2.9 * num_panels),
        sharex=True,
    )
    fig.suptitle("Spatial EA — evolution statistics", fontsize=15)

    # -- Population -----------------------------------------------------------
    ax = axes[0]
    ax.plot(
        *_aligned(generations, collector.population_size),
        "-",
        color="tab:blue",
        linewidth=2,
        label="population",
    )
    # Births and deaths are recorded one generation behind the population.
    # Births and deaths often coincide exactly, so give them distinct markers
    # rather than two dashed lines that hide one another.
    for series, color, marker, label in (
        (collector.births, "tab:green", "^", "births"),
        (collector.deaths, "tab:red", "v", "deaths"),
    ):
        count = min(len(series), max(0, len(generations) - 1))
        if count:
            ax.plot(
                generations[1 : count + 1],
                series[:count],
                "--",
                marker=marker,
                markersize=5,
                color=color,
                alpha=0.8,
                label=label,
            )
    ax.set_ylabel("count")
    ax.set_title("Population dynamics", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(visible=True, alpha=0.3)

    # -- Fitness --------------------------------------------------------------
    ax = axes[1]
    average = np.asarray(collector.fitness_avg, dtype=float)
    spread = np.asarray(collector.fitness_std, dtype=float)
    ax.plot(
        *_aligned(generations, collector.fitness_best),
        "-",
        color="tab:green",
        linewidth=2,
        label="best",
    )
    ax.plot(
        *_aligned(generations, list(average)),
        "-",
        color="tab:blue",
        linewidth=2,
        label="mean",
    )
    ax.plot(
        *_aligned(generations, collector.fitness_worst),
        "-",
        color="tab:red",
        linewidth=1,
        alpha=0.7,
        label="worst",
    )
    if len(average) == len(generations) and len(spread) == len(generations):
        ax.fill_between(
            generations,
            average - spread,
            average + spread,
            alpha=0.18,
            color="tab:blue",
            label="±1 sd",
        )
    ax.set_ylabel("fitness")
    ax.set_title("Fitness", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(visible=True, alpha=0.3)

    # -- Age ------------------------------------------------------------------
    ax = axes[2]
    ax.plot(
        *_aligned(generations, collector.age_avg),
        "-",
        color="tab:purple",
        linewidth=2,
        label="mean age",
    )
    ax.plot(
        *_aligned(generations, collector.age_max),
        "--",
        color="tab:red",
        alpha=0.7,
        label="oldest",
    )
    ax.set_ylabel("generations")
    ax.set_title("Age", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(visible=True, alpha=0.3)

    # -- Mating ---------------------------------------------------------------
    ax = axes[3]
    mating_gens, mating_pairs = _aligned(generations, collector.mating_pairs)
    ax.plot(
        mating_gens,
        mating_pairs,
        "-",
        color="tab:blue",
        linewidth=2,
        label="pairs formed",
    )
    ax.plot(
        mating_gens,
        collector.unpaired_individuals[: len(mating_gens)],
        "--",
        color="tab:red",
        alpha=0.7,
        label="unpaired",
    )
    ax.set_ylabel("count")
    ax.set_title("Mating", fontsize=11)
    ax.grid(visible=True, alpha=0.3)

    rate_ax = ax.twinx()
    rate_ax.plot(
        mating_gens,
        collector.mating_success_rate[: len(mating_gens)],
        "-",
        color="tab:green",
        alpha=0.8,
        label="success rate",
    )
    rate_ax.set_ylabel("success rate (%)", color="tab:green")
    handles = (
        ax.get_legend_handles_labels()[0]
        + (rate_ax.get_legend_handles_labels()[0])
    )
    labels = (
        ax.get_legend_handles_labels()[1]
        + (rate_ax.get_legend_handles_labels()[1])
    )
    ax.legend(handles, labels, fontsize=9, loc="upper left")

    # -- Diversity ------------------------------------------------------------
    next_panel = 4
    if has_diversity:
        ax = axes[next_panel]
        next_panel += 1
        ax.plot(
            *_aligned(generations, collector.genotype_diversity),
            "-",
            color="tab:brown",
            linewidth=2,
            label="genome diversity",
        )
        ax.set_ylabel("weight spread")
        ax.set_title(
            "Genome diversity (spread of enabled connection weights)",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        ax.grid(visible=True, alpha=0.3)

    # -- Energy ---------------------------------------------------------------
    if has_energy:
        ax = axes[next_panel]
        # Energy is sampled more than once per generation (after depletion and
        # again after mating). Spread the samples across the generation span so
        # this panel stays aligned with the ones above it.
        count = len(collector.energy_avg)
        if len(generations) > 1 and count > 1:
            energy_gens = np.linspace(generations[0], generations[-1], count)
        else:
            energy_gens = np.arange(count, dtype=float)
        ax.plot(
            energy_gens,
            collector.energy_avg,
            "-",
            color="tab:orange",
            linewidth=2,
            label="mean energy",
        )
        ax.plot(
            energy_gens,
            collector.energy_min,
            "--",
            color="tab:red",
            alpha=0.7,
            label="lowest",
        )
        ax.axhline(0.0, color="black", linewidth=1, linestyle=":")
        ax.set_ylabel("energy")
        ax.set_title(
            "Energy (sampled after depletion and again after mating)",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        ax.grid(visible=True, alpha=0.3)

    axes[-1].set_xlabel("generation")

    if collector.stopped_early:
        fig.text(
            0.5,
            0.005,
            f"Run stopped early: {collector.stop_reason}",
            ha="center",
            fontsize=10,
            color="tab:red",
        )

    fig.tight_layout()

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    msg = f"Evolution statistics saved to {path}"
    log.info(msg)
    return path


def plot_mating_zones(
    world_size: tuple[float, float],
    zone_centers: list[tuple[float, float]],
    zone_radius: float,
    save_path: str | Path,
    positions: list[FloatArray] | None = None,
    *,
    use_periodic_boundaries: bool = False,
    title_suffix: str = "",
) -> Path:
    """Plot the mating-zone layout, optionally with a population on it.

    Parameters
    ----------
    world_size
        World dimensions ``(width, height)``.
    zone_centers
        Centre of each zone.
    zone_radius
        Radius shared by every zone.
    save_path
        Where to write the figure.
    positions
        Optional population to overlay, coloured by zone membership.
    use_periodic_boundaries
        Whether zone membership wraps around the world edges.
    title_suffix
        Extra text appended to the title.

    Returns
    -------
        The path the figure was written to.
    """
    from ariel.spatial_ea.interaction import is_in_mating_zone

    fig, ax = plt.subplots(figsize=(9, 9))
    _draw_world(ax, world_size)
    _draw_zones(
        ax,
        zone_centers,
        zone_radius,
        world_size,
        use_periodic_boundaries=use_periodic_boundaries,
    )

    if positions:
        inside_seen = False
        outside_seen = False
        for position in positions:
            inside = any(
                is_in_mating_zone(
                    position,
                    center,
                    zone_radius,
                    world_size,
                    use_periodic_boundaries=use_periodic_boundaries,
                )
                for center in zone_centers
            )
            label = None
            if inside and not inside_seen:
                label, inside_seen = "in a zone", True
            elif not inside and not outside_seen:
                label, outside_seen = "outside every zone", True

            ax.plot(
                position[0],
                position[1],
                marker="o" if inside else "x",
                color="green" if inside else "gray",
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=label,
            )
        ax.legend(loc="upper right", fontsize=9)

    coverage = (
        len(zone_centers)
        * np.pi
        * zone_radius**2
        / (world_size[0] * world_size[1])
    )
    title = (
        f"{len(zone_centers)} mating zone(s), radius {zone_radius:g} m — "
        f"{coverage:.1%} of the world covered"
    )
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(visible=True, alpha=0.2)

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path


def plot_aggregated_results(
    aggregated: Mapping[str, AggregatedResults],
    save_path: str | Path,
    metric: str = "fitness_best",
) -> Path:
    """Compare several experiments' pooled statistics over generations.

    The mean is drawn with a one-standard-deviation band, and the number of
    runs still active is drawn underneath. That second panel matters: once most
    runs of an experiment have died out, its mean is an average over a handful
    of survivors and should be read with suspicion.

    Parameters
    ----------
    aggregated
        Pooled statistics per experiment.
    save_path
        Where to write the figure.
    metric
        Which series to plot, from
        :data:`ariel.spatial_ea.experiment.SERIES_NAMES`.

    Returns
    -------
        The path the figure was written to.

    Raises
    ------
    ValueError
        If there is nothing to plot.
    """
    if not aggregated:
        msg = "No aggregated results to plot"
        raise ValueError(msg)

    fig, (ax, active_ax) = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        height_ratios=[3, 1],
    )

    for index, (name, stats) in enumerate(aggregated.items()):
        color = TAB10(
            index % EXPERIMENT_COLOR_COUNT / EXPERIMENT_COLOR_COUNT,
        )
        generations = stats.generations
        mean = stats.mean.get(metric)
        if mean is None or not len(mean):
            continue

        ax.plot(generations, mean, "-", color=color, linewidth=2, label=name)

        spread = stats.std.get(metric)
        if spread is not None and len(spread) == len(mean):
            ax.fill_between(
                generations,
                mean - spread,
                mean + spread,
                color=color,
                alpha=0.15,
            )

        if stats.runs_active is not None:
            active_ax.plot(
                generations,
                stats.runs_active,
                "-",
                color=color,
                linewidth=1.6,
            )

    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(
        f"{metric.replace('_', ' ')} across experiments (mean ±1 sd over runs)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(visible=True, alpha=0.3)

    active_ax.set_ylabel("runs active")
    active_ax.set_xlabel("generation")
    active_ax.grid(visible=True, alpha=0.3)
    active_ax.set_ylim(bottom=0)

    fig.tight_layout()

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    msg = f"Aggregated comparison saved to {path}"
    log.info(msg)
    return path


# -- Clustering ----------------------------------------------------------------
def _cluster_color(label: int) -> Any:
    """Pick a stable colour for a cluster label.

    Parameters
    ----------
    label
        Cluster index, or ``-1`` for noise.

    Returns
    -------
        An RGBA colour; grey for noise.
    """
    if label < 0:
        return (0.6, 0.6, 0.6, 1.0)
    return TAB10(label % EXPERIMENT_COLOR_COUNT / EXPERIMENT_COLOR_COUNT)


def plot_spatial_clusters(
    positions: FloatArray,
    cluster_labels: FloatArray,
    save_path: str | Path,
    *,
    world_size: tuple[float, float],
    spatial_silhouette: float | None = None,
    cluster_centroids: dict[int, FloatArray] | None = None,
    title_suffix: str = "",
) -> Path:
    """Draw the population in the world, coloured by genotype cluster.

    This is the figure the research question lives in: colour comes only from
    genome similarity, position only from where the robots actually are. If the
    colours separate into regions, genetic structure has become spatial
    structure.

    Parameters
    ----------
    positions
        One position row per individual.
    cluster_labels
        Genotype cluster label per individual; ``-1`` is noise.
    save_path
        Where to write the figure.
    world_size
        World dimensions ``(width, height)``.
    spatial_silhouette
        Coherence score to report in the title.
    cluster_centroids
        Optional per-cluster centroids to mark.
    title_suffix
        Extra text appended to the title.

    Returns
    -------
        The path the figure was written to.
    """
    points = np.asarray(positions, dtype=float)
    labels = np.asarray(cluster_labels)

    fig, ax = plt.subplots(figsize=(9, 9))
    _draw_world(ax, world_size)

    for label in sorted({int(v) for v in labels}):
        members = points[labels == label]
        name = "noise" if label < 0 else f"cluster {label}"
        # Noise uses an unfilled marker, which has no edge to colour.
        if label < 0:
            ax.scatter(
                members[:, 0],
                members[:, 1],
                s=90,
                color=_cluster_color(label),
                marker="x",
                label=f"{name} (n={len(members)})",
                zorder=3,
            )
        else:
            ax.scatter(
                members[:, 0],
                members[:, 1],
                s=90,
                color=_cluster_color(label),
                marker="o",
                edgecolors="black",
                linewidths=0.6,
                label=f"{name} (n={len(members)})",
                zorder=3,
            )

    if cluster_centroids:
        for label, centroid in cluster_centroids.items():
            ax.plot(
                centroid[0],
                centroid[1],
                marker="P",
                markersize=14,
                color=_cluster_color(int(label)),
                markeredgecolor="black",
                markeredgewidth=1.2,
                zorder=5,
            )

    title = ["Genotype clusters in space"]
    if spatial_silhouette is not None:
        # Say what the number means; a bare score invites over-reading.
        reading = (
            "clusters occupy distinct regions"
            if spatial_silhouette > 0.25
            else "clusters are spatially mixed"
        )
        title.append(
            f"spatial silhouette {spatial_silhouette:+.3f} — {reading}",
        )
    if title_suffix:
        title.append(title_suffix)
    ax.set_title("\n".join(title), fontsize=12)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(visible=True, alpha=0.2)

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cluster_embedding(
    coordinates: FloatArray,
    cluster_labels: FloatArray,
    save_path: str | Path,
    *,
    method: str = "PCA",
    individual_ids: list[int] | None = None,
) -> Path:
    """Draw a two-dimensional embedding of genome distances.

    Parameters
    ----------
    coordinates
        Embedded coordinates, one row per individual.
    cluster_labels
        Cluster label per individual.
    save_path
        Where to write the figure.
    method
        Name of the embedding, for the axis labels.
    individual_ids
        Optional identifiers to annotate points with.

    Returns
    -------
        The path the figure was written to.
    """
    points = np.asarray(coordinates, dtype=float)
    labels = np.asarray(cluster_labels)

    fig, ax = plt.subplots(figsize=(9, 7))

    for label in sorted({int(v) for v in labels}):
        members = points[labels == label]
        name = "noise" if label < 0 else f"cluster {label}"
        if label < 0:
            ax.scatter(
                members[:, 0],
                members[:, 1],
                s=80,
                color=_cluster_color(label),
                marker="x",
                label=f"{name} (n={len(members)})",
            )
        else:
            ax.scatter(
                members[:, 0],
                members[:, 1],
                s=80,
                color=_cluster_color(label),
                marker="o",
                edgecolors="black",
                linewidths=0.6,
                label=f"{name} (n={len(members)})",
            )

    if individual_ids is not None:
        for point, identifier in zip(points, individual_ids, strict=False):
            ax.annotate(
                str(identifier),
                point[:2],
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
                color="black",
            )

    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(
        f"Genome distances embedded with {method}\n"
        f"axes carry no units; only the grouping is meaningful",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(visible=True, alpha=0.25)

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_distance_heatmap(
    distance_matrix: FloatArray,
    save_path: str | Path,
    cluster_labels: FloatArray | None = None,
    *,
    distance_type: str = "combined",
) -> Path:
    """Draw the pairwise genome distance matrix.

    Sorting rows by cluster turns the matrix into a block structure: dark
    blocks on the diagonal are the clusters.

    Parameters
    ----------
    distance_matrix
        Pairwise genome distances.
    save_path
        Where to write the figure.
    cluster_labels
        Optional labels used to order the rows and columns.
    distance_type
        Name of the metric, for the title.

    Returns
    -------
        The path the figure was written to.
    """
    matrix = np.asarray(distance_matrix, dtype=float)

    order = np.arange(len(matrix))
    if cluster_labels is not None:
        order = np.argsort(np.asarray(cluster_labels), kind="stable")
        matrix = matrix[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)

    if cluster_labels is not None:
        sorted_labels = np.asarray(cluster_labels)[order]
        boundaries = np.flatnonzero(np.diff(sorted_labels)) + 0.5
        for boundary in boundaries:
            ax.axhline(boundary, color="white", linewidth=1.2)
            ax.axvline(boundary, color="white", linewidth=1.2)

    fig.colorbar(image, ax=ax, label=f"{distance_type} distance")
    ax.set_title(
        f"Pairwise genome distance ({distance_type})"
        + ("\nordered by cluster" if cluster_labels is not None else ""),
        fontsize=12,
    )
    ax.set_xlabel("individual")
    ax.set_ylabel("individual")

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cluster_quality(
    scores: dict[int, float],
    save_path: str | Path,
    chosen: int | None = None,
) -> Path:
    """Draw silhouette score against the number of clusters.

    Parameters
    ----------
    scores
        Silhouette score per cluster count.
    save_path
        Where to write the figure.
    chosen
        The count that was selected, marked on the plot.

    Returns
    -------
        The path the figure was written to.

    Raises
    ------
    ValueError
        If there are no scores to plot.
    """
    if not scores:
        msg = "No cluster quality scores to plot"
        raise ValueError(msg)

    counts = sorted(scores)
    values = [scores[k] for k in counts]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(counts, values, "-o", color="tab:blue", linewidth=2)

    if chosen is not None and chosen in scores:
        ax.axvline(chosen, color="tab:red", linestyle="--", alpha=0.8)
        ax.annotate(
            f"chosen: {chosen}",
            (chosen, scores[chosen]),
            textcoords="offset points",
            xytext=(8, 8),
            color="tab:red",
            fontsize=9,
        )

    ax.set_xlabel("number of clusters")
    ax.set_ylabel("silhouette score")
    ax.set_title("Cluster count selection (higher is better)", fontsize=12)
    ax.set_xticks(counts)
    ax.grid(visible=True, alpha=0.3)

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dendrogram(
    distance_matrix: FloatArray,
    save_path: str | Path,
    labels: list[str] | None = None,
    linkage_method: DendrogramLinkage = "average",
) -> Path:
    """Draw the merge tree behind hierarchical clustering.

    Shows what any particular cluster count is cutting through, which a fixed
    ``k`` hides.

    Parameters
    ----------
    distance_matrix
        Pairwise genome distances.
    save_path
        Where to write the figure.
    labels
        Optional leaf labels.
    linkage_method
        Merge criterion.

    Returns
    -------
        The path the figure was written to.
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    condensed = squareform(
        np.asarray(distance_matrix, dtype=float),
        checks=False,
    )
    tree = linkage(condensed, method=linkage_method)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    dendrogram(tree, labels=labels, ax=ax, color_threshold=None)

    ax.set_ylabel("genome distance at merge")
    ax.set_title(
        f"Hierarchical merge tree ({linkage_method} linkage)",
        fontsize=12,
    )
    ax.grid(visible=True, alpha=0.25, axis="y")

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_final_population(
    controllers: list[dict[str, Any]],
    save_path: str | Path,
    *,
    stop_reason: str = "",
) -> Path:
    """Summarise the population a run ended with.

    The per-generation figures show how the run moved; this one shows who was
    left standing. The two scatter panels are the ones worth reading: whether
    fitness tracks age tells you if selection is actually cumulative, and
    whether it tracks energy tells you whether the energy rule is selecting or
    just decorating.

    Parameters
    ----------
    controllers
        Controller records from a saved run, each with ``fitness``, ``age``
        and ``energy``.
    save_path
        Where to write the figure.
    stop_reason
        Why the run ended, shown in the summary panel.

    Returns
    -------
        The path the figure was written to.

    Raises
    ------
    ValueError
        If there are no controllers to summarise.
    """
    if not controllers:
        msg = "No controllers to summarise"
        raise ValueError(msg)

    fitness = np.array([float(c.get("fitness", 0.0)) for c in controllers])
    ages = np.array([float(c.get("age", 0)) for c in controllers])
    energy = np.array([float(c.get("energy", 0.0)) for c in controllers])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle("Final population", fontsize=15)

    ax = axes[0][0]
    ax.hist(fitness, bins=min(20, max(5, len(fitness) // 2)), color="tab:blue")
    ax.axvline(
        float(np.mean(fitness)),
        color="tab:red",
        linestyle="--",
        label=f"mean {np.mean(fitness):.3f}",
    )
    ax.set_xlabel("fitness")
    ax.set_ylabel("individuals")
    ax.set_title("Fitness distribution", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(visible=True, alpha=0.3)

    ax = axes[0][1]
    ax.scatter(
        ages,
        fitness,
        s=55,
        color="tab:purple",
        edgecolors="black",
        linewidths=0.5,
    )
    ax.set_xlabel("age (generations survived)")
    ax.set_ylabel("fitness")
    ax.set_title("Fitness against age", fontsize=11)
    ax.grid(visible=True, alpha=0.3)

    ax = axes[1][0]
    ax.scatter(
        energy,
        fitness,
        s=55,
        color="tab:orange",
        edgecolors="black",
        linewidths=0.5,
    )
    ax.axvline(0.0, color="black", linewidth=1, linestyle=":")
    ax.set_xlabel("energy")
    ax.set_ylabel("fitness")
    ax.set_title("Fitness against energy", fontsize=11)
    ax.grid(visible=True, alpha=0.3)

    ax = axes[1][1]
    ax.axis("off")
    depleted = int(np.sum(energy <= 0))
    lines = [
        f"individuals      {len(controllers)}",
        f"fitness  best    {fitness.max():.4f}",
        f"         mean    {fitness.mean():.4f}",
        f"         worst   {fitness.min():.4f}",
        f"age      oldest  {int(ages.max())}",
        f"         mean    {ages.mean():.1f}",
        f"energy   mean    {energy.mean():.1f}",
        f"         depleted {depleted}",
    ]
    if stop_reason:
        lines.extend(["", f"stopped: {stop_reason}"])
    ax.text(
        0.02,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
    )
    ax.set_title("Summary", fontsize=11)

    fig.tight_layout()

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    msg = f"Final-population summary saved to {path}"
    log.info(msg)
    return path
