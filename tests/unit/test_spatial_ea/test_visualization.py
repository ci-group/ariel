"""Test: figures produced from a spatial EA run."""

# Standard library
from pathlib import Path

# Third-party libraries
import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

# Local libraries
from ariel.spatial_ea.config import SpatialEAConfig
from ariel.spatial_ea.data import EvolutionDataCollector
from ariel.spatial_ea.engine import SpatialEA
from ariel.spatial_ea.individual import SpatialIndividual
from ariel.spatial_ea.visualization import (
    plot_cluster_embedding,
    plot_cluster_quality,
    plot_dendrogram,
    plot_distance_heatmap,
    plot_evolution_statistics,
    plot_final_population,
    plot_mating_trajectories,
    plot_mating_zones,
    plot_spatial_clusters,
)

WORLD = (10.0, 10.0)


def _population(count: int) -> list[SpatialIndividual]:
    """Build individuals with distinct ids and fitness."""
    return [
        SpatialIndividual(unique_id=i, fitness=float(i) / 10.0)
        for i in range(count)
    ]


def _trajectories(count: int, steps: int = 5) -> list[list[np.ndarray]]:
    """Build simple diagonal trajectories."""
    return [
        [
            np.array([1.0 + i + step * 0.1, 1.0 + i + step * 0.1])
            for step in range(steps)
        ]
        for i in range(count)
    ]


def _collector(
    generations: int = 4,
    *,
    with_energy: bool = True,
) -> EvolutionDataCollector:
    """Build a collector holding a short synthetic run."""
    collector = EvolutionDataCollector()
    population = _population(5)
    for generation in range(generations):
        collector.record_generation_start(generation, 10 + generation)
        collector.record_fitness_stats(population, generation)
        collector.record_age_stats(population, generation)
        collector.record_genotype_diversity([])
        collector.record_mating_stats(2, 1, 10)
        if generation > 0:
            collector.record_reproduction(4, 10)
            collector.record_selection(14, 10)
        if with_energy:
            collector.record_energy_stats(population, "after_depletion")
            collector.record_energy_stats(population, "after_mating")
    return collector


def test_trajectory_plot_is_written(tmp_path: Path) -> None:
    """The trajectory plot should land on disk."""
    path = plot_mating_trajectories(
        _trajectories(4),
        _population(4),
        2,
        tmp_path / "traj.png",
        world_size=WORLD,
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_trajectory_plot_creates_missing_directories(tmp_path: Path) -> None:
    """Writing into a directory that does not exist yet should work."""
    path = plot_mating_trajectories(
        _trajectories(2),
        _population(2),
        0,
        tmp_path / "a" / "b" / "traj.png",
        world_size=WORLD,
    )

    assert path.exists()


def test_trajectory_plot_accepts_every_option(tmp_path: Path) -> None:
    """Zones, pairs and periodic boundaries should all render."""
    path = plot_mating_trajectories(
        _trajectories(4),
        _population(4),
        3,
        tmp_path / "full.png",
        world_size=WORLD,
        robot_size=0.3,
        simulation_time=5.0,
        use_periodic_boundaries=True,
        mating_zone_centers=[(2.0, 2.0), (8.0, 8.0)],
        mating_zone_radius=1.5,
        pairs=[(0, 1), (2, 3)],
        pairing_method="mating_zone",
    )

    assert path.exists()


def test_trajectory_plot_tolerates_ragged_input(tmp_path: Path) -> None:
    """Empty and single-point trajectories must not raise."""
    trajectories = [
        [],
        [np.array([1.0, 1.0])],
        [np.array([2.0, 2.0]), np.array([2.5, 2.5])],
    ]

    path = plot_mating_trajectories(
        trajectories,
        _population(3),
        0,
        tmp_path / "ragged.png",
        world_size=WORLD,
    )

    assert path.exists()


def test_trajectory_plot_ignores_out_of_range_pairs(tmp_path: Path) -> None:
    """A pair index past the end of the trajectories is skipped, not fatal."""
    path = plot_mating_trajectories(
        _trajectories(2),
        _population(2),
        0,
        tmp_path / "pairs.png",
        world_size=WORLD,
        pairs=[(0, 1), (0, 99)],
    )

    assert path.exists()


def test_statistics_plot_with_and_without_energy(tmp_path: Path) -> None:
    """The energy panel appears only when energy was recorded."""
    with_energy = plot_evolution_statistics(
        _collector(with_energy=True),
        tmp_path / "with.png",
    )
    without_energy = plot_evolution_statistics(
        _collector(with_energy=False),
        tmp_path / "without.png",
    )

    assert with_energy.exists()
    assert without_energy.exists()


def test_statistics_plot_of_a_single_generation(tmp_path: Path) -> None:
    """A one-generation run should not break the axis scaling."""
    path = plot_evolution_statistics(
        _collector(generations=1),
        tmp_path / "one.png",
    )

    assert path.exists()


def test_statistics_plot_notes_an_early_stop(tmp_path: Path) -> None:
    """A run that ended early still renders."""
    collector = _collector()
    collector.record_early_stop(3, "Population extinction", 50)

    path = plot_evolution_statistics(collector, tmp_path / "stopped.png")

    assert path.exists()


def test_zone_plot(tmp_path: Path) -> None:
    """The zone layout plot should render, with and without a population."""
    bare = plot_mating_zones(
        WORLD,
        [(2.0, 2.0), (8.0, 8.0)],
        1.5,
        tmp_path / "zones.png",
    )
    populated = plot_mating_zones(
        WORLD,
        [(2.0, 2.0)],
        1.5,
        tmp_path / "zones_pop.png",
        [np.array([2.0, 2.0, 0.1]), np.array([9.0, 9.0, 0.1])],
        use_periodic_boundaries=True,
        title_suffix="starting population",
    )

    assert bare.exists()
    assert populated.exists()


def test_engine_writes_a_plot_per_generation(tmp_path: Path) -> None:
    """Turning on generation plots should leave one figure per generation."""
    config = SpatialEAConfig(
        population_size=4,
        num_generations=2,
        simulation_time=0.15,
        world_size=(4.0, 4.0),
        spawn_x_min=0.5,
        spawn_x_max=3.5,
        spawn_y_min=0.5,
        spawn_y_max=3.5,
        min_spawn_distance=0.5,
        pairing_radius=100.0,
        figure_folder=tmp_path,
        save_results=False,
        save_generation_plots=True,
        print_generation_stats=False,
    )
    engine = SpatialEA(config=config)
    engine.run(generations=2)

    figures = sorted(tmp_path.glob("mating_generation_*.png"))
    assert [path.name for path in figures] == [
        "mating_generation_000.png",
        "mating_generation_001.png",
    ]


def test_engine_plot_helpers_are_opt_in(tmp_path: Path) -> None:
    """With plotting off, a run should write no figures at all."""
    config = SpatialEAConfig(
        population_size=3,
        num_generations=1,
        simulation_time=0.15,
        world_size=(4.0, 4.0),
        figure_folder=tmp_path,
        save_results=False,
        save_plots=False,
        save_generation_plots=False,
        print_generation_stats=False,
    )
    SpatialEA(config=config).run(generations=1)

    assert list(tmp_path.glob("*.png")) == []


def test_figures_are_independent_of_saving_data(tmp_path: Path) -> None:
    """Plots and data files are separate switches, not nested."""
    config = SpatialEAConfig(
        population_size=3,
        num_generations=1,
        simulation_time=0.15,
        world_size=(4.0, 4.0),
        figure_folder=tmp_path / "figures",
        result_folder=tmp_path / "results",
        save_results=False,
        save_plots=True,
        print_generation_stats=False,
    )
    SpatialEA(config=config).run(generations=1)

    assert (tmp_path / "figures" / "evolution_statistics.png").exists()
    assert not (tmp_path / "results").exists()


def test_statistics_plot_survives_an_immediate_stop(tmp_path: Path) -> None:
    """A run that stops on generation 0 still plots.

    The collector records the generation before it records any fitness, so the
    series are of unequal length at that point.
    """
    config = SpatialEAConfig(
        population_size=4,
        num_generations=3,
        simulation_time=0.15,
        world_size=(4.0, 4.0),
        max_population_limit=3,
        figure_folder=tmp_path,
        save_results=False,
        save_plots=True,
        print_generation_stats=False,
    )
    engine = SpatialEA(config=config)
    engine.run(generations=3)

    assert engine.data_collector.stopped_early is True
    assert engine.data_collector.fitness_best == []
    assert (tmp_path / "evolution_statistics.png").exists()


def test_engine_statistics_plot(tmp_path: Path) -> None:
    """The engine should be able to plot its own collected statistics."""
    config = SpatialEAConfig(
        population_size=3,
        num_generations=2,
        simulation_time=0.15,
        world_size=(4.0, 4.0),
        figure_folder=tmp_path,
        save_results=False,
        print_generation_stats=False,
    )
    engine = SpatialEA(config=config)
    engine.run(generations=2)

    path = engine.save_statistics_plot()

    assert path.exists()
    assert path.name == "evolution_statistics.png"


def test_generation_plot_without_trajectories_is_a_no_op(
    tmp_path: Path,
) -> None:
    """Nothing to plot means nothing written, and no exception."""
    config = SpatialEAConfig(figure_folder=tmp_path, save_results=False)
    engine = SpatialEA(config=config)

    assert engine.save_generation_plot() is None


def test_analytical_movement_still_records_trajectories() -> None:
    """The cheap movement mode must stay plottable.

    It records two-point trajectories so the same plotting path serves both
    movement modes.
    """
    config = SpatialEAConfig(
        population_size=4,
        simulation_time=0.15,
        world_size=(4.0, 4.0),
        spawn_x_min=0.5,
        spawn_x_max=3.5,
        spawn_y_min=0.5,
        spawn_y_max=3.5,
        min_spawn_distance=0.5,
        use_physical_movement_phase=False,
        movement_bias="nearest_zone",
        movement_step_size=0.5,
        num_mating_zones=1,
        mating_zone_center=(2.0, 2.0),
        pairing_radius=1e-6,
        save_results=False,
        print_generation_stats=False,
    )
    engine = SpatialEA(config=config)
    engine.initialize_population()
    spawned = engine.spawn_population()
    engine.evaluate_population_fitness()
    engine.create_next_generation(spawned)

    assert len(engine.trajectories) == 4
    assert all(len(path) == 2 for path in engine.trajectories)
    # The nudge actually moved somebody.
    moved = [
        float(np.linalg.norm(np.asarray(path[1]) - np.asarray(path[0])))
        for path in engine.trajectories
    ]
    assert max(moved) == pytest.approx(0.5, abs=1e-6)


# -- Clustering figures --------------------------------------------------------
def _clustered() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build positions, labels and a distance matrix for two clusters."""
    positions = np.array([
        [1.0, 1.0, 0.1],
        [1.4, 1.2, 0.1],
        [8.0, 8.0, 0.1],
        [8.3, 7.8, 0.1],
    ])
    labels = np.array([0, 0, 1, 1])
    distances = np.array([
        [0.0, 0.1, 0.8, 0.9],
        [0.1, 0.0, 0.9, 0.8],
        [0.8, 0.9, 0.0, 0.1],
        [0.9, 0.8, 0.1, 0.0],
    ])
    return positions, labels, distances


def test_spatial_cluster_plot(tmp_path: Path) -> None:
    """The world map coloured by genotype cluster renders."""
    positions, labels, _ = _clustered()

    path = plot_spatial_clusters(
        positions,
        labels,
        tmp_path / "spatial.png",
        world_size=(10.0, 10.0),
        spatial_silhouette=0.81,
        cluster_centroids={0: np.array([1.2, 1.1]), 1: np.array([8.15, 7.9])},
        title_suffix="combined distance",
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_spatial_cluster_plot_marks_noise(tmp_path: Path) -> None:
    """Unassigned individuals render without breaking the legend."""
    positions = np.array([
        [1.0, 1.0, 0.1],
        [8.0, 8.0, 0.1],
        [5.0, 5.0, 0.1],
    ])

    path = plot_spatial_clusters(
        positions,
        np.array([0, 1, -1]),
        tmp_path / "noise.png",
        world_size=(10.0, 10.0),
    )

    assert path.exists()


def test_cluster_embedding_plot(tmp_path: Path) -> None:
    """The embedding scatter renders, with identifiers."""
    _, labels, _ = _clustered()
    coordinates = np.array([[0.0, 0.0], [0.2, 0.1], [3.0, 3.0], [3.2, 2.9]])

    path = plot_cluster_embedding(
        coordinates,
        labels,
        tmp_path / "embedding.png",
        method="PCA",
        individual_ids=[10, 11, 12, 13],
    )

    assert path.exists()


def test_distance_heatmap(tmp_path: Path) -> None:
    """The heatmap renders, ordered and unordered."""
    _, labels, distances = _clustered()

    plain = plot_distance_heatmap(distances, tmp_path / "plain.png")
    ordered = plot_distance_heatmap(
        distances,
        tmp_path / "ordered.png",
        labels,
        distance_type="structural",
    )

    assert plain.exists()
    assert ordered.exists()


def test_cluster_quality_plot(tmp_path: Path) -> None:
    """Silhouette against cluster count renders, marking the choice."""
    path = plot_cluster_quality(
        {2: 0.8, 3: 0.6, 4: 0.4},
        tmp_path / "quality.png",
        chosen=2,
    )

    assert path.exists()


def test_cluster_quality_plot_needs_scores(tmp_path: Path) -> None:
    """There is nothing to draw without scores."""
    with pytest.raises(ValueError, match="No cluster quality scores"):
        plot_cluster_quality({}, tmp_path / "empty.png")


def test_dendrogram(tmp_path: Path) -> None:
    """The merge tree renders from a distance matrix."""
    _, _, distances = _clustered()

    path = plot_dendrogram(
        distances,
        tmp_path / "dendrogram.png",
        labels=["a", "b", "c", "d"],
    )

    assert path.exists()


def _controllers(count: int = 6) -> list[dict]:
    """Build saved-controller records."""
    return [
        {
            "unique_id": i,
            "fitness": 0.05 * i,
            "age": i % 3,
            "energy": 100.0 - 15.0 * i,
        }
        for i in range(count)
    ]


def test_final_population_figure(tmp_path: Path) -> None:
    """The end-of-run summary renders."""
    path = plot_final_population(
        _controllers(),
        tmp_path / "final.png",
        stop_reason="Completed all generations",
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_final_population_needs_a_population(tmp_path: Path) -> None:
    """There is nothing to summarise for an extinct run."""
    with pytest.raises(ValueError, match="No controllers"):
        plot_final_population([], tmp_path / "empty.png")


def test_final_population_of_a_single_individual(tmp_path: Path) -> None:
    """A lone survivor must not break the histogram binning."""
    path = plot_final_population(_controllers(1), tmp_path / "one.png")

    assert path.exists()


def test_statistics_plot_gains_a_diversity_panel(tmp_path: Path) -> None:
    """Recorded genome diversity earns its own panel."""
    collector = _collector()
    collector.genotype_diversity = [1.0, 1.2, 0.9, 1.1]

    path = plot_evolution_statistics(collector, tmp_path / "diverse.png")

    assert path.exists()
