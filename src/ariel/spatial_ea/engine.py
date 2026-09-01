"""The spatial evolutionary algorithm.

Each generation: check the population limits, move the mating zones, spawn the
whole population into one world, evaluate any new individuals in isolation,
record statistics, apply survivor selection, then reproduce. Reproduction is
where the spatial part lives — robots are simulated together, physically
approach one another, and only those that end up close enough actually mate.

Notes
-----
    * Position and orientation live on the individual rather than in parallel
      lists, so selection cannot desynchronise them from the population.
    * Population size is not fixed. Extinction and runaway growth are recorded
      outcomes, not errors.

"""

# Standard library
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

# Third-party libraries
import numpy as np

from ariel import log

# Local libraries
from ariel.spatial_ea.config import SpatialEAConfig
from ariel.spatial_ea.data import EvolutionDataCollector
from ariel.spatial_ea.evaluation import evaluate_population
from ariel.spatial_ea.genetics import (
    clone_individual,
    create_initial_hyperneat_genome,
    crossover_hyperneat,
    mutate_hyperneat,
)
from ariel.spatial_ea.incubation import (
    IncubationEvolution,
    seed_spatial_population_from_incubation,
)
from ariel.spatial_ea.individual import SpatialIndividual
from ariel.spatial_ea.interaction import (
    apply_movement_bias,
    apply_world_boundaries,
    calculate_offspring_positions,
    calculate_periodic_distance,
    find_pairs_with_strategy,
    generate_random_zone_centers,
    relocate_zone_centers,
)
from ariel.spatial_ea.movement import (
    MatingController,
    build_substrates,
    run_mating_movement_phase,
)
from ariel.spatial_ea.persistence import save_final_controllers
from ariel.spatial_ea.recording import GenerationRecorder
from ariel.spatial_ea.selection import select_individuals
from ariel.spatial_ea.world import (
    generate_spawn_positions,
    spawn_population_in_world,
)

# Evaluate type annotations in a deferred manner (ruff: UP037)
if TYPE_CHECKING:
    from ariel.parameters.ariel_types import FloatArray
    from ariel.spatial_ea.world import SpawnedPopulation


@dataclass
class SpatialEA:
    """A single run of the spatial evolutionary algorithm.

    Parameters
    ----------
    config
        Run configuration.
    population
        Current population.
    current_zone_centers
        Positions of the mating zones.
    assigned_zones
        Mapping from ``unique_id`` to the zone an individual is bound to.
    generation
        Current generation index.
    next_unique_id
        Identifier counter for newly created individuals.
    data_collector
        Per-generation statistics record.
    num_joints
        Number of actuated joints per robot, resolved on first spawn.
    trajectories
        Trajectories recorded during the most recent movement phase.
    paired_indices
        Indices of the individuals that reproduced in the previous
        generation, consumed by ``parents_die`` selection.
    pairs
        Index pairs that reproduced in the most recent generation, kept so
        that the movement phase can be plotted with its outcome.
    """

    config: SpatialEAConfig = field(default_factory=SpatialEAConfig)
    population: list[SpatialIndividual] = field(default_factory=list)
    current_zone_centers: list[tuple[float, float]] = field(
        default_factory=list,
    )
    assigned_zones: dict[int, int] = field(default_factory=dict)
    generation: int = 0
    next_unique_id: int = 0
    data_collector: EvolutionDataCollector = field(
        default_factory=EvolutionDataCollector,
    )
    num_joints: int = 0
    trajectories: list[list[FloatArray]] = field(
        default_factory=list,
    )
    paired_indices: set[int] = field(default_factory=set)
    pairs: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Attach the configuration to the data collector."""
        self.data_collector.config = self.config

    # -- Positions -------------------------------------------------------------
    @property
    def population_size(self) -> int:
        """Current population size.

        Returns
        -------
            Number of living individuals.
        """
        return len(self.population)

    @property
    def world_xy(self) -> tuple[float, float]:
        """Planar world dimensions.

        Returns
        -------
            The world's ``(width, height)``.
        """
        return (self.config.world_size[0], self.config.world_size[1])

    @property
    def positions(self) -> list[FloatArray]:
        """Current position of every individual.

        Returns
        -------
            One position per individual, defaulting to the world centre for
            individuals that have not been placed yet.
        """
        default = np.array([
            self.config.world_size[0] / 2.0,
            self.config.world_size[1] / 2.0,
            self.config.spawn_z,
        ])
        return [
            individual.spawn_position.copy()
            if individual.spawn_position is not None
            else default.copy()
            for individual in self.population
        ]

    # -- Mating zones ----------------------------------------------------------
    def _initialize_mating_zones(self) -> None:
        """Place the mating zones at the start of a run."""
        if self.config.num_mating_zones <= 1:
            self.current_zone_centers = [self.config.mating_zone_center]
            return

        self.current_zone_centers = generate_random_zone_centers(
            num_zones=self.config.num_mating_zones,
            world_size=self.world_xy,
            zone_radius=self.config.mating_zone_radius,
            min_zone_distance=self.config.min_zone_distance,
        )

    def _update_mating_zones(self) -> None:
        """Move the mating zones if the relocation strategy calls for it."""
        if (
            self.config.zone_relocation_strategy != "generation_interval"
            or not self.current_zone_centers
        ):
            return

        interval = max(1, self.config.zone_change_interval)
        if self.generation > 0 and self.generation % interval == 0:
            self.current_zone_centers = generate_random_zone_centers(
                num_zones=len(self.current_zone_centers),
                world_size=self.world_xy,
                zone_radius=self.config.mating_zone_radius,
                min_zone_distance=self.config.min_zone_distance,
            )
            self._assign_zones_to_population()

    def relocate_mating_zones(self, zone_indices: set[int]) -> None:
        """Move the zones in which a mating just happened.

        Parameters
        ----------
        zone_indices
            Indices of the zones to relocate.
        """
        self.current_zone_centers = relocate_zone_centers(
            self.current_zone_centers,
            zone_indices,
            world_size=self.world_xy,
            zone_radius=self.config.mating_zone_radius,
            min_zone_distance=self.config.min_zone_distance,
        )

    def _zone_index_for_position(
        self,
        position: FloatArray,
    ) -> int:
        """Find the mating zone nearest a position.

        Parameters
        ----------
        position
            Position to classify.

        Returns
        -------
            Index of the nearest zone, or ``0`` when no zones exist.
        """
        if not self.current_zone_centers:
            return 0

        nearest_zone = 0
        nearest_distance = float("inf")
        for zone_index, (cx, cy) in enumerate(self.current_zone_centers):
            zone_pos = np.array([cx, cy, position[2]], dtype=float)
            if self.config.use_periodic_boundaries:
                distance = calculate_periodic_distance(
                    position,
                    zone_pos,
                    self.world_xy,
                )
            else:
                distance = float(
                    np.linalg.norm(position[:2] - zone_pos[:2]),
                )
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_zone = zone_index

        return nearest_zone

    def _assign_zones_to_population(self) -> None:
        """Bind every individual to its nearest mating zone."""
        if not self.current_zone_centers:
            self.assigned_zones = {}
            return

        active_ids: set[int] = set()
        for individual in self.population:
            if individual.unique_id is None:
                continue
            position = (
                individual.spawn_position
                if individual.spawn_position is not None
                else np.zeros(3)
            )
            zone_index = self._zone_index_for_position(position)
            self.assigned_zones[individual.unique_id] = zone_index
            individual.assigned_zone = zone_index
            active_ids.add(individual.unique_id)

        for stale_id in set(self.assigned_zones) - active_ids:
            del self.assigned_zones[stale_id]

    # -- Population ------------------------------------------------------------
    def create_individual(self) -> SpatialIndividual:
        """Create one individual with a fresh random genome.

        Returns
        -------
            The new individual.
        """
        individual = SpatialIndividual(
            unique_id=self.next_unique_id,
            generation=self.generation,
            genotype=create_initial_hyperneat_genome(),
            energy=self.config.initial_energy,
        )
        self.next_unique_id += 1
        return individual

    def _place_population(
        self,
        individuals: list[SpatialIndividual],
    ) -> None:
        """Give each individual a non-overlapping position and heading.

        Parameters
        ----------
        individuals
            Individuals to place, updated in place.
        """
        positions = generate_spawn_positions(
            population_size=len(individuals),
            spawn_x_range=(self.config.spawn_x_min, self.config.spawn_x_max),
            spawn_y_range=(self.config.spawn_y_min, self.config.spawn_y_max),
            spawn_z=self.config.spawn_z,
            min_spawn_distance=self.config.min_spawn_distance,
        )
        for individual, position in zip(
            individuals,
            positions,
            strict=False,
        ):
            individual.spawn_position = apply_world_boundaries(
                position,
                self.world_xy,
                use_periodic_boundaries=self.config.use_periodic_boundaries,
            )
            individual.orientation = float(np.random.uniform(0, 2 * np.pi))

    def initialize_population(
        self,
        population_size: int | None = None,
    ) -> None:
        """Build the starting population and place it in the world.

        Parameters
        ----------
        population_size
            Size to build, defaulting to ``config.population_size``.
        """
        size = population_size or self.config.population_size

        if self.config.incubation_enabled:
            incubator = IncubationEvolution(
                config=self.config,
                next_unique_id=self.next_unique_id,
            )
            incubated, self.next_unique_id = incubator.run()
            self.population, self.next_unique_id = (
                seed_spatial_population_from_incubation(
                    incubated,
                    target_population_size=size,
                    generation=self.generation,
                    next_unique_id=self.next_unique_id,
                    initial_energy=self.config.initial_energy,
                )
            )
        else:
            self.population = [self.create_individual() for _ in range(size)]

        self._place_population(self.population)
        self._initialize_mating_zones()
        self._assign_zones_to_population()

    # -- Simulation ------------------------------------------------------------
    def spawn_population(self) -> SpawnedPopulation:
        """Put the whole population into one compiled world.

        Returns
        -------
            Handles into the compiled shared world.
        """
        spawned = spawn_population_in_world(
            self.population,
            self.positions,
            (
                self.config.world_size[0],
                self.config.world_size[1],
                self.config.world_z,
            ),
            orientations=[
                individual.orientation for individual in self.population
            ],
        )
        self.num_joints = spawned.num_joints
        return spawned

    def evaluate_population_fitness(self) -> list[float]:
        """Evaluate every individual that has not been evaluated yet.

        Fitness is inherited rather than recomputed, so an individual is only
        ever simulated once.

        Returns
        -------
            Fitness of every individual, in population order.
        """
        unevaluated = [
            individual
            for individual in self.population
            if not individual.evaluated
        ]
        if unevaluated:
            evaluate_population(unevaluated, self.config)

        return [individual.fitness for individual in self.population]

    def mating_movement_phase(
        self,
        spawned: SpawnedPopulation,
        duration: float | None = None,
    ) -> None:
        """Let the population move toward their mating targets.

        Positions after this phase are what pairing acts on, so whether two
        robots reproduce is decided by physics rather than by assumption.

        Parameters
        ----------
        spawned
            The compiled shared world holding the population.
        duration
            Length of the phase in simulated seconds, defaulting to
            ``config.simulation_time``.
        """
        if not self.population:
            return

        controller = MatingController(
            population=self.population,
            substrates=build_substrates(self.population, spawned.num_joints),
            num_joints=spawned.num_joints,
            control_clip_min=self.config.control_clip_min,
            control_clip_max=self.config.control_clip_max,
            movement_bias=self.config.movement_bias,
            world_size=self.world_xy,
            use_periodic_boundaries=self.config.use_periodic_boundaries,
            mating_zone_centers=list(self.current_zone_centers),
            assigned_zones=dict(self.assigned_zones),
        )

        # The recorder is a no-op unless a video or snapshot was asked for, and
        # disables itself if no GL context is available.
        with GenerationRecorder(
            model=spawned.model,
            generation=self.generation,
            world_size=self.world_xy,
            record_video=self.config.record_generation_videos,
            save_snapshot=self.config.save_generation_snapshots,
            video_folder=Path(self.config.video_folder),
            figure_folder=Path(self.config.figure_folder),
            width=self.config.video_width,
            height=self.config.video_height,
            fps=self.config.video_fps,
        ) as recorder:
            self.trajectories = run_mating_movement_phase(
                spawned,
                controller,
                duration
                if duration is not None
                else self.config.simulation_time,
                use_periodic_boundaries=self.config.use_periodic_boundaries,
                world_size=self.world_xy,
                recorder=recorder,
            )

        for individual, position in zip(
            self.population,
            spawned.core_positions(),
            strict=False,
        ):
            individual.spawn_position = apply_world_boundaries(
                position,
                self.world_xy,
                use_periodic_boundaries=self.config.use_periodic_boundaries,
            )

    def _apply_analytical_movement(self) -> None:
        """Nudge positions geometrically instead of simulating movement.

        A cheap stand-in for :meth:`mating_movement_phase`, used when
        ``config.use_physical_movement_phase`` is off. The before and after
        positions are still recorded as two-point trajectories, so the same
        plotting path works for either movement mode.
        """
        if (
            self.config.movement_bias == "none"
            or self.config.movement_step_size <= 0.0
        ):
            self.trajectories = [
                [position[:2].copy(), position[:2].copy()]
                for position in self.positions
            ]
            return

        assigned = [
            self.assigned_zones.get(individual.unique_id, 0)
            if individual.unique_id is not None
            else 0
            for individual in self.population
        ]
        before = self.positions
        moved = apply_movement_bias(
            before,
            movement_bias=self.config.movement_bias,
            movement_step_size=self.config.movement_step_size,
            world_size=self.world_xy,
            use_periodic_boundaries=self.config.use_periodic_boundaries,
            mating_zone_centers=list(self.current_zone_centers),
            assigned_zone_indices=assigned,
        )
        for individual, position in zip(self.population, moved, strict=False):
            individual.spawn_position = position

        self.trajectories = [
            [start[:2].copy(), end[:2].copy()]
            for start, end in zip(before, moved, strict=False)
        ]

    # -- Reproduction ----------------------------------------------------------
    def _apply_energy_depletion(self) -> None:
        """Charge every individual the per-generation energy cost."""
        if not self.config.enable_energy:
            return

        for individual in self.population:
            individual.energy -= self.config.energy_depletion_rate

        self.data_collector.record_energy_stats(
            self.population,
            "after_depletion",
        )

    def _apply_mating_energy_effect(
        self,
        parent1: SpatialIndividual,
        parent2: SpatialIndividual,
    ) -> None:
        """Apply the energy consequence of a mating to both parents.

        Parameters
        ----------
        parent1
            First parent.
        parent2
            Second parent.
        """
        if not self.config.enable_energy:
            return

        if self.config.mating_energy_effect == "restore":
            parent1.energy = self.config.initial_energy
            parent2.energy = self.config.initial_energy
        elif self.config.mating_energy_effect == "cost":
            parent1.energy -= self.config.mating_energy_amount
            parent2.energy -= self.config.mating_energy_amount

    def _breed(
        self,
        parent1: SpatialIndividual,
        parent2: SpatialIndividual,
    ) -> tuple[SpatialIndividual, SpatialIndividual]:
        """Produce two mutated offspring from a mated pair.

        Parameters
        ----------
        parent1
            First parent.
        parent2
            Second parent.

        Returns
        -------
            The two offspring.
        """
        if np.random.random() < self.config.crossover_rate:
            child1, child2, self.next_unique_id = crossover_hyperneat(
                parent1,
                parent2,
                self.next_unique_id,
                self.generation + 1,
                initial_energy=self.config.initial_energy,
            )
        else:
            child1, self.next_unique_id = clone_individual(
                parent1,
                self.next_unique_id,
                self.generation + 1,
                initial_energy=self.config.initial_energy,
            )
            child2, self.next_unique_id = clone_individual(
                parent2,
                self.next_unique_id,
                self.generation + 1,
                initial_energy=self.config.initial_energy,
            )

        mutated: list[SpatialIndividual] = []
        for child in (child1, child2):
            mutant, self.next_unique_id = mutate_hyperneat(
                child,
                next_unique_id=self.next_unique_id,
                weight_mutation_rate=self.config.mutation_rate,
                weight_mutation_power=self.config.mutation_strength,
                add_connection_rate=self.config.add_connection_rate,
                add_node_rate=self.config.add_node_rate,
                initial_energy=self.config.initial_energy,
            )
            mutant.generation = self.generation + 1
            mutant.parent_ids = list(child.parent_ids)
            mutated.append(mutant)

        return mutated[0], mutated[1]

    def create_next_generation(self, spawned: SpawnedPopulation) -> None:
        """Move, pair, and breed the population.

        Parameters
        ----------
        spawned
            The compiled shared world holding the population.
        """
        if not self.population:
            log.warning("Population is extinct; skipping reproduction")
            return

        self._apply_energy_depletion()

        if self.config.use_physical_movement_phase:
            self.mating_movement_phase(spawned)
        else:
            self._apply_analytical_movement()

        population_before = len(self.population)
        zone_centers = self.current_zone_centers or [
            self.config.mating_zone_center,
        ]

        pairs, paired_indices, zones_with_matings = find_pairs_with_strategy(
            self.population,
            self.positions,
            pairing_radius=self.config.pairing_radius,
            world_size=self.world_xy,
            use_periodic_boundaries=self.config.use_periodic_boundaries,
            method=self.config.pairing_method,
            mating_zone_centers=zone_centers,
            mating_zone_radius=self.config.mating_zone_radius,
        )
        self.paired_indices = paired_indices
        self.pairs = pairs

        # Plot before any zone relocation, so the figure shows the zones the
        # robots were actually navigating toward.
        if self.config.save_generation_plots:
            self.save_generation_plot()

        if (
            self.config.zone_relocation_strategy == "event_driven"
            and zones_with_matings
        ):
            self.relocate_mating_zones(zones_with_matings)

        self.data_collector.record_mating_stats(
            num_pairs=len(pairs),
            num_unpaired=population_before - len(paired_indices),
            population_size=population_before,
        )

        offspring_positions = calculate_offspring_positions(
            pairs,
            self.positions,
            offspring_radius=self.config.offspring_radius,
            world_size=self.world_xy,
            use_periodic_boundaries=self.config.use_periodic_boundaries,
        )

        offspring: list[SpatialIndividual] = []
        for pair_index, (parent1_idx, parent2_idx) in enumerate(pairs):
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]

            self._apply_mating_energy_effect(parent1, parent2)
            child1, child2 = self._breed(parent1, parent2)

            for child, position in zip(
                (child1, child2),
                offspring_positions[pair_index],
                strict=False,
            ):
                child.spawn_position = position
                child.orientation = float(np.random.uniform(0, 2 * np.pi))
                offspring.append(child)

        if self.config.movement_bias == "assigned_zone" and zone_centers:
            for child in offspring:
                if child.unique_id is not None:
                    zone_idx = int(np.random.randint(0, len(zone_centers)))
                    self.assigned_zones[child.unique_id] = zone_idx
                    child.assigned_zone = zone_idx

        self.population.extend(offspring)

        self.data_collector.record_reproduction(
            num_offspring=len(offspring),
            population_before=population_before,
        )

        if self.config.enable_energy:
            self.data_collector.record_energy_stats(
                self.population,
                "after_mating",
            )

        if self.config.print_generation_stats:
            msg = (
                f"Generation {self.generation}: {len(pairs)} pairs, "
                f"{len(offspring)} offspring, population "
                f"{population_before} -> {len(self.population)}"
            )
            log.info(msg)

    # -- Selection -------------------------------------------------------------
    def apply_selection(self) -> None:
        """Thin the population down to the survivors of this generation."""
        population_before = len(self.population)

        zone_indices = [
            self._zone_index_for_position(position)
            for position in self.positions
        ]

        self.population = select_individuals(
            self.population,
            self.config.effective_target_population_size,
            method=self.config.selection_method,
            current_generation=self.generation,
            paired_indices=self.paired_indices,
            max_age=self.config.max_age,
            zone_indices=zone_indices,
            num_zones=max(1, len(self.current_zone_centers)),
            zone_capacity_softness=self.config.zone_capacity_softness,
            positions=self.positions,
            world_size=self.world_xy,
            use_periodic_boundaries=self.config.use_periodic_boundaries,
            density_locality_radius=self.config.density_locality_radius,
            density_critical_density=self.config.density_critical_density,
            density_base_death_prob=self.config.density_base_death_prob,
            density_max_death_prob=self.config.density_max_death_prob,
            density_fitness_protection=self.config.density_fitness_protection,
        )
        self.paired_indices = set()

        self.data_collector.record_selection(
            population_before=population_before,
            population_after=len(self.population),
        )
        self._assign_zones_to_population()

    # -- Run -------------------------------------------------------------------
    def _check_population_limits(self) -> str | None:
        """Test whether the run should stop on a population limit.

        Returns
        -------
            A description of the limit that was hit, or ``None`` to continue.
        """
        if not self.config.stop_on_limits:
            return None

        if self.population_size >= self.config.max_population_limit:
            return (
                f"Population reached maximum limit "
                f"({self.config.max_population_limit})"
            )
        if self.population_size < self.config.min_population_limit:
            return (
                f"Population extinction (below "
                f"{self.config.min_population_limit})"
            )
        return None

    def run(self, generations: int | None = None) -> SpatialIndividual | None:
        """Run the algorithm to completion.

        Parameters
        ----------
        generations
            Number of generations to run, defaulting to
            ``config.num_generations``.

        Returns
        -------
            The fittest surviving individual, or ``None`` if none survived.
        """
        num_generations = (
            generations
            if generations is not None
            else self.config.num_generations
        )

        if not self.population:
            self.initialize_population()

        for generation in range(num_generations):
            self.generation = generation

            stop_reason = self._check_population_limits()
            if stop_reason is not None:
                if self.population_size < self.config.min_population_limit:
                    self.data_collector.record_extinct_generation(generation)
                else:
                    self.data_collector.record_generation_start(
                        generation,
                        self.population_size,
                    )
                self.data_collector.record_early_stop(
                    generation,
                    stop_reason,
                    num_generations,
                )
                log.warning(stop_reason)
                break

            self.data_collector.record_generation_start(
                generation,
                self.population_size,
            )
            self._update_mating_zones()

            spawned = self.spawn_population()
            self.evaluate_population_fitness()

            self.data_collector.record_fitness_stats(
                self.population,
                generation,
            )
            self.data_collector.record_age_stats(self.population, generation)
            self.data_collector.record_genotype_diversity(self.population)

            if generation > 0:
                self.apply_selection()

            self.create_next_generation(spawned)

            del spawned
            gc.collect()

        # Data and figures are separate concerns: either can be wanted alone.
        if self.config.save_results:
            self.save_results()
        if self.config.save_plots:
            self.save_statistics_plot()

        return self.get_best_individual()

    # Kept for symmetry with the research prototype's entry point.
    run_evolution = run

    def get_best_individual(self) -> SpatialIndividual | None:
        """Return the fittest individual currently alive.

        Returns
        -------
            The fittest individual, or ``None`` when the population is empty.
        """
        if not self.population:
            return None
        return max(
            self.population,
            key=lambda individual: individual.fitness,
        )

    def save_generation_plot(
        self,
        save_path: Path | None = None,
    ) -> Path | None:
        """Plot the movement phase that has just finished.

        Parameters
        ----------
        save_path
            Where to write the figure. Defaults to a per-generation file under
            ``config.figure_folder``.

        Returns
        -------
            The path written, or ``None`` if there was nothing to plot.

        Notes
        -----
            The plotting module is imported here rather than at module scope so
            that runs which never plot do not pay for ``matplotlib.pyplot``.
        """
        if not self.trajectories:
            return None

        from ariel.spatial_ea.visualization import plot_mating_trajectories

        path = save_path or (
            Path(self.config.figure_folder)
            / f"mating_generation_{self.generation:03d}.png"
        )
        return plot_mating_trajectories(
            self.trajectories,
            self.population,
            self.generation,
            path,
            world_size=self.world_xy,
            robot_size=self.config.robot_size,
            simulation_time=self.config.simulation_time,
            use_periodic_boundaries=self.config.use_periodic_boundaries,
            mating_zone_centers=list(self.current_zone_centers),
            mating_zone_radius=self.config.mating_zone_radius,
            pairs=self.pairs,
            pairing_method=self.config.pairing_method,
        )

    def save_statistics_plot(self, save_path: Path | None = None) -> Path:
        """Plot the run's per-generation statistics.

        Parameters
        ----------
        save_path
            Where to write the figure. Defaults to
            ``config.figure_folder / "evolution_statistics.png"``.

        Returns
        -------
            The path the figure was written to.
        """
        from ariel.spatial_ea.visualization import plot_evolution_statistics

        path = save_path or (
            Path(self.config.figure_folder) / "evolution_statistics.png"
        )
        return plot_evolution_statistics(self.data_collector, path)

    def save_results(self) -> None:
        """Write the run's data — statistics and final controllers — to disk.

        Every file of a run shares one timestamp. Analysis tooling pairs the
        CSV, the NPZ and the controller export by that exact string, so
        letting each call stamp itself would silently orphan them whenever a
        run happened to cross a second boundary mid-save.

        Figures are written separately by :meth:`save_statistics_plot` and
        :meth:`save_generation_plot`.
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        self.data_collector.save_to_csv(self.config.result_folder, timestamp)
        self.data_collector.save_to_npz(self.config.result_folder, timestamp)
        save_final_controllers(
            self.population,
            self.config,
            self.generation,
            self.num_joints,
            timestamp=timestamp,
        )
