"""Configuration for the spatial evolutionary algorithm.

The settings are flat, so every knob can be overridden from the command line,
from the environment, or with ``model_copy(update=...)``. The research
prototype's nested YAML schema is still readable through
:meth:`SpatialEAConfig.from_legacy_dict`.
"""

# Standard library
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

# Third-party libraries
import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings

# Local libraries
from ariel import log

# Type Aliases
type PairingMethod = Literal["proximity_pairing", "random", "mating_zone"]
type SelectionMethod = Literal[
    "parents_die",
    "fitness_based",
    "age_based",
    "probabilistic_age",
    "energy_based",
    "density_based",
    "zone_capacity",
]
type MovementBias = Literal[
    "nearest_neighbor",
    "nearest_zone",
    "assigned_zone",
    "none",
]
type ZoneRelocationStrategy = Literal[
    "static",
    "generation_interval",
    "event_driven",
]
type MatingEnergyEffect = Literal["restore", "cost", "none"]

# Global constants
SPAWN_MARGIN = 0.1
LEGACY_SECTION_KEYS = (
    "population",
    "simulation",
    "selection",
    "multi_robot",
    "incubation",
    "crossover",
    "mutation",
    "video",
)


class SpatialEAConfig(BaseSettings):
    """Settings for one spatial evolutionary algorithm run."""

    # -- Incubation ------------------------------------------------------------
    incubation_enabled: bool = False
    incubation_population_size: int = 30
    incubation_num_generations: int = 50
    incubation_mutation_probability: float = 0.8
    incubation_mutation_strength: float = 0.2
    incubation_add_connection_rate: float = 0.05
    incubation_add_node_rate: float = 0.03
    incubation_crossover_rate: float = 0.9
    incubation_tournament_size: int = 3
    incubation_elitism_count: int = 2

    # -- Population ------------------------------------------------------------
    population_size: int = 20
    num_generations: int = 50
    target_population_size: int | None = None
    max_population_limit: int = 100
    min_population_limit: int = 1
    stop_on_limits: bool = True

    # -- Variation -------------------------------------------------------------
    crossover_rate: float = 0.9
    mutation_rate: float = 0.8
    mutation_strength: float = 0.5
    add_connection_rate: float = 0.05
    add_node_rate: float = 0.03

    # -- Simulation ------------------------------------------------------------
    simulation_time: float = 1.0
    control_clip_min: float = -1.5708
    control_clip_max: float = 1.5708
    use_periodic_boundaries: bool = False
    use_directional_fitness: bool = False
    target_distance_min: float = 5.0
    target_distance_max: float = 10.0
    progress_weight: float = 0.5

    # -- World and spawning ----------------------------------------------------
    world_size: tuple[float, float] = (10.0, 10.0)
    world_z: float = 0.1
    robot_size: float = 0.4
    spawn_z: float = 0.1
    spawn_x_min: float = 0.1
    spawn_x_max: float = 9.9
    spawn_y_min: float = 0.1
    spawn_y_max: float = 9.9
    min_spawn_distance: float = 1.0

    # -- Pairing and movement --------------------------------------------------
    pairing_radius: float = 0.5
    offspring_radius: float = 0.3
    pairing_method: PairingMethod = "proximity_pairing"
    movement_bias: MovementBias = "none"
    movement_step_size: float = 0.0
    use_physical_movement_phase: bool = True

    # -- Mating zones ----------------------------------------------------------
    num_mating_zones: int = 1
    mating_zone_center: tuple[float, float] = (5.0, 5.0)
    mating_zone_radius: float = 3.0
    min_zone_distance: float = 2.0
    zone_relocation_strategy: ZoneRelocationStrategy = "static"
    zone_change_interval: int = 5
    zone_capacity_softness: int = 1

    # -- Selection -------------------------------------------------------------
    selection_method: SelectionMethod = "parents_die"
    max_age: int = 10

    # -- Energy ----------------------------------------------------------------
    enable_energy: bool = True
    initial_energy: float = 100.0
    energy_depletion_rate: float = 10.0
    mating_energy_effect: MatingEnergyEffect = "cost"
    mating_energy_amount: float = 35.0

    # -- Density-based selection -----------------------------------------------
    density_locality_radius: float = 3.0
    density_critical_density: float = 5.0
    density_base_death_prob: float = 0.05
    density_max_death_prob: float = 0.8
    density_fitness_protection: float = 0.0

    # -- Output ----------------------------------------------------------------
    result_folder: Path = Path.cwd() / "__results__"
    figure_folder: Path = Path.cwd() / "__figures__"
    video_folder: Path = Path.cwd() / "__videos__"
    print_generation_stats: bool = True
    save_results: bool = True
    save_plots: bool = True
    save_generation_plots: bool = False

    # -- Recording -------------------------------------------------------------
    record_generation_videos: bool = False
    save_generation_snapshots: bool = False
    video_width: int = 640
    video_height: int = 480
    video_fps: int = 30

    @model_validator(mode="after")
    def _clamp_spawn_area_to_world(self) -> SpatialEAConfig:
        """Keep the spawn area inside the world.

        The spawn bounds and ``world_size`` are independent fields, so changing
        one alone — ``--world-size 4 4`` while the bounds still default to a
        ten-metre world — used to leave the whole population spawning outside
        the world and then being clipped onto its edge by
        ``apply_world_boundaries``. That is silent and it ruins a run, so the
        bounds are clamped here and the correction is logged.

        Returns
        -------
            This configuration, with the spawn area inside the world.
        """
        width, height = self.world_size
        margin = min(SPAWN_MARGIN, width / 4.0, height / 4.0)

        clamped = {
            "spawn_x_min": min(max(self.spawn_x_min, 0.0), width - margin),
            "spawn_x_max": min(max(self.spawn_x_max, margin), width - margin),
            "spawn_y_min": min(max(self.spawn_y_min, 0.0), height - margin),
            "spawn_y_max": min(max(self.spawn_y_max, margin), height - margin),
        }

        changed = {
            name: value
            for name, value in clamped.items()
            if abs(getattr(self, name) - value) > 1e-9
        }
        if changed:
            msg = (
                f"Spawn area fell outside the {width:g}x{height:g} world; "
                f"clamped {changed}"
            )
            log.warning(msg)
            for name, value in changed.items():
                object.__setattr__(self, name, value)

        return self

    @property
    def effective_target_population_size(self) -> int:
        """Population size that survivor selection aims for.

        Returns
        -------
            ``target_population_size`` when set, otherwise
            ``population_size``.
        """
        if self.target_population_size is not None:
            return self.target_population_size
        return self.population_size

    @classmethod
    def from_legacy_dict(cls, raw: dict[str, Any]) -> SpatialEAConfig:
        """Build a configuration from the prototype's nested YAML schema.

        Parameters
        ----------
        raw
            The parsed nested configuration.

        Returns
        -------
            The equivalent flat configuration.
        """

        def pick(path: tuple[str, ...], default: Any = None) -> Any:
            """Read a nested key, falling back to a default.

            Parameters
            ----------
            path
                Successive dictionary keys.
            default
                Value returned when the path is absent.

            Returns
            -------
                The value at ``path``, or ``default``.
            """
            value: Any = raw
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    return default
                value = value[key]
            return value

        world_size = pick(("multi_robot", "world_size"), [10.0, 10.0, 0.1])
        if isinstance(world_size, (list, tuple)) and len(world_size) >= 2:
            world_xy = (float(world_size[0]), float(world_size[1]))
            world_z = float(world_size[2]) if len(world_size) > 2 else 0.1
        else:
            world_xy, world_z = (10.0, 10.0), 0.1

        zone_center_raw = pick(("selection", "mating_zone_center"), None)
        if (
            isinstance(zone_center_raw, (list, tuple))
            and len(zone_center_raw) >= 2
        ):
            zone_center = (
                float(zone_center_raw[0]),
                float(zone_center_raw[1]),
            )
        else:
            zone_center = (world_xy[0] / 2.0, world_xy[1] / 2.0)

        # The deprecated boolean spelling of the relocation strategy.
        relocation = pick(("selection", "zone_relocation_strategy"), None)
        if relocation is None:
            dynamic_zones = pick(("selection", "dynamic_mating_zones"), False)
            relocation = (
                "generation_interval" if bool(dynamic_zones) else "static"
            )

        movement_bias = cast(
            "MovementBias",
            str(pick(("selection", "movement_bias"), "none")),
        )
        # The prototype fed the heading to the controller as a neural input and
        # had no analytical step size. Give the fallback nudge a usable
        # default whenever a bias is requested, so that loading a legacy file
        # does not silently disable movement.
        movement_step_size = float(
            pick(
                ("selection", "movement_step_size"),
                0.0 if movement_bias == "none" else 0.5,
            ),
        )

        return cls(
            incubation_enabled=bool(pick(("incubation", "enabled"), False)),
            incubation_population_size=int(
                pick(("incubation", "population_size"), 30),
            ),
            incubation_num_generations=int(
                pick(("incubation", "num_generations"), 50),
            ),
            incubation_mutation_probability=float(
                pick(("incubation", "mutation_rate"), 0.8),
            ),
            incubation_mutation_strength=float(
                pick(("incubation", "mutation_power"), 0.2),
            ),
            incubation_add_connection_rate=float(
                pick(("incubation", "add_connection_rate"), 0.05),
            ),
            incubation_add_node_rate=float(
                pick(("incubation", "add_node_rate"), 0.03),
            ),
            incubation_crossover_rate=float(
                pick(("incubation", "crossover_rate"), 0.9),
            ),
            incubation_tournament_size=int(
                pick(("incubation", "tournament_size"), 3),
            ),
            incubation_elitism_count=int(
                pick(("incubation", "elitism_count"), 2),
            ),
            population_size=int(pick(("population", "size"), 20)),
            num_generations=int(pick(("population", "num_generations"), 50)),
            target_population_size=pick(
                ("selection", "target_population_size"),
                None,
            ),
            max_population_limit=int(
                pick(("population", "max_population_limit"), 100),
            ),
            min_population_limit=int(
                pick(("population", "min_population_limit"), 1),
            ),
            stop_on_limits=bool(pick(("population", "stop_on_limits"), True)),
            crossover_rate=float(pick(("crossover", "rate"), 0.9)),
            mutation_rate=float(pick(("mutation", "rate"), 0.8)),
            mutation_strength=float(pick(("mutation", "strength"), 0.5)),
            add_connection_rate=float(
                pick(("mutation", "add_connection_rate"), 0.05),
            ),
            add_node_rate=float(pick(("mutation", "add_node_rate"), 0.03)),
            simulation_time=float(pick(("simulation", "time"), 1.0)),
            control_clip_min=float(
                pick(("simulation", "control_clip_min"), -1.5708),
            ),
            control_clip_max=float(
                pick(("simulation", "control_clip_max"), 1.5708),
            ),
            use_periodic_boundaries=bool(
                pick(("simulation", "use_periodic_boundaries"), False),
            ),
            use_directional_fitness=bool(
                pick(("incubation", "use_directional_fitness"), False),
            ),
            target_distance_min=float(
                pick(("incubation", "target_distance_min"), 5.0),
            ),
            target_distance_max=float(
                pick(("incubation", "target_distance_max"), 10.0),
            ),
            progress_weight=float(pick(("incubation", "progress_weight"), 0.5)),
            world_size=world_xy,
            world_z=world_z,
            robot_size=float(pick(("multi_robot", "robot_size"), 0.4)),
            spawn_z=float(pick(("multi_robot", "spawn_area", "z"), 0.1)),
            spawn_x_min=float(
                pick(("multi_robot", "spawn_area", "x_min"), 0.1),
            ),
            spawn_x_max=float(
                pick(("multi_robot", "spawn_area", "x_max"), world_xy[0] - 0.1),
            ),
            spawn_y_min=float(
                pick(("multi_robot", "spawn_area", "y_min"), 0.1),
            ),
            spawn_y_max=float(
                pick(("multi_robot", "spawn_area", "y_max"), world_xy[1] - 0.1),
            ),
            min_spawn_distance=float(
                pick(("multi_robot", "min_spawn_distance"), 1.0),
            ),
            pairing_radius=float(pick(("selection", "pairing_radius"), 0.5)),
            offspring_radius=float(
                pick(("selection", "offspring_radius"), 0.3),
            ),
            pairing_method=cast(
                "PairingMethod",
                str(pick(("selection", "pairing_method"), "proximity_pairing")),
            ),
            movement_bias=movement_bias,
            movement_step_size=movement_step_size,
            num_mating_zones=int(pick(("selection", "num_mating_zones"), 1)),
            mating_zone_center=zone_center,
            mating_zone_radius=float(
                pick(("selection", "mating_zone_radius"), 3.0),
            ),
            min_zone_distance=float(
                pick(("selection", "min_zone_distance"), 2.0),
            ),
            zone_relocation_strategy=cast(
                "ZoneRelocationStrategy",
                str(relocation),
            ),
            zone_change_interval=int(
                pick(("selection", "zone_change_interval"), 5),
            ),
            zone_capacity_softness=int(
                pick(("selection", "zone_capacity_softness"), 1),
            ),
            selection_method=cast(
                "SelectionMethod",
                str(pick(("selection", "selection_method"), "parents_die")),
            ),
            max_age=int(pick(("selection", "max_age"), 10)),
            enable_energy=bool(pick(("selection", "enable_energy"), True)),
            initial_energy=float(pick(("selection", "initial_energy"), 100.0)),
            energy_depletion_rate=float(
                pick(("selection", "energy_depletion_rate"), 10.0),
            ),
            mating_energy_effect=cast(
                "MatingEnergyEffect",
                str(pick(("selection", "mating_energy_effect"), "cost")),
            ),
            mating_energy_amount=float(
                pick(("selection", "mating_energy_amount"), 35.0),
            ),
            density_locality_radius=float(
                pick(("selection", "locality_radius"), 3.0),
            ),
            density_critical_density=float(
                pick(("selection", "critical_density"), 5.0),
            ),
            density_base_death_prob=float(
                pick(("selection", "base_death_prob"), 0.05),
            ),
            density_max_death_prob=float(
                pick(("selection", "max_density_death_prob"), 0.8),
            ),
            density_fitness_protection=float(
                pick(("selection", "density_fitness_protection"), 0.0),
            ),
            result_folder=Path(
                str(
                    pick(
                        ("output", "results_folder"),
                        Path.cwd() / "__results__",
                    ),
                ),
            ),
            figure_folder=Path(
                str(
                    pick(
                        ("output", "figures_folder"),
                        Path.cwd() / "__figures__",
                    ),
                ),
            ),
            video_folder=Path(
                str(
                    pick(("output", "video_folder"), Path.cwd() / "__videos__"),
                ),
            ),
            print_generation_stats=bool(
                pick(("logging", "print_generation_stats"), True),
            ),
            record_generation_videos=bool(
                pick(("video", "record_generation_videos"), False),
            ),
            save_generation_snapshots=bool(
                pick(("video", "save_generation_snapshots"), False),
            ),
        )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> SpatialEAConfig:
        """Load a configuration from a YAML file.

        Both the flat native schema and the prototype's nested schema are
        accepted; the nested one is recognised by its section keys.

        Parameters
        ----------
        config_path
            Path to the YAML file.

        Returns
        -------
            The parsed configuration.

        Raises
        ------
        ValueError
            If the document's root is not a mapping.
        """
        with Path(config_path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        if not isinstance(payload, dict):
            msg = (
                f"Expected a mapping at the root of {config_path}, "
                f"got {type(payload).__name__}"
            )
            raise ValueError(msg)

        if any(key in payload for key in LEGACY_SECTION_KEYS):
            return cls.from_legacy_dict(payload)

        return cls(**payload)


spatial_ea_config: SpatialEAConfig = SpatialEAConfig()
