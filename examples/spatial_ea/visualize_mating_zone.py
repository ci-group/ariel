r"""Draw the mating-zone layout a configuration produces.

Mating zones are the rendezvous points that make reproduction a navigation
problem, so seeing where they land — and how much of the world they cover — is
the quickest way to sanity-check a configuration before spending simulation
time on it.

Examples
--------
Default configuration::

    python examples/spatial_ea/visualize_mating_zone.py

A research-prototype configuration, with a sample population drawn on top::

    python examples/spatial_ea/visualize_mating_zone.py \\
        --config examples/spatial_ea/ea_config.yaml --show-population

"""

# Standard library
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Third-party libraries
import matplotlib as mpl

mpl.use("Agg")

# Local libraries
from ariel import console
from ariel.spatial_ea.config import SpatialEAConfig
from ariel.spatial_ea.interaction import generate_random_zone_centers
from ariel.spatial_ea.visualization import plot_mating_zones
from ariel.spatial_ea.world import generate_spawn_positions

# Global constants
SCRIPT_NAME = Path(__file__).stem


def zone_centers(config: SpatialEAConfig) -> list[tuple[float, float]]:
    """Resolve the zone layout a configuration asks for.

    Parameters
    ----------
    config
        Run configuration.

    Returns
    -------
        One centre per mating zone.
    """
    if config.num_mating_zones <= 1:
        return [config.mating_zone_center]

    return generate_random_zone_centers(
        num_zones=config.num_mating_zones,
        world_size=(config.world_size[0], config.world_size[1]),
        zone_radius=config.mating_zone_radius,
        min_zone_distance=config.min_zone_distance,
    )


def main(argv: list[str] | None = None) -> int:
    """Draw the mating zones described by a configuration.

    Parameters
    ----------
    argv
        Argument list, defaulting to ``sys.argv[1:]``.

    Returns
    -------
        Always ``0``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show-population", action="store_true")
    args = parser.parse_args(argv)

    if args.config is not None:
        config = SpatialEAConfig.from_yaml(args.config)
    else:
        config = SpatialEAConfig(num_mating_zones=5, mating_zone_radius=1.5)

    positions = None
    if args.show_population:
        positions = generate_spawn_positions(
            population_size=config.population_size,
            spawn_x_range=(config.spawn_x_min, config.spawn_x_max),
            spawn_y_range=(config.spawn_y_min, config.spawn_y_max),
            spawn_z=config.spawn_z,
            min_spawn_distance=config.min_spawn_distance,
        )

    written = plot_mating_zones(
        (config.world_size[0], config.world_size[1]),
        zone_centers(config),
        config.mating_zone_radius,
        args.output or (Path(config.figure_folder) / f"{SCRIPT_NAME}.png"),
        positions,
        use_periodic_boundaries=config.use_periodic_boundaries,
        title_suffix=(
            f"relocation: {config.zone_relocation_strategy}, "
            f"pairing: {config.pairing_method}"
        ),
    )

    console.print(f"[bold green]Saved:[/] {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
