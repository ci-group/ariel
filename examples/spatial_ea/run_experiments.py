r"""Run named batch experiments over the spatial EA.

An experiment is a set of configuration overrides run several times with
different seeds. Results and pooled statistics are written per experiment, and
a comparison figure is drawn across whichever experiments were run.

The experiments defined here mirror the questions the research prototype was
built to ask: does spatial pairing differ from random pairing, does giving
robots a zone to navigate to change anything, and which death rule keeps a
population alive.

Examples
--------
List what is available, then run one::

    python examples/spatial_ea/run_experiments.py --list
    python examples/spatial_ea/run_experiments.py --experiment baseline_random

Run everything in parallel, with more repeats::

    python examples/spatial_ea/run_experiments.py --experiment all \\
        --runs 5 --parallel

Sweep a parameter instead::

    python examples/spatial_ea/run_experiments.py \\
        --grid density_base_death_prob=0.02,0.05,0.1

"""

# Standard library
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Third-party libraries
import matplotlib as mpl

mpl.use("Agg")

# Local libraries
from ariel import console
from ariel.spatial_ea.config import SpatialEAConfig
from ariel.spatial_ea.experiment import (
    AggregatedResults,
    ExperimentRunner,
    ExperimentSpec,
)
from ariel.spatial_ea.visualization import plot_aggregated_results

# Global constants
SCRIPT_NAME = Path(__file__).stem


def base_config(args: argparse.Namespace) -> SpatialEAConfig:
    """Build the configuration every experiment starts from.

    A small world keeps the centimetre-scale movement of untrained controllers
    meaningful relative to the pairing radius.

    Parameters
    ----------
    args
        Parsed command-line arguments.

    Returns
    -------
        The base configuration.
    """
    return SpatialEAConfig(
        population_size=args.population_size,
        num_generations=args.generations,
        simulation_time=args.simulation_time,
        world_size=(6.0, 6.0),
        min_spawn_distance=0.7,
        pairing_radius=0.8,
        offspring_radius=0.3,
        selection_method="fitness_based",
        target_population_size=args.population_size,
        max_population_limit=args.population_size * 5,
        use_periodic_boundaries=True,
        enable_energy=True,
        energy_depletion_rate=8.0,
        mating_energy_effect="cost",
        mating_energy_amount=10.0,
        incubation_enabled=args.incubation,
        use_directional_fitness=args.incubation,
        # Per-run output is written by the runner into each run's own folder.
        save_results=True,
        save_plots=True,
        save_generation_plots=False,
        print_generation_stats=False,
    )


def define_experiments() -> dict[str, ExperimentSpec]:
    """Describe the experiments this script can run.

    Returns
    -------
        Experiment specifications, keyed by name.
    """
    specs = [
        ExperimentSpec(
            name="baseline_random",
            description=(
                "Non-spatial control: partners are drawn at random, so "
                "distance cannot matter."
            ),
            overrides={"pairing_method": "random", "movement_bias": "none"},
        ),
        ExperimentSpec(
            name="proximity",
            description="Robots pair with whoever is nearest.",
            overrides={
                "pairing_method": "proximity_pairing",
                "movement_bias": "none",
            },
        ),
        ExperimentSpec(
            name="proximity_seeking",
            description=(
                "As proximity, but controllers are told which way their "
                "nearest neighbour lies."
            ),
            overrides={
                "pairing_method": "proximity_pairing",
                "movement_bias": "nearest_neighbor",
            },
        ),
        ExperimentSpec(
            name="mating_zones",
            description=(
                "Reproduction is a rendezvous problem: only robots sharing a "
                "zone can pair."
            ),
            overrides={
                "pairing_method": "mating_zone",
                "movement_bias": "assigned_zone",
                "num_mating_zones": 3,
                "mating_zone_radius": 1.0,
            },
        ),
        ExperimentSpec(
            name="zones_event_driven",
            description=(
                "As mating_zones, but a zone moves once it has been used, so "
                "the population must keep re-finding them."
            ),
            overrides={
                "pairing_method": "mating_zone",
                "movement_bias": "assigned_zone",
                "num_mating_zones": 3,
                "mating_zone_radius": 1.0,
                "zone_relocation_strategy": "event_driven",
            },
        ),
        ExperimentSpec(
            name="density_death",
            description="Crowding kills, so the population self-limits.",
            overrides={
                "pairing_method": "proximity_pairing",
                "movement_bias": "nearest_neighbor",
                "selection_method": "density_based",
            },
        ),
        ExperimentSpec(
            name="energy_death",
            description="Individuals starve unless they keep reproducing.",
            overrides={
                "pairing_method": "proximity_pairing",
                "movement_bias": "nearest_neighbor",
                "selection_method": "energy_based",
            },
        ),
    ]
    return {spec.name: spec for spec in specs}


def parse_grid(spec: str) -> dict[str, list[Any]]:
    """Parse a ``field=v1,v2`` sweep specification.

    Parameters
    ----------
    spec
        One or more ``field=value,value`` clauses separated by semicolons.

    Returns
    -------
        Field name to the values it should take.

    Raises
    ------
    ValueError
        If a clause has no ``=``, or names an unknown configuration field.
    """
    grid: dict[str, list[Any]] = {}
    for clause in spec.split(";"):
        if "=" not in clause:
            msg = f"Grid clause {clause!r} is not of the form field=v1,v2"
            raise ValueError(msg)

        field_name, raw_values = clause.split("=", 1)
        field_name = field_name.strip()
        if field_name not in SpatialEAConfig.model_fields:
            msg = f"Unknown configuration field {field_name!r}"
            raise ValueError(msg)

        grid[field_name] = [
            _coerce(value.strip()) for value in raw_values.split(",")
        ]

    return grid


def _coerce(text: str) -> Any:
    """Turn a command-line grid value into a number or string.

    Parameters
    ----------
    text
        The raw value.

    Returns
    -------
        An ``int`` or ``float`` when the text looks numeric, else the text.
    """
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            continue
    return text


def main(argv: list[str] | None = None) -> int:
    """Run the requested experiments.

    Parameters
    ----------
    argv
        Argument list, defaulting to ``sys.argv[1:]``.

    Returns
    -------
        ``0`` on success, ``1`` if nothing could be run.
    """
    experiments = define_experiments()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="mating_zones")
    parser.add_argument("--grid", default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--simulation-time", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "__experiments__",
    )
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--incubation", action="store_true")
    parser.add_argument(
        "--padding",
        choices=["nan", "forward_fill", "terminal_state"],
        default="forward_fill",
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        console.print("[bold]Available experiments[/]")
        for name, spec in experiments.items():
            console.print(f"  [bold cyan]{name}[/]\n    {spec.description}")
        return 0

    runner = ExperimentRunner(
        base_config=base_config(args),
        output_folder=args.output,
        seed=args.seed,
    )

    aggregated: dict[str, AggregatedResults] = {}

    if args.grid is not None:
        grid = parse_grid(args.grid)
        console.print(f"[bold]Grid sweep[/] over {grid}")
        aggregated = runner.grid_search(
            "grid",
            grid,
            num_runs=args.runs,
            parallel=args.parallel,
            num_workers=args.num_workers,
        )
    else:
        if args.experiment == "all":
            selected = list(experiments.values())
        elif args.experiment in experiments:
            selected = [experiments[args.experiment]]
        else:
            console.print(
                f"[bold red]Unknown experiment[/] {args.experiment!r}. "
                f"Use --list to see the available ones.",
            )
            return 1

        for spec in selected:
            spec.num_runs = args.runs
            console.print(f"\n[bold]{spec.name}[/] — {spec.description}")
            _, stats = runner.run_and_save(
                spec,
                parallel=args.parallel,
                num_workers=args.num_workers,
                padding_strategy=args.padding,
            )
            aggregated[spec.name] = stats

    if not aggregated:
        console.print("[bold red]Nothing was run[/]")
        return 1

    rows = runner.compare(aggregated, args.output / "comparison.json")
    console.print("\n[bold]Results[/] (best fitness first)")
    for row in rows:
        console.print(
            f"  {row['experiment']:<40} "
            f"best={row['best_fitness_ever']:.4f}  "
            f"completed={row['completion_rate']:.0%}  "
            f"extinct={row['num_extinctions']}  "
            f"exploded={row['num_explosions']}",
        )

    figure = plot_aggregated_results(
        aggregated,
        args.output / "comparison_fitness.png",
        metric="fitness_best",
    )
    population_figure = plot_aggregated_results(
        aggregated,
        args.output / "comparison_population.png",
        metric="population_size",
    )
    console.print(f"\n[bold green]Figures:[/] {figure}, {population_figure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
