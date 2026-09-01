"""Run a small spatial EA and render every stage, to confirm it works.

This is a smoke test you can look at. It runs a short evolution and writes a
figure for each mechanism the algorithm depends on, then prints a checklist of
what the run actually exercised.

What it produces
----------------
``01_mating_zones.png``
    The zone layout the configuration asks for, with the starting population
    marked as inside or outside a zone.
``02_physical/mating_generation_*.png``
    One frame per generation of the shared-world movement phase: real MuJoCo
    trajectories, and a green line joining every pair that reproduced.
``03_analytical/mating_generation_*.png``
    The same run with the physics phase swapped for the analytical nudge, where
    zone-seeking is exaggerated and therefore easy to read.
``04_statistics_physical.png``, ``04_statistics_analytical.png``
    Population, fitness, age, mating and energy over the whole run.

A note on scale
---------------
Untrained HyperNEAT controllers barely locomote — roughly a centimetre or two
per simulated second. The demo therefore uses a small world so that real
movement is visible at the plotted scale. Robots that walk properly are what
the incubation phase is for, and it is far too slow to run here; pass
``--incubation`` if you want to watch it try.

Examples
--------
::

    python examples/spatial_ea/visual_experiment.py
    python examples/spatial_ea/visual_experiment.py --generations 10 --output /tmp/demo

"""

# Standard library
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

# Third-party libraries
import matplotlib as mpl
import numpy as np

mpl.use("Agg")

# Local libraries
from ariel import console
from ariel.spatial_ea import SpatialEA, SpatialEAConfig
from ariel.spatial_ea.visualization import (
    plot_evolution_statistics,
    plot_mating_zones,
)

# Global constants
SCRIPT_NAME = Path(__file__).stem
DEFAULT_SEED = 11


def build_config(
    args: argparse.Namespace,
    figure_folder: Path,
) -> SpatialEAConfig:
    """Build the demo configuration.

    The world is deliberately small so that the little distance an untrained
    controller covers is still legible on the plots.

    Parameters
    ----------
    args
        Parsed command-line arguments.
    figure_folder
        Where per-generation figures should be written.

    Returns
    -------
        The configuration for one demo run.
    """
    return SpatialEAConfig(
        population_size=args.population_size,
        num_generations=args.generations,
        simulation_time=args.simulation_time,
        # A 3 m world keeps centimetre-scale movement visible.
        world_size=(3.0, 3.0),
        world_z=0.1,
        robot_size=0.25,
        spawn_x_min=0.3,
        spawn_x_max=2.7,
        spawn_y_min=0.3,
        spawn_y_max=2.7,
        min_spawn_distance=0.55,
        # Reproduction is a rendezvous problem: only robots sharing a zone pair.
        pairing_method="mating_zone",
        num_mating_zones=2,
        mating_zone_radius=0.7,
        min_zone_distance=1.5,
        movement_bias="assigned_zone",
        pairing_radius=0.6,
        offspring_radius=0.25,
        zone_relocation_strategy="static",
        selection_method="fitness_based",
        target_population_size=args.population_size,
        max_population_limit=40,
        use_periodic_boundaries=True,
        enable_energy=True,
        initial_energy=100.0,
        energy_depletion_rate=8.0,
        mating_energy_effect="cost",
        mating_energy_amount=10.0,
        incubation_enabled=args.incubation,
        incubation_population_size=12,
        incubation_num_generations=15,
        use_directional_fitness=args.incubation,
        figure_folder=figure_folder,
        save_results=False,
        save_generation_plots=True,
        print_generation_stats=False,
    )


def run_variant(
    args: argparse.Namespace,
    output: Path,
    *,
    physical: bool,
    seed: int,
) -> SpatialEA:
    """Run one demo variant and write its figures.

    Parameters
    ----------
    args
        Parsed command-line arguments.
    output
        Directory for this variant's per-generation figures.
    physical
        Whether to simulate the movement phase in MuJoCo, or to use the cheap
        analytical nudge instead.
    seed
        Random seed, so the two variants start from the same population.

    Returns
    -------
        The engine, after the run.
    """
    np.random.seed(seed)
    random.seed(seed)

    config = build_config(args, output)
    if not physical:
        config = config.model_copy(
            update={
                "use_physical_movement_phase": False,
                "movement_step_size": 0.35,
            },
        )

    label = "physical (MuJoCo)" if physical else "analytical (nudge)"
    console.print(f"\n[bold]Running {label} movement…[/]")

    started = time.time()
    engine = SpatialEA(config=config)
    engine.run()
    elapsed = time.time() - started

    console.print(f"  finished in {elapsed:.1f}s")
    return engine


def summarise(
    engine: SpatialEA,
    label: str,
    *,
    physical: bool,
) -> dict[str, bool | None]:
    """Check that the run actually exercised each mechanism.

    Parameters
    ----------
    engine
        The engine after a completed run.
    label
        Name of the variant, used in the printed heading.
    physical
        Whether this variant simulated movement, which decides if the
        trajectory check applies at all.

    Returns
    -------
        One flag per mechanism: ``True`` exercised, ``False`` not exercised,
        ``None`` not applicable to this variant.
    """
    data = engine.data_collector

    # The last generation's offspring are created after the final evaluation,
    # so they are unevaluated by design; check that evaluation ran at all.
    evaluated = [ind for ind in engine.population if ind.evaluated]

    energy_moved = (
        len({round(value, 3) for value in data.energy_avg}) > 1
        if data.energy_avg
        else False
    )

    moved: bool | None = None
    if physical:
        moved = any(
            float(np.linalg.norm(np.asarray(path[-1]) - np.asarray(path[0])))
            > 1e-4
            for path in engine.trajectories
        )

    checks: dict[str, bool | None] = {
        "generations recorded": len(data.generations) > 0,
        "fitness evaluated": bool(evaluated)
        and all(np.isfinite(value) for value in data.fitness_best),
        "robots moved in physics": moved,
        "pairs formed": sum(data.mating_pairs) > 0,
        "offspring born": sum(data.births) > 0,
        "selection removed individuals": sum(data.deaths) > 0,
        "energy changed": energy_moved,
        "diversity tracked": any(
            value > 0 for value in data.genotype_diversity
        ),
    }

    console.print(f"\n[bold]{label}[/]")
    for name, passed in checks.items():
        if passed is None:
            mark = "[dim]n/a[/]"
        elif passed:
            mark = "[green]ok[/]"
        else:
            mark = "[yellow]not exercised[/]"
        console.print(f"  {mark:<30} {name}")

    console.print(
        f"  population per generation : {data.population_size}\n"
        f"  pairs per generation      : {data.mating_pairs}\n"
        f"  best fitness              : "
        f"{[round(value, 3) for value in data.fitness_best]}",
    )
    return checks


def main(argv: list[str] | None = None) -> int:
    """Run the visual confirmation experiment.

    Parameters
    ----------
    argv
        Argument list, defaulting to ``sys.argv[1:]``.

    Returns
    -------
        ``0`` when every mechanism was exercised in at least one variant,
        ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "__figures__" / SCRIPT_NAME,
    )
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--simulation-time", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--incubation",
        action="store_true",
        help="Pre-evolve locomotion first. Much slower, still not fast robots.",
    )
    parser.add_argument(
        "--skip-analytical",
        action="store_true",
        help="Only run the physics variant.",
    )
    args = parser.parse_args(argv)

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Writing figures to[/] {output}")

    # -- 1. The zone layout the configuration asks for ------------------------
    np.random.seed(args.seed)
    random.seed(args.seed)
    preview = SpatialEA(config=build_config(args, output))
    preview.initialize_population()
    plot_mating_zones(
        (preview.config.world_size[0], preview.config.world_size[1]),
        preview.current_zone_centers,
        preview.config.mating_zone_radius,
        output / "01_mating_zones.png",
        preview.positions,
        use_periodic_boundaries=preview.config.use_periodic_boundaries,
        title_suffix="starting population, before any movement",
    )
    console.print("  wrote 01_mating_zones.png")

    # -- 2. The real thing ----------------------------------------------------
    variants: dict[str, tuple[SpatialEA, bool]] = {}
    physical = run_variant(
        args,
        output / "02_physical",
        physical=True,
        seed=args.seed,
    )
    plot_evolution_statistics(
        physical.data_collector,
        output / "04_statistics_physical.png",
    )
    variants["Physical movement (MuJoCo)"] = (physical, True)

    # -- 3. The same run with movement exaggerated ----------------------------
    if not args.skip_analytical:
        analytical = run_variant(
            args,
            output / "03_analytical",
            physical=False,
            seed=args.seed,
        )
        plot_evolution_statistics(
            analytical.data_collector,
            output / "04_statistics_analytical.png",
        )
        variants["Analytical movement (nudge)"] = (analytical, False)

    # -- 4. Report ------------------------------------------------------------
    all_checks: dict[str, bool] = {}
    for label, (engine, is_physical) in variants.items():
        results = summarise(engine, label, physical=is_physical)
        for name, passed in results.items():
            if passed is None:
                continue
            all_checks[name] = all_checks.get(name, False) or passed

    unexercised = [name for name, passed in all_checks.items() if not passed]
    console.print("")
    if unexercised:
        console.print(
            "[bold yellow]Not exercised by any variant:[/] "
            + ", ".join(unexercised),
        )
        console.print(
            "Try --generations 12, a larger --population-size, "
            "or a longer --simulation-time.",
        )
        return 1

    console.print("[bold green]Every mechanism was exercised.[/]")
    console.print(f"Open the figures in {output} to confirm visually.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
