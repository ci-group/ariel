r"""Re-draw the figures for a run that has already finished.

Reads the CSV and controller export a run saved and redraws its figures, so an
old result can be re-examined without re-running it. Replaces the research
prototype's post-hoc viewer.

Examples
--------
Most recent run in a results folder::

    python examples/spatial_ea/visualize_run.py --results __results__

A specific run, written somewhere else::

    python examples/spatial_ea/visualize_run.py \\
        --csv __results__/evolution_data_20260101_120000.csv \\
        --output ./report

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
from ariel.spatial_ea.data import EvolutionDataCollector
from ariel.spatial_ea.persistence import load_controllers_from_json
from ariel.spatial_ea.visualization import (
    plot_evolution_statistics,
    plot_final_population,
)


def matching_controllers(csv_path: Path) -> Path | None:
    """Find the controller export that belongs to a statistics CSV.

    Every file of a run shares one timestamp, so the controllers are found by
    substituting the prefix rather than by guessing the newest file.

    Parameters
    ----------
    csv_path
        Path to an ``evolution_data_*.csv`` file.

    Returns
    -------
        The matching ``final_controllers_*.json``, or ``None``.
    """
    timestamp = csv_path.stem.replace("evolution_data_", "")
    candidate = csv_path.parent / f"final_controllers_{timestamp}.json"
    return candidate if candidate.exists() else None


def main(argv: list[str] | None = None) -> int:
    """Redraw a finished run's figures.

    Parameters
    ----------
    argv
        Argument list, defaulting to ``sys.argv[1:]``.

    Returns
    -------
        ``0`` on success, ``1`` when no run could be found.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path.cwd() / "__results__",
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "__figures__" / "run_report",
    )
    args = parser.parse_args(argv)

    csv_path = args.csv or EvolutionDataCollector.latest_csv(args.results)
    if csv_path is None or not Path(csv_path).exists():
        console.print(
            f"[bold red]No run found[/] in {args.results}. "
            f"Run the EA with --save-results true first.",
        )
        return 1

    csv_path = Path(csv_path)
    collector = EvolutionDataCollector.from_csv(csv_path)
    console.print(
        f"[bold]Loaded[/] {len(collector.generations)} generations "
        f"from {csv_path.name}",
    )

    args.output.mkdir(parents=True, exist_ok=True)
    written = [
        plot_evolution_statistics(
            collector,
            args.output / "evolution_statistics.png",
        ),
    ]

    controllers_path = matching_controllers(csv_path)
    if controllers_path is None:
        console.print(
            "[yellow]No matching controller export[/]; skipping the "
            "final-population figure.",
        )
    else:
        payload = load_controllers_from_json(controllers_path)
        controllers = payload.get("controllers", [])
        if controllers:
            written.append(
                plot_final_population(
                    controllers,
                    args.output / "final_population.png",
                    stop_reason=collector.stop_reason,
                ),
            )
            console.print(
                f"  final population: {len(controllers)} controllers",
            )

    summary = collector.get_summary_stats()
    console.print(
        f"  best fitness {summary['fitness']['best_ever']:.4f} | "
        f"population {summary['population']['initial']} → "
        f"{summary['population']['final']} | "
        f"{summary['stop_reason']}",
    )
    console.print(
        f"\n[bold green]Wrote {len(written)} figures to[/] {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
