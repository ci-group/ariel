"""Show what periodic boundaries do to distance, pairing and trajectories.

The spatial EA treats the world as a torus so that the edges do not create
artificial corners where robots pile up or become permanently isolated. Any
clustering that appears is then a property of the algorithm rather than of the
walls. This script draws the three consequences that matter:

1. Two robots near opposite edges are actually neighbours.
2. A pairing radius near an edge reaches around to the far side.
3. A trajectory that wraps must be drawn in segments, not as one line.

Examples
--------
::

    python examples/spatial_ea/visualize_periodic_boundaries.py

"""

# Standard library
import argparse
import sys
from pathlib import Path

# Third-party libraries
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

# Local libraries
from ariel import console
from ariel.spatial_ea.interaction import (
    apply_world_boundaries,
    calculate_periodic_displacement,
    calculate_periodic_distance,
    split_trajectory_at_wraps,
)

# Global constants
SCRIPT_NAME = Path(__file__).stem
WORLD_SIZE = (10.0, 10.0)


def _draw_world(ax: plt.Axes, title: str) -> None:
    """Draw the world outline and label a panel.

    Parameters
    ----------
    ax
        Axes to draw on.
    title
        Panel title.
    """
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            WORLD_SIZE[0],
            WORLD_SIZE[1],
            linewidth=0,
            facecolor="lightgray",
            alpha=0.25,
        ),
    )
    # Drawn separately so the outline is not washed out by the fill alpha.
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            WORLD_SIZE[0],
            WORLD_SIZE[1],
            linewidth=1.5,
            edgecolor="black",
            facecolor="none",
        ),
    )
    ax.set_xlim(-1.0, WORLD_SIZE[0] + 1.0)
    ax.set_ylim(-1.0, WORLD_SIZE[1] + 1.0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.grid(visible=True, alpha=0.2)


def _panel_distance(ax: plt.Axes) -> None:
    """Contrast straight-line distance with toroidal distance.

    Parameters
    ----------
    ax
        Axes to draw on.
    """
    first = np.array([1.0, 5.0, 0.0])
    second = np.array([9.0, 5.0, 0.0])

    direct = float(np.linalg.norm(second[:2] - first[:2]))
    toroidal = calculate_periodic_distance(first, second, WORLD_SIZE)

    _draw_world(
        ax,
        f"Distance\ndirect {direct:.1f} m, through the edge {toroidal:.1f} m",
    )
    ax.plot(*first[:2], "o", color="tab:blue", markersize=12)
    ax.plot(*second[:2], "o", color="tab:orange", markersize=12)

    ax.plot(
        [first[0], second[0]],
        [first[1], second[1]],
        "--",
        color="gray",
        label=f"across the world ({direct:.1f} m)",
    )
    # The short way runs off one edge and back on through the other.
    ax.annotate(
        "",
        xy=(-1.0, 5.0),
        xytext=(1.0, 5.0),
        arrowprops={"arrowstyle": "->", "color": "tab:green", "lw": 2},
    )
    ax.annotate(
        "",
        xy=(9.0, 5.0),
        xytext=(11.0, 5.0),
        arrowprops={"arrowstyle": "->", "color": "tab:green", "lw": 2},
    )
    ax.plot(
        [],
        [],
        "-",
        color="tab:green",
        lw=2,
        label=f"wrap ({toroidal:.1f} m)",
    )
    ax.legend(loc="lower center", fontsize=8)


def _panel_pairing(ax: plt.Axes) -> None:
    """Show a pairing radius reaching around the world edge.

    Parameters
    ----------
    ax
        Axes to draw on.
    """
    radius = 2.5
    center = np.array([0.8, 5.0, 0.0])
    candidates = [
        np.array([2.5, 5.0, 0.0]),
        np.array([9.2, 5.0, 0.0]),
        np.array([5.0, 5.0, 0.0]),
    ]

    _draw_world(ax, f"Pairing within {radius} m\nthe radius wraps too")

    for offset in (0.0, WORLD_SIZE[0], -WORLD_SIZE[0]):
        ax.add_patch(
            patches.Circle(
                (center[0] + offset, center[1]),
                radius,
                edgecolor="tab:red",
                facecolor="tab:red",
                alpha=0.12,
                linestyle="--",
            ),
        )

    ax.plot(*center[:2], "*", color="tab:red", markersize=18, label="searcher")
    for candidate in candidates:
        distance = calculate_periodic_distance(center, candidate, WORLD_SIZE)
        reachable = distance <= radius
        ax.plot(
            *candidate[:2],
            "o" if reachable else "x",
            color="tab:green" if reachable else "gray",
            markersize=10,
        )
        ax.annotate(
            f"{distance:.1f}",
            candidate[:2],
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )
    ax.legend(loc="lower center", fontsize=8)


def _panel_trajectory(ax: plt.Axes) -> None:
    """Show why a wrapped trajectory must be split before plotting.

    Parameters
    ----------
    ax
        Axes to draw on.
    """
    raw = [
        apply_world_boundaries(
            np.array([7.0 + step * 0.6, 5.0 + np.sin(step * 0.4), 0.0]),
            WORLD_SIZE,
            use_periodic_boundaries=True,
        )[:2]
        for step in range(14)
    ]

    _draw_world(ax, "A wrapping trajectory\ndrawn as separate segments")

    # One polyline would streak straight back across the world.
    ax.plot(
        [point[0] for point in raw],
        [point[1] for point in raw],
        "-",
        color="lightgray",
        lw=3,
        label="naive single line",
    )

    for i, segment in enumerate(split_trajectory_at_wraps(raw, WORLD_SIZE)):
        ax.plot(
            [point[0] for point in segment],
            [point[1] for point in segment],
            "-o",
            markersize=4,
            lw=2,
            label=f"segment {i + 1}",
        )

    ax.legend(loc="lower center", fontsize=8)


def main(argv: list[str] | None = None) -> int:
    """Render the periodic-boundary explainer.

    Parameters
    ----------
    argv
        Argument list, defaulting to ``sys.argv[1:]``.

    Returns
    -------
        Always ``0``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "__figures__" / f"{SCRIPT_NAME}.png",
    )
    args = parser.parse_args(argv)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    _panel_distance(axes[0])
    _panel_pairing(axes[1])
    _panel_trajectory(axes[2])
    fig.suptitle(
        "Periodic (toroidal) boundaries in the spatial EA",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # The displacement helper is what feeds the controller its heading.
    heading = calculate_periodic_displacement(
        np.array([0.5, 5.0, 0.0]),
        np.array([9.5, 5.0, 0.0]),
        WORLD_SIZE,
    )
    console.print(
        f"Heading from x=0.5 to x=9.5: {heading[:2]} "
        f"(left through the edge, not right across the world)",
    )
    console.print(f"[bold green]Saved:[/] {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
