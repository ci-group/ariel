"""3D bar plots of final reward over (width × depth), one figure per insect_n.

Reads the same run directories as plot_sweep.py

    insect_n_n{N}_w{W}_d{D}_seed{S}-<timestamp>/training_history.json

and writes ``final_reward_3d_n{N}.png`` for every n found. Bar height is the
final-eval reward averaged over seeds; the label on top of each bar is
mean ± std across seeds.

Depth 0 (linear policy) is only run once per n, at the first width, because
width is irrelevant there. By default that result is drawn at every width
(semi-transparent) so the depth-0 row is complete; pass --no-broadcast-linear
to draw it only at the width it was actually run with.

Usage
-----
    experiment/mujoco_playground/.venv/bin/python experiment/original_idea/plot_sweep_3d.py \
        --data-dir experiment/__data__/directed_capacity_sweep \
        --out-dir  experiment/original_idea/__plots_directed__
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from plot_sweep import _aggregate, _collect


def plot_final_reward_3d(
    agg: dict,
    n: int,
    widths: list[int],
    depths: list[int],
    out_path: Path,
    zlim: tuple[float, float] | None,
    norm: colors.Normalize,
    broadcast_linear: bool,
) -> None:
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(projection="3d")
    # Manual draw order: matplotlib's depth sorting hides the value labels
    # behind nearer bars, so draw bars back-to-front and labels last.
    ax.computed_zorder = False
    cmap = plt.get_cmap("viridis")
    bar = 0.6

    # Linear-policy result for this n (whatever width it was run at).
    linear = next((agg[(n, w, 0)] for w in widths if (n, w, 0) in agg), None)

    # With view azim=-130 the (width[0], depth[0]) corner is nearest the viewer.
    cells = sorted(
        ((i, w, j, d) for i, w in enumerate(widths) for j, d in enumerate(depths)),
        key=lambda c: -(c[0] + c[2]),
    )
    labels = []
    for order, (i, w, j, d) in enumerate(cells):
        rec = agg.get((n, w, d))
        alpha = 0.9
        if rec is None and d == 0 and broadcast_linear and linear is not None:
            rec, alpha = linear, 0.35
        if rec is None:
            continue
        z = rec["final_mean"]
        ax.bar3d(
            i - bar / 2, j - bar / 2, 0.0, bar, bar, z,
            color=cmap(norm(z)), alpha=alpha, shade=True,
            edgecolor="k", linewidth=0.3, zorder=order,
        )
        labels.append((i, j, z, rec["final_std"]))
    for i, j, z, sd in labels:
        ax.text(
            i, j, max(z, 0.0), f"{z:.1f}\n±{sd:.1f}",
            ha="center", va="bottom", fontsize=7, zorder=1000,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        )

    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([("0 (linear)" if d == 0 else str(d)) for d in depths])
    ax.set_xlabel("network width")
    ax.set_ylabel("network depth")
    ax.set_zlabel("final eval reward")
    if zlim is not None:
        ax.set_zlim(*zlim)
    ax.view_init(elev=25, azim=-130)
    ax.set_title(f"Directed capacity sweep — insect_n n={n}\n"
                 "final reward (mean ± std over seeds)")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 shrink=0.6, pad=0.1, label="final eval reward")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path,
                   default=root.parent / "__data__" / "directed_capacity_sweep")
    p.add_argument("--out-dir", type=Path, default=root / "__plots_directed__")
    p.add_argument("--independent-z", action="store_true",
                   help="Give each n its own z-axis/colour scale instead of a shared one.")
    p.add_argument("--no-broadcast-linear", dest="broadcast_linear",
                   action="store_false",
                   help="Draw the depth-0 bar only at the width it was run with.")
    args = p.parse_args()

    runs = _collect(args.data_dir, None)
    if not runs:
        raise SystemExit(f"no sweep runs under {args.data_dir}")
    widths = sorted({r["w"] for r in runs})
    depths = sorted({r["d"] for r in runs})
    ns = sorted({r["n"] for r in runs})
    agg = _aggregate(runs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    finals = [rec["final_mean"] for rec in agg.values()]
    shared_lo, shared_hi = min(0.0, min(finals)), max(finals) * 1.15

    for n in ns:
        if args.independent_z:
            vals = [rec["final_mean"] for (nn, _, _), rec in agg.items() if nn == n]
            lo, hi = min(0.0, min(vals)), max(vals) * 1.15
        else:
            lo, hi = shared_lo, shared_hi
        out = args.out_dir / f"final_reward_3d_n{n}.png"
        plot_final_reward_3d(
            agg, n, widths, depths, out,
            zlim=(lo, hi), norm=colors.Normalize(lo, hi),
            broadcast_linear=args.broadcast_linear,
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
