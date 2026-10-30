"""Aggregate the architecture × morphology sweep and plot sample efficiency.

Reads every run directory under --data-dir matching the pattern

    insect_n_n{N}_w{W}_d{D}_seed{S}-<timestamp>/

parses ``training_history.json`` from each, groups runs by (n, w, d), and
renders four comparison figures:

    learning_curves_grid.png
        3×4 subplot grid (widths × depths); each subplot shows one curve
        per insect_n (mean reward across seeds, ± std shaded).

    final_reward_heatmaps.png
        One heatmap per insect_n, cells = final-eval reward averaged over
        seeds. Reveals how terminal performance tracks capacity.

    steps_to_threshold_heatmaps.png
        Same (width × depth) layout per n, cell = environment steps to
        first exceed THRESHOLD_PCT of the best-arch final reward at that
        n. The core "sample efficiency" plot.

    capacity_vs_reward.png
        Scatter of (width × depth) capacity proxy vs. final reward,
        coloured by n, marker shape by depth. Shows the diminishing
        returns of capacity per morphology.

Note on 0 hidden layers: with ``hidden_layer_sizes=()`` brax emits a
single ``Dense(param_size)`` linear head, so w16/w32/w64 at d=0 train
identical architectures. They are plotted as separate cells for sanity;
the three cells of each d=0 row should overlap within seed noise.

Usage
-----
    uv run experiment/original_idea/plot_sweep.py \
        --data-dir experiment/__data__/undirected_locomotion_ppo_jax \
        --out-dir  experiment/original_idea/__plots__
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

RUN_RE = re.compile(r"^insect_n_n(\d+)_w(\d+)_d(\d+)_seed(\d+)-")


def _load_run(run_dir: Path) -> dict[str, Any] | None:
    m = RUN_RE.match(run_dir.name)
    if m is None:
        return None
    n_pairs, width, depth, seed = (int(x) for x in m.groups())
    hist_path = run_dir / "training_history.json"
    if not hist_path.exists():
        return None
    hist = json.loads(hist_path.read_text())
    steps = hist.get("steps") or []
    if not steps:
        return None
    return {
        "n": n_pairs,
        "w": width,
        "d": depth,
        "seed": seed,
        "steps": np.asarray(steps, dtype=np.int64),
        "reward": np.asarray(hist["reward"], dtype=np.float64),
        "reward_std": np.asarray(hist["reward_std"], dtype=np.float64),
        "dir": run_dir.name,
    }


def _collect(data_dir: Path, run_filter: str | None) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        if run_filter and run_filter not in sub.name:
            continue
        rec = _load_run(sub)
        if rec is not None:
            runs.append(rec)
    return runs


def _aggregate(runs: list[dict[str, Any]]) -> dict[tuple[int, int, int], dict]:
    """Group runs by (n, w, d); interpolate reward curves to a common step grid."""
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        groups[(r["n"], r["w"], r["d"])].append(r)

    agg: dict[tuple[int, int, int], dict] = {}
    for key, seed_runs in groups.items():
        # Shared step grid = union of all eval steps across seeds. np.interp
        # extrapolates with the end-point value, which is fine here because
        # all seeds target the same --num-timesteps.
        common_steps = np.unique(np.concatenate([r["steps"] for r in seed_runs]))
        reward_mat = np.stack([
            np.interp(common_steps, r["steps"], r["reward"]) for r in seed_runs
        ])
        agg[key] = {
            "n_seeds": len(seed_runs),
            "steps": common_steps,
            "reward_mean": reward_mat.mean(axis=0),
            "reward_std": reward_mat.std(axis=0) if len(seed_runs) > 1 else np.zeros_like(reward_mat[0]),
            "final_mean": float(reward_mat[:, -1].mean()),
            "final_std": float(reward_mat[:, -1].std() if len(seed_runs) > 1 else 0.0),
            "reward_mat": reward_mat,  # (seeds, steps)
        }
    return agg


# ------------------------------------------------------------------------- #
#                               Plots                                       #
# ------------------------------------------------------------------------- #

def _n_color(n: int, ns: list[int]) -> Any:
    cmap = plt.get_cmap("viridis")
    return cmap(ns.index(n) / max(len(ns) - 1, 1))


def plot_learning_curves_grid(
    agg: dict,
    widths: list[int],
    depths: list[int],
    ns: list[int],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(widths), len(depths),
        figsize=(3.2 * len(depths), 2.6 * len(widths)),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)
    for i, w in enumerate(widths):
        for j, d in enumerate(depths):
            ax = axes[i, j]
            for n in ns:
                rec = agg.get((n, w, d))
                if rec is None:
                    continue
                color = _n_color(n, ns)
                ax.plot(rec["steps"], rec["reward_mean"], color=color, lw=1.3,
                        label=f"n={n}")
                ax.fill_between(
                    rec["steps"],
                    rec["reward_mean"] - rec["reward_std"],
                    rec["reward_mean"] + rec["reward_std"],
                    color=color, alpha=0.18,
                )
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(f"depth={d}")
            if j == 0:
                ax.set_ylabel(f"width={w}\nreward")
            if i == len(widths) - 1:
                ax.set_xlabel("env steps")
    # One legend on the top-right subplot.
    axes[0, -1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Learning curves across (width × depth) and insect_n", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _heatmap_grid_per_n(
    agg: dict,
    widths: list[int],
    depths: list[int],
    ns: list[int],
    value_fn,
    title: str,
    cbar_label: str,
    out_path: Path,
    cmap: str = "viridis",
    annotate: bool = True,
    missing_val: float = np.nan,
) -> None:
    """Side-by-side heatmaps (one per n) of some scalar(width, depth)."""
    fig, axes = plt.subplots(
        1, len(ns),
        figsize=(2.8 * len(ns) + 0.6, 2.6),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    # Shared colour scale across subplots (easier to compare morphologies).
    all_vals = []
    for n in ns:
        for w in widths:
            for d in depths:
                rec = agg.get((n, w, d))
                v = value_fn(rec) if rec is not None else missing_val
                if not np.isnan(v):
                    all_vals.append(v)
    vmin = min(all_vals) if all_vals else 0.0
    vmax = max(all_vals) if all_vals else 1.0

    for k, n in enumerate(ns):
        ax = axes[k]
        grid = np.full((len(widths), len(depths)), np.nan)
        for i, w in enumerate(widths):
            for j, d in enumerate(depths):
                rec = agg.get((n, w, d))
                grid[i, j] = value_fn(rec) if rec is not None else missing_val
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([str(d) for d in depths])
        ax.set_yticks(range(len(widths)))
        ax.set_yticklabels([str(w) for w in widths])
        ax.set_title(f"n={n}")
        ax.set_xlabel("depth")
        if k == 0:
            ax.set_ylabel("width")
        if annotate:
            for i in range(len(widths)):
                for j in range(len(depths)):
                    v = grid[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.2g}", ha="center", va="center",
                                color="white" if (v - vmin) / (vmax - vmin + 1e-9) < 0.55 else "black",
                                fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.9, label=cbar_label)
    fig.suptitle(title, y=1.03)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_final_reward_heatmaps(agg, widths, depths, ns, out_path):
    _heatmap_grid_per_n(
        agg, widths, depths, ns,
        value_fn=lambda rec: rec["final_mean"],
        title="Final eval reward (mean over seeds)",
        cbar_label="reward",
        out_path=out_path,
    )


def _steps_to_threshold(rec: dict, threshold: float) -> float:
    """First step at which the seed-averaged reward crosses `threshold`."""
    mean = rec["reward_mean"]
    hits = np.where(mean >= threshold)[0]
    if hits.size == 0:
        return np.nan
    return float(rec["steps"][hits[0]])


def plot_steps_to_threshold_heatmaps(
    agg, widths, depths, ns, threshold_pct, out_path,
):
    def value_fn_for_n(n_val):
        # Per-n threshold: fraction of the best final-reward across the grid at that n.
        best = max(
            (agg[(n_val, w, d)]["final_mean"] for w in widths for d in depths
             if (n_val, w, d) in agg),
            default=0.0,
        )
        threshold = threshold_pct * best

        def value_fn(rec):
            return _steps_to_threshold(rec, threshold) if rec is not None else np.nan
        return value_fn, threshold

    # Fallback: can't share a value_fn across n's; render manually.
    fig, axes = plt.subplots(
        1, len(ns),
        figsize=(2.8 * len(ns) + 0.6, 2.8),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    all_vals: list[float] = []
    per_n_grids: list[tuple[int, np.ndarray, float]] = []
    for n in ns:
        value_fn, thr = value_fn_for_n(n)
        grid = np.full((len(widths), len(depths)), np.nan)
        for i, w in enumerate(widths):
            for j, d in enumerate(depths):
                rec = agg.get((n, w, d))
                grid[i, j] = value_fn(rec)
        per_n_grids.append((n, grid, thr))
        all_vals.extend(v for v in grid.flatten() if not np.isnan(v))

    vmin = min(all_vals) if all_vals else 0.0
    vmax = max(all_vals) if all_vals else 1.0

    for k, (n, grid, thr) in enumerate(per_n_grids):
        ax = axes[k]
        # Lower = faster, so invert the colormap.
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis_r",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels([str(d) for d in depths])
        ax.set_yticks(range(len(widths)))
        ax.set_yticklabels([str(w) for w in widths])
        ax.set_title(f"n={n}\nthr={thr:.2f}")
        ax.set_xlabel("depth")
        if k == 0:
            ax.set_ylabel("width")
        for i in range(len(widths)):
            for j in range(len(depths)):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v / 1e6:.0f}M",
                            ha="center", va="center",
                            color="white" if (v - vmin) / (vmax - vmin + 1e-9) > 0.45 else "black",
                            fontsize=8)
                else:
                    ax.text(j, i, "—", ha="center", va="center",
                            color="white", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.9, label="env steps to threshold")
    fig.suptitle(
        f"Steps to reach {threshold_pct * 100:.0f}% of best-arch final reward (per n)",
        y=1.05,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_vs_reward(agg, widths, depths, ns, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    depth_markers = {0: "o", 1: "s", 2: "^", 3: "D"}
    for n in ns:
        color = _n_color(n, ns)
        for d in depths:
            xs: list[float] = []
            ys: list[float] = []
            yerrs: list[float] = []
            for w in widths:
                rec = agg.get((n, w, d))
                if rec is None:
                    continue
                # Capacity proxy: hidden params ≈ w*w*(d-1) + w*d; ignores
                # obs/action-dim IO weights so n-families are comparable on
                # the x-axis. For d=0 → capacity = 0 (linear baseline).
                cap = (w * w * max(d - 1, 0)) + (w * d)
                xs.append(max(cap, 0.5))  # jitter so log-scale shows d=0 at 0.5
                ys.append(rec["final_mean"])
                yerrs.append(rec["final_std"])
            if xs:
                ax.errorbar(
                    xs, ys, yerr=yerrs,
                    color=color, marker=depth_markers.get(d, "x"),
                    linestyle="-", lw=1.1, capsize=2, ms=5,
                    label=f"n={n}, d={d}" if d == depths[0] else None,
                )
    # Build a two-level legend: colour=n, marker=depth.
    from matplotlib.lines import Line2D
    n_handles = [Line2D([0], [0], color=_n_color(n, ns), lw=2, label=f"n={n}") for n in ns]
    d_handles = [Line2D([0], [0], color="gray", marker=depth_markers.get(d, "x"),
                        linestyle="", label=f"depth={d}") for d in depths]
    leg1 = ax.legend(handles=n_handles, loc="lower right", title="morphology", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=d_handles, loc="lower left", title="depth", fontsize=8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("hidden-param capacity proxy  (w²(d-1) + w·d)")
    ax.set_ylabel("final eval reward")
    ax.set_title("Capacity vs final performance")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------------- #
#                               CLI                                         #
# ------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    default_data = (
        Path(__file__).resolve().parent.parent
        / "__data__" / "undirected_locomotion_ppo_jax"
    )
    default_out = Path(__file__).resolve().parent / "__plots__"
    p.add_argument("--data-dir", type=Path, default=default_data)
    p.add_argument("--out-dir", type=Path, default=default_out)
    p.add_argument(
        "--run-filter", type=str, default=None,
        help="Substring filter on run-dir name (e.g. 'n3' to keep only insect_n n=3).",
    )
    p.add_argument(
        "--threshold-pct", type=float, default=0.80,
        help="Fraction of best-arch final reward used as the sample-efficiency "
        "threshold (default 0.80).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(f"no data dir at {args.data_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = _collect(args.data_dir, args.run_filter)
    if not runs:
        raise SystemExit(
            f"no sweep runs under {args.data_dir}"
            + (f" matching {args.run_filter!r}" if args.run_filter else "")
        )

    widths = sorted({r["w"] for r in runs})
    depths = sorted({r["d"] for r in runs})
    ns = sorted({r["n"] for r in runs})
    print(
        f"collected {len(runs)} runs: "
        f"widths={widths}  depths={depths}  n={ns}"
    )
    agg = _aggregate(runs)
    # Report per-group seed counts to flag missing cells.
    missing = []
    for n in ns:
        for w in widths:
            for d in depths:
                key = (n, w, d)
                k = agg.get(key, {}).get("n_seeds", 0)
                if k == 0:
                    missing.append(f"n={n} w={w} d={d}")
    if missing:
        print(f"warning: {len(missing)} empty (n,w,d) cells: {missing[:6]}"
              + (" ..." if len(missing) > 6 else ""))

    out = args.out_dir
    plot_learning_curves_grid(agg, widths, depths, ns, out / "learning_curves_grid.png")
    plot_final_reward_heatmaps(agg, widths, depths, ns, out / "final_reward_heatmaps.png")
    plot_steps_to_threshold_heatmaps(
        agg, widths, depths, ns, args.threshold_pct,
        out / "steps_to_threshold_heatmaps.png",
    )
    plot_capacity_vs_reward(agg, widths, depths, ns, out / "capacity_vs_reward.png")

    # Compact summary table printed to stdout + csv.
    rows = ["n,width,depth,n_seeds,final_mean,final_std"]
    for n in ns:
        for w in widths:
            for d in depths:
                rec = agg.get((n, w, d))
                if rec is None:
                    continue
                rows.append(
                    f"{n},{w},{d},{rec['n_seeds']},"
                    f"{rec['final_mean']:.4f},{rec['final_std']:.4f}"
                )
    (out / "summary.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\nwrote: {out}/{{learning_curves_grid,final_reward_heatmaps,"
          f"steps_to_threshold_heatmaps,capacity_vs_reward}}.png and summary.csv")


if __name__ == "__main__":
    main()
