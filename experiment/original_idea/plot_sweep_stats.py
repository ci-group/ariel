"""Statistically careful analysis of the directed capacity sweep.

Companion to plot_sweep.py / plot_sweep_3d.py. Those use the *last* eval
point of the *shaped reward*; this script addresses the problems with that:

1. Metric. The shaped reward is not comparable across morphologies: the
   feet_air_time, torques, joint_vel and action_rate terms all sum over
   feet/actuators, so their magnitude scales with n. The primary metric here
   is ``targets_reached`` (targets reached per eval episode, mean over the
   eval envs), which measures task success in the same units for every n.
   Reward plots are still produced, but labelled as within-n only.

2. Endpoint noise. Each eval is one draw of eval envs at one checkpoint.
   A run's score is the mean of its last K evals (default 4, i.e. the final
   ~20% of training), not the single last eval.

3. Seed uncertainty. Cells show the mean over seeds, the individual seeds
   (dots) and a 95% t-interval (with 3 seeds t_{0.975,2} = 4.30, so the
   intervals are wide, and should be).

4. Convergence. Cells whose seeds were still improving at the end are
   flagged with "↑": a least-squares fit over the last K evals predicts a
   gain of more than --converge-tol (default 5%) of the cell mean across
   that window. For flagged cells the number is a lower bound, and
   comparisons involving them partly measure learning speed.

5. Measured cells only. The linear policy (depth 0) was only run at width
   16; it is drawn only there (no copying across widths).

6. The "gap" claim. Whether the advantage of a large network over a linear
   policy shrinks with n is tested on a pre-specified comparison (the
   largest arch, w=max d=max, vs linear) rather than the post-hoc best
   cell, using OLS with an interaction term:
       y = b0 + b1·log2(n) + b2·linear + b3·log2(n)·linear
   b3 > 0 means the linear policy gains more with n than the large network.
   Sensitivity to K and to excluding non-learning seeds is reported.

Outputs (in --out-dir):
    targets_3d_n{N}.png        3D bars, targets/episode, shared z across n
    reward_3d_n{N}.png         3D bars, shaped reward (compare within n only)
    targets_by_capacity.png    2D: targets vs width, one line per depth, panel per n
    linear_vs_largest.png      2D: the gap comparison with seeds and CIs
    cells.csv                  per (n, w, d) stats
    stats.txt                  test results and sensitivity analyses

Usage
-----
    experiment/mujoco_playground/.venv/bin/python experiment/original_idea/plot_sweep_stats.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from scipy import stats

from plot_sweep import RUN_RE

import json

METRICS = {
    "targets": ("targets_reached", "targets reached / episode"),
    "reward": ("reward", "shaped eval reward"),
}


# ------------------------------------------------------------------------- #
#                               Data                                        #
# ------------------------------------------------------------------------- #

def load_runs(data_dir: Path) -> list[dict]:
    runs = []
    for sub in sorted(data_dir.iterdir()):
        m = RUN_RE.match(sub.name)
        hist_path = sub / "training_history.json"
        if m is None or not hist_path.exists():
            continue
        h = json.loads(hist_path.read_text())
        if not h.get("steps"):
            continue
        n, w, d, s = (int(x) for x in m.groups())
        runs.append({
            "n": n, "w": w, "d": d, "seed": s,
            "steps": np.asarray(h["steps"], float),
            "reward": np.asarray(h["reward"], float),
            "targets_reached": np.asarray(h["targets_reached"], float),
        })
    return runs


def run_score(run: dict, key: str, k: int) -> tuple[float, float]:
    """(mean of last k evals, least-squares gain predicted across that window)."""
    y = run[key][-k:]
    x = run["steps"][-k:]
    gain = np.polyfit(x, y, 1)[0] * (x[-1] - x[0]) if k > 1 else 0.0
    return float(y.mean()), float(gain)


def t_ci_half(vals: np.ndarray, level: float = 0.95) -> float:
    if len(vals) < 2:
        return np.nan
    return float(stats.t.ppf(0.5 + level / 2, len(vals) - 1)
                 * vals.std(ddof=1) / np.sqrt(len(vals)))


def cell_stats(runs: list[dict], key: str, k: int, tol: float) -> dict:
    groups: dict[tuple, list] = defaultdict(list)
    for r in runs:
        groups[(r["n"], r["w"], r["d"])].append(r)
    out = {}
    for cell, rs in groups.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        scores, gains = zip(*(run_score(r, key, k) for r in rs))
        scores, gains = np.array(scores), np.array(gains)
        mean = scores.mean()
        out[cell] = {
            "seeds": [r["seed"] for r in rs],
            "scores": scores,
            "mean": float(mean),
            "sd": float(scores.std(ddof=1)) if len(scores) > 1 else np.nan,
            "ci": t_ci_half(scores),
            "gain": float(gains.mean()),
            "improving": bool(gains.mean() > tol * max(abs(mean), 1e-9)),
        }
    return out


# ------------------------------------------------------------------------- #
#                               3D bars                                     #
# ------------------------------------------------------------------------- #

def plot_3d(cells, n, widths, depths, zlim, label, title, out_path):
    fig = plt.figure(figsize=(7.8, 6.2))
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False  # draw back-to-front by hand (see below)
    norm = colors.Normalize(*zlim)
    cmap = plt.get_cmap("viridis")
    bar = 0.55

    # View azim=-130: (width[0], depth[0]) is nearest the viewer.
    order = sorted(
        ((i, w, j, d) for i, w in enumerate(widths) for j, d in enumerate(depths)),
        key=lambda c: -(c[0] + c[2]),
    )
    texts = []
    for z_order, (i, w, j, d) in enumerate(order):
        c = cells.get((n, w, d))
        if c is None:
            continue
        zo = 10 * z_order
        m = c["mean"]
        ax.bar3d(i - bar / 2, j - bar / 2, 0.0, bar, bar, m,
                 color=cmap(norm(m)), alpha=0.55, shade=True,
                 edgecolor="k", linewidth=0.3, zorder=zo)
        # 95% CI whisker and individual seeds, drawn in front of their own bar.
        if not np.isnan(c["ci"]):
            ax.plot([i, i], [j, j], [m - c["ci"], m + c["ci"]],
                    color="crimson", lw=2, zorder=zo + 1)
            for dz in (-c["ci"], c["ci"]):
                ax.plot([i - 0.08, i + 0.08], [j, j], [m + dz] * 2,
                        color="crimson", lw=2, zorder=zo + 1)
        ax.scatter(np.full(len(c["scores"]), i + 0.12), np.full(len(c["scores"]), j),
                   c["scores"], color="k", s=10, depthshade=False, zorder=zo + 2)
        texts.append((i, j, max(m + (0 if np.isnan(c["ci"]) else c["ci"]), 0.0), c))

    for i, j, z, c in texts:
        flag = " ↑" if c["improving"] else ""
        ax.text(i, j, z, f"{c['mean']:.2f}{flag}\n±{c['ci']:.2f}",
                ha="center", va="bottom", fontsize=7, zorder=10_000,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels(["0 (linear)" if d == 0 else str(d) for d in depths])
    ax.set_xlabel("network width")
    ax.set_ylabel("network depth")
    ax.set_zlabel(label)
    ax.set_zlim(*zlim)
    ax.view_init(elev=25, azim=-130)
    ax.set_title(title, fontsize=10)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 shrink=0.55, pad=0.1, label=label)
    fig.text(0.02, 0.02,
             "bar = mean over seeds · red = 95% t-CI · dots = seeds · "
             "↑ = still improving at end of training (value is a lower bound)",
             fontsize=7, color="0.3")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------- #
#                               2D figures                                  #
# ------------------------------------------------------------------------- #

def plot_by_capacity(cells, ns, widths, depths, label, k, out_path):
    fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    hidden = [d for d in depths if d > 0]
    dcol = {d: plt.get_cmap("tab10")(q) for q, d in enumerate(hidden)}
    xs = np.log2(widths)
    for ax, n in zip(axes, ns):
        lin = next((cells[(n, w, 0)] for w in widths if (n, w, 0) in cells), None)
        if lin is not None:
            ax.axhspan(lin["mean"] - lin["ci"], lin["mean"] + lin["ci"],
                       color="0.6", alpha=0.25, lw=0)
            ax.axhline(lin["mean"], color="0.35", ls="--", lw=1.2,
                       label="linear (d=0)")
            ax.scatter(np.full(len(lin["scores"]), xs[0] - 0.25), lin["scores"],
                       color="0.35", s=12, marker="x")
        for q, d in enumerate(hidden):
            off = (q - (len(hidden) - 1) / 2) * 0.12
            pts = [(x, cells[(n, w, d)]) for x, w in zip(xs, widths) if (n, w, d) in cells]
            if not pts:
                continue
            px = np.array([p[0] for p in pts]) + off
            ax.errorbar(px, [p[1]["mean"] for p in pts],
                        yerr=[p[1]["ci"] for p in pts], color=dcol[d],
                        marker="o", capsize=3, lw=1.5, label=f"depth {d}")
            for x, c in zip(px, (p[1] for p in pts)):
                ax.scatter(np.full(len(c["scores"]), x), c["scores"],
                           color=dcol[d], s=9, alpha=0.5, zorder=3)
                if c["improving"]:
                    ax.annotate("↑", (x, c["mean"] + c["ci"]), ha="center",
                                va="bottom", fontsize=9, color=dcol[d])
        ax.set_xticks(xs)
        ax.set_xticklabels([str(w) for w in widths])
        ax.set_xlabel("network width")
        ax.set_title(f"n={n}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(f"{label}\n(mean of last {k} evals)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Capacity vs task success — mean ± 95% t-CI over seeds "
                 "(dots/×: seeds, band: linear CI, ↑: still improving)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_linear_vs_largest(cells, ns, wmax, dmax, w_lin, label, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, key_fn, col, off in (
        ("linear (d=0)", lambda n: (n, w_lin, 0), "0.4", -0.08),
        (f"w={wmax}, d={dmax}", lambda n: (n, wmax, dmax), "tab:blue", 0.08),
    ):
        pts = [(q, cells[key_fn(n)]) for q, n in enumerate(ns) if key_fn(n) in cells]
        x = np.array([p[0] for p in pts]) + off
        ax.errorbar(x, [p[1]["mean"] for p in pts], yerr=[p[1]["ci"] for p in pts],
                    color=col, marker="o", capsize=4, lw=1.6, label=name)
        for xi, (_, c) in zip(x, pts):
            ax.scatter(np.full(len(c["scores"]), xi), c["scores"], color=col,
                       s=14, alpha=0.6, zorder=3)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f"n={n}" for n in ns])
    ax.set_ylabel(label)
    ax.set_title("Linear policy vs largest network across morphologies\n"
                 "mean ± 95% t-CI, dots = seeds", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------- #
#                               Statistics                                  #
# ------------------------------------------------------------------------- #

def interaction_test(cells, ns, wmax, dmax, w_lin, drop_dead=False):
    """OLS y ~ log2(n) * is_linear on per-seed scores; returns dict for b3."""
    X, y = [], []
    for n in ns:
        for is_lin, key in ((1, (n, w_lin, 0)), (0, (n, wmax, dmax))):
            c = cells.get(key)
            if c is None:
                continue
            for s in c["scores"]:
                if drop_dead and is_lin and s < 0.05:
                    continue
                x = np.log2(n)
                X.append([1.0, x, is_lin, x * is_lin])
                y.append(s)
    X, y = np.asarray(X), np.asarray(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    tval = beta[3] / se[3]
    p = 2 * stats.t.sf(abs(tval), dof)
    half = stats.t.ppf(0.975, dof) * se[3]
    return {"b3": beta[3], "lo": beta[3] - half, "hi": beta[3] + half,
            "p": p, "dof": dof, "N": len(y),
            "slope_large": beta[1], "slope_linear": beta[1] + beta[3]}


def welch(a, b):
    r = stats.ttest_ind(a, b, equal_var=False)
    return float(r.statistic), float(r.pvalue)


def main() -> None:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path,
                   default=root.parent / "__data__" / "directed_capacity_sweep")
    p.add_argument("--out-dir", type=Path, default=root / "__plots_directed_stats__")
    p.add_argument("--last-k", type=int, default=4,
                   help="Average each run over its last K evals (default 4).")
    p.add_argument("--converge-tol", type=float, default=0.05,
                   help="Flag a cell as still improving if the fitted gain over the "
                        "last K evals exceeds this fraction of its mean (default 0.05).")
    args = p.parse_args()

    runs = load_runs(args.data_dir)
    if not runs:
        raise SystemExit(f"no sweep runs under {args.data_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ns = sorted({r["n"] for r in runs})
    widths = sorted({r["w"] for r in runs})
    depths = sorted({r["d"] for r in runs})
    wmax, dmax = max(widths), max(depths)
    w_lin = min(r["w"] for r in runs if r["d"] == 0)
    k = args.last_k
    first_eval = runs[0]["steps"][-k] / 1e6
    last_eval = runs[0]["steps"][-1] / 1e6

    all_cells = {}
    for mname, (key, label) in METRICS.items():
        cells = cell_stats(runs, key, k, args.converge_tol)
        all_cells[mname] = cells
        hi = max(c["mean"] + np.nan_to_num(c["ci"]) for c in cells.values())
        lo = min(0.0, min(c["mean"] - np.nan_to_num(c["ci"]) for c in cells.values()))
        hi = max(hi, max(s for c in cells.values() for s in c["scores"]))
        zlim = (lo, hi * 1.1)
        for n in ns:
            note = ("comparable across n" if mname == "targets"
                    else "NOT comparable across n — terms scale with #legs")
            plot_3d(
                cells, n, widths, depths, zlim, label,
                f"Directed capacity sweep — insect_n n={n}\n{label}, mean of last {k} "
                f"evals ({first_eval:.0f}–{last_eval:.0f}M steps)\n[{note}]",
                args.out_dir / f"{mname}_3d_n{n}.png",
            )

    tcells = all_cells["targets"]
    tlabel = METRICS["targets"][1]
    plot_by_capacity(tcells, ns, widths, depths, tlabel, k,
                     args.out_dir / "targets_by_capacity.png")
    plot_linear_vs_largest(tcells, ns, wmax, dmax, w_lin, tlabel,
                           args.out_dir / "linear_vs_largest.png")

    # ---- cells.csv --------------------------------------------------------
    rows = ["metric,n,width,depth,n_seeds,mean,sd,ci95_half,still_improving,seed_scores"]
    for mname, cells in all_cells.items():
        for (n, w, d), c in sorted(cells.items()):
            rows.append(f"{mname},{n},{w},{d},{len(c['scores'])},{c['mean']:.4f},"
                        f"{c['sd']:.4f},{c['ci']:.4f},{int(c['improving'])},"
                        + " ".join(f"{s:.4f}" for s in c["scores"]))
    (args.out_dir / "cells.csv").write_text("\n".join(rows) + "\n")

    # ---- stats.txt --------------------------------------------------------
    L = [f"Directed capacity sweep — {len(runs)} runs, n={ns}, widths={widths}, "
         f"depths={depths}",
         f"Primary metric: {tlabel}; per-run score = mean of last {k} evals "
         f"({first_eval:.0f}–{last_eval:.0f}M steps).", ""]

    L.append(f"Pre-specified comparison: linear (w={w_lin}, d=0) vs largest "
             f"(w={wmax}, d={dmax}).")
    for n in ns:
        a, b = tcells.get((n, wmax, dmax)), tcells.get((n, w_lin, 0))
        if a is None or b is None:
            continue
        diff = a["mean"] - b["mean"]
        t, pv = welch(a["scores"], b["scores"])
        L.append(f"  n={n}: largest {a['mean']:.2f} ±{a['ci']:.2f}   linear "
                 f"{b['mean']:.2f} ±{b['ci']:.2f}   gap {diff:.2f}  "
                 f"(Welch t={t:.2f}, p={pv:.3g})   linear/largest = "
                 f"{b['mean'] / a['mean']:.2f}")
    L.append("")
    lin_scores = {n: tcells[(n, w_lin, 0)]["scores"] for n in ns if (n, w_lin, 0) in tcells}
    if len(lin_scores) >= 2:
        n_lo, n_hi = min(lin_scores), max(lin_scores)
        t, pv = welch(lin_scores[n_hi], lin_scores[n_lo])
        L.append(f"Linear policy, n={n_hi} vs n={n_lo}: Welch t={t:.2f}, p={pv:.3g}")
        L.append("")

    L.append("Interaction test  y ~ log2(n) * is_linear  (OLS on per-seed scores).")
    L.append("b3 > 0  <=>  the linear policy improves more with n than the largest network,")
    L.append("i.e. the capacity advantage shrinks as n grows.")
    for kk in sorted({1, 2, k, 6} & set(range(1, len(runs[0]['steps']) + 1))):
        for drop in (False, True):
            cs = cell_stats(runs, "targets_reached", kk, args.converge_tol)
            r = interaction_test(cs, ns, wmax, dmax, w_lin, drop_dead=drop)
            tag = f"K={kk}" + (", non-learning linear seeds dropped" if drop else "")
            L.append(f"  [{tag:<42}] b3={r['b3']:+.3f} per doubling of n  "
                     f"95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  p={r['p']:.3g}  "
                     f"(N={r['N']}, dof={r['dof']})   slope: largest {r['slope_large']:+.3f}, "
                     f"linear {r['slope_linear']:+.3f}")
    L.append("  (non-learning = linear seed with score < 0.05 targets/episode)")
    L.append("")

    flagged = [(key, c) for key, c in sorted(tcells.items()) if c["improving"]]
    L.append(f"Cells still improving at end of training (fitted gain over last {k} evals "
             f"> {args.converge_tol:.0%} of mean): {len(flagged)}/{len(tcells)}")
    for (n, w, d), c in flagged:
        L.append(f"  n={n} w={w} d={d}: mean {c['mean']:.2f}, fitted gain "
                 f"{c['gain']:+.2f} over window")
    L.append("")
    L.append("Caveats: 3 seeds per cell (t_{0.975,2}=4.30 -> wide CIs); n has only 3 "
             "levels, so the trend in n is not resolved beyond 'monotone over 1,2,4'; "
             "the largest arch is the pre-specified comparison, other arch choices "
             "are exploratory.")
    text = "\n".join(L)
    (args.out_dir / "stats.txt").write_text(text + "\n")
    print(text)
    print(f"\nwrote figures, cells.csv, stats.txt to {args.out_dir}")


if __name__ == "__main__":
    main()
