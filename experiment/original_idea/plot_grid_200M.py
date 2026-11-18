"""Same-budget (200M) width x depth grid for the directed capacity sweep.

Combines
  * __data__/directed_capacity_sweep_200M           w16/w64 x d1/d3 + w256 d1
  * __data__/directed_capacity_sweep_linear300M     linear (w16 d0) and w256 d3,
    truncated to the first 34 evals (identical eval grid, constant LR, so the
    first 200M of a 300M run is a 200M run)
  * __data__/directed_capacity_n1linear_continue    n=1 linear continued from
    its 300M params to ~600M (shown separately; Adam state was reset)

Metric: targets reached / episode (comparable across n; shaped reward is not).
Score per run: mean over evals in the last 25% of training.
Convergence (stricter than plot_sweep_stats): least-squares fit over that same
last 25%; a cell is flagged "↑ still improving" when the fitted gain across
the window exceeds max(5% of the cell mean, 0.05 targets).

Outputs in __plots_directed_200M__/:
    targets_3d_n{N}.png, targets_by_capacity.png, learning_curves.png,
    n1_linear_continuation.png, cells.csv, stats.txt

Usage:
    experiment/mujoco_playground/.venv/bin/python experiment/original_idea/plot_grid_200M.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_sweep_stats import interaction_test, load_runs, plot_3d, t_ci_half, welch

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "__data__"
N_EVALS = 34
WIN = 0.25
TOL_REL, TOL_ABS = 0.05, 0.05
KEY = "targets_reached"
LABEL = "targets reached / episode"


def truncate(run, n_evals):
    out = dict(run)
    for k in ("steps", "reward", "targets_reached"):
        out[k] = run[k][:n_evals]
    return out


def run_score(run):
    x, y = run["steps"], run[KEY]
    m = x >= (1 - WIN) * x[-1]
    xs, ys = x[m], y[m]
    gain = np.polyfit(xs, ys, 1)[0] * (xs[-1] - xs[0])
    return float(ys.mean()), float(gain)


def build_cells(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[(r["n"], r["w"], r["d"])].append(r)
    cells = {}
    for key, rs in groups.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        sc, gn = map(np.array, zip(*(run_score(r) for r in rs)))
        mean = float(sc.mean())
        cells[key] = {
            "seeds": [r["seed"] for r in rs], "scores": sc, "mean": mean,
            "sd": float(sc.std(ddof=1)), "ci": t_ci_half(sc),
            "gain": float(gn.mean()), "seed_gains": gn,
            "improving": bool(gn.mean() > max(TOL_REL * abs(mean), TOL_ABS)),
        }
    return cells


def arch_name(w, d):
    return "linear" if d == 0 else f"w{w} d{d}"


def plot_by_capacity(cells, ns, out):
    fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 3.8), sharey=True)
    widths = sorted({w for (_, w, d) in cells if d > 0})
    for ax, n in zip(axes, ns):
        for d, col in ((1, "tab:blue"), (3, "tab:orange")):
            ws = [w for w in widths if (n, w, d) in cells]
            m = [cells[(n, w, d)]["mean"] for w in ws]
            ci = [cells[(n, w, d)]["ci"] for w in ws]
            ax.errorbar(ws, m, yerr=ci, color=col, marker="o", capsize=3, label=f"depth {d}")
            for w in ws:
                c = cells[(n, w, d)]
                ax.scatter([w] * len(c["scores"]), c["scores"], color=col, s=10, alpha=0.5)
                if c["improving"]:
                    ax.annotate("↑", (w, c["mean"]), xytext=(4, 4),
                                textcoords="offset points", color=col)
        c = cells.get((n, 16, 0))
        if c:
            ax.errorbar([16], [c["mean"]], yerr=[c["ci"]], color="k", marker="s",
                        capsize=3, label="linear", ls="none")
            ax.scatter([16] * 3, c["scores"], color="k", s=10, alpha=0.5)
            if c["improving"]:
                ax.annotate("↑", (16, c["mean"]), xytext=(-12, 4),
                            textcoords="offset points")
        ax.set_xscale("log", base=2)
        ax.set_xticks(widths, [str(w) for w in widths])
        ax.set_title(f"insect_n n={n}  ({2 * (n + 1)} legs)")
        ax.set_xlabel("policy width")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(LABEL)
    axes[0].legend(fontsize=8)
    fig.suptitle("200M steps, score = mean of last 25% of evals · bars = 95% t-CI (3 seeds) · ↑ = still improving",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_curves(runs, ns, archs, out):
    fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 3.8), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, n in zip(axes, ns):
        for i, (w, d) in enumerate(archs):
            col = "k" if d == 0 else cmap(i / (len(archs) - 1))
            rs = [r for r in runs if (r["n"], r["w"], r["d"]) == (n, w, d)]
            if not rs:
                continue
            Y = np.stack([r[KEY] for r in rs])
            x = rs[0]["steps"] / 1e6
            ax.plot(x, Y.mean(0), color=col, lw=1.8, label=arch_name(w, d))
            ax.fill_between(x, Y.min(0), Y.max(0), color=col, alpha=0.12)
        ax.axvspan((1 - WIN) * x[-1], x[-1], color="0.85", alpha=0.5, lw=0)
        ax.set_title(f"n={n}")
        ax.set_xlabel("env steps (M)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(LABEL)
    axes[-1].legend(fontsize=7)
    fig.suptitle("Learning curves (mean over 3 seeds, band = min–max) · grey = scoring window", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def load_continuation():
    out = []
    for sub in sorted((DATA / "directed_capacity_n1linear_continue").glob("insect_n_n1_w16_d0_seed*")):
        h = json.loads((sub / "training_history.json").read_text())
        off = json.loads((sub / "restored_from.json").read_text())["step_offset"]
        seed = int(sub.name.split("seed")[1].split("-")[0])
        out.append({"seed": seed, "steps": np.asarray(h["steps"], float) + off,
                    KEY: np.asarray(h[KEY], float)})
    return out


def plot_continuation(parents, conts, ref, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    cols = plt.get_cmap("tab10")
    for p in sorted(parents, key=lambda r: r["seed"]):
        c = next(c for c in conts if c["seed"] == p["seed"])
        col = cols(p["seed"])
        ax.plot(p["steps"] / 1e6, p[KEY], color=col, lw=1.2, label=f"seed {p['seed']}")
        ax.plot(c["steps"] / 1e6, c[KEY], color=col, lw=1.2, ls="--")
    ax.axvline(parents[0]["steps"][-1] / 1e6, color="0.5", lw=0.8)
    ax.text(parents[0]["steps"][-1] / 1e6, ax.get_ylim()[1] * 0.97, " restart (Adam reset)",
            fontsize=7, va="top", color="0.4")
    if ref:
        ax.axhspan(ref["mean"] - ref["ci"], ref["mean"] + ref["ci"], color="tab:green", alpha=0.15)
        ax.axhline(ref["mean"], color="tab:green", lw=1, label="w256 d3 @200M (95% CI)")
    ax.set_xlabel("env steps (M)")
    ax.set_ylabel(LABEL)
    ax.set_title("n=1 linear policy: 300M run (solid) + continuation (dashed)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "__plots_directed_200M__")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)

    grid = load_runs(DATA / "directed_capacity_sweep_200M")
    long = [truncate(r, N_EVALS) for r in load_runs(DATA / "directed_capacity_sweep_linear300M")]
    runs = grid + long
    assert all(len(r["steps"]) == N_EVALS for r in runs), "eval grids differ"
    cells = build_cells(runs)

    ns = sorted({r["n"] for r in runs})
    widths = sorted({r["w"] for r in runs})
    depths = sorted({r["d"] for r in runs})
    archs = sorted({(w, d) for (_, w, d) in cells}, key=lambda a: (a[1] > 0, a[0] * max(a[1], 1), a[1]))

    top = max(c["mean"] + c["ci"] for c in cells.values())
    zlim = (0.0, float(np.ceil(top + 0.5)))
    for n in ns:
        plot_3d(cells, n, widths, depths, zlim, LABEL,
                f"insect_n n={n} · 200M steps · targets/episode (last 25% of evals)",
                args.out_dir / f"targets_3d_n{n}.png")
    plot_by_capacity(cells, ns, args.out_dir / "targets_by_capacity.png")
    plot_curves(runs, ns, archs, args.out_dir / "learning_curves.png")

    parents = [r for r in load_runs(DATA / "directed_capacity_sweep_linear300M") if (r["n"], r["d"]) == (1, 0)]
    conts = load_continuation()
    plot_continuation(parents, conts, cells.get((1, 256, 3)), args.out_dir / "n1_linear_continuation.png")

    # ---- tables & tests ----
    lines = ["cell scores: mean of evals in last 25% of 200M, 95% t-CI over 3 seeds",
             f"{'n':>2} {'arch':>9} {'mean':>6} {'±CI':>6} {'sd':>5}  seeds               gain/window  flag"]
    with open(args.out_dir / "cells.csv", "w") as f:
        f.write("n,width,depth,mean,ci95,sd,seed_scores,gain_window,improving\n")
        for key in sorted(cells, key=lambda k: (k[0], archs.index(k[1:]))):
            c = cells[key]
            n, w, d = key
            seeds = " ".join(f"{s:.2f}" for s in c["scores"])
            f.write(f"{n},{w},{d},{c['mean']:.3f},{c['ci']:.3f},{c['sd']:.3f},{seeds},{c['gain']:.3f},{int(c['improving'])}\n")
            lines.append(f"{n:>2} {arch_name(w, d):>9} {c['mean']:6.2f} {c['ci']:6.2f} {c['sd']:5.2f}  "
                         f"{seeds:<18}  {c['gain']:+7.2f}     {'↑' if c['improving'] else ''}")
        lines.append("")
    nflag = sum(c["improving"] for c in cells.values())
    lines.append(f"still improving: {nflag}/{len(cells)} cells")
    lines.append("")

    lines.append("best vs worst non-linear arch per n, and each arch vs linear (Welch t)")
    for n in ns:
        lin = cells[(n, 16, 0)]
        for (w, d) in archs:
            if d == 0 or (n, w, d) not in cells:
                continue
            t, p = welch(cells[(n, w, d)]["scores"], lin["scores"])
            lines.append(f"  n={n} {arch_name(w, d):>9} - linear = {cells[(n, w, d)]['mean'] - lin['mean']:+.2f}  t={t:+.2f} p={p:.3f}")
    lines.append("")

    it = interaction_test(cells, ns, 256, 3, 16)
    lines.append("interaction y ~ log2(n) * linear, linear vs w256 d3, per-seed scores @200M:")
    lines.append(f"  b3 = {it['b3']:+.3f}  95% CI [{it['lo']:+.3f}, {it['hi']:+.3f}]  p={it['p']:.4f}  "
                 f"(slope w256d3 {it['slope_large']:+.3f}/doubling, linear {it['slope_linear']:+.3f}/doubling, N={it['N']})")
    lines.append("")

    lines.append("n=1 linear continuation (mean of last 25% of continuation evals):")
    for c in sorted(conts, key=lambda r: r["seed"]):
        m = c["steps"] >= c["steps"][0] + (1 - WIN) * (c["steps"][-1] - c["steps"][0])
        p = next(p for p in parents if p["seed"] == c["seed"])
        pm = p["steps"] >= (1 - WIN) * p["steps"][-1]
        lines.append(f"  seed {c['seed']}: 300M {p[KEY][pm].mean():.2f} -> ~{c['steps'][-1] / 1e6:.0f}M {c[KEY][m].mean():.2f}")
    text = "\n".join(lines)
    (args.out_dir / "stats.txt").write_text(text + "\n")
    print(text)
    print(f"\nwrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
