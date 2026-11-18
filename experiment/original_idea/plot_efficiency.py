"""Efficiency vs efficacy for the 200M directed capacity grid.

Efficacy  = final performance: mean targets/episode over the last 25% of the
            200M budget (same score as plot_grid_200M.py).
Efficiency, two kinds:
  * sample efficiency  - env steps until the run first reaches a fixed task
                         threshold (default 2.0 targets/episode, on a 3-eval
                         moving average so a single lucky eval doesn't count);
                         also the area under the learning curve (mean targets
                         over all 200M steps, "anytime performance").
  * compute efficiency - wall-clock seconds to the same threshold, from the
                         'elapsed=' column of the training logs (one GPU, runs
                         sequential). Runs that shared the GPU with the
                         distillation job (2026-11-17 15:41-16:45) are excluded
                         from wall-clock numbers only.

Runs that never reach the threshold are censored: drawn at the right edge with
an open marker and excluded from the cell mean (the count reaching is printed).

Outputs in __plots_directed_200M__/:
    efficiency_vs_efficacy.png  rows: steps-to-threshold, wall-clock-to-threshold
    anytime_vs_final.png        AUC (mean over training) vs final score
    efficiency.csv

Usage:
    experiment/mujoco_playground/.venv/bin/python experiment/original_idea/plot_efficiency.py
"""

import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_grid_200M import (DATA, HERE, KEY, N_EVALS, TOL_ABS, TOL_REL, WIN, arch_name,
                            run_score, truncate)

CONV_FRAC = 0.9
from plot_sweep import RUN_RE
from plot_sweep_stats import t_ci_half

LINE_RE = re.compile(r"^\[\s*(\d+)\].*elapsed=\s*([\d.]+)s")
GPU_SHARED = (datetime(2026, 11, 17, 15, 41, 15), datetime(2026, 11, 17, 16, 45, 14))
ARCHS = [(16, 0), (16, 1), (16, 3), (64, 1), (64, 3), (256, 1), (256, 3)]


def load(data_dir):
    runs = []
    for sub in sorted(data_dir.iterdir()):
        m = RUN_RE.match(sub.name)
        hist = sub / "training_history.json"
        if m is None or not hist.exists():
            continue
        n, w, d, s = (int(x) for x in m.groups())
        h = __import__("json").loads(hist.read_text())
        log = data_dir / "_logs" / f"insect_n_n{n}_w{w}_d{d}_seed{s}.log"
        el = [float(mm.group(2)) for line in log.read_text().splitlines()
              if (mm := LINE_RE.match(line))]
        start = datetime.strptime(sub.name.rsplit("-", 2)[-2] + sub.name.rsplit("-", 1)[-1], "%Y%m%d%H%M%S")
        end = start.timestamp() + (el[-1] if el else 0)
        shared = start < GPU_SHARED[1] and end > GPU_SHARED[0].timestamp()
        runs.append(truncate({
            "n": n, "w": w, "d": d, "seed": s,
            "steps": np.asarray(h["steps"], float),
            "reward": np.asarray(h["reward"], float),
            KEY: np.asarray(h[KEY], float),
            "elapsed": np.asarray(el, float),
            "gpu_shared": shared,
        }, N_EVALS) | {"elapsed": np.asarray(el[:N_EVALS], float)})
    return runs


def first_crossing(x, y, thr):
    """Interpolated x where the 3-eval moving average first reaches thr (nan if never)."""
    ys = np.convolve(y, np.ones(3) / 3, mode="valid")
    xs = x[2:]
    idx = np.nonzero(ys >= thr)[0]
    if not len(idx):
        return np.nan
    i = idx[0]
    if i == 0:
        return float(xs[0])
    f = (thr - ys[i - 1]) / (ys[i] - ys[i - 1])
    return float(xs[i - 1] + f * (xs[i] - xs[i - 1]))


def per_run(r, thr):
    x, y = r["steps"], r[KEY]
    final = float(y[x >= (1 - WIN) * x[-1]].mean())
    auc = float(np.trapezoid(y, x) / (x[-1] - x[0]))
    t_steps = first_crossing(x, y, thr)
    t_wall = np.nan
    if not np.isnan(t_steps) and not r["gpu_shared"] and len(r["elapsed"]) == len(x):
        t_wall = float(np.interp(t_steps, x, r["elapsed"]))
    # convergence: first time the 3-eval moving average reaches CONV_FRAC of the run's own final
    t_conv = first_crossing(x, y, CONV_FRAC * final) if final >= 0.1 else np.nan
    w_conv = (float(np.interp(t_conv, x, r["elapsed"]))
              if not np.isnan(t_conv) and not r["gpu_shared"] and len(r["elapsed"]) == len(x) else np.nan)
    _, gain = run_score(r)
    return {"final": final, "auc": auc, "steps_thr": t_steps, "wall_thr": t_wall,
            "steps_conv": t_conv, "wall_conv": w_conv,
            "improving": bool(gain > max(TOL_REL * final, TOL_ABS)),
            "gpu_shared": r["gpu_shared"],
            "wall_total": float(r["elapsed"][-1]) if not r["gpu_shared"] else np.nan}


def summarise(vals):
    v = np.asarray([x for x in vals if not np.isnan(x)])
    if not len(v):
        return np.nan, np.nan, 0
    return float(v.mean()), (t_ci_half(v) if len(v) > 1 else np.nan), len(v)


def colours():
    cmap = plt.get_cmap("viridis")
    return {a: ("k" if a[1] == 0 else cmap(i / (len(ARCHS) - 1))) for i, a in enumerate(ARCHS)}


def marker(a):
    return {0: "s", 1: "o", 3: "^"}[a[1]]


def scatter_panel(ax, cells, n, xkey, xscale, xmax, col):
    for a in ARCHS:
        rs = cells.get((n,) + a, [])
        if xkey == "wall_thr":  # timing of GPU-shared runs is invalid: drop, don't censor
            rs = [r for r in rs if not r["gpu_shared"]]
        if not rs:
            continue
        xs = np.array([r[xkey] for r in rs]) * xscale
        ys = np.array([r["final"] for r in rs])
        reached = ~np.isnan(xs)
        ax.scatter(xs[reached], ys[reached], color=col[a], marker=marker(a), s=45,
                   edgecolors="k", linewidths=0.5)
        ax.scatter(np.full((~reached).sum(), xmax), ys[~reached], facecolors="none",
                   edgecolors=col[a], marker=marker(a), s=45, lw=1.4)
    ax.axvline(xmax, color="0.6", ls=":", lw=0.8)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--out-dir", type=Path, default=HERE / "__plots_directed_200M__")
    args = ap.parse_args()
    args.out_dir.mkdir(exist_ok=True)
    thr = args.threshold

    runs = load(DATA / "directed_capacity_sweep_200M") + load(DATA / "directed_capacity_sweep_linear300M")
    cells = defaultdict(list)
    for r in runs:
        cells[(r["n"], r["w"], r["d"])].append(per_run(r, thr) | {"seed": r["seed"]})
    ns = sorted({k[0] for k in cells})
    col = colours()
    budget = max(r["steps"][-1] for r in runs)
    wall_max = np.nanmax([x["wall_total"] for v in cells.values() for x in v])

    # ---- figure 1: efficiency (steps, wall-clock) vs efficacy ----
    fig, axes = plt.subplots(2, len(ns), figsize=(4.3 * len(ns), 7.6), sharey=True)
    for j, n in enumerate(ns):
        scatter_panel(axes[0, j], cells, n, "steps_thr", 1e-6, budget / 1e6, col)
        scatter_panel(axes[1, j], cells, n, "wall_thr", 1 / 60, wall_max / 60, col)
        axes[0, j].set_title(f"n={n}  ({2 * (n + 1)} legs)")
        axes[0, j].set_xlabel(f"env steps to reach {thr:g} targets/ep (M)")
        axes[1, j].set_xlabel(f"wall-clock to reach {thr:g} targets/ep (min)")
        dropped = [f"{arch_name(*k[1:])} ×{sum(r['gpu_shared'] for r in v)}" for k, v in cells.items()
                   if k[0] == n and any(r["gpu_shared"] for r in v)]
        if dropped:
            axes[1, j].text(0.98, 0.02, "GPU-shared, omitted: " + ", ".join(dropped),
                            transform=axes[1, j].transAxes, ha="right", fontsize=7, color="0.4")
    for i in range(2):
        axes[i, 0].set_ylabel("final targets / episode")
    handles = [Line2D([], [], color=col[a], marker=marker(a), ls="none", mec="k", mew=0.6,
                      label=arch_name(*a)) for a in ARCHS]
    handles.append(Line2D([], [], color="0.4", marker="o", mfc="none", ls="none",
                          label="never reached (at right edge)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle("Efficiency vs efficacy · top-left is better · one point per run (3 seeds per architecture)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(args.out_dir / "efficiency_vs_efficacy.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: anytime (AUC) vs final ----
    fig, axes = plt.subplots(1, len(ns), figsize=(4.3 * len(ns), 4.0), sharex=True, sharey=True)
    for ax, n in zip(axes, ns):
        for a in ARCHS:
            rs = cells.get((n,) + a)
            if not rs:
                continue
            xs = np.array([r["auc"] for r in rs])
            ys = np.array([r["final"] for r in rs])
            ax.scatter(xs, ys, color=col[a], marker=marker(a), s=45, edgecolors="k", linewidths=0.5)
        lim = [0, 6]
        ax.plot(lim, lim, color="0.7", ls="--", lw=0.8)
        ax.set_title(f"n={n}")
        ax.set_xlabel("mean targets/ep over all 200M (learning-curve area)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("final targets / episode")
    axes[0].set_xlim(0, 6)
    axes[0].set_ylim(0, 6)
    fig.legend(handles=handles[:-1], loc="lower center", ncol=len(ARCHS), fontsize=8, frameon=False)
    fig.suptitle("Anytime vs final performance · distance below the diagonal = time spent learning", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(args.out_dir / "anytime_vs_final.png", dpi=150)
    plt.close(fig)

    # ---- table ----
    lines = [f"threshold = {thr:g} targets/ep (3-eval moving average); means over seeds that reached",
             f"{'n':>2} {'arch':>9} {'final':>6} {'auc':>5} {'reached':>7} {'steps(M)':>9} {'wall(min)':>9} {'total wall(min)':>15}"]
    with open(args.out_dir / "efficiency.csv", "w") as f:
        f.write("n,width,depth,seed,final,auc,steps_to_thr,wall_to_thr_s,wall_total_s\n")
        for n in ns:
            for a in ARCHS:
                rs = cells.get((n,) + a, [])
                for r in rs:
                    f.write(f"{n},{a[0]},{a[1]},{r['seed']},{r['final']:.3f},{r['auc']:.3f},"
                            f"{r['steps_thr']:.0f},{r['wall_thr']:.1f},{r['wall_total']:.1f}\n")
                if not rs:
                    continue
                ms, _, k = summarise([r["steps_thr"] for r in rs])
                mw, _, kw = summarise([r["wall_thr"] for r in rs])
                mt, _, kt = summarise([r["wall_total"] for r in rs])
                lines.append(f"{n:>2} {arch_name(*a):>9} {np.mean([r['final'] for r in rs]):6.2f} "
                             f"{np.mean([r['auc'] for r in rs]):5.2f} {k:>5}/{len(rs)} {ms / 1e6:9.0f} "
                             f"{mw / 60:9.1f} {mt / 60:11.1f} ({kt})")
    text = "\n".join(lines)
    (args.out_dir / "efficiency.txt").write_text(text + "\n")
    print(text)

    # ---- convergence: steps to 90% of each run's own final ----
    conv = [f"convergence = first step where the 3-eval moving average reaches {CONV_FRAC:.0%} "
            "of that run's own final score (mean of last 25% of 200M)",
            "* = run still improving at 200M (its final is an underestimate, so 'converged' is premature)",
            f"{'n':>2} {'arch':>9} {'final (seeds)':>24} {'mean':>5} | {'steps to 90% (M), seeds':>24} "
            f"{'mean':>5} | {'wall (min)':>10}"]
    with open(args.out_dir / "convergence.csv", "w") as f:
        f.write("n,width,depth,seed,final,steps_to_90pct,wall_to_90pct_s,still_improving\n")
        for n in ns:
            for a in ARCHS:
                rs = sorted(cells.get((n,) + a, []), key=lambda r: r["seed"])
                for r in rs:
                    f.write(f"{n},{a[0]},{a[1]},{r['seed']},{r['final']:.3f},{r['steps_conv']:.0f},"
                            f"{r['wall_conv']:.1f},{int(r['improving'])}\n")
                fin = " ".join(f"{r['final']:.2f}{'*' if r['improving'] else ' '}" for r in rs)
                st = " ".join(f"{r['steps_conv'] / 1e6:5.0f}" for r in rs)
                mw, _, kw = summarise([r["wall_conv"] for r in rs])
                conv.append(f"{n:>2} {arch_name(*a):>9} {fin:>24} {np.mean([r['final'] for r in rs]):5.2f} | "
                            f"{st:>24} {np.nanmean([r['steps_conv'] for r in rs]) / 1e6:5.0f} | "
                            f"{mw / 60:6.1f} ({kw})")
            conv.append("")
    ctext = "\n".join(conv)
    (args.out_dir / "convergence.txt").write_text(ctext + "\n")
    print("\n" + ctext)

    fig, axes = plt.subplots(1, len(ns), figsize=(4.3 * len(ns), 4.0), sharex=True, sharey=True)
    for ax, n in zip(axes, ns):
        for a in ARCHS:
            rs = cells.get((n,) + a, [])
            if rs:
                ax.scatter(np.nanmean([r["steps_conv"] for r in rs]) / 1e6,
                           np.mean([r["final"] for r in rs]), marker=marker(a), s=80,
                           color=col[a], edgecolors="k", linewidths=0.6)
        ax.set_title(f"n={n}  ({2 * (n + 1)} legs)")
        ax.set_xlabel(f"env steps to reach {CONV_FRAC:.0%} of own final (M)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("final targets / episode")
    fig.legend(handles=handles[:-1], loc="lower center", ncol=len(ARCHS), fontsize=8, frameon=False)
    fig.suptitle(f"Time to convergence ({CONV_FRAC:.0%} of each run's own final) vs final performance · "
                 "mean of 3 seeds per architecture · top-left is better", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(args.out_dir / "convergence_vs_final.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
