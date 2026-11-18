"""Linear vs largest MLP at 100M and 300M steps (directed capacity sweep).

Reads the 300M runs in --data-dir (default
experiment/__data__/directed_capacity_sweep_linear300M), which hold the two
architectures the "capacity advantage shrinks with n" claim rests on:
linear (w16 d0) and the largest MLP (w256 d3).

The learning rate is constant (3e-4), so a 300M run's eval at ~100M steps is
a valid 100M-budget measurement. Both budgets therefore come from the SAME
runs, which makes the 100M-vs-300M comparison paired.

Score of a run at budget B = mean targets-reached/episode over its evals with
step in [0.8·B, 1.15·B] (the final stretch before B, plus brax's closing
eval, which lands a little past num_timesteps).

Figures (in --out-dir):
    learning_curves.png     one panel per n: per-seed curves (thin) + mean
                            (bold), linear vs w256 d3, dotted line at 100M
    budget_vs_n.png         score vs n for both archs at 100M and 300M,
                            mean ± 95% t-CI, seeds as dots
    gap_vs_n.png            (w256 d3 − linear) and linear/w256 d3 ratio vs n,
                            at 100M and 300M
    stats_300M.txt          per-cell numbers + interaction test per budget

Usable while the sweep is still running: cells with no finished runs are
skipped and the per-cell seed count is printed.

Usage
-----
    experiment/mujoco_playground/.venv/bin/python experiment/original_idea/plot_300M.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from plot_sweep_stats import interaction_test, load_runs, t_ci_half, welch

ARCHS = {  # (w, d) -> (label, colour)
    (16, 0): ("linear (d=0)", "0.35"),
    (256, 3): ("w=256, d=3", "tab:blue"),
}
BUDGETS = (100e6, 300e6)
BUDGET_STYLE = {100e6: ("100M", "--", "s"), 300e6: ("300M", "-", "o")}


def score_at(run: dict, budget: float) -> float:
    s = run["steps"]
    mask = (s >= 0.8 * budget) & (s <= 1.15 * budget)
    if not mask.any() or s[-1] < 0.95 * budget:
        return np.nan  # run hasn't reached this budget
    return float(run["targets_reached"][mask].mean())


def build_cells(runs, budget):
    cells = {}
    for (w, d) in ARCHS:
        for n in sorted({r["n"] for r in runs}):
            rs = sorted((r for r in runs if (r["n"], r["w"], r["d"]) == (n, w, d)),
                        key=lambda r: r["seed"])
            sc = np.array([score_at(r, budget) for r in rs])
            sc = sc[~np.isnan(sc)]
            if len(sc):
                cells[(n, w, d)] = {"scores": sc, "mean": float(sc.mean()),
                                    "ci": t_ci_half(sc)}
    return cells


def xpos(ns):
    """n on a log2 axis (n=1,2,4 evenly spaced; 3 and 6 in between)."""
    return {n: np.log2(n) for n in ns}


def plot_learning_curves(runs, ns, out):
    fig, axes = plt.subplots(1, len(ns), figsize=(3.6 * len(ns), 3.4),
                             sharey=True, squeeze=False)
    for ax, n in zip(axes[0], ns):
        for (w, d), (lab, col) in ARCHS.items():
            rs = [r for r in runs if (r["n"], r["w"], r["d"]) == (n, w, d)]
            if not rs:
                continue
            for r in rs:
                ax.plot(r["steps"] / 1e6, r["targets_reached"], color=col,
                        lw=0.7, alpha=0.4)
            # Mean over seeds on the shortest common step grid.
            m = min(len(r["steps"]) for r in rs)
            mean = np.mean([r["targets_reached"][:m] for r in rs], axis=0)
            ax.plot(rs[0]["steps"][:m] / 1e6, mean, color=col, lw=2,
                    label=f"{lab} ({len(rs)} seeds)")
        ax.axvline(100, color="k", ls=":", lw=1)
        ax.set_title(f"n={n}  ({2 * (n + 1)} legs)")
        ax.set_xlabel("env steps (M)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[0, 0].set_ylabel("targets reached / episode")
    fig.suptitle("Learning curves to 300M steps (thin: seeds, bold: mean; "
                 "dotted: 100M budget of the original sweep)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_budget_vs_n(cells_by_b, ns, out):
    xp = xpos(ns)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for q, ((w, d), (lab, col)) in enumerate(ARCHS.items()):
        for b in BUDGETS:
            bl, ls, mk = BUDGET_STYLE[b]
            off = (q - 0.5) * 0.08 + (0.02 if b == 300e6 else -0.02)
            pts = [(n, cells_by_b[b][(n, w, d)]) for n in ns if (n, w, d) in cells_by_b[b]]
            if not pts:
                continue
            x = np.array([xp[n] for n, _ in pts]) + off
            ax.errorbar(x, [c["mean"] for _, c in pts],
                        yerr=[np.nan_to_num(c["ci"]) for _, c in pts],
                        color=col, ls=ls, marker=mk, capsize=3, lw=1.5,
                        mfc=col if b == 300e6 else "white", label=f"{lab} @ {bl}")
            for xi, (_, c) in zip(x, pts):
                ax.scatter(np.full(len(c["scores"]), xi), c["scores"], color=col,
                           s=9, alpha=0.5, zorder=3)
    ax.set_xticks([xp[n] for n in ns])
    ax.set_xticklabels([f"n={n}\n({2 * (n + 1)} legs)" for n in ns])
    ax.set_ylabel("targets reached / episode")
    ax.set_title("Task success vs morphology at 100M and 300M steps\n"
                 "mean ± 95% t-CI, dots = seeds", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _paired_gap(cells, n):
    a, b = cells.get((n, 256, 3)), cells.get((n, 16, 0))
    if a is None or b is None:
        return None
    gap = a["mean"] - b["mean"]
    # Welch 95% CI on a difference of independent means (Welch–Satterthwaite
    # dof), consistent with the t-intervals used everywhere else.
    na, nb = len(a["scores"]), len(b["scores"])
    if na < 2 or nb < 2:
        return gap, np.nan, b["mean"] / a["mean"]
    va = a["scores"].var(ddof=1) / na
    vb = b["scores"].var(ddof=1) / nb
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1)) if se > 0 else 1
    return gap, float(stats.t.ppf(0.975, dof) * se), b["mean"] / a["mean"]


def plot_gap_vs_n(cells_by_b, ns, out):
    xp = xpos(ns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    if not any(_paired_gap(cells_by_b[b], n) for b in BUDGETS for n in ns):
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "no n has both linear and w256 d3 runs yet",
                    ha="center", va="center", transform=ax.transAxes, color="0.4")
            ax.set_axis_off()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return
    for b in BUDGETS:
        bl, ls, mk = BUDGET_STYLE[b]
        rows = [(n, _paired_gap(cells_by_b[b], n)) for n in ns]
        rows = [(n, g) for n, g in rows if g is not None]
        if not rows:
            continue
        x = np.array([xp[n] for n, _ in rows]) + (0.03 if b == 300e6 else -0.03)
        ax1.errorbar(x, [g[0] for _, g in rows], yerr=[np.nan_to_num(g[1]) for _, g in rows],
                     color="tab:purple", ls=ls, marker=mk, capsize=3,
                     mfc="tab:purple" if b == 300e6 else "white", label=bl)
        ax2.plot(x, [g[2] for _, g in rows], color="tab:green", ls=ls, marker=mk,
                 mfc="tab:green" if b == 300e6 else "white", label=bl)
    for ax in (ax1, ax2):
        ax.set_xticks([xp[n] for n in ns])
        ax.set_xticklabels([f"n={n}" for n in ns])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, title="budget")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_ylabel("w256 d3 − linear  (targets / episode)")
    ax1.set_title("Capacity advantage  (± 95% Welch CI)", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("linear / w256 d3")
    ax2.set_title("Fraction of large-net performance\nreached by a linear policy",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path,
                   default=root.parent / "__data__" / "directed_capacity_sweep_linear300M")
    p.add_argument("--out-dir", type=Path, default=root / "__plots_directed_300M__")
    args = p.parse_args()

    runs = [r for r in load_runs(args.data_dir) if (r["w"], r["d"]) in ARCHS]
    if not runs:
        raise SystemExit(f"no finished 300M runs under {args.data_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ns = sorted({r["n"] for r in runs})
    cells_by_b = {b: build_cells(runs, b) for b in BUDGETS}

    plot_learning_curves(runs, ns, args.out_dir / "learning_curves.png")
    plot_budget_vs_n(cells_by_b, ns, args.out_dir / "budget_vs_n.png")
    plot_gap_vs_n(cells_by_b, ns, args.out_dir / "gap_vs_n.png")

    L = [f"{len(runs)} finished 300M runs in {args.data_dir.name}", ""]
    for b in BUDGETS:
        cells = cells_by_b[b]
        L.append(f"== budget {BUDGET_STYLE[b][0]} (score = mean targets/ep over evals "
                 f"in [{0.8 * b / 1e6:.0f}M, {1.15 * b / 1e6:.0f}M]) ==")
        for n in ns:
            parts = []
            for (w, d), (lab, _) in ARCHS.items():
                c = cells.get((n, w, d))
                parts.append(f"{lab}: " + ("—" if c is None else
                             f"{c['mean']:.2f} ±{c['ci']:.2f} (k={len(c['scores'])}, "
                             f"seeds " + " ".join(f"{s:.2f}" for s in c['scores']) + ")"))
            g = _paired_gap(cells, n)
            if g is not None and len(cells[(n, 256, 3)]["scores"]) > 1 \
                    and len(cells[(n, 16, 0)]["scores"]) > 1:
                t, pv = welch(cells[(n, 256, 3)]["scores"], cells[(n, 16, 0)]["scores"])
                parts.append(f"gap {g[0]:.2f} (Welch p={pv:.3g}), ratio {g[2]:.2f}")
            L.append(f"  n={n}: " + "   ".join(parts))
        both = [n for n in ns if (n, 256, 3) in cells and (n, 16, 0) in cells]
        if len(both) >= 2:
            r = interaction_test(cells, both, 256, 3, 16)
            if r["dof"] > 0:
                L.append(f"  interaction b3 = {r['b3']:+.3f} per doubling of n, 95% CI "
                         f"[{r['lo']:+.3f}, {r['hi']:+.3f}], p={r['p']:.3g} "
                         f"(N={r['N']}, n levels={both})")
                L.append(f"    slope per doubling: largest {r['slope_large']:+.3f}, "
                         f"linear {r['slope_linear']:+.3f}")
        L.append("")
    text = "\n".join(L)
    (args.out_dir / "stats_300M.txt").write_text(text + "\n")
    print(text)
    print(f"wrote figures + stats_300M.txt to {args.out_dir}")


if __name__ == "__main__":
    main()
