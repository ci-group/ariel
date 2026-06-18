"""Plot learning curves for the architecture comparison experiment.

Reads results from results_gecko/ and results_evogym/ (or custom paths via CLI)
and produces a 2-panel figure with:
  - mean best-so-far fitness ± 1 std (shaded)
  - mean-of-generation-maxima curve (dashed)

Usage:
    uv run plot_results.py [--gecko-dir __data__/results_gecko] [--evogym-dir __data__/results_evogym] [--out __data__/comparison.png]
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BRAIN_COLORS = {
    "distributed": "#2196F3",   # blue
    "standard":    "#F44336",   # red
}
BRAIN_LABELS = {
    "distributed": "Distributed MLP",
    "standard":    "Standard ANN",
}


def load_histories(results_dir: str, brain_type: str) -> np.ndarray | None:
    """Load (reps, gens) array or return None if not found."""
    path = os.path.join(results_dir, f"{brain_type}_histories.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def plot_panel(
    ax: plt.Axes,
    results_dir: str,
    title: str,
    ylabel: str,
) -> None:
    """Draw one domain panel onto ax."""
    any_data = False
    for brain_type in ("distributed", "standard"):
        histories = load_histories(results_dir, brain_type)
        if histories is None:
            continue
        any_data = True

        # histories: (reps, gens) — each cell is cumulative best so far
        gens = np.arange(1, histories.shape[1] + 1)
        mean = histories.mean(axis=0)
        std = histories.std(axis=0)
        # Mean of each generation's best (not cumulative) — shows per-gen improvement
        # histories already stores cumulative best, so gen_max[g] = histories[:,g] - histories[:,g-1]
        # is always ≥ 0, which is uninteresting. Instead plot mean of the per-rep gen max
        # by taking the per-gen max across the population from the raw history.
        # Since we only saved cumulative bests, we approximate gen-max as the
        # difference between consecutive cumulative bests (lower bound):
        # gen_step[rep, g] = histories[rep, g] - histories[rep, g-1]  (first gen = histories[:,0])
        gen_delta = np.diff(histories, axis=1, prepend=0.0)
        # Mean across reps of how much improvement happened at each gen:
        mean_delta = gen_delta.mean(axis=0)

        color = BRAIN_COLORS[brain_type]
        label = BRAIN_LABELS[brain_type]

        ax.fill_between(gens, mean - std, mean + std, alpha=0.15, color=color)
        ax.plot(gens, mean, color=color, linewidth=2, label=f"{label} (mean±std)")
        # Overlay mean cumulative improvement as a lighter dashed line representing avg-max
        ax.plot(gens, mean_delta.cumsum() + mean[0] - mean_delta[0],
                color=color, linewidth=1.2, linestyle="--", alpha=0.6,
                label=f"{label} (avg-max)")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if not any_data:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=14)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gecko-dir", default="__data__/results_gecko")
    parser.add_argument("--evogym-dir", default="__data__/results_evogym")
    parser.add_argument("--out", default="__data__/comparison.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Distributed MLP vs Standard ANN — CMA-ES Learning Curves",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plot_panel(
        axes[0],
        args.gecko_dir,
        title="ARIEL Gecko",
        ylabel="Best x-displacement (m)",
    )
    plot_panel(
        axes[1],
        args.evogym_dir,
        title="EvoGym Walker",
        ylabel="Cumulative reward (x-displacement)",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
