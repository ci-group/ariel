"""Side-by-side visual comparison of planner_generator vs path_curvature_based.

Generates N paths from each and plots them in a 2-column grid:
  Left  → planner_generator   (numpy, uniform-t sampling)
  Right → path_curvature_based (torch, equidistant resampling)

Usage
-----
    uv run examples/spear/8c_compare_generators.py
    uv run examples/spear/8c_compare_generators.py --n 12 --steps 20 --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "goal_generator_ltu" / "polynomial_goal_generator"))

import planner_generator as pg
import path_curvature_based as pcb

parser = argparse.ArgumentParser(description="Compare gate generators side by side")
parser.add_argument("--n",     type=int,   default=6,    help="Number of paths per generator")
parser.add_argument("--steps", type=int,   default=20,   help="Gates per path")
parser.add_argument("--seed",  type=int,   default=None, help="Random seed")
parser.add_argument("--scale", type=float, default=5.0,  help="Path scale (metres)")
parser.add_argument("--coeffs",
                    default=str(_REPO_ROOT / "goal_generator_ltu"
                                / "polynomial_goal_generator" / "quintic_coeffs.npy"))
args = parser.parse_args()

rng     = np.random.default_rng(args.seed)
seed_pg  = int(rng.integers(0, 2**31))
seed_pcb = int(rng.integers(0, 2**31))

numpy_coeffs = np.load(args.coeffs)
torch_coeffs = torch.from_numpy(numpy_coeffs).to(torch.float64)

print(f"planner_generator    seed={seed_pg}")
paths_pg, _ = pg.generate_paths_from_coefficients(
    numpy_coeffs,
    num_generate=args.n,
    steps=args.steps,
    seed=seed_pg,
    clip_range=(-1.0, 1.0),
)
paths_pg = paths_pg * args.scale   # (N, steps, 2) numpy

print(f"path_curvature_based seed={seed_pcb}")
paths_pcb, _ = pcb.generate_paths_from_coefficients(
    coefficients=torch_coeffs,
    num_generate=args.n,
    steps=args.steps,
    seed=seed_pcb,
    clip_range=(-1.0, 1.0),
)
paths_pcb = paths_pcb.numpy() * args.scale   # (N, steps, 2) numpy

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(args.n, 2, figsize=(9, 3.0 * args.n), squeeze=False)

axes[0, 0].set_title("planner_generator\n(uniform-t)",      fontsize=11, fontweight="bold", color="#1f77b4")
axes[0, 1].set_title("path_curvature_based\n(equidistant)", fontsize=11, fontweight="bold", color="#d62728")

lim = args.scale * 1.15

for row in range(args.n):
    for ax, path, color in [
        (axes[row, 0], paths_pg[row],  "#1f77b4"),
        (axes[row, 1], paths_pcb[row], "#d62728"),
    ]:
        ax.plot(path[:, 0], path[:, 1], "-o",
                color=color, markersize=3, linewidth=1.4, alpha=0.85)
        ax.plot(path[0, 0], path[0, 1], "s", color="black", markersize=5, zorder=5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    axes[row, 0].set_ylabel(f"path {row}", fontsize=9)

fig.tight_layout()
plt.show()
