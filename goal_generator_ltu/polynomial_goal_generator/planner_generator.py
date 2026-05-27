#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import argparse
import numpy as np
import matplotlib.pyplot as plt

###

def compute_yaws(path: np.ndarray) -> np.ndarray:
    """
    Compute yaw using points at t and t+2, matching the stored-path logic.
    Returns a yaw array with the same length as the path.
    """
    path = np.asarray(path, dtype=np.float64)

    if path.ndim != 2 or path.shape[1] != 2 or len(path) == 0:
        return np.array([], dtype=np.float64)

    if len(path) == 1:
        return np.array([0.0], dtype=np.float64)

    if len(path) == 2:
        dx = path[1, 0] - path[0, 0]
        dy = path[1, 1] - path[0, 1]
        yaw = np.arctan2(dy, dx)
        return np.array([yaw, yaw], dtype=np.float64)

    dx = path[2:, 0] - path[:-2, 0]
    dy = path[2:, 1] - path[:-2, 1]
    yaw = np.arctan2(dy, dx).astype(np.float64)

    # Pad to match length
    yaw = np.concatenate([yaw, yaw[-1:], yaw[-1:]], axis=0)
    return yaw

###

def plot_path_with_yaw(
    path: np.ndarray,
    yaw: np.ndarray,
    outpath: Path | None = None,
    arrow_step: int = 6,
    title: str = "Generated path with estimated yaw",
) -> None:
    path = np.asarray(path, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(path[:, 0], path[:, 1], linewidth=2.0, label="path")

    idx = np.arange(0, len(path), max(1, arrow_step))
    ax.quiver(
        path[idx, 0],
        path[idx, 1],
        np.cos(yaw[idx]),
        np.sin(yaw[idx]),
        angles="xy",
        scale_units="xy",
        scale=8.0,
        width=0.004,
        alpha=0.85,
        label="estimated yaw",
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    if outpath is not None:
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.show()

###

def sample_joint_coeff_vector(coeffs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Bootstrap sample one full coefficient vector from the empirical set.
    """
    idx = rng.integers(0, len(coeffs))
    return coeffs[idx].copy()

def reconstruct_path_from_coeffs(coeff_row: np.ndarray, steps: int) -> np.ndarray:
    """
    coeff_row must be shape (12,) = [x_c0..x_c5, y_c0..y_c5]
    """
    coeff_row = np.asarray(coeff_row, dtype=np.float64)
    if coeff_row.shape != (12,):
        raise ValueError(f"Expected coeff_row shape (12,), got {coeff_row.shape}")

    t = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    cx = coeff_row[:6]
    cy = coeff_row[6:]

    x = np.polynomial.polynomial.polyval(t, cx)
    y = np.polynomial.polynomial.polyval(t, cy)
    return np.stack([x, y], axis=1)

def generate_paths_from_coefficients(
    coeffs: np.ndarray,
    num_generate: int,
    steps: int,
    seed: int | None = None,
    clip_range: Tuple[float, float] | None = (-1.0, 1.0),
    max_attempts_per_path: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate new paths by jointly sampling full coefficient vectors.
    Returns:
        paths: (K, steps, 2)
        yaws:  (K, steps)
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.ndim != 2 or coeffs.shape[1] != 12:
        raise ValueError(f"Expected coeffs shape (N,12), got {coeffs.shape}")

    rng = np.random.default_rng(seed)
    generated_paths = []
    generated_yaws = []

    attempts = 0
    max_attempts = num_generate * max_attempts_per_path

    while len(generated_paths) < num_generate and attempts < max_attempts:
        attempts += 1

        row = sample_joint_coeff_vector(coeffs, rng)
        p = reconstruct_path_from_coeffs(row, steps=steps)

        if clip_range is not None:
            p = np.clip(p, clip_range[0], clip_range[1])

        if path_has_local_loops(p):
            continue

        yaw = compute_yaws(p)
        generated_paths.append(p)
        generated_yaws.append(yaw)

    if len(generated_paths) < num_generate:
        raise RuntimeError(
            f"Could only generate {len(generated_paths)} valid paths after {attempts} attempts."
        )

    return np.stack(generated_paths, axis=0), np.stack(generated_yaws, axis=0)

###

def path_has_self_intersection(path: np.ndarray, min_gap: int = 2) -> bool:
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 4:
        return False

    def _ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    def _segments_intersect(p1, p2, p3, p4) -> bool:
        return (_ccw(p1, p3, p4) != _ccw(p2, p3, p4)) and (_ccw(p1, p2, p3) != _ccw(p1, p2, p4))

    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        for j in range(i + min_gap, len(path) - 1):
            if j == i or j == i + 1:
                continue
            p3, p4 = path[j], path[j + 1]
            if _segments_intersect(p1, p2, p3, p4):
                return True

    return False

def path_has_local_loops(
    path: np.ndarray,
    max_local_turn_deg: float = 120.0,
    max_turn_spike_deg: float = 90.0,
) -> bool:
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 4:
        return False

    yaw = compute_yaws(path)
    if len(yaw) < 3:
        return False

    yaw_unwrapped = np.unwrap(yaw)
    dyaw = np.abs(np.diff(yaw_unwrapped))

    max_local_turn = np.deg2rad(max_local_turn_deg)
    max_turn_spike = np.deg2rad(max_turn_spike_deg)

    if np.max(dyaw) > max_local_turn:
        return True

    if np.sum(dyaw > max_turn_spike) >= 2:
        return True

    if path_has_self_intersection(path):
        return True

    return False

###

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paths from saved quintic coefficients.")
    parser.add_argument(
        "--coeffs",
        required=True,
        help="Path to the saved quintic coefficients .npy file (shape: N x 12).",
    )
    parser.add_argument("--steps", type=int, default=100, help="Number of points per generated path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-generate", type=int, default=1, help="Number of paths to generate.")
    parser.add_argument("--clip-min", type=float, default=-1.0, help="Minimum allowed coordinate.")
    parser.add_argument("--clip-max", type=float, default=1.0, help="Maximum allowed coordinate.")
    parser.add_argument("--outpath", default=None, help="Optional path to save the plot image.")
    args = parser.parse_args()

    coeffs = np.load(args.coeffs)
    generated_paths, generated_yaws = generate_paths_from_coefficients(
        coeffs=coeffs,
        num_generate=args.num_generate,
        steps=args.steps,
        seed=args.seed,
        clip_range=(args.clip_min, args.clip_max),
    )

    # Show the first generated path
    path = generated_paths[0]
    yaw = generated_yaws[0]

    print(f"Generated path shape: {path.shape}")
    print(f"Generated yaw shape:  {yaw.shape}")
    print("First 5 path points:\n", path[:5])
    print("First 5 yaw values:\n", yaw[:5])

    plot_path_with_yaw(
        path,
        yaw,
        outpath=Path(args.outpath) if args.outpath else None,
        arrow_step=max(1, args.steps // 20),
    )

if __name__ == "__main__":
    main()