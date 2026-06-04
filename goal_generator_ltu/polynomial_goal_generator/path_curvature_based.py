#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from itertools import combinations, islice
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def calculate_cumulative_distances(path: torch.Tensor) -> torch.Tensor:
    distances = torch.linalg.norm(torch.diff(path, dim=0), dim=1)
    return torch.cat((torch.tensor([0.0], dtype=path.dtype, device=path.device), torch.cumsum(distances, dim=0)))


def interpolate_1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    idx = torch.searchsorted(xp, x)
    idx = torch.clamp(idx, 1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def resample_path_equidistant(path: torch.Tensor, num_points: int) -> torch.Tensor:
    cum_distances = calculate_cumulative_distances(path)
    total_distance = cum_distances[-1]
    
    target_distances = torch.linspace(0.0, total_distance.item(), num_points, dtype=path.dtype, device=path.device)

    x_resampled = interpolate_1d(target_distances, cum_distances, path[:, 0])
    y_resampled = interpolate_1d(target_distances, cum_distances, path[:, 1])

    return torch.stack([x_resampled, y_resampled], dim=1)


def compute_segment_angles(path: torch.Tensor) -> torch.Tensor:
    deltas = torch.diff(path, dim=0)
    return torch.atan2(deltas[:, 1], deltas[:, 0])


def compute_yaws(path: torch.Tensor) -> torch.Tensor:
    if path.ndim != 2 or path.shape[1] != 2 or len(path) == 0:
        return torch.empty(0, dtype=torch.float64, device=path.device)

    if len(path) == 1:
        return torch.tensor([0.0], dtype=torch.float64, device=path.device)

    if len(path) == 2:
        angles = compute_segment_angles(path)
        return torch.tensor([angles[0], angles[0]], dtype=torch.float64, device=path.device)

    dx = path[2:, 0] - path[:-2, 0]
    dy = path[2:, 1] - path[:-2, 1]
    yaw = torch.atan2(dy, dx).to(torch.float64)

    return torch.cat([yaw, yaw[-1:], yaw[-1:]], dim=0)


def create_gates(path: torch.Tensor, yaw: torch.Tensor, gate_width: float = 0.05) -> torch.Tensor:
    half_gate_width = gate_width / 2.0
    dx = -torch.sin(yaw) * half_gate_width
    dy = torch.cos(yaw) * half_gate_width
    offsets = torch.stack([dx, dy], dim=1)

    start_points = path - offsets
    end_points = path + offsets

    return torch.stack([start_points, end_points], dim=1)


def plot_path_with_yaw(
        path: torch.Tensor,
        yaw: torch.Tensor,
        outpath: Path | None = None,
        arrow_step: int = 6,
        scale: float = 1.0,
        title: str = "Generated path with estimated yaw",
) -> None:
    path_array = path.detach().cpu().numpy()
    yaw_array = yaw.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(path_array[:, 0], path_array[:, 1], linewidth=2.0, label="path")

    gates = create_gates(path, yaw, gate_width=0.05 * scale).detach().cpu().numpy()
    segments = LineCollection(
        gates,
        colors="red",
        linewidths=1.5,
        alpha=0.85,
        label="gates",
    )
    ax.add_collection(segments)

    idx = np.arange(0, len(path_array), max(1, arrow_step))
    ax.quiver(
        path_array[idx, 0],
        path_array[idx, 1],
        np.cos(yaw_array[idx]),
        np.sin(yaw_array[idx]),
        angles="xy",
        scale_units="xy",
        scale=15.0 / scale,
        width=0.003,
        alpha=0.85,
        label="estimated yaw",
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    if outpath is not None:
        fig.savefig(outpath, dpi=200, bbox_inches="tight")

    plt.show()


def sample_joint_coeff_vector(coefficients: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    idx = torch.randint(0, len(coefficients), (1,), generator=generator).item()
    return coefficients[idx].clone()


def evaluate_polynomial(t: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    powers = torch.arange(len(coefficients), dtype=t.dtype, device=t.device)
    vandermonde_matrix = t.unsqueeze(-1) ** powers
    return vandermonde_matrix @ coefficients


def reconstruct_path_from_coeffs(coeff_row: torch.Tensor, steps: int, high_res_steps: int = 1000) -> torch.Tensor:
    if coeff_row.shape != (12,):
        raise ValueError(f"Expected coeff_row shape (12,), got {coeff_row.shape}")

    t_high = torch.linspace(0.0, 1.0, high_res_steps, dtype=torch.float64, device=coeff_row.device)
    cx, cy = coeff_row[:6], coeff_row[6:]

    x_high = evaluate_polynomial(t_high, cx)
    y_high = evaluate_polynomial(t_high, cy)
    high_res_path = torch.stack([x_high, y_high], dim=1)

    return resample_path_equidistant(high_res_path, steps)


def evaluate_counter_clockwise(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
    return (evaluate_counter_clockwise(p1, p3, p4) != evaluate_counter_clockwise(p2, p3, p4)) & \
           (evaluate_counter_clockwise(p1, p2, p3) != evaluate_counter_clockwise(p1, p2, p4))


def path_has_self_intersection(path: torch.Tensor, min_gap: int = 2) -> bool:
    if len(path) < 4:
        return False

    segments = torch.stack([path[:-1], path[1:]], dim=1)
    num_segments = len(segments)

    def is_valid_gap(pair: tuple[int, int]) -> bool:
        return (pair[1] - pair[0]) >= min_gap

    def check_intersection(pair: tuple[int, int]) -> bool:
        i, j = pair
        return segments_intersect(segments[i, 0], segments[i, 1], segments[j, 0], segments[j, 1]).item()

    index_pairs = filter(is_valid_gap, combinations(range(num_segments), 2))
    return any(map(check_intersection, index_pairs))


def unwrap_angles(angles: torch.Tensor) -> torch.Tensor:
    diffs = torch.diff(angles)
    diffs_wrapped = (diffs + torch.pi) % (2 * torch.pi) - torch.pi
    
    unwrapped = torch.zeros_like(angles)
    unwrapped[0] = angles[0]
    unwrapped[1:] = diffs_wrapped
    
    return torch.cumsum(unwrapped, dim=0)


def path_has_local_loops(
        path: torch.Tensor,
        max_local_turn_deg: float = 120.0,
        max_turn_spike_deg: float = 90.0,
) -> bool:
    if len(path) < 4:
        return False

    if path_has_self_intersection(path):
        return True

    angles = compute_segment_angles(path)
    angles_unwrapped = unwrap_angles(angles)
    dyaw = torch.abs(torch.diff(angles_unwrapped))

    max_local_turn = math.radians(max_local_turn_deg)
    max_turn_spike = math.radians(max_turn_spike_deg)

    if torch.max(dyaw) > max_local_turn:
        return True

    if torch.sum(dyaw > max_turn_spike) >= 2:
        return True

    return False


def path_exceeds_bounds(path: torch.Tensor, clip_range: tuple[float, float]) -> bool:
    return torch.any((path < clip_range[0]) | (path > clip_range[1])).item()


def generate_valid_path(
        coefficients: torch.Tensor, 
        steps: int, 
        clip_range: tuple[float, float], 
        generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor] | None:
    row = sample_joint_coeff_vector(coefficients, generator)
    path = reconstruct_path_from_coeffs(row, steps=steps)

    if clip_range is not None and path_exceeds_bounds(path, clip_range):
        return None

    if path_has_local_loops(path):
        return None

    yaw = compute_yaws(path)
    return path, yaw


def generate_paths_from_coefficients(
        coefficients: torch.Tensor,
        num_generate: int,
        steps: int,
        seed: int | None = None,
        clip_range: tuple[float, float] | None = (-1.0, 1.0),
        max_attempts_per_path: int = 500,
) -> tuple[torch.Tensor, torch.Tensor]:
    if coefficients.ndim != 2 or coefficients.shape[1] != 12:
        raise ValueError(f"Expected coefficients shape (N,12), got {coefficients.shape}")

    generator = torch.Generator(device=coefficients.device)
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()

    max_attempts = num_generate * max_attempts_per_path

    def attempt_generation(_: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        return generate_valid_path(coefficients, steps, clip_range, generator)

    attempts_iterator = map(attempt_generation, range(max_attempts))
    valid_paths_iterator = filter(lambda result: result is not None, attempts_iterator)
    collected_results = list(islice(valid_paths_iterator, num_generate))

    if len(collected_results) < num_generate:
        raise RuntimeError(
            f"Could only generate {len(collected_results)} valid paths after {max_attempts} attempts."
        )

    generated_paths, generated_yaws = zip(*collected_results)
    return torch.stack(generated_paths, dim=0), torch.stack(generated_yaws, dim=0)


def generate_deltas_path(
        coefficients: torch.Tensor,
        num_generate: int,
        steps: int,
        seed: int | None = None,
        clip_range: tuple[float, float] | None = (-1.0, 1.0),
        max_attempts_per_path: int = 500,
) -> torch.Tensor:
    paths, _ = generate_paths_from_coefficients(
        coefficients=coefficients,
        num_generate=num_generate,
        steps=steps,
        seed=seed,
        clip_range=clip_range,
        max_attempts_per_path=max_attempts_per_path,
    )

    delta_xy = torch.diff(paths, dim=1)
    zeros = torch.zeros((*delta_xy.shape[:2], 2), dtype=torch.float64, device=paths.device)
    return torch.squeeze(torch.cat([delta_xy, zeros], dim=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paths from saved quintic coefficients.")
    parser.add_argument("--coeffs", required=True, help="Path to the saved quintic coefficients .npy file.")
    parser.add_argument("--steps", type=int, default=100, help="Number of points per generated path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-generate", type=int, default=1, help="Number of paths to generate.")
    parser.add_argument("--clip-min", type=float, default=-1.0, help="Minimum allowed raw coordinate.")
    parser.add_argument("--clip-max", type=float, default=1.0, help="Maximum allowed raw coordinate.")
    parser.add_argument("--outpath", default=None, help="Optional path to save the plot image.")
    parser.add_argument("--scale", type=float, default=10.0, help="Scale factor for the generated path.")
    parser.add_argument("--raw", action="store_true", help="Plot raw values without applying the scale factor.")
    args = parser.parse_args()

    numpy_coefficients = np.load(args.coeffs)
    coefficients = torch.from_numpy(numpy_coefficients).to(torch.float64)

    generated_paths, generated_yaws = generate_paths_from_coefficients(
        coefficients=coefficients,
        num_generate=args.num_generate,
        steps=args.steps,
        seed=args.seed,
        clip_range=(args.clip_min, args.clip_max),
    )

    deltas_path = generate_deltas_path(
        coefficients=coefficients,
        num_generate=args.num_generate,
        steps=args.steps,
        seed=args.seed,
        clip_range=(args.clip_min, args.clip_max),
    )

    path = generated_paths[0]
    yaw = generated_yaws[0]

    final_scale = 1.0 if args.raw else args.scale
    final_path = path * final_scale

    print(f"Generated path shape: {final_path.shape}")
    print(f"Generated yaw shape:  {yaw.shape}")
    print(f"First 5 path points:\n{final_path[:5]}")
    print(f"First 5 yaw values:\n{yaw[:5]}")
    print(f"First 5 deltas:\n{deltas_path[:5]}")

    plot_path_with_yaw(
        path=final_path,
        yaw=yaw,
        outpath=Path(args.outpath) if args.outpath else None,
        arrow_step=max(1, args.steps // 20),
        scale=final_scale
    )

if __name__ == "__main__":
    main()