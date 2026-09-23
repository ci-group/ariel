"""Create plots from ARIEL random morphology-space sampling data."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("__data__") / "variable_bones" / "random_sampling",
    )
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def finite_float(value: str) -> float | None:
    if not value:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def save(fig, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    robots = read_csv(args.input / "robots.csv")
    bones = read_csv(args.input / "bones.csv")
    attempts = read_csv(args.input / "attempts.csv")

    out = args.input / "plots"
    out.mkdir(parents=True, exist_ok=True)

    bone_lengths = np.asarray(
        [float(row["length_mm"]) for row in bones], dtype=float
    )

    robot_means = np.asarray(
        [
            value
            for row in robots
            if (value := finite_float(row["mean_bone_length_mm"])) is not None
        ],
        dtype=float,
    )

    # 1. Bone-level length distribution.
    if bone_lengths.size:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(bone_lengths, bins=args.bins)
        ax.set_xlabel("Bone length (mm)")
        ax.set_ylabel("Number of bones")
        ax.set_title("Bone-level length distribution")
        save(fig, out / "01_bone_length_histogram.png", args.dpi)

    # 2. Robot-level mean bone-length distribution.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(robot_means, bins=args.bins)
    ax.set_xlabel("Mean bone length per robot (mm)")
    ax.set_ylabel("Number of robots")
    ax.set_title("Robot-level mean bone-length distribution")
    save(fig, out / "02_robot_mean_bone_length_histogram.png", args.dpi)

    # 3. Brick count distribution.
    brick_counts = Counter(int(float(row["num_bricks"])) for row in robots)
    xs = sorted(brick_counts)
    ys = [brick_counts[x] for x in xs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(xs, ys)
    ax.set_xlabel("Number of bricks/bones in robot")
    ax.set_ylabel("Number of robots")
    ax.set_title("Brick-count distribution")
    save(fig, out / "03_brick_count_distribution.png", args.dpi)

    # 4. Mean bone length vs brick count.
    by_bricks = defaultdict(list)
    for row in robots:
        value = finite_float(row["mean_bone_length_mm"])
        if value is not None:
            by_bricks[int(float(row["num_bricks"]))].append(value)

    xs = sorted(by_bricks)
    ys = [float(np.mean(by_bricks[x])) for x in xs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Number of bricks/bones")
    ax.set_ylabel("Mean bone length (mm)")
    ax.set_title("Mean bone length vs brick count")
    save(fig, out / "04_mean_length_vs_brick_count.png", args.dpi)

    # 5. Mean bone length vs total module count.
    by_modules = defaultdict(list)
    for row in robots:
        value = finite_float(row["mean_bone_length_mm"])
        if value is not None:
            by_modules[int(float(row["num_modules"]))].append(value)

    xs = sorted(by_modules)
    ys = [float(np.mean(by_modules[x])) for x in xs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Total modules")
    ax.set_ylabel("Mean bone length (mm)")
    ax.set_title("Mean bone length vs total module count")
    save(fig, out / "05_mean_length_vs_module_count.png", args.dpi)

    # 6. Sampling acceptance/rejection reasons.
    statuses = Counter(row["status"] for row in attempts)
    labels = [label for label, _ in statuses.most_common()]
    values = [statuses[label] for label in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(labels))
    ax.barh(positions, values)
    ax.set_yticks(positions, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of attempts")
    ax.set_title("Sampling acceptance and rejection reasons")
    save(fig, out / "06_attempt_status_distribution.png", args.dpi)

    # 7. ECDF of all bone lengths.
    if bone_lengths.size:
        ordered = np.sort(bone_lengths)
        y = np.arange(1, ordered.size + 1) / ordered.size

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ordered, y)
        ax.set_xlabel("Bone length (mm)")
        ax.set_ylabel("Cumulative proportion")
        ax.set_title("Empirical CDF of bone lengths")
        save(fig, out / "07_bone_length_ecdf.png", args.dpi)

    print(f"Wrote plots to: {out}")


if __name__ == "__main__":
    main()
