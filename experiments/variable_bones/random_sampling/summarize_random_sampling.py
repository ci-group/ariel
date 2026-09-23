"""Summarize raw ARIEL random morphology-space sampling data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("__data__") / "variable_bones" / "random_sampling",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def numeric(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key, "")
        if not value:
            continue
        parsed = float(value)
        if math.isfinite(parsed):
            values.append(parsed)
    return np.asarray(values, dtype=float)


def descriptive(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }

    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    args = parse_args()

    robots = read_csv(args.input / "robots.csv")
    bones = read_csv(args.input / "bones.csv")
    attempts = read_csv(args.input / "attempts.csv")

    status_counts = Counter(row["status"] for row in attempts)
    brick_count_distribution = Counter(
        int(float(row["num_bricks"]))
        for row in robots
        if row["num_bricks"] != ""
    )
    hinge_count_distribution = Counter(
        int(float(row["num_hinges"]))
        for row in robots
        if row["num_hinges"] != ""
    )

    summary = {
        "num_robots": len(robots),
        "num_bones": len(bones),
        "num_attempts": len(attempts),
        "attempt_status_counts": dict(sorted(status_counts.items())),
        "bone_level_length_mm": descriptive(numeric(bones, "length_mm")),
        "robot_level_mean_bone_length_mm": descriptive(
            numeric(robots, "mean_bone_length_mm")
        ),
        "robot_num_modules": descriptive(numeric(robots, "num_modules")),
        "robot_num_bricks": descriptive(numeric(robots, "num_bricks")),
        "robot_num_hinges": descriptive(numeric(robots, "num_hinges")),
        "brick_count_distribution": {
            str(key): value for key, value in sorted(brick_count_distribution.items())
        },
        "hinge_count_distribution": {
            str(key): value for key, value in sorted(hinge_count_distribution.items())
        },
    }

    (args.input / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with (args.input / "analysis_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])

        writer.writerow(["num_robots", len(robots)])
        writer.writerow(["num_bones", len(bones)])
        writer.writerow(["num_attempts", len(attempts)])

        for prefix, stats in [
            ("bone_length_mm", summary["bone_level_length_mm"]),
            (
                "robot_mean_bone_length_mm",
                summary["robot_level_mean_bone_length_mm"],
            ),
            ("num_modules", summary["robot_num_modules"]),
            ("num_bricks", summary["robot_num_bricks"]),
            ("num_hinges", summary["robot_num_hinges"]),
        ]:
            for key, value in stats.items():
                writer.writerow([f"{prefix}_{key}", value])

    print(f"Wrote {args.input / 'analysis_summary.json'}")
    print(f"Wrote {args.input / 'analysis_summary.csv'}")


if __name__ == "__main__":
    main()
