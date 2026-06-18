"""Create visualisation videos for a trained drone individual.

Mirrors src/airevolve/examples/videos/make_video.py using ARIEL imports.

Expects an ``individual_dir`` containing:
  - ``individual.npy`` (or ``genome.npy``) — drone genome
  - ``policy.zip``                          — trained SB3 PPO policy

Usage:
    uv run examples/d_drones/7_make_video.py __data__/my_individual \\
        --gate-cfg figure8

    # With explicit device:
    uv run examples/d_drones/7_make_video.py __data__/my_individual \\
        --gate-cfg circle --device cpu
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Create drone visualisation videos")
parser.add_argument("individual_dir", nargs="?", default="__data__/drone_individual",
                    help="Directory with individual.npy/genome.npy and policy.zip")
parser.add_argument("--gate-cfg",
                    choices=["slalom", "figure8", "circle", "backandforth"],
                    default="figure8",
                    help="Gate configuration used during training (default figure8)")
parser.add_argument("--device", default=None,
                    help="Compute device (e.g. cpu, cuda:0). Auto-detected if omitted.")
parser.add_argument("--fps", type=int, default=100,
                    help="Frames per second (default 100)")
parser.add_argument("--width", type=int, default=864, help="Video width  (default 864)")
parser.add_argument("--height", type=int, default=700, help="Video height (default 700)")
parser.add_argument("--color", default="blue", help="Primary trajectory color")
args = parser.parse_args()

individual_dir = Path(args.individual_dir)
if not individual_dir.exists():
    raise SystemExit(f"Directory not found: {individual_dir}")

# ---------------------------------------------------------------------------
# Load individual
# ---------------------------------------------------------------------------

body_file = individual_dir / "individual.npy"
if not body_file.exists():
    body_file = individual_dir / "genome.npy"
if not body_file.exists():
    raise SystemExit(f"No individual.npy or genome.npy found in {individual_dir}")

policy_file = individual_dir / "policy.zip"
if not policy_file.exists():
    raise SystemExit(f"No policy.zip found in {individual_dir}")

individual = np.load(body_file, allow_pickle=True).astype(np.float32)
print(f"Loaded individual: {body_file}  shape={individual.shape}")
print(f"Gate cfg:  {args.gate_cfg}")
print(f"Device:    {args.device or 'auto'}")

# ---------------------------------------------------------------------------
# Animate (uses DroneGateEnv from ariel.simulation.tasks)
# ---------------------------------------------------------------------------

from ariel.simulation.tasks.drone_gate_env import DroneGateEnv
from ariel.ec.drone.inspection.behavioural_analysis.gate_based.animate_individual_with_gates import (
    animate_individual,
)
from ariel.ec.drone.inspection.behavioural_analysis.gate_based.combine_videos import (
    combine_videos_from_directory,
)
from ariel.ec.drone.inspection.behavioural_analysis.gate_based.calculate_stats import (
    calculate_stats,
)

vid_dir = individual_dir / "videos"
vid_dir.mkdir(exist_ok=True)

motor_colors = ["red", "blue", "green", "orange", "purple", "brown"]

print("\nRendering top-view animation …")
try:
    animate_individual(
        gate_cfg=args.gate_cfg,
        individual_dir=str(individual_dir),
        save_dir=str(vid_dir),
        file_name="/top_view.mp4",
        device=args.device,
        view_type="top",
        follow=True,
        draw_forces=False,
        draw_path=True,
        auto_play=True,
        record=True,
        motor_colors=motor_colors,
    )

    print("Rendering iso-view animation …")
    animate_individual(
        gate_cfg=args.gate_cfg,
        individual_dir=str(individual_dir),
        save_dir=str(vid_dir),
        file_name="/iso_view.mp4",
        device=args.device,
        view_type="iso",
        follow=True,
        draw_forces=False,
        draw_path=True,
        auto_play=True,
        record=True,
        motor_colors=motor_colors,
    )

    print("Combining videos …")
    combine_videos_from_directory(str(vid_dir))
    print(f"Videos saved to: {vid_dir}")

except Exception as exc:
    print(f"Warning: Animation failed: {exc}")
    print("Check that OpenCV, ffmpeg, and the policy dependencies are installed.")
