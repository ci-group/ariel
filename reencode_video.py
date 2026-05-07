"""
Re-encode MuJoCo output videos to a web-compatible H.264 format
that can be played in VS Code's Video Preview extension.

Usage:
    uv run reencode_video.py                          # re-encodes latest .mp4 in current dir
    uv run reencode_video.py path/to/video.mp4        # re-encodes specific file
    uv run reencode_video.py path/to/dir/             # re-encodes all .mp4s in a directory
"""

import subprocess
import sys
from pathlib import Path


def reencode(input_path: Path) -> Path:
    output_path = input_path.with_stem(input_path.stem + "_web")
    print(f"Re-encoding: {input_path.name} → {output_path.name}")
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",                    # overwrite output if exists
            "-i", str(input_path),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",   # required for browser/VS Code compat
            "-movflags", "+faststart", # enables progressive loading
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ✗ Failed:\n{result.stderr}")
    else:
        print(f"  ✓ Saved to: {output_path}")
    return output_path


def get_targets(arg: str | None) -> list[Path]:
    if arg is None:
        # No argument: pick the most recently modified .mp4 in current dir
        mp4s = sorted(Path(".").glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not mp4s:
            print("No .mp4 files found in current directory.")
            sys.exit(1)
        latest = mp4s[-1]
        print(f"No file specified — using latest: {latest.name}")
        return [latest]

    path = Path(arg)

    if path.is_dir():
        targets = list(path.glob("*.mp4"))
        if not targets:
            print(f"No .mp4 files found in {path}")
            sys.exit(1)
        return targets

    if path.is_file() and path.suffix == ".mp4":
        return [path]

    print(f"Invalid input: {arg}")
    sys.exit(1)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    targets = get_targets(arg)
    for target in targets:
        reencode(target)