#!/usr/bin/env python3
"""Convert a URDF file to USD via Isaac Lab's UrdfConverter.

Isaac-Lab-env side of the two-step Blueprint → URDF → USD pipeline.
Pairs with ``scripts/blueprint_to_urdf.py`` (run in the ariel env).
Pure stdlib + Isaac Lab — no ariel dependency.

Examples:

    # Default conda-env path
    /home/keiichi-ito/miniconda3/envs/isaaclab/bin/python \\
        scripts/urdf_to_usd.py --headless \\
        --input /tmp/quad.urdf --output_dir /tmp/quad_usd
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def _log(msg: str) -> None:
    """Print to stderr (Isaac Sim captures stdout but leaves stderr alone)."""
    sys.stderr.write(f"[urdf_to_usd] {msg}\n")
    sys.stderr.flush()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=str, required=True,
                        help="Path to source .urdf file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write the .usd into (created if missing).")
    parser.add_argument("--usd_name", type=str, default=None,
                        help="Output USD filename (default: <input_stem>.usd).")
    parser.add_argument("--merge_fixed_joints", action="store_true",
                        help="Collapse fixed-joint chains into a single rigid body. "
                             "Off by default so each motor stays a separate link "
                             "(needed for per-motor thrust application in Isaac Lab).")
    parser.add_argument("--fix_base", action="store_true",
                        help="Treat the URDF root as a fixed base. Off by default "
                             "(drones are floating-base; Isaac Lab adds the freejoint).")
    return parser


def main() -> None:
    _log("starting")
    parser = _build_parser()

    # Isaac Lab adds --headless, --device, etc.
    _log("importing isaaclab.app.AppLauncher")
    from isaaclab.app import AppLauncher  # noqa: PLC0415
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    urdf_path = Path(args.input).resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    usd_name = args.usd_name or f"{urdf_path.stem}.usd"
    expected_out = output_dir / usd_name

    _log(f"input : {urdf_path}")
    _log(f"output: {expected_out}")

    _log("launching Isaac Sim app...")
    app_launcher = AppLauncher(args, multi_gpu=False)
    simulation_app = app_launcher.app
    _log("Isaac Sim app launched")

    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: PLC0415

        _log("building UrdfConverterCfg")
        # All joints in our v1 URDF are fixed, so stiffness is unused; we set
        # it to 0 just to satisfy the configclass validator (the field is
        # MISSING by default).
        cfg = UrdfConverterCfg(
            asset_path=str(urdf_path),
            usd_dir=str(output_dir),
            usd_file_name=usd_name,
            force_usd_conversion=True,
            merge_fixed_joints=args.merge_fixed_joints,
            fix_base=args.fix_base,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                ),
            ),
        )
        _log("invoking UrdfConverter")
        converter = UrdfConverter(cfg)
        result_path = Path(converter.usd_path)
        if result_path.exists():
            _log(f"USD written: {result_path} (size={result_path.stat().st_size} bytes)")
        else:
            _log(f"WARNING: converter returned path {result_path} but file does not exist")
    except Exception as e:
        _log(f"ERROR: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        _log("closing simulation app")
        simulation_app.close()


if __name__ == "__main__":
    main()
