from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from ariel.body_phenotypes.lynx_mjspec.unified_pipeline.common import (
    DEFAULT_CTRL_FREQ,
    DEFAULT_SIM_STEPS,
    DEFAULT_TARGET,
    DEFAULT_TOUCH_THRESHOLD,
    PolicySpec,
    FastNumpyNetwork,
    apply_position_delta_control,
    build_model,
    build_observation,
    set_target_position,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Lynx replay pipeline")
    parser.add_argument("--artifact-dir", type=str, default="__data__/lynx_mjspec/unified")
    parser.add_argument("--sim-steps", type=int, default=DEFAULT_SIM_STEPS)
    parser.add_argument("--ctrl-freq", type=int, default=DEFAULT_CTRL_FREQ)
    parser.add_argument("--target-x", type=float, default=float(DEFAULT_TARGET[0]))
    parser.add_argument("--target-y", type=float, default=float(DEFAULT_TARGET[1]))
    parser.add_argument("--target-z", type=float, default=float(DEFAULT_TARGET[2]))
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def resolve_artifact_dir(artifact_dir: Path) -> Path:
    if (artifact_dir / "metadata.json").exists():
        return artifact_dir

    run_dirs = [path for path in artifact_dir.iterdir() if path.is_dir() and (path / "metadata.json").exists()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directory with metadata.json found under: {artifact_dir}")

    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def load_artifacts(artifact_dir: Path) -> tuple[np.ndarray, np.ndarray, PolicySpec, dict[str, object]]:
    artifact_dir = resolve_artifact_dir(artifact_dir)

    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    policy_dict = metadata.get("policy", {})

    policy = PolicySpec(
        input_size=int(policy_dict.get("input_size", PolicySpec.input_size)),
        hidden_size=int(policy_dict.get("hidden_size", PolicySpec.hidden_size)),
        output_size=int(policy_dict.get("output_size", PolicySpec.output_size)),
        action_scale=float(policy_dict.get("action_scale", PolicySpec.action_scale)),
        max_delta=float(policy_dict.get("max_delta", PolicySpec.max_delta)),
    )

    tubes = np.load(artifact_dir / "best_tube_lengths.npy")
    brain = np.load(artifact_dir / "best_brain_weights.npy")
    return tubes, brain, policy, metadata


def run_rollout(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    net: FastNumpyNetwork,
    policy: PolicySpec,
    tcp_sid: int,
    tgt_sid: int,
    joint_ids: list[int],
    target: np.ndarray,
    sim_steps: int,
    ctrl_freq: int,
    touch_threshold: float,
) -> tuple[float, float, float, float | None]:
    set_target_position(model, data, tgt_sid, target)

    initial_distance = float(np.linalg.norm(data.site_xpos[tcp_sid] - data.site_xpos[tgt_sid]))
    min_distance = initial_distance
    final_distance = initial_distance
    first_touch_time: float | None = None
    controls_locked = False
    locked_ctrl = np.zeros(model.nu, dtype=np.float64)

    for step in range(sim_steps):
        if step % ctrl_freq == 0 and not controls_locked:
            obs = build_observation(model, data, joint_ids, tcp_sid, tgt_sid)
            action = net.forward(obs)
            apply_position_delta_control(
                model=model,
                data=data,
                joint_ids=joint_ids,
                action=action,
                action_scale=policy.action_scale,
                max_delta=policy.max_delta,
            )
            locked_ctrl[:] = data.ctrl[:]

        if controls_locked:
            data.ctrl[:] = locked_ctrl

        mujoco.mj_step(model, data)

        d = float(np.linalg.norm(data.site_xpos[tcp_sid] - data.site_xpos[tgt_sid]))
        final_distance = d
        min_distance = min(min_distance, d)
        if first_touch_time is None and d <= touch_threshold:
            first_touch_time = float(data.time)
            controls_locked = True
            locked_ctrl[:] = data.ctrl[:]

    return initial_distance, min_distance, final_distance, first_touch_time


def main() -> None:
    args = parse_args()

    artifact_dir = Path(args.artifact_dir)
    tubes, brain, policy, metadata = load_artifacts(artifact_dir)

    default_target = np.array(metadata.get("target", DEFAULT_TARGET.tolist()), dtype=np.float64)
    touch_threshold = float(metadata.get("touch_threshold", DEFAULT_TOUCH_THRESHOLD))
    target = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float64)
    if np.allclose(target, DEFAULT_TARGET):
        target = default_target

    model, data, tcp_sid, tgt_sid, joint_ids = build_model(tubes)
    net = FastNumpyNetwork(
        input_size=policy.input_size,
        hidden_size=policy.hidden_size,
        output_size=policy.output_size,
        weights=brain,
    )

    if args.eval_only:
        init_d, min_d, final_d, touch_t = run_rollout(
            model=model,
            data=data,
            net=net,
            policy=policy,
            tcp_sid=tcp_sid,
            tgt_sid=tgt_sid,
            joint_ids=joint_ids,
            target=target,
            sim_steps=int(args.sim_steps),
            ctrl_freq=int(args.ctrl_freq),
            touch_threshold=touch_threshold,
        )
        print(f"initial_distance={init_d:.6f}")
        print(f"min_distance={min_d:.6f}")
        print(f"final_distance={final_d:.6f}")
        print(f"touch_time={touch_t if touch_t is not None else 'none'}")
        return

    set_target_position(model, data, tgt_sid, target)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if hasattr(mujoco.mjtVisFlag, "mjVIS_SITE"):
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SITE] = True
        else:
            viewer.opt.sitegroup[:] = 1
        step = 0
        controls_locked = False
        locked_ctrl = np.zeros(model.nu, dtype=np.float64)
        while viewer.is_running():
            if step % int(args.ctrl_freq) == 0 and not controls_locked:
                obs = build_observation(model, data, joint_ids, tcp_sid, tgt_sid)
                action = net.forward(obs)
                apply_position_delta_control(
                    model=model,
                    data=data,
                    joint_ids=joint_ids,
                    action=action,
                    action_scale=policy.action_scale,
                    max_delta=policy.max_delta,
                )
                locked_ctrl[:] = data.ctrl[:]

            if controls_locked:
                data.ctrl[:] = locked_ctrl

            d = float(np.linalg.norm(data.site_xpos[tcp_sid] - data.site_xpos[tgt_sid]))
            if not controls_locked and d <= touch_threshold:
                controls_locked = True
                locked_ctrl[:] = data.ctrl[:]

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1 / 60.0)
            step += 1


if __name__ == "__main__":
    main()
