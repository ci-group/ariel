from __future__ import annotations

import time
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import mujoco
import nevergrad as ng
import numpy as np

from ariel.body_phenotypes.lynx_mjspec.unified_pipeline.common import (
    DEFAULT_CTRL_FREQ,
    DEFAULT_SIM_STEPS,
    DEFAULT_TOUCH_THRESHOLD,
    DEFAULT_TARGET,
    NUM_TUBES,
    PolicySpec,
    FastNumpyNetwork,
    apply_position_delta_control,
    build_model,
    build_observation,
    num_network_params,
    set_target_position,
)

INVALID_FITNESS = 999.0
DEFAULT_HOLD_THRESHOLD = 0.02
DEFAULT_HOLD_STEPS_TO_STOP = 60
DEFAULT_TIME_BONUS_WEIGHT = 0.08


def evaluate_candidate(
    genome: np.ndarray,
    policy_spec: PolicySpec,
    target: np.ndarray,
    sim_steps: int,
    ctrl_freq: int,
    touch_threshold: float,
    hold_threshold: float,
    hold_steps_to_stop: int,
    time_bonus_weight: float,
) -> float:
    try:
        tube_lengths = genome[:NUM_TUBES]
        weights = genome[NUM_TUBES:]

        model, data, tcp_sid, tgt_sid, joint_ids = build_model(tube_lengths)
        set_target_position(model, data, tgt_sid, target)

        net = FastNumpyNetwork(
            input_size=policy_spec.input_size,
            hidden_size=policy_spec.hidden_size,
            output_size=policy_spec.output_size,
            weights=weights,
        )

        min_distance = float("inf")
        final_distance = float("inf")
        first_touch_step = sim_steps
        touch_steps = 0
        consecutive_hold_steps = 0
        executed_steps = 0

        for step in range(sim_steps):
            if step % ctrl_freq == 0:
                obs = build_observation(model, data, joint_ids, tcp_sid, tgt_sid)
                action = net.forward(obs)
                apply_position_delta_control(
                    model=model,
                    data=data,
                    joint_ids=joint_ids,
                    action=action,
                    action_scale=policy_spec.action_scale,
                    max_delta=policy_spec.max_delta,
                )

            mujoco.mj_step(model, data)
            executed_steps = step + 1
            d = float(np.linalg.norm(data.site_xpos[tcp_sid] - data.site_xpos[tgt_sid]))
            final_distance = d
            min_distance = min(min_distance, d)

            if d <= hold_threshold:
                touch_steps += 1
                consecutive_hold_steps += 1
            else:
                consecutive_hold_steps = 0

            if d <= touch_threshold and first_touch_step == sim_steps:
                first_touch_step = step

            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                return INVALID_FITNESS

            # End the rollout early once the policy has reached and stably held near target.
            if d <= touch_threshold and consecutive_hold_steps >= hold_steps_to_stop:
                break

        touched = first_touch_step < sim_steps
        touch_latency = (first_touch_step / sim_steps) if touched else 1.0
        hold_ratio = touch_steps / max(1, executed_steps)
        remaining_ratio = (sim_steps - executed_steps) / sim_steps
        time_bonus = time_bonus_weight * remaining_ratio if touched else 0.0

        return (
            0.50 * min_distance
            + 0.35 * final_distance
            + 0.15 * touch_latency
            - 0.12 * hold_ratio
            - time_bonus
            + (0.10 if not touched else 0.0)
        )
    except Exception:
        return INVALID_FITNESS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Lynx evolution pipeline")
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--sim-steps", type=int, default=DEFAULT_SIM_STEPS)
    parser.add_argument("--ctrl-freq", type=int, default=DEFAULT_CTRL_FREQ)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument("--max-delta", type=float, default=0.12)
    parser.add_argument("--target-x", type=float, default=float(DEFAULT_TARGET[0]))
    parser.add_argument("--target-y", type=float, default=float(DEFAULT_TARGET[1]))
    parser.add_argument("--target-z", type=float, default=float(DEFAULT_TARGET[2]))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="__data__/lynx_mjspec/unified")
    parser.add_argument("--touch-threshold", type=float, default=DEFAULT_TOUCH_THRESHOLD)
    parser.add_argument("--hold-threshold", type=float, default=DEFAULT_HOLD_THRESHOLD)
    parser.add_argument("--hold-steps-to-stop", type=int, default=DEFAULT_HOLD_STEPS_TO_STOP)
    parser.add_argument("--time-bonus-weight", type=float, default=DEFAULT_TIME_BONUS_WEIGHT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    target = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float64)

    policy_spec = PolicySpec(
        hidden_size=int(args.hidden_size),
        action_scale=float(args.action_scale),
        max_delta=float(args.max_delta),
    )

    num_weights = num_network_params(policy_spec)
    genome_size = NUM_TUBES + num_weights

    init = np.zeros(genome_size, dtype=np.float64)
    init[:NUM_TUBES] = 0.10
    init[NUM_TUBES:] = rng.uniform(-0.1, 0.1, size=num_weights)

    parametrization = ng.p.Array(init=init)
    parametrization.set_mutation(sigma=float(args.sigma))

    optimizer = ng.optimizers.CMA(
        parametrization=parametrization,
        budget=int(args.generations) * int(args.population),
        num_workers=min(int(args.population), int(args.workers)),
    )

    workers = min(int(args.population), int(args.workers))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for gen in range(1, int(args.generations) + 1):
            candidates = [optimizer.ask() for _ in range(int(args.population))]
            genomes = [np.asarray(c.value, dtype=np.float64) for c in candidates]

            fitnesses = list(
                executor.map(
                    evaluate_candidate,
                    genomes,
                    [policy_spec] * len(genomes),
                    [target] * len(genomes),
                    [int(args.sim_steps)] * len(genomes),
                    [int(args.ctrl_freq)] * len(genomes),
                    [float(args.touch_threshold)] * len(genomes),
                    [float(args.hold_threshold)] * len(genomes),
                    [int(args.hold_steps_to_stop)] * len(genomes),
                    [float(args.time_bonus_weight)] * len(genomes),
                )
            )

            for cand, fit in zip(candidates, fitnesses):
                optimizer.tell(cand, float(fit))

            print(
                f"gen={gen:03d} best={np.min(fitnesses):.5f} "
                f"mean={np.mean(fitnesses):.5f} worst={np.max(fitnesses):.5f}"
            )

    best_genome = np.asarray(optimizer.provide_recommendation().value, dtype=np.float64)
    best_tube_lengths = best_genome[:NUM_TUBES]
    best_weights = best_genome[NUM_TUBES:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = time.time()
    print(f"Saving unified artifacts to: {out_dir} at time {t}")

    np.save(out_dir / f"best_genome_{t}.npy", best_genome)
    np.save(out_dir / f"best_tube_lengths_{t}.npy", best_tube_lengths)
    np.save(out_dir / f"best_brain_weights_{t}.npy", best_weights)

    metadata = {
        "policy": asdict(policy_spec),
        "target": target.tolist(),
        "sim_steps": int(args.sim_steps),
        "ctrl_freq": int(args.ctrl_freq),
        "touch_threshold": float(args.touch_threshold),
        "hold_threshold": float(args.hold_threshold),
        "hold_steps_to_stop": int(args.hold_steps_to_stop),
        "time_bonus_weight": float(args.time_bonus_weight),
    }
    
    (out_dir / f"metadata_{t}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved unified artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
