from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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
    TUBE_MIN,
    TUBE_MAX,
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


def make_run_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Lynx evolution pipeline")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--weight-sigma", type=float, default=0.10)
    parser.add_argument("--optimizer", type=str, default="CMA",
                        choices=["CMA", "TBPSA", "TwoPointsDE", "NGOpt"])
    parser.add_argument("--sim-steps", type=int, default=DEFAULT_SIM_STEPS)
    parser.add_argument("--ctrl-freq", type=int, default=DEFAULT_CTRL_FREQ)
    parser.add_argument("--hidden-size", type=int, default=4)
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
        tube_lengths_raw = genome[:NUM_TUBES]
        weights = genome[NUM_TUBES:]

        # Penalize out-of-bounds tubes instead of silently clipping.
        # Per CMA-ES tutorial (B.5): simple repair violates distributional assumptions.
        # Evaluate at repaired point, add penalty proportional to violation distance.
        tube_lengths = np.clip(tube_lengths_raw, TUBE_MIN, TUBE_MAX)
        boundary_penalty = 2.0 * float(np.sum((tube_lengths_raw - tube_lengths) ** 2))

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
                # Return progress-aware penalty so CMA-ES can still rank diverging candidates.
                return min_distance + 1.0

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
            + boundary_penalty
        )
    except Exception:
        return INVALID_FITNESS

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

    tubes_init   = rng.uniform(TUBE_MIN, TUBE_MAX, size=NUM_TUBES).astype(np.float64)
    weights_init = rng.uniform(-0.1, 0.1, size=num_weights).astype(np.float64)

    # Split parametrization so tubes and weights get independent sigma values.
    # Tube sigma: 0.3 * (TUBE_MAX - TUBE_MIN) = 0.27 (per CMA-ES tutorial Eq. sigma=0.3*(b-a)).
    # Weight sigma: kept small to avoid tanh saturation in early generations.
    # set_bounds tells CMA the valid range; clipping keeps proposals in-bounds before penalization.
    tubes_p = (
        ng.p.Array(init=tubes_init)
        .set_bounds(TUBE_MIN, TUBE_MAX, method="clipping")
        .set_mutation(sigma=0.27)
    )
    weights_p = ng.p.Array(init=weights_init).set_mutation(sigma=float(args.weight_sigma))
    parametrization = ng.p.Tuple(tubes_p, weights_p)

    # num_workers must equal population so the optimizer's internal λ matches the batch size.
    optimizer = ng.optimizers.registry[args.optimizer](
        parametrization=parametrization,
        budget=int(args.generations) * int(args.population),
        num_workers=int(args.population),
    )

    workers = min(int(args.population), int(args.workers))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for gen in range(1, int(args.generations) + 1):
            candidates = [optimizer.ask() for _ in range(int(args.population))]
            genomes = [
                np.concatenate([np.asarray(c.value[0]), np.asarray(c.value[1])]).astype(np.float64)
                for c in candidates
            ]

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

            valid = [f for f in fitnesses if f < INVALID_FITNESS]
            try:
                sigma_str = f" σ={optimizer.optim.es.sigma:.4f}"
            except AttributeError:
                sigma_str = ""
            best_idx = int(np.argmin(fitnesses))
            best_tubes = np.round(np.asarray(candidates[best_idx].value[0]), 3)
            print(
                f"gen={gen:03d} best={np.min(fitnesses):.5f} "
                f"mean={np.mean(fitnesses):.5f} std={np.std(fitnesses):.5f} "
                f"worst={np.max(fitnesses):.5f} valid={len(valid)}/{len(fitnesses)}"
                f"{sigma_str}"
            )
            print(f"         tubes={best_tubes}")

    rec = optimizer.provide_recommendation()
    best_tube_lengths = np.asarray(rec.value[0], dtype=np.float64)
    best_weights      = np.asarray(rec.value[1], dtype=np.float64)
    best_genome       = np.concatenate([best_tube_lengths, best_weights])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = make_run_name()
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"Saving unified artifacts to: {run_dir}")

    np.save(run_dir / "best_genome.npy", best_genome)
    np.save(run_dir / "best_tube_lengths.npy", best_tube_lengths)
    np.save(run_dir / "best_brain_weights.npy", best_weights)

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

    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved unified artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
