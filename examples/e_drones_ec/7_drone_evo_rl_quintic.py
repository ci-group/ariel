"""Evolve drone morphology with ARIEL EA; evaluate each body by training a
short PPO policy using quintic-generated gate tracks.

Two modes (--mode):
  naive   Sample N gates from a quintic path once per run; pass as fixed
          gate_pos / gate_yaw to TorchDroneGateEnv — all physics run on the
          target device (GPU or CPU) via hand-written PyTorch dynamics.
  online  QuinticGateEnv subclass: generate the next gate on-the-fly each
          time the drone passes through a gate.  The policy sees an
          ever-changing course, which may promote more general controllers.
          (Uses the NumPy-backed DroneGateEnv for now.)

Smoke test (CPU, ~2 min):
    uv run examples/e_drones_ec/7_drone_evo_rl_quintic.py \\
        --pop 4 --gens 2 --ppo-steps 20000 --num-envs 20 --device cpu

Moderate run (GPU, online mode):
    uv run examples/e_drones_ec/7_drone_evo_rl_quintic.py \\
        --pop 8 --gens 5 --ppo-steps 200000 --num-envs 100 \\
        --device cuda:0 --mode online
"""
# NOTE: no ``from __future__ import annotations`` — ariel's EAOperation
# inspects real annotation objects at decoration time; stringified annotations
# (PEP 563) would break the introspection.

import argparse
import base64
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from rich.console import Console
from rich.progress import track
from stable_baselines3 import PPO

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from ariel.body_phenotypes.drone import (
    crossover_drones,
    deserialize_genome,
    initialize_drones,
    mutate_drones,
    parent_tag,
    truncation_select,
)
from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.drone.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)
from ariel.simulation.tasks.drone_gate_env import DroneGateEnv
from ariel.simulation.tasks.torch_drone_gate_env import TorchDroneGateEnv

# Add the LTU quintic goal generator to the path so we can import it directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "goal_generator_ltu" / "polynomial_goal_generator"))
from planner_generator import generate_paths_from_coefficients  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
time_format = "%Y%m%d_%H%M%S"
curr_time = time.strftime(time_format)


console = Console()
parser = argparse.ArgumentParser(
    description="Drone morphology evolution evaluated via PPO on quintic gate tracks"
)
parser.add_argument("--pop", type=int, default=4,
                    help="Body population size (default 4)")
parser.add_argument("--gens", type=int, default=3,
                    help="EA generations (default 3)")
parser.add_argument("--ppo-steps", type=int, default=200_000,
                    help="PPO training timesteps per morphology (default 200 000)")
parser.add_argument("--num-envs", type=int, default=500,
                    help="Parallel environments inside the gate env (default 100)")
parser.add_argument("--eval-steps", type=int, default=20_000,
                    help="Deterministic rollout steps used to score each policy (default 2 000)")
parser.add_argument("--prop-size", type=int, default=2,
                    help="Propeller size in inches (default 2)")
parser.add_argument("--device", default="cpu",
                    help="Torch device: cpu or cuda:0 (default cpu)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out-dir", default=f"__data__/drone_evo_rl_quintic/{curr_time}",)

# Quintic-specific
_DEFAULT_COEFFS = str(
    _REPO_ROOT / "goal_generator_ltu" / "polynomial_goal_generator" / "quintic_coeffs.npy"
)
parser.add_argument("--coeffs", default=_DEFAULT_COEFFS,
                    help="Path to quintic_coeffs.npy (shape N×12)")
parser.add_argument("--path-steps", type=int, default=15,
                    help="Number of gates sampled from the quintic path (default 15)")
parser.add_argument("--path-scale", type=float, default=5.0,
                    help="Spatial scale applied to quintic x,y ∈ [-1,1] (default 5.0 → ≈ ±5 m)")
parser.add_argument("--mode", choices=["naive", "online"], default="naive",
                    help="naive: fixed quintic gates per run; online: generate next gate on each pass")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

DATA = Path(args.out_dir)
DATA.mkdir(parents=True, exist_ok=True)
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
DB_PATH = DATA / f"database_{RUN_ID}.db"

COEFFS = np.load(args.coeffs)
Z_HEIGHT = -1.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_to_b64(model: PPO) -> str:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        return base64.b64encode(Path(tmp_path).read_bytes()).decode("ascii")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _b64_to_policy_file(b64: str, out_path: Path) -> None:
    out_path.write_bytes(base64.b64decode(b64))


def _quintic_to_gates(
    coeffs: np.ndarray,
    n_gates: int,
    scale: float,
    z: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one quintic path, return gate_pos (N×3, float32) and gate_yaw (N, float32).

    Gate yaw is taken directly from the planner's tangent at each sampled
    point on the dense path.  This is the correct source: the planner computes
    the smooth quintic tangent over many points, so it is stable even when only
    a handful of waypoints are ultimately kept.  Computing yaw from the noisy
    gate-to-gate difference vector breaks down when consecutive sampled gates
    are close together and gives a yaw that mismatches the actual travel
    direction, making gates impossible to pass in sequence.
    """
    # Sample at higher density then downsample to n_gates with minimum spacing,
    # so that consecutive gates are never nearly co-located.
    OVERSAMPLE = max(n_gates * 8, 64)
    MIN_SPACING = 0.3                                               # metres, after scaling
    paths, yaws_dense_arr = generate_paths_from_coefficients(
        coeffs, num_generate=1, steps=OVERSAMPLE, seed=seed, clip_range=(-1.0, 1.0),
    )
    xy_dense  = paths[0] * scale                                   # (OVERSAMPLE, 2)
    yaw_dense = yaws_dense_arr[0]                                  # (OVERSAMPLE,)  planner tangents

    # Greedy stride-based downsample: walk the dense path and only keep a
    # point when it is at least MIN_SPACING away from the last kept point.
    kept = [0]
    for idx in range(1, len(xy_dense)):
        if np.linalg.norm(xy_dense[idx] - xy_dense[kept[-1]]) >= MIN_SPACING:
            kept.append(idx)
        if len(kept) == n_gates:
            break
    # Fallback: if path is too short, spread evenly (shouldn't happen in practice).
    if len(kept) < n_gates:
        kept = list(np.linspace(0, len(xy_dense) - 1, n_gates, dtype=int))
    xy = xy_dense[kept]                                            # (n_gates, 2)

    gate_pos = np.column_stack(
        [xy, np.full(n_gates, z, dtype=np.float64)]
    ).astype(np.float32)

    # Use the planner's smooth tangent yaw at the sampled indices — not the
    # noisy gate-to-gate difference vector.
    gate_yaw = yaw_dense[kept].astype(np.float32)
    return gate_pos, gate_yaw


# ---------------------------------------------------------------------------
# QuinticGateEnv — online gate generation
# ---------------------------------------------------------------------------

class QuinticGateEnv(DroneGateEnv):
    """DroneGateEnv that replaces each gate slot on-the-fly as the drone passes it.

    Maintains a circular buffer of ``num_gates`` slots.  When any env passes
    gate slot *i*, that slot is overwritten with the next gate drawn from the
    quintic stream and the relative-gate observation arrays are recomputed so
    the policy always sees consistent look-ahead information.

    The quintic stream is refilled in batches of ``_BATCH_STEPS`` path points
    whenever the internal queue runs dry.
    """

    _BATCH_STEPS: int = 100

    def __init__(
        self,
        coeffs: np.ndarray,
        path_scale: float = 3.0,
        z_height: float = Z_HEIGHT,
        num_gates: int = 8,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._coeffs = coeffs
        self._path_scale = path_scale
        self._z_height = z_height
        # Independent RNG so global np.random state (set by DroneGateEnv) does
        # not interfere with gate generation.
        self._gate_rng = np.random.default_rng(seed)
        self._gate_queue: list[tuple[np.ndarray, float]] = []
        self._refill_queue()

        init_gate_pos, init_gate_yaw = self._pull_n_gates(num_gates)
        start_pos = (init_gate_pos[0] + np.array([0.0, -1.0, 0.0])).astype(np.float32)

        super().__init__(
            gates_pos=init_gate_pos,
            gate_yaw=init_gate_yaw,
            start_pos=start_pos,
            seed=seed,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Gate stream management
    # ------------------------------------------------------------------

    def _refill_queue(self) -> None:
        """Generate one quintic path batch and append its points to the queue.

        Gate yaw is taken directly from the planner's tangent at each point —
        not from the gate-to-gate difference vector, which is noisy when
        consecutive points are close together and produces yaws that mismatch
        the actual travel direction.
        """
        paths, yaws_arr = generate_paths_from_coefficients(
            self._coeffs,
            num_generate=1,
            steps=self._BATCH_STEPS,
            seed=int(self._gate_rng.integers(0, 2**31)),
            clip_range=(-1.0, 1.0),
        )
        xy        = paths[0] * self._path_scale                    # (BATCH_STEPS, 2)
        gate_yaws = yaws_arr[0]                                    # (BATCH_STEPS,) planner tangents
        for i in range(len(xy)):
            pos = np.array([xy[i, 0], xy[i, 1], self._z_height], dtype=np.float32)
            self._gate_queue.append((pos, float(gate_yaws[i])))

    def _pull_one_gate(self) -> tuple[np.ndarray, float]:
        if not self._gate_queue:
            self._refill_queue()
        return self._gate_queue.pop(0)

    def _pull_n_gates(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        while len(self._gate_queue) < n:
            self._refill_queue()
        gates = [self._gate_queue.pop(0) for _ in range(n)]
        pos = np.array([g[0] for g in gates], dtype=np.float32)
        yaw = np.array([g[1] for g in gates], dtype=np.float32)
        return pos, yaw

    # ------------------------------------------------------------------
    # Relative-gate bookkeeping
    # ------------------------------------------------------------------

    def _recompute_rel_for_slot(self, slot: int) -> None:
        """Recompute gate_pos_rel / gate_yaw_rel for *slot* and its successor.

        Must be called after gate_pos[slot] / gate_yaw[slot] are updated so
        that the policy observation stays geometrically consistent.
        """
        n = self.num_gates
        for i in (slot, (slot + 1) % n):
            prev_i = (i - 1) % n
            self.gate_pos_rel[i] = self.gate_pos[i] - self.gate_pos[prev_i]
            c = np.cos(self.gate_yaw[prev_i])
            s = np.sin(self.gate_yaw[prev_i])
            R = np.array([[c, s], [-s, c]])
            self.gate_pos_rel[i, 0:2] = R @ self.gate_pos_rel[i, 0:2]
            diff = float(self.gate_yaw[i]) - float(self.gate_yaw[prev_i])
            diff %= 2 * np.pi
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi
            self.gate_yaw_rel[i] = diff

    # ------------------------------------------------------------------
    # Step override
    # ------------------------------------------------------------------

    def step_wait(self):
        prev_targets = self.target_gates.copy()
        states, rewards, dones, infos = super().step_wait()

        # Identify slots that were just passed by non-terminal envs.
        # Done envs have their target_gates reset inside super().step_wait()
        # (via reset_()); we exclude them so we don't fire on reset noise.
        passed_mask = (self.target_gates != prev_targets) & ~dones
        if np.any(passed_mask):
            for slot in np.unique(prev_targets[passed_mask]):
                new_pos, new_yaw = self._pull_one_gate()
                self.gate_pos[int(slot)] = new_pos
                self.gate_yaw[int(slot)] = new_yaw
                self._recompute_rel_for_slot(int(slot))

        return states, rewards, dones, infos


# ---------------------------------------------------------------------------
# Per-individual PPO evaluation
# ---------------------------------------------------------------------------

PARAMETER_LIMITS = np.array([
    [0.055, 0.17],           # arm length (m)
    [-np.pi, np.pi],         # arm azimuth
    [-np.pi / 2, np.pi / 2], # arm elevation
    [-np.pi, np.pi],         # motor disc azimuth
    [-np.pi, np.pi],         # motor disc pitch
    [0, 1],                  # propeller spin direction
])


def _train_and_eval(
    genotype: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[float, Any, int]:
    """Decode genome, train PPO for cfg['ppo_steps'], evaluate deterministically.

    Returns (fitness, trained_model, n_motors).
    fitness = mean episode reward (higher is better).
    Returns (-inf, None, 0) for invalid or undecodable genomes.

    Warm-start: if cfg['policy_bank'] contains a policy trained on a
    morphology with the same number of motors, its network weights are
    copied into the freshly-constructed PPO before training begins.  This
    transfers learned flight dynamics without inheriting stale optimizer
    state — offspring (which share arm count with their parent ~90% of the
    time) typically converge faster.
    """
    try:
        genome = deserialize_genome(genotype)
    except Exception:
        return float("-inf"), None, 0

    valid = np.isfinite(genome.arms[:, 0])
    if not valid.any():
        return float("-inf"), None, 0

    try:
        bp = spherical_angular_to_blueprint(genome.arms, propsize=cfg["prop_size"])
        propellers = blueprint_to_propellers(bp, convention="ned")
    except Exception:
        return float("-inf"), None, 0

    n_motors = len(propellers)
    env = cfg["make_env"](propellers)

    n_steps = max(64, cfg["ppo_steps"] // (cfg["num_envs"] * 10))
    rollout_size = n_steps * cfg["num_envs"]
    # Pick the largest batch size ≤ rollout_size//4 that divides the rollout
    # exactly — no truncated tail batches, and large enough for good GPU util.
    _target = max(4096, rollout_size // 4)
    batch_size = next(b for b in range(_target, 0, -1) if rollout_size % b == 0)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            activation_fn=torch.nn.SiLU,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            log_std_init=0.0,
        ),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=20,
        gamma=0.999,
        learning_rate=1e-3,
        clip_range=0.2,
        ent_coef=0.01,
        device=cfg["device"],
        verbose=0,
    )

    # Warm-start: overlay weights from the best known policy for this motor
    # count.  We copy only the actor-critic network (policy.state_dict()),
    # not the optimizer state, so the fresh optimizer explores around the
    # warm-start point rather than inheriting momentum from a different body.
    warm_b64: str | None = cfg["policy_bank"].get(n_motors)
    if warm_b64 is not None:
        _tmp = Path(tempfile.mktemp(suffix=".zip"))
        try:
            _tmp.write_bytes(base64.b64decode(warm_b64))
            _warm = PPO.load(str(_tmp), env=env, device=cfg["device"])
            model.policy.load_state_dict(_warm.policy.state_dict())
            del _warm
        except Exception:
            pass  # fall back to random init if loading fails for any reason
        finally:
            _tmp.unlink(missing_ok=True)

    model.learn(total_timesteps=cfg["ppo_steps"])

    obs = env.reset()
    ep_rewards: list[float] = []
    cur_rews = np.zeros(cfg["num_envs"], dtype=np.float32)
    # eval_steps is a budget of *transitions* (env.step calls × num_envs).
    # Divide by num_envs so wall-clock cost stays constant regardless of how
    # many parallel envs are used (20 000 iterations × 2000 envs = 40 M
    # transitions; with 500 envs that's only 10 M for the same loop count).
    # Clamp to at least 500 env.step() calls so episodes actually complete —
    # untrained drones can take 50-200 steps before going OOB, and at fewer
    # than ~200 steps there's a real risk of no completions at all.
    eval_loop_steps = max(cfg["eval_steps"] // cfg["num_envs"], 500)
    for _ in range(eval_loop_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rews, dones, _ = env.step(action)
        cur_rews += rews
        for i, done in enumerate(dones):
            if done:
                ep_rewards.append(float(cur_rews[i]))
                cur_rews[i] = 0.0

    mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
    return mean_reward, model, n_motors


# ---------------------------------------------------------------------------
# EA genome handler + evaluation operation
# ---------------------------------------------------------------------------

N_ARMS = 6
template_handler = SphericalAngularDroneGenomeHandler(
    min_max_narms=(N_ARMS, N_ARMS),
    parameter_limits=PARAMETER_LIMITS,
    append_arm_chance=0.0,   # arm count is fixed; mutations only perturb arm params
    bilateral_plane_for_symmetry=None,
    repair=False,
    rnd=np.random.default_rng(args.seed),
)

# Build the env factory for the chosen mode.
if args.mode == "naive":
    _gate_pos, _gate_yaw = _quintic_to_gates(
        COEFFS, args.path_steps, args.path_scale, Z_HEIGHT, seed=args.seed,
    )
    _start_pos = (_gate_pos[0] + np.array([0.0, -1.0, 0.0])).astype(np.float32)

    def _make_env(propellers: Any) -> TorchDroneGateEnv:
        return TorchDroneGateEnv(
            propellers=propellers,
            num_envs=args.num_envs,
            gates_pos=_gate_pos,
            gate_yaw=_gate_yaw,
            start_pos=_start_pos,
            device=args.device,
            dt=0.01,
            seed=args.seed,
        )
else:
    def _make_env(propellers: Any) -> QuinticGateEnv:  # type: ignore[misc]
        return QuinticGateEnv(
            coeffs=COEFFS,
            path_scale=args.path_scale,
            z_height=Z_HEIGHT,
            num_gates=args.path_steps,
            propellers=propellers,
            num_envs=args.num_envs,
            device=args.device,
            dt=0.01,
            seed=args.seed,
        )

cfg: dict[str, Any] = {
    "prop_size": args.prop_size,
    "num_envs": args.num_envs,
    "device": args.device,
    "ppo_steps": args.ppo_steps,
    "eval_steps": args.eval_steps,
    "seed": args.seed,
    "make_env": _make_env,
    # Warm-start policy bank: n_motors → base64-encoded best policy zip.
    # Updated after each evaluation; shared across all generations so
    # offspring always start from the best known weights for their motor count.
    "policy_bank": {},
    "policy_bank_fitness": {},  # n_motors → best fitness seen so far
}

console.rule("[bold blue]Drone Body Evolution + PPO Controller (quintic gate task)")
console.log(
    f"mode={args.mode}  pop={args.pop}  gens={args.gens}  "
    f"ppo_steps={args.ppo_steps:,}  num_envs={args.num_envs}  "
    f"path_steps={args.path_steps}  path_scale={args.path_scale}  device={args.device}"
)
console.log(f"DB → {DB_PATH}")


def evaluate_with_ppo(population: Population) -> Population:
    """Train a PPO policy for every unevaluated body; assign fitness."""
    to_eval = list(population.alive.unevaluated)
    if not to_eval:
        return population

    for _i, ind in enumerate(track(to_eval, description="PPO evaluation…", console=console)):
        t0 = time.time()
        fit, model, n_motors = _train_and_eval(ind.genotype, cfg)
        ind.fitness = fit
        warm_tag = ""
        if model is not None:
            ind.tags["policy_b64"] = _model_to_b64(model)
            # Update the warm-start bank if this is the best policy seen for
            # this motor count — benefits all future individuals with same dims.
            if np.isfinite(fit):
                prev_best = cfg["policy_bank_fitness"].get(n_motors, float("-inf"))
                if fit > prev_best:
                    cfg["policy_bank"][n_motors] = ind.tags["policy_b64"]
                    cfg["policy_bank_fitness"][n_motors] = fit
                    warm_tag = " [new bank best]"
        ws = "warm" if cfg["policy_bank"].get(n_motors) and warm_tag == "" else "cold"
        # ind.id is the SQLite PK — None until EA.run() commits the row.
        # Show the DB id when available, otherwise a local 1-based counter.
        display_id = ind.id if ind.id is not None else f"#{_i + 1}"
        console.log(
            f"  id={display_id}  motors={n_motors}  mean_reward={fit:.3f}"
            f"  start={ws}{warm_tag}  ({time.time() - t0:.1f}s)"
        )

    finite = [
        ind.fitness_ for ind in population.alive.evaluated
        if ind.fitness_ is not None and np.isfinite(ind.fitness_)
    ]
    if finite:
        console.log(
            f"  batch done — max={max(finite):.4f}  mean={np.mean(finite):.4f}"
        )
    return population


# ---------------------------------------------------------------------------
# Run evolution
# ---------------------------------------------------------------------------

initial_pop = Population([Individual() for _ in range(args.pop)])
init_op = initialize_drones(template_handler=template_handler)
eval_op = EAOperation(evaluate_with_ppo)

console.log("Initializing + evaluating initial population …")
initial_pop = init_op(initial_pop)
initial_pop = eval_op(initial_pop)

finite0 = [
    ind.fitness_ for ind in initial_pop
    if ind.fitness_ is not None and np.isfinite(ind.fitness_)
]
if finite0:
    console.log(f"Initial — max={max(finite0):.4f}  mean={np.mean(finite0):.4f}")

generation_ops = [
    parent_tag(n=args.pop),
    crossover_drones(template_handler=template_handler),
    mutate_drones(template_handler=template_handler),
    eval_op,
    truncation_select(n=args.pop),
]

ea = EA(
    population=initial_pop,
    operations=generation_ops,
    num_steps=args.gens,
    db_file_path=DB_PATH,
    db_handling="delete",
)

console.rule("[bold green]Evolving")
ea.run()


# ---------------------------------------------------------------------------
# Save best individual
# ---------------------------------------------------------------------------

ea.fetch_population(only_alive=False, requires_eval=False)
finite = [
    ind for ind in ea.population
    if ind.fitness_ is not None and np.isfinite(ind.fitness_)
]
if not finite:
    console.log("[red]No valid individuals found — aborting.[/red]")
    raise SystemExit(1)

best = sorted(finite, key=lambda ind: ind.fitness_, reverse=True)[0]
console.rule("[bold cyan]Best individual")
console.log(f"  id={best.id}  mean_reward={best.fitness_:.4f}  born={best.time_of_birth}")

best_genome = deserialize_genome(best.genotype)
valid_mask = ~np.isnan(best_genome.arms[:, 0])
console.log(f"  active arms: {int(valid_mask.sum())} / {best_genome.arms.shape[0]}")

best_bp = spherical_angular_to_blueprint(best_genome.arms, propsize=args.prop_size)
bp_path = DATA / f"best_blueprint_{RUN_ID}.json"
best_bp.save_json(bp_path)
console.log(f"  blueprint → {bp_path}")

# Save gate configuration so the MuJoCo visualizer can render the track.
# Both modes call _quintic_to_gates with the same seed used at training time,
# so naive mode reproduces the identical fixed track; online mode produces a
# representative snapshot of the quintic distribution.
_vis_gate_pos, _vis_gate_yaw = _quintic_to_gates(
    COEFFS, args.path_steps, args.path_scale, Z_HEIGHT, seed=args.seed
)
gate_pos_path = DATA / f"gate_pos_{RUN_ID}.npy"
gate_yaw_path = DATA / f"gate_yaw_{RUN_ID}.npy"
np.save(gate_pos_path, _vis_gate_pos)
np.save(gate_yaw_path, _vis_gate_yaw)
console.log(f"  gate_pos  → {gate_pos_path}")
console.log(f"  gate_yaw  → {gate_yaw_path}")

policy_b64 = best.tags.get("policy_b64", "")
if policy_b64:
    policy_path = DATA / f"best_policy_{RUN_ID}.zip"
    _b64_to_policy_file(policy_b64, policy_path)
    console.log(f"  policy    → {policy_path}")
    console.log(
        f"\n  visualize:\n"
        f"    uv run examples/e_drones_ec/6_visualize_evo_results.py \\\n"
        f"        --blueprint {bp_path} --policy {policy_path} \\\n"
        f"        --gate-pos {gate_pos_path} --gate-yaw {gate_yaw_path} --no-show"
    )
else:
    console.log("[yellow]  policy_b64 tag missing — policy not saved[/yellow]")

console.log(f"[bold green]Done. DB → {DB_PATH}")
