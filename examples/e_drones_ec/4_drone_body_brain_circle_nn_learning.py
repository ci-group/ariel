"""Joint drone body + brain evolution: spherical body, NN circle-controller.

Mirrors the c_genotypes/4 pattern (body EA + per-individual NN brain learned
with CMA-ES + multiprocessing) but for drones — and, unlike the hover
examples in this folder, the NN *replaces* the Lee geometric controller:

    SphericalAngularDroneGenomeHandler        (direct body encoding)
        │   ariel.ec EA pipeline (parent_tag → crossover → mutate → …)
        ▼
    per individual:  inner CMA-ES loop  ─────────────────────────────┐
        │   genome → DroneBlueprint → MuJoCo model                   │
        │   CircleBrain (torch NN, CMA-tuned weights)                │
        │       state(pos,vel,quat,omega) + circle phase             │
        │           ▼                                                │
        │       data.ctrl[n_motors]   (per-motor thrust, no Lee)     │
        │           ▼                                                │
        │       phase-locked circle-tracking fitness  ◀──────────────┘
        ▼
    body fitness = best brain it could learn  (lower is better)

The brain is the *full* flight controller: CMA-ES must learn to both
stabilise the (evolved, possibly asymmetric) airframe and track a
horizontal circle. Body fitness is therefore the score of the best brain
that morphology could be taught — co-selecting for easy-to-control bodies.

Circle reference (ENU, Z-up)::

    ref(t) = center + R · [cos(2πt/T), sin(2πt/T), 0]
    fit    = mean‖pos − ref(t)‖ + tilt_weight·mean(1−cosθ)
                                + ctrl_weight·mean(ctrl²)

Outputs: SQLite DB, best DroneBlueprint JSON, best brain ``.npy``, MP4
video, optional passive viewer behind ``--visualize``.

This is compute-heavy: every body runs a full nested CMA-ES. Keep
``--pop``/``--budget``/``--learn-bud`` small for a quick smoke test.

Run::

    uv run examples/e_drones_ec/4_drone_body_brain_circle_nn_learning.py
    uv run examples/e_drones_ec/4_drone_body_brain_circle_nn_learning.py --pop 4 --budget 3 --workers 4
    uv run examples/e_drones_ec/4_drone_body_brain_circle_nn_learning.py --no-visualize
"""
# NOTE: no ``from __future__ import annotations`` — ariel's @EAOperation
# introspects the real ``Population`` annotation object at decoration time,
# which a stringified annotation would break.

import argparse
import multiprocessing as mp
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import nevergrad as ng
import numpy as np
import torch
from rich.console import Console
from rich.progress import track
from torch import nn

from ariel.body_phenotypes.drone import (
    crossover_drones,
    deserialize_genome,
    initialize_drones,
    mutate_drones,
    parent_tag,
)
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.drone.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)

# CMA-ES mirrored sampling (TPA) emits a harmless consistency UserWarning on
# noisy stochastic simulators — silence it so it doesn't drown real output.
warnings.filterwarnings(
    "ignore", message="TPA: apparent inconsistency", category=UserWarning, module="cma",
)


# ---------------------------------------------------------------------------
# Brain: a small NN that maps drone state + circle phase → per-motor thrust
# ---------------------------------------------------------------------------

# Feature layout fed to the brain each step (fixed size, body-independent):
#   pos error (3) | linear vel (3) | quaternion w,x,y,z (4) | body angvel (3)
#   | sin(phase), cos(phase) (2)                                  = 15
BRAIN_INPUT_SIZE = 15


class CircleBrain(nn.Module):
    """Weight-only NN controller — every motor's normalised thrust in [0, 1].

    Weights are never trained by gradient descent; they are filled in bulk
    from a CMA-ES candidate vector via :func:`fill_parameters`.
    """

    def __init__(self, n_motors: int, hidden_size: int = 12) -> None:
        super().__init__()
        self.fc1 = nn.Linear(BRAIN_INPUT_SIZE, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, n_motors)
        self.act = nn.Tanh()
        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, features: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        # Sigmoid keeps each command inside the actuator's [0, 1] ctrlrange.
        return torch.sigmoid(self.fc_out(x)).numpy()


@torch.no_grad()
def fill_parameters(net: nn.Module, vector: np.ndarray) -> None:
    """Overwrite every parameter of ``net`` from a flat CMA-ES vector."""
    address = 0
    for p in net.parameters():
        flat = p.data.view(-1)
        n = len(flat)
        flat[:] = torch.as_tensor(vector[address : address + n], dtype=flat.dtype)
        address += n


# ---------------------------------------------------------------------------
# Circle reference + feature extraction
# ---------------------------------------------------------------------------

def circle_reference(
    t: float,
    center: tuple[float, float, float],
    radius: float,
    period: float,
) -> np.ndarray:
    """Position of the moving reference point on the horizontal circle."""
    ang = 2.0 * np.pi * t / period
    return np.asarray(center, dtype=np.float64) + radius * np.array(
        [np.cos(ang), np.sin(ang), 0.0], dtype=np.float64,
    )


def circle_features(
    data: Any,
    center: tuple[float, float, float],
    radius: float,
    period: float,
) -> np.ndarray:
    """Build the 15-element brain input from the live MuJoCo state."""
    t = float(data.time)
    ref = circle_reference(t, center, radius, period)
    pos = np.asarray(data.qpos[0:3], dtype=np.float64)
    quat = np.asarray(data.qpos[3:7], dtype=np.float64)  # (w, x, y, z)
    vel = np.asarray(data.qvel[0:3], dtype=np.float64)
    angvel = np.asarray(data.qvel[3:6], dtype=np.float64)
    phase = 2.0 * np.pi * t / period
    return np.concatenate(
        [ref - pos, vel, quat, angvel, [np.sin(phase), np.cos(phase)]],
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Decoding + per-individual brain learning (worker-side, picklable)
# ---------------------------------------------------------------------------

def _genome_to_blueprint(genotype: dict[str, Any], propsize: int):
    """Stored spherical genome → DroneBlueprint, or ``None`` if invalid."""
    try:
        genome = deserialize_genome(genotype)
    except Exception:  # noqa: BLE001 - any decode failure → invalid body
        return None
    arms = getattr(genome, "arms", None)
    if arms is None or not np.isfinite(arms[:, 0]).any():
        return None
    try:
        return spherical_angular_to_blueprint(arms, propsize=int(propsize))
    except Exception:  # noqa: BLE001
        return None


def _rollout_circle(model: Any, data: Any, net: CircleBrain, cfg: dict[str, Any]) -> float:
    """Fly one circle attempt with ``net`` driving the motors; return fitness.

    Lower is better. A drone that diverges (NaN state / flies off) gets a
    large penalty that grows the earlier it failed, so CMA-ES still sees a
    gradient between "crashed instantly" and "crashed near the end".
    """
    import mujoco

    center, radius, period = cfg["center"], cfg["radius"], cfg["period"]
    dt = float(model.opt.timestep)
    steps = max(1, int(round(cfg["duration"] / dt)))
    warm_steps = int(round(max(0.0, cfg["warm_up"]) / dt))

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = circle_reference(0.0, center, radius, period)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    dist_acc = tilt_acc = ctrl_acc = 0.0
    n_logged = 0
    for i in range(steps):
        ctrl = np.clip(net.forward(circle_features(data, center, radius, period)), 0.0, 1.0)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if not np.isfinite(data.qpos).all() or np.abs(data.qpos[0:3]).max() > 50.0:
            return 1000.0 + 1000.0 * (steps - i) / steps

        if i >= warm_steps:
            ref = circle_reference(float(data.time), center, radius, period)
            dist_acc += float(np.linalg.norm(np.asarray(data.qpos[0:3]) - ref))
            x, y = float(data.qpos[4]), float(data.qpos[5])
            # Tilt deviation 1 - cosθ, where cosθ = 1 - 2(x² + y²) for (w,x,y,z).
            tilt_acc += 2.0 * (x * x + y * y)
            ctrl_acc += float(np.mean(ctrl ** 2))
            n_logged += 1

    if n_logged == 0:
        return float("inf")
    return (
        dist_acc / n_logged
        + cfg["tilt_weight"] * (tilt_acc / n_logged)
        + cfg["ctrl_weight"] * (ctrl_acc / n_logged)
    )


def _learn_circle_brain(
    genotype: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[float, list[float], list[float]]:
    """Inner CMA-ES loop: tune a brain for one fixed body.

    Returns ``(best_fitness, best_weight_vector, per_iteration_best)``.
    """
    from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
    from ariel.simulation.drone.controllers.lee_control.mujoco_bridge import (
        spawn_blueprint_in_world,
    )
    from ariel.simulation.drone.drone_interface import DroneInterface

    bp = _genome_to_blueprint(genotype, cfg["propsize"])
    if bp is None:
        return float("inf"), [], []

    propellers = blueprint_to_propellers(bp, convention="ned")
    if not propellers:
        return float("inf"), [], []

    try:
        # DroneInterface is used only as a source of the canonical drone mass
        # for mass-matched spawning — no Lee controller is built.
        target_mass = float(DroneInterface(0, propellers=propellers).params["mB"])
        spawned = spawn_blueprint_in_world(
            bp,
            propellers=propellers,
            target_mass=target_mass,
            spawn_position=tuple(
                float(v) for v in circle_reference(
                    0.0, cfg["center"], cfg["radius"], cfg["period"],
                )
            ),
        )
    except Exception:  # noqa: BLE001 - bad geometry → invalid body
        return float("inf"), [], []

    model, data = spawned.model, spawned.data
    n_motors = int(model.nu)
    if n_motors == 0:
        return float("inf"), [], []

    net = CircleBrain(n_motors, hidden_size=cfg["hidden"])
    # Warm-start the output layer so an untrained brain already commands
    # near-hover thrust. CMA-ES then only learns the tracking correction
    # instead of rediscovering how to stay airborne from scratch.
    g = abs(float(model.opt.gravity[2])) or 9.81
    hover_ctrl = float(np.clip(
        (target_mass * g / n_motors) / spawned.max_thrust_per_motor, 0.05, 0.95,
    ))
    with torch.no_grad():
        net.fc_out.weight.mul_(0.1)
        net.fc_out.bias.fill_(float(np.log(hover_ctrl / (1.0 - hover_ctrl))))

    init_vec = np.concatenate(
        [p.detach().cpu().numpy().ravel() for p in net.parameters()],
    ).astype(np.float64)
    num_params = int(init_vec.size)

    # CMA-ES needs λ ≥ 4 + ⌊3·ln(n)⌋ for healthy covariance adaptation;
    # round up to an even λ so mirrored sampling has paired candidates.
    learn_pop = max(cfg["learn_pop"], 4 + int(3 * np.log(max(num_params, 2))))
    if learn_pop % 2 != 0:
        learn_pop += 1

    param = ng.p.Array(init=init_vec).set_mutation(sigma=0.3)
    learner = ng.optimizers.registry["CMA"](
        parametrization=param,
        budget=cfg["learn_budget"] * learn_pop,
        num_workers=learn_pop,
    )

    best_fit = float("inf")
    best_vec = init_vec.tolist()
    iter_best: list[float] = []
    for _ in range(cfg["learn_budget"]):
        candidates = [learner.ask() for _ in range(learn_pop)]
        it_best = float("inf")
        for cand in candidates:
            vec = np.asarray(cand.value, dtype=np.float64)
            fill_parameters(net, vec)
            fit = _rollout_circle(model, data, net, cfg)
            learner.tell(cand, fit)  # nevergrad minimises the told loss
            if fit < best_fit:
                best_fit, best_vec = fit, vec.tolist()
            it_best = min(it_best, fit)
        iter_best.append(it_best)

    return best_fit, best_vec, iter_best


def _evaluate_individual_process(
    task: tuple[dict[str, Any], dict[str, Any]],
) -> tuple[float, list[float], list[float]]:
    """Worker entry point: learn a brain for one genome, return its score."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    genotype, cfg = task
    try:
        torch.set_num_threads(1)
        torch.manual_seed(int(cfg["seed"]))
        fit, vec, iter_best = _learn_circle_brain(genotype, cfg)
        if not np.isfinite(fit):
            return float("inf"), [], []
        # Per-iteration improvement (positive = CMA got better that round).
        deltas: list[float] = []
        prev: float | None = None
        for score in iter_best:
            deltas.append(0.0 if prev is None else prev - score)
            prev = score
        return float(fit), vec, deltas
    except Exception:  # noqa: BLE001 - never let one bad body kill the pool
        return float("inf"), [], []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint drone body + NN-brain evolution (spherical body, CMA-ES circle controller)",
    )
    parser.add_argument("--pop", type=int, default=4, help="Body population size")
    parser.add_argument("--budget", type=int, default=3, help="Body EA generations")
    parser.add_argument("--min-arms", type=int, default=4, help="Minimum rotor arms")
    parser.add_argument("--max-arms", type=int, default=6, help="Maximum rotor arms")
    parser.add_argument("--learn-bud", type=int, default=3, help="CMA-ES iterations per body")
    parser.add_argument(
        "--learn-pop", type=int, default=8,
        help="CMA-ES candidates per iteration (raised to the CMA minimum λ)",
    )
    parser.add_argument("--hidden", type=int, default=12, help="Brain hidden-layer width")
    parser.add_argument("--dur", type=float, default=6.0, help="Circle-tracking rollout seconds")
    parser.add_argument("--warm-up", type=float, default=0.5, help="Warm-up seconds discarded from fitness")
    parser.add_argument("--target-alt", type=float, default=1.5, help="Circle-plane altitude (m, ENU)")
    parser.add_argument("--radius", type=float, default=1.0, help="Circle radius (m)")
    parser.add_argument("--period", type=float, default=6.0, help="Circle period (s per loop)")
    parser.add_argument("--tilt-weight", type=float, default=1.0, help="Weight for (1 - cos θ_tilt)")
    parser.add_argument("--ctrl-weight", type=float, default=0.05, help="Weight for mean(ctrl²)")
    parser.add_argument("--prop-size", type=int, default=2, help="Propeller size (inches)")
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2),
        help="Parallel body-evaluation worker processes (1 = single-process)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--visualize", action=argparse.BooleanOptionalAction, default=True,
        help="Render best individual: MP4 + passive viewer (--no-visualize to skip)",
    )
    parser.add_argument("--viewer-duration", type=float, default=12.0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    console = Console()

    POP_SIZE = args.pop
    BUDGET = args.budget
    WORKERS = max(1, min(args.workers, POP_SIZE * 2))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    DATA = Path.cwd() / "__data__" / Path(__file__).stem
    DATA.mkdir(parents=True, exist_ok=True)
    RUN_ID = time.strftime("%Y%m%d_%H%M%S")
    DB_PATH = DATA / f"database_{RUN_ID}.db"

    CENTER = (0.0, 0.0, float(args.target_alt))

    # Shared config passed to every worker (must be plain / picklable).
    cfg: dict[str, Any] = {
        "propsize": args.prop_size,
        "center": CENTER,
        "radius": float(args.radius),
        "period": float(args.period),
        "duration": float(args.dur),
        "warm_up": float(args.warm_up),
        "tilt_weight": float(args.tilt_weight),
        "ctrl_weight": float(args.ctrl_weight),
        "learn_budget": int(args.learn_bud),
        "learn_pop": int(args.learn_pop),
        "hidden": int(args.hidden),
        "seed": int(args.seed),
    }

    # -- Body genome handler (direct spherical-angular encoding) -------------
    PARAMETER_LIMITS = np.array([
        [0.055, 0.17],            # arm magnitude (m)
        [-np.pi, np.pi],          # arm azimuth
        [-np.pi / 2, np.pi / 2],  # arm elevation
        [-np.pi, np.pi],          # motor disc azimuth
        [-np.pi, np.pi],          # motor disc pitch
        [0, 1],                   # propeller spin direction
    ])
    template_handler = SphericalAngularDroneGenomeHandler(
        min_max_narms=(args.min_arms, args.max_arms),
        parameter_limits=PARAMETER_LIMITS,
        append_arm_chance=0.1,
        bilateral_plane_for_symmetry=None,
        repair=False,
        rnd=np.random.default_rng(args.seed),
    )

    console.rule("[bold blue]Drone Body+Brain Evolution (spherical body, CMA-ES circle NN)")
    console.log(
        f"pop={POP_SIZE}  budget={BUDGET}  arms=[{args.min_arms}, {args.max_arms}]  "
        f"learn_bud={args.learn_bud}  learn_pop≥{args.learn_pop}  workers={WORKERS}",
    )
    console.log(
        f"circle: R={args.radius}m  T={args.period}s  alt={args.target_alt}m  "
        f"rollout={args.dur}s",
    )
    console.log(f"DB → {DB_PATH}")

    # -- EA operations ------------------------------------------------------

    def evaluate_circle_brains(population: Population) -> Population:
        """Learn a brain for every unevaluated body; fitness = best brain."""
        to_eval = list(population.alive.unevaluated)
        if not to_eval:
            return population

        tasks = [(ind.genotype, cfg) for ind in to_eval]
        results: list[tuple[float, list[float], list[float]]] = [None] * len(tasks)  # type: ignore[list-item]

        if WORKERS == 1:
            for i, t in enumerate(track(tasks, description="Learning brains…", console=console)):
                results[i] = _evaluate_individual_process(t)
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as executor:
                future_to_idx = {
                    executor.submit(_evaluate_individual_process, t): i
                    for i, t in enumerate(tasks)
                }
                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    try:
                        results[idx] = fut.result()
                    except Exception:  # noqa: BLE001
                        results[idx] = (float("inf"), [], [])

        for ind, (fit, brain_vec, deltas) in zip(to_eval, results, strict=True):
            ind.fitness = fit
            ind.tags["best_brain"] = brain_vec
            ind.tags["learning_deltas"] = deltas

        finite = [
            ind.fitness_ for ind in population.alive.evaluated
            if ind.fitness_ is not None and np.isfinite(ind.fitness_)
        ]
        if finite:
            console.log(
                f"  evaluated {len(to_eval)} bodies — "
                f"min={min(finite):.4f}  mean={np.mean(finite):.4f}",
            )
        return population

    def truncation_select_min(population: Population) -> Population:
        """Keep the POP_SIZE lowest-fitness alive bodies; kill the rest."""
        ranked = population.alive.evaluated.sort(sort="min", attribute="fitness_")
        for i, ind in enumerate(ranked):
            if i >= POP_SIZE:
                ind.alive = False
        return population

    # -- Initial population (init + evaluate pre-loop) ----------------------

    initial_pop = Population([Individual() for _ in range(POP_SIZE)])
    init_op = initialize_drones(template_handler=template_handler)
    eval_op = EAOperation(evaluate_circle_brains)

    console.log("Initializing + learning brains for the initial population …")
    initial_pop = init_op(initial_pop)
    initial_pop = eval_op(initial_pop)

    # -- Generational pipeline (lower-is-better; mu+lambda truncation) ------

    generation_ops = [
        parent_tag(n=POP_SIZE),
        crossover_drones(template_handler=template_handler),
        mutate_drones(template_handler=template_handler),
        eval_op,
        EAOperation(truncation_select_min),
    ]

    ea = EA(
        population=initial_pop,
        operations=generation_ops,
        num_steps=BUDGET,
        db_file_path=DB_PATH,
        db_handling="delete",
    )

    console.rule("[bold green]Evolving body + brain")
    ea.run()

    # -- Pick best (minimisation → smallest fitness) ------------------------

    ea.fetch_population(only_alive=False, requires_eval=False)
    finite = [
        ind for ind in ea.population
        if ind.fitness_ is not None and np.isfinite(ind.fitness_)
    ]
    if not finite:
        console.log("[red]No individuals with finite fitness — aborting.[/red]")
        return
    best = sorted(finite, key=lambda i: i.fitness_ or float("inf"))[0]

    console.rule("[bold cyan]Best individual")
    console.log(f"  id={best.id}  fitness={best.fitness_:.4f}  born={best.time_of_birth}")

    best_genome = deserialize_genome(best.genotype)
    valid_mask = np.isfinite(best_genome.arms[:, 0])
    console.log(f"  active arms: {int(valid_mask.sum())} / {best_genome.arms.shape[0]}")

    best_bp = spherical_angular_to_blueprint(best_genome.arms, propsize=args.prop_size)
    bp_path = DATA / f"best_blueprint_{RUN_ID}.json"
    brain_path = DATA / f"best_brain_{RUN_ID}.npy"
    best_bp.save_json(bp_path)
    best_brain = np.asarray(best.tags.get("best_brain", []), dtype=np.float32)
    np.save(brain_path, best_brain)
    console.log(f"  blueprint → {bp_path}")
    console.log(f"  brain     → {brain_path}  ({best_brain.size} weights)")

    # -- Replay (MP4 + optional passive viewer) -----------------------------

    if not args.visualize:
        console.rule("[bold]Done")
        return
    if best_brain.size == 0:
        console.log("[yellow]Best individual has no learned brain — skipping replay.[/yellow]")
        console.rule("[bold]Done")
        return

    import mujoco

    from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
    from ariel.simulation.drone.controllers.lee_control.mujoco_bridge import (
        spawn_blueprint_in_world,
    )
    from ariel.simulation.drone.drone_interface import DroneInterface
    from ariel.utils.video_recorder import VideoRecorder

    propellers = blueprint_to_propellers(best_bp, convention="ned")
    if not propellers:
        console.log("[yellow]Best individual has no propellers — skipping replay.[/yellow]")
        console.rule("[bold]Done")
        return

    target_mass = float(DroneInterface(0, propellers=propellers).params["mB"])
    spawned = spawn_blueprint_in_world(
        best_bp, propellers=propellers, target_mass=target_mass,
        spawn_position=tuple(float(v) for v in circle_reference(
            0.0, CENTER, args.radius, args.period)),
    )
    model, data = spawned.model, spawned.data
    net = CircleBrain(int(model.nu), hidden_size=args.hidden)
    fill_parameters(net, best_brain.astype(np.float64))

    def reset_to_circle_start() -> None:
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = circle_reference(0.0, CENTER, args.radius, args.period)
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)

    def drive_one_step() -> None:
        feats = circle_features(data, CENTER, args.radius, args.period)
        data.ctrl[:] = np.clip(net.forward(feats), 0.0, 1.0)
        mujoco.mj_step(model, data)

    # MP4
    videos_dir = DATA / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    video = VideoRecorder(file_name=f"best_circle_{RUN_ID}", output_folder=videos_dir)
    reset_to_circle_start()
    steps_per_frame = max(1, int(round(1.0 / (model.opt.timestep * video.fps))))
    total_steps = int(args.viewer_duration / model.opt.timestep)
    console.log(f"Recording MP4 ({args.viewer_duration}s) …")
    with mujoco.Renderer(model, width=video.width, height=video.height) as renderer:
        for step_i in range(total_steps):
            drive_one_step()
            if step_i % steps_per_frame == 0:
                renderer.update_scene(data)
                video.write(renderer.render())
    video.release()
    console.log(f"  MP4 → {videos_dir}")

    # Passive viewer (skip on macOS / headless — no GUI backend)
    try:
        import mujoco.viewer as mj_viewer
    except Exception:  # noqa: BLE001
        mj_viewer = None

    if sys.platform != "darwin" and mj_viewer is not None and hasattr(mj_viewer, "launch_passive"):
        reset_to_circle_start()
        console.log("Launching passive viewer (close window to exit) …")
        with mj_viewer.launch_passive(model, data) as v:
            t_start = time.time()
            while v.is_running() and (time.time() - t_start) < args.viewer_duration:
                step_start = time.time()
                drive_one_step()
                v.sync()
                slack = model.opt.timestep - (time.time() - step_start)
                if slack > 0:
                    time.sleep(slack)

    console.rule("[bold]Done")


if __name__ == "__main__":
    main()
