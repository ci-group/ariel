"""
Sequential waypoint navigation task.

The spider robot must visit three target points arranged in a triangle,
in order.  A single green mocap marker is moved to the next waypoint the
moment the current one is reached.  The controller uses the camera-visible
green marker to guide locomotion, plus a normalised waypoint-progress
signal so the network knows how far through the course it is.

Fitness function (lower is better, CMA-ES minimises):

  • Incomplete run:  min_distance_to_next − waypoints_reached × 10
  • Complete run:    −waypoints_reached × 10 − (DURATION − t_done) / DURATION

The first term creates a strict hierarchy across waypoint counts (each
waypoint credit = 10, max possible distance ≈ few metres).  The second
term is a continuous tiebreaker within the fully-solved class: faster
completions receive a larger bonus in (−1, 0], keeping them below every
incomplete-run score while preserving a gradient between fast and slow
solutions.
"""

# Standard library
import gc
import os
import random
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Optional, cast

# Third-party
import cv2
import multiprocessing as mp
import mujoco
import nevergrad as ng
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from rich.console import Console
from rich.traceback import install
from torch import nn

# ARIEL
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider_with_blocks import body_spider45
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import VideoRecorder

install()
console = Console()

# TPA fires check_consistency on noisy stochastic simulators — harmless.
# See wiki/CMA-ES_Mirrored_Sampling.md.
warnings.filterwarnings(
    "ignore",
    message="TPA: apparent inconsistency",
    category=UserWarning,
    module="cma",
)

# ── CLI args ──────────────────────────────────────────────────────────────────

import argparse
parser = argparse.ArgumentParser(description="Sequential waypoint navigation")
parser.add_argument("--budget", type=int, default=200, help="CMA generations")
parser.add_argument("--population", type=int, default=48, help="Requested CMA population (enforced ≥ min-lambda, even)")
parser.add_argument("--dur", type=float, default=60.0, help="Max episode duration (s)")
parser.add_argument("--reach-radius", type=float, default=0.35, help="Planar reach radius (m)")
parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="Parallel worker processes")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no-video", action="store_true", help="Skip video recording after evolution")
args = parser.parse_args()

BUDGET        = args.budget
POP_SIZE      = args.population
DURATION      = args.dur
REACH_RADIUS  = max(0.05, args.reach_radius)
NUM_WORKERS   = max(1, args.workers)
BASE_SEED     = args.seed

# Triangle of waypoints (x, y, z).  Robot spawns at origin.
#
#   WP2 (2, 2) ──── WP3 (0, 2)
#       │                │
#   WP1 (2, 0) ── start (0, 0)
#
WAYPOINTS: list[np.ndarray] = [
    np.array([2.0, 0.0, 0.1]),   # straight ahead
    np.array([2.0, 2.0, 0.1]),   # right turn
    np.array([0.0, 2.0, 0.1]),   # closing leg back toward start
]
NUM_WAYPOINTS = len(WAYPOINTS)

SCRIPT_NAME = Path(__file__).stem
DATA = Path.cwd() / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)


# ── Network ───────────────────────────────────────────────────────────────────

class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.hidden_act = nn.ELU()
        self.out_act = nn.Tanh()
        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, model, data, state: np.ndarray) -> np.ndarray:  # noqa: ARG002
        x = torch.as_tensor(state, dtype=torch.float32)
        x = self.hidden_act(self.fc1(x))
        x = self.hidden_act(self.fc2(x))
        return (self.out_act(self.fc_out(x)) * (torch.pi / 2)).numpy()


@torch.no_grad()
def fill_parameters(net: nn.Module, vector: np.ndarray) -> None:
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(vector[address : address + n])
        address += n
    if address != len(vector):
        raise IndexError("Parameter vector length mismatch")


# ── Vision helpers ────────────────────────────────────────────────────────────

def isolate_green(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    return cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))


def analyze_sections(mask: np.ndarray) -> list[float]:
    """Split mask into left / centre / right thirds; return green fraction each."""
    sections = np.array_split(mask, 3, axis=1)
    return [cv2.countNonZero(s) / max(s.size, 1) for s in sections]


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    network: Network,
    waypoints: list[np.ndarray],
    duration: float,
    reach_radius: float,
    target_mocap_id: int,
    renderer: mujoco.Renderer,
    cam_name: Optional[str],
    control_step_freq: int = 50,
) -> dict[str, Any]:
    """
    Simulate one episode.  Moves the green marker to the next waypoint each
    time the robot steps within `reach_radius` of the current one.

    Returns:
        waypoints_reached  – int, how many waypoints were touched in order
        min_dist_to_current – float, closest approach to the next un-reached
                              waypoint (0 if all reached)
    """
    current_wp_idx = 0
    waypoints_reached = 0
    current_target = waypoints[0]
    data.mocap_pos[target_mocap_id] = current_target

    current_action = np.zeros(model.nu)
    min_dist_to_current = float("inf")
    completion_time: Optional[float] = None
    step = 0

    while data.time < duration and current_wp_idx < len(waypoints):
        # ── Control (vision + network) every N steps ──────────────────────
        if step % control_step_freq == 0:
            renderer.update_scene(data, camera=cam_name)
            img = renderer.render()
            vision = analyze_sections(isolate_green(img))

            robot_state = get_robot_state(data)
            phase = [
                2.0 * np.sin(data.time * 2.0 * np.pi),
                2.0 * np.cos(data.time * 2.0 * np.pi),
            ]
            # Normalised waypoint index: gives the network a sense of progress
            # through the course (0.0 = heading to WP1, 1.0 = heading to WP3).
            progress = [current_wp_idx / max(NUM_WAYPOINTS - 1, 1)]

            state = np.concatenate([robot_state, vision, phase, progress]).astype(np.float32)
            current_action = network.forward(model, data, state)

        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)
        step += 1

        # ── Waypoint check (every step; cheap) ────────────────────────────
        dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
        min_dist_to_current = min(min_dist_to_current, dist)

        if dist <= reach_radius:
            waypoints_reached += 1
            current_wp_idx += 1
            if current_wp_idx < len(waypoints):
                current_target = waypoints[current_wp_idx]
                data.mocap_pos[target_mocap_id] = current_target
                min_dist_to_current = float("inf")  # reset for the new target
            else:
                # All waypoints reached — record the moment for speed tiebreak.
                completion_time = data.time

    final_dist = 0.0 if current_wp_idx >= len(waypoints) else min_dist_to_current

    return {
        "waypoints_reached": waypoints_reached,
        "min_dist_to_current": final_dist,
        "completion_time": completion_time,
    }


def compute_fitness(
    waypoints_reached: int,
    min_dist_to_current: float,
    completion_time: Optional[float] = None,
    duration: float = DURATION,
) -> float:
    """
    Lower is better.

    Incomplete run: min_dist − waypoints_reached × 10.
      Each waypoint credit (−10) dominates the distance term (max ~few m),
      so the optimiser is never rewarded for camping near one waypoint.

    Complete run (all waypoints reached):
      −waypoints_reached × 10 − (duration − completion_time) / duration
      The bonus lies in (−1, 0]: faster completion → more negative.
      This keeps every completed score below any incomplete score while
      letting CMA-ES distinguish fast solutions from slow ones.
    """
    if waypoints_reached == NUM_WAYPOINTS and completion_time is not None:
        time_bonus = (duration - completion_time) / duration  # 0..1, higher = faster
        return -waypoints_reached * 10.0 - time_bonus
    return min_dist_to_current - waypoints_reached * 10.0


# ── Per-process context (built once per worker, cached) ───────────────────────

_RENDER_INIT_LOCK = threading.Lock()
_process_local_ctx: Optional[dict[str, Any]] = None


def _build_context() -> dict[str, Any]:
    world = SimpleFlatWorld()

    # Robot
    spider = body_spider45()
    world.spawn(spider.spec, position=[0.0, 0.0, 0.1])

    # Single green mocap marker — repositioned to each waypoint in sequence.
    marker = world.spec.worldbody.add_body(
        name="green_target", mocap=True, pos=WAYPOINTS[0].tolist()
    )
    marker.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.15, 0.15, 0.15],
        rgba=[0, 1, 0, 1],
    )

    # Top-down overview camera for video recording.
    # Centred above the triangle (centroid ≈ (1.3, 1.3)).
    world.spec.worldbody.add_camera(
        name="overview_cam",
        pos=[1.0, 1.0, 7.0],
        xyaxes=[1, 0, 0, 0, 1, 0],
    )

    model = world.spec.compile()
    data  = mujoco.MjData(model)

    # Detect the robot's onboard camera (used for vision control).
    cam_name: Optional[str] = None
    for i in range(model.ncam):
        name = model.camera(i).name
        if ("camera" in name or "core" in name) and "overview" not in name:
            cam_name = name
            break

    target_mocap_id = model.body("green_target").mocapid[0]

    # Derive input_dim from an actual state read (robust to robot changes).
    robot_state_size = len(get_robot_state(data))
    # robot_state  +  3 vision bins  +  2 phase  +  1 waypoint progress
    input_dim = robot_state_size + 3 + 2 + 1

    network = Network(input_size=input_dim, output_size=model.nu)

    with _RENDER_INIT_LOCK:
        renderer = mujoco.Renderer(model, height=24, width=32)

    return {
        "model":           model,
        "data":            data,
        "network":         network,
        "renderer":        renderer,
        "cam_name":        cam_name,
        "target_mocap_id": target_mocap_id,
        "input_dim":       input_dim,
    }


def _get_ctx() -> dict[str, Any]:
    global _process_local_ctx
    if _process_local_ctx is None:
        _process_local_ctx = _build_context()
    return cast(dict[str, Any], _process_local_ctx)


def _init_worker(base_seed: int) -> None:
    torch.set_num_threads(1)
    seed = (base_seed + os.getpid()) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _evaluate_candidate(weights: np.ndarray) -> float:
    ctx              = _get_ctx()
    model            = cast(mujoco.MjModel,         ctx["model"])
    data             = cast(mujoco.MjData,          ctx["data"])
    network          = cast(Network,                ctx["network"])
    renderer         = cast(mujoco.Renderer,        ctx["renderer"])
    cam_name         = cast(Optional[str],          ctx["cam_name"])
    target_mocap_id  = cast(int,                    ctx["target_mocap_id"])

    fill_parameters(network, weights)
    mujoco.mj_resetData(model, data)

    result = run_episode(
        model=model,
        data=data,
        network=network,
        waypoints=WAYPOINTS,
        duration=DURATION,
        reach_radius=REACH_RADIUS,
        target_mocap_id=target_mocap_id,
        renderer=renderer,
        cam_name=cam_name,
    )

    return compute_fitness(
        waypoints_reached=result["waypoints_reached"],
        min_dist_to_current=result["min_dist_to_current"],
        completion_time=result["completion_time"],
        duration=DURATION,
    )


# ── Evolution ─────────────────────────────────────────────────────────────────

def evolve() -> tuple[np.ndarray, int]:
    # Build a throwaway context in the main process to get dimensions.
    ctx       = _build_context()
    input_dim = ctx["input_dim"]
    model     = ctx["model"]

    dummy_net  = Network(input_size=input_dim, output_size=model.nu)
    num_params = sum(p.numel() for p in dummy_net.parameters())

    # CMA-ES minimum population (wiki/CMA-ES_Parameters.md).
    min_lambda = 4 + int(3 * np.log(max(num_params, 2)))
    pop_size   = max(POP_SIZE, min_lambda)
    # TPA requires even λ for mirrored pairs (wiki/CMA-ES_Mirrored_Sampling.md).
    if pop_size % 2 != 0:
        pop_size += 1

    # σ⁰ = 0.3·(b−a) = 0.3 for weights in [−0.5, 0.5] (wiki/CMA-ES_Parameters.md).
    initial_guess = np.random.uniform(-0.5, 0.5, size=num_params)
    param = ng.p.Array(init=initial_guess).set_mutation(sigma=0.3)

    cma_config = ng.optimizers.ParametrizedCMA(popsize=pop_size)
    optimizer  = cma_config(
        parametrization=param,
        budget=BUDGET * pop_size,
        num_workers=pop_size,
    )

    console.rule("[bold magenta]Sequential Waypoint Navigation[/bold magenta]")
    console.log(
        f"params={num_params}  pop_size={pop_size} (requested {POP_SIZE})  "
        f"budget={BUDGET} gens  workers={NUM_WORKERS}"
    )
    console.log(f"Waypoints: {[w.tolist() for w in WAYPOINTS]}")
    console.log(f"Reach radius: {REACH_RADIUS} m  |  Episode duration: {DURATION} s")

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(BASE_SEED,),
    ) as executor:
        for gen in range(BUDGET):
            candidates = [optimizer.ask() for _ in range(pop_size)]
            fitnesses  = list(executor.map(_evaluate_candidate, [c.value for c in candidates]))

            for cand, fit in zip(candidates, fitnesses):
                optimizer.tell(cand, fit)

            best     = float(np.min(fitnesses))
            # Recover waypoints_reached from fitness.
            # Incomplete: fitness = min_dist − wps×10, so wps = ceil(-fitness/10)
            # Complete:   fitness = −wps×10 − bonus, bonus ∈ (0,1]
            #             → -fitness ∈ (wps×10, wps×10+1], ceil(-fitness/10) = wps+1
            #             but capped at NUM_WAYPOINTS.
            best_wps = (
                min(NUM_WAYPOINTS, max(0, int(np.ceil(-best / 10))))
                if best < 0 else 0
            )
            console.rule(f"Gen {gen + 1}/{BUDGET}")
            if best_wps == NUM_WAYPOINTS:
                # Extract the time bonus: bonus = -(fitness + NUM_WAYPOINTS*10)
                speed_pct = (-best - NUM_WAYPOINTS * 10.0) * 100.0
                console.log(
                    f"Best fitness: {best:.3f}  "
                    f"({best_wps}/{NUM_WAYPOINTS} waypoints — {speed_pct:.1f}% speed bonus)"
                )
            else:
                console.log(
                    f"Best fitness: {best:.3f}  ({best_wps}/{NUM_WAYPOINTS} waypoints reached)"
                )

    return optimizer.provide_recommendation().value, input_dim


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)

    start = time.time()
    best_weights, input_dim = evolve()
    elapsed = time.time() - start

    weights_path = DATA / "best_weights.npy"
    np.save(weights_path, best_weights)
    console.log(f"Evolution finished in {elapsed / 60:.1f} min. Weights → {weights_path}")

    if args.no_video:
        return

    # ── Replay best individual and record video ───────────────────────────────
    ctx             = _build_context()
    model           = ctx["model"]
    data            = ctx["data"]
    cam_name        = ctx["cam_name"]
    target_mocap_id = ctx["target_mocap_id"]

    net = Network(input_size=input_dim, output_size=model.nu)
    fill_parameters(net, best_weights)

    mujoco.mj_resetData(model, data)
    data.mocap_pos[target_mocap_id] = WAYPOINTS[0]

    videos_dir = DATA / "videos"
    videos_dir.mkdir(exist_ok=True)
    video_recorder = VideoRecorder(
        file_name="sequential_best", output_folder=str(videos_dir)
    )

    fps             = 30
    dt              = model.opt.timestep
    steps_per_frame = max(1, int(round(1.0 / (fps * dt))))
    control_step_freq = 50
    current_ctrl    = np.zeros(model.nu)
    render_step     = 0
    wp_idx          = 0
    current_target  = WAYPOINTS[0]

    control_renderer = mujoco.Renderer(model, height=24, width=32)

    def get_control(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
        nonlocal wp_idx, current_target
        control_renderer.update_scene(d, camera=cam_name)
        img    = control_renderer.render()
        vision = analyze_sections(isolate_green(img))
        rs     = get_robot_state(d)
        phase  = [2.0 * np.sin(d.time * 2.0 * np.pi), 2.0 * np.cos(d.time * 2.0 * np.pi)]
        prog   = [wp_idx / max(NUM_WAYPOINTS - 1, 1)]
        state  = np.concatenate([rs, vision, phase, prog]).astype(np.float32)
        return net.forward(m, d, state)

    try:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview_cam")
    except Exception:
        camera_id = -1

    with mujoco.Renderer(model, height=480, width=640) as renderer:
        while data.time < DURATION:
            for _ in range(steps_per_frame):
                if render_step % control_step_freq == 0:
                    current_ctrl = get_control(model, data)
                np.copyto(data.ctrl, current_ctrl)
                mujoco.mj_step(model, data)
                render_step += 1

                # Keep the green marker on the current active waypoint.
                if wp_idx < NUM_WAYPOINTS:
                    dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
                    if dist <= REACH_RADIUS:
                        wp_idx += 1
                        if wp_idx < NUM_WAYPOINTS:
                            current_target = WAYPOINTS[wp_idx]
                            data.mocap_pos[target_mocap_id] = current_target

            renderer.update_scene(data, camera=camera_id)
            video_recorder.write(renderer.render())

    video_recorder.release()
    control_renderer.close()
    console.log(f"[green]Video saved to {videos_dir}[/green]")


if __name__ == "__main__":
    main()
    gc.disable()
