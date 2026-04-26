"""
Randomised-waypoints navigation task.

The spider must visit N waypoints in order, guided by a green mocap
marker.  Waypoint positions are re-sampled **once per generation** using a
deterministic per-generation seed (BASE_SEED + gen).  All candidates in a
generation are evaluated on the same waypoint layout (fair selection), but
every generation presents a different layout (no memorised path generalises
across generations).

Why this matters:
  In 4_sequential_waypoints.py the waypoints form a fixed triangle.  A
  network can learn a timed open-loop motor program that traces the shape
  without ever consulting the camera.  Here the layout changes every
  generation, so the only consistent signal available is the green marker
  visible in the robot's camera.  Visual tracking is no longer optional.

Fitness function (lower is better, CMA-ES minimises):

  • Incomplete run:  min_distance_to_next − waypoints_reached × 10
  • Complete run:    −waypoints_reached × 10 − (DURATION − t_done) / DURATION

Same hierarchy as file 4: each waypoint credit (−10) dominates the
continuous distance term, and the speed bonus (∈ (−1, 0]) breaks ties
among fully-solved runs without violating the waypoint ordering.
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
# from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider_with_blocks import body_spider45
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko as body_spider45
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
parser = argparse.ArgumentParser(description="Randomised waypoint navigation")
parser.add_argument("--budget",        type=int,   default=100,  help="CMA generations")
parser.add_argument("--population",    type=int,   default=40,   help="Requested CMA population (enforced ≥ min-lambda, even)")
parser.add_argument("--dur",           type=float, default=60.0, help="Max episode duration (s)")
parser.add_argument("--reach-radius",  type=float, default=0.35, help="Planar reach radius (m)")
parser.add_argument("--num-waypoints", type=int,   default=3,    help="Waypoints per episode")
parser.add_argument("--arena-radius",  type=float, default=3.0,  help="Max distance from origin for waypoints (m)")
parser.add_argument("--workers",       type=int,   default=max(1, os.cpu_count() or 1), help="Parallel worker processes")
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--warm-start",    type=str,   default=None, help="Path to a .npy weights file to use as the CMA-ES starting point")
parser.add_argument("--no-video",      action="store_true", help="Skip video recording after evolution")
parser.add_argument("--replay",        type=str,   default=None, help="Path to a .npy weights file; skips evolution and opens an interactive MuJoCo viewer")
parser.add_argument("--replay-waypoints", type=str, default=None, help="Path to a .npy waypoints file for --replay (default: auto-load best_seen_waypoints.npy from same dir)")
args = parser.parse_args()

BUDGET        = args.budget
POP_SIZE      = args.population
DURATION      = args.dur
REACH_RADIUS  = max(0.05, args.reach_radius)
NUM_WAYPOINTS = args.num_waypoints
ARENA_RADIUS  = max(1.0, args.arena_radius)
NUM_WORKERS   = max(1, args.workers)
BASE_SEED     = args.seed

SCRIPT_NAME = Path(__file__).stem
DATA = Path.cwd() / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)


# ── Waypoint sampling ─────────────────────────────────────────────────────────

def sample_waypoints(
    rng: np.random.Generator,
    n: int = NUM_WAYPOINTS,
    radius: float = ARENA_RADIUS,
    min_sep: float = 0.8,
) -> list[np.ndarray]:
    """
    Sample n waypoints at random positions in a disc of given radius.

    Each waypoint is at least `min_sep` metres from every other in the xy
    plane.  The robot spawns at the origin, so waypoints are placed at
    radial distance [0.8, radius] to keep them reachable but not trivial.

    Returns a list of np.ndarray([x, y, 0.1]).
    """
    wps: list[np.ndarray] = []
    for _ in range(n):
        for _ in range(2000):
            r     = rng.uniform(0.8, radius)
            theta = rng.uniform(0.0, 2.0 * np.pi)
            p     = np.array([r * np.cos(theta), r * np.sin(theta), 0.1])
            if all(np.linalg.norm(p[:2] - w[:2]) >= min_sep for w in wps):
                wps.append(p)
                break
        else:
            # Separation guarantee relaxed as a last resort (arena too small).
            r = rng.uniform(0.8, radius)
            theta = rng.uniform(0.0, 2.0 * np.pi)
            wps.append(np.array([r * np.cos(theta), r * np.sin(theta), 0.1]))
    return wps


# ── Network ───────────────────────────────────────────────────────────────────

class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.fc1    = nn.Linear(input_size, hidden_size)
        self.fc2    = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.hidden_act = nn.ELU()
        self.out_act    = nn.Tanh()
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
        waypoints_reached   – int
        min_dist_to_current – float (0 if all reached)
        completion_time     – float or None (sim time when last WP reached)
    """
    num_wps        = len(waypoints)
    current_wp_idx = 0
    waypoints_reached   = 0
    current_target = waypoints[0]
    data.mocap_pos[target_mocap_id] = current_target

    current_action      = np.zeros(model.nu)
    min_dist_to_current = float("inf")
    completion_time: Optional[float] = None
    step = 0

    while data.time < duration and current_wp_idx < num_wps:
        # ── Control (vision + network) every N steps ──────────────────────
        if step % control_step_freq == 0:
            renderer.update_scene(data, camera=cam_name)
            img    = renderer.render()
            vision = analyze_sections(isolate_green(img))

            robot_state = get_robot_state(data)
            phase = [
                2.0 * np.sin(data.time * 2.0 * np.pi),
                2.0 * np.cos(data.time * 2.0 * np.pi),
            ]
            # Normalised waypoint index (0 = first WP, 1 = last WP).
            # Uses the actual episode length so it adapts to --num-waypoints.
            progress = [current_wp_idx / max(num_wps - 1, 1)]

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
            current_wp_idx    += 1
            if current_wp_idx < num_wps:
                current_target = waypoints[current_wp_idx]
                data.mocap_pos[target_mocap_id] = current_target
                min_dist_to_current = float("inf")
            else:
                completion_time = data.time

    final_dist = 0.0 if current_wp_idx >= num_wps else min_dist_to_current

    return {
        "waypoints_reached":   waypoints_reached,
        "min_dist_to_current": final_dist,
        "completion_time":     completion_time,
    }


def compute_fitness(
    waypoints_reached: int,
    min_dist_to_current: float,
    num_waypoints: int,
    completion_time: Optional[float] = None,
    duration: float = DURATION,
) -> float:
    """
    Lower is better.

    Incomplete: min_dist − waypoints_reached × 10
    Complete:   −waypoints_reached × 10 − (duration − t_done) / duration
      Speed bonus ∈ (−1, 0]: faster → more negative → strictly better than
      any incomplete run score.
    """
    if waypoints_reached == num_waypoints and completion_time is not None:
        time_bonus = (duration - completion_time) / duration
        return -waypoints_reached * 10.0 - time_bonus
    return min_dist_to_current - waypoints_reached * 10.0


# ── Per-process context (built once per worker, cached) ───────────────────────

_RENDER_INIT_LOCK = threading.Lock()
_process_local_ctx: Optional[dict[str, Any]] = None


def _build_context() -> dict[str, Any]:
    world  = SimpleFlatWorld()
    spider = body_spider45()
    world.spawn(spider.spec, position=[0.0, 0.0, 0.1])

    # Green mocap marker — initial position is arbitrary; run_episode moves it.
    marker = world.spec.worldbody.add_body(
        name="green_target", mocap=True, pos=[0.0, 0.0, 0.1]
    )
    marker.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.15, 0.15, 0.15],
        rgba=[0, 1, 0, 1],
    )

    # Top-down overview camera centred at origin, high enough to show the full
    # arena regardless of where waypoints are placed.
    world.spec.worldbody.add_camera(
        name="overview_cam",
        pos=[0.0, 0.0, ARENA_RADIUS * 3.5],
        xyaxes=[1, 0, 0, 0, 1, 0],
    )

    model = world.spec.compile()
    data  = mujoco.MjData(model)

    cam_name: Optional[str] = None
    for i in range(model.ncam):
        name = model.camera(i).name
        if ("camera" in name or "core" in name) and "overview" not in name:
            cam_name = name
            break

    target_mocap_id  = model.body("green_target").mocapid[0]
    robot_state_size = len(get_robot_state(data))
    # robot_state  +  3 vision bins  +  2 phase  +  1 waypoint progress
    input_dim = robot_state_size + 3 + 2 + 1

    network = Network(input_size=input_dim, output_size=model.nu)

    with _RENDER_INIT_LOCK:
        renderer = mujoco.Renderer(model, height=48*2, width=64*2)

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


def _evaluate_candidate(task: tuple[np.ndarray, list[np.ndarray]]) -> float:
    """
    task = (weights, waypoints)

    Waypoints are passed explicitly (not a global) so that per-generation
    randomisation is correctly reflected inside each worker process.
    """
    weights, waypoints = task

    ctx             = _get_ctx()
    model           = cast(mujoco.MjModel,  ctx["model"])
    data            = cast(mujoco.MjData,   ctx["data"])
    network         = cast(Network,         ctx["network"])
    renderer        = cast(mujoco.Renderer, ctx["renderer"])
    cam_name        = cast(Optional[str],   ctx["cam_name"])
    target_mocap_id = cast(int,             ctx["target_mocap_id"])

    fill_parameters(network, weights)
    mujoco.mj_resetData(model, data)

    result = run_episode(
        model=model,
        data=data,
        network=network,
        waypoints=waypoints,
        duration=DURATION,
        reach_radius=REACH_RADIUS,
        target_mocap_id=target_mocap_id,
        renderer=renderer,
        cam_name=cam_name,
    )

    return compute_fitness(
        waypoints_reached=result["waypoints_reached"],
        min_dist_to_current=result["min_dist_to_current"],
        num_waypoints=len(waypoints),
        completion_time=result["completion_time"],
        duration=DURATION,
    )


# ── Warm-start helpers ────────────────────────────────────────────────────────

def _embed_weights_add_input(
    weights: np.ndarray,
    old_input_size: int,
    hidden_size: int,
) -> np.ndarray:
    """
    Expand a flat weight vector to match a network with one extra input neuron.

    The only difference between a file-1 (no progress) and file-5 (with
    progress) network is that fc1.weight gains one column.  PyTorch stores
    Linear weights row-major as (out_features, in_features), so the flat
    layout is:

        [fc1.weight  (hidden × old_in)]  [fc1.bias (hidden)]
        [fc2.weight  (hidden × hidden)]  [fc2.bias (hidden)]
        [fc_out.weight (out × hidden)]   [fc_out.bias (out)]

    We reshape the fc1.weight block, append a column of zeros for the new
    input at the END (progress is the last element of the state vector),
    and concatenate the rest unchanged.  The new input starts with zero
    weight — CMA-ES learns how to use it during the run.
    """
    fc1_w_size = hidden_size * old_input_size
    fc1_w = weights[:fc1_w_size].reshape(hidden_size, old_input_size)
    new_col = np.zeros((hidden_size, 1), dtype=fc1_w.dtype)
    fc1_w_expanded = np.concatenate([fc1_w, new_col], axis=1)  # (hidden, old_in+1)
    return np.concatenate([fc1_w_expanded.ravel(), weights[fc1_w_size:]])


# ── Evolution ─────────────────────────────────────────────────────────────────

def evolve() -> tuple[np.ndarray, Optional[np.ndarray], Optional[list[np.ndarray]], int]:
    ctx       = _build_context()
    input_dim = ctx["input_dim"]
    model     = ctx["model"]

    dummy_net  = Network(input_size=input_dim, output_size=model.nu)
    num_params = sum(p.numel() for p in dummy_net.parameters())

    min_lambda = 4 + int(3 * np.log(max(num_params, 2)))
    pop_size   = max(POP_SIZE, min_lambda)
    if pop_size % 2 != 0:
        pop_size += 1

    if args.warm_start is not None:
        warm_path = Path(args.warm_start)
        loaded    = np.load(warm_path)
        hidden_size = 32  # must match Network hidden_size

        # Params expected from a network without the progress input (file 1).
        num_params_no_progress = num_params - hidden_size

        if loaded.shape == (num_params,):
            # Exact match — architectures are identical.
            initial_guess = loaded
            console.log(f"[yellow]Warm start: {num_params} params from {warm_path}[/yellow]")

        elif loaded.shape == (num_params_no_progress,):
            # One fewer input neuron (no waypoint-progress signal, e.g. from
            # 1_brain_evolution_multiprocess.py).  Embed by inserting a zero
            # column for the progress input into fc1.weight.
            initial_guess = _embed_weights_add_input(
                loaded,
                old_input_size=input_dim - 1,
                hidden_size=hidden_size,
            )
            console.log(
                f"[yellow]Warm start: embedded {loaded.shape[0]}-param weights "
                f"(no-progress → progress network) from {warm_path}[/yellow]"
            )

        else:
            raise ValueError(
                f"Warm-start file has {loaded.shape[0]} params; "
                f"expected {num_params} (exact) or {num_params_no_progress} "
                f"(no-progress network from file 1). "
                "Check that hidden_size=32 and the same robot body were used."
            )
    else:
        initial_guess = np.random.uniform(-0.5, 0.5, size=num_params)
    param = ng.p.Array(init=initial_guess).set_mutation(sigma=0.3)

    cma_config = ng.optimizers.ParametrizedCMA(popsize=pop_size)
    optimizer  = cma_config(
        parametrization=param,
        budget=BUDGET * pop_size,
        num_workers=pop_size,
    )

    console.rule("[bold magenta]Randomised Waypoint Navigation[/bold magenta]")
    console.log(
        f"params={num_params}  pop_size={pop_size} (requested {POP_SIZE})  "
        f"budget={BUDGET} gens  workers={NUM_WORKERS}"
    )
    console.log(
        f"waypoints/ep={NUM_WAYPOINTS}  arena_radius={ARENA_RADIUS} m  "
        f"reach_radius={REACH_RADIUS} m  duration={DURATION} s"
    )

    # Track the best-evaluated individual and the exact waypoints it was scored on.
    # optimizer.provide_recommendation() returns the CMA distribution mean (good
    # for warm-starting future runs), but it may never have been directly evaluated
    # and its waypoints would be unknown.  We keep a separate record for the video.
    best_seen_fitness:  float                    = float("inf")
    best_seen_weights:  Optional[np.ndarray]     = None
    best_seen_waypoints: Optional[list[np.ndarray]] = None

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(BASE_SEED,),
    ) as executor:
        for gen in range(BUDGET):
            # Fresh waypoints this generation — same layout for every candidate
            # so within-generation selection is fair.  Different per generation
            # so no fixed motor program can generalise across the run.
            gen_rng       = np.random.default_rng(BASE_SEED + gen)
            gen_waypoints = sample_waypoints(gen_rng, n=NUM_WAYPOINTS, radius=ARENA_RADIUS)

            candidates = [optimizer.ask() for _ in range(pop_size)]
            tasks      = [(c.value, gen_waypoints) for c in candidates]
            fitnesses  = list(executor.map(_evaluate_candidate, tasks))

            for cand, fit in zip(candidates, fitnesses):
                optimizer.tell(cand, fit)

            best     = float(np.min(fitnesses))
            best_idx = int(np.argmin(fitnesses))

            # Update best-seen record whenever this generation's best is better.
            if best < best_seen_fitness:
                best_seen_fitness   = best
                best_seen_weights   = candidates[best_idx].value.copy()
                best_seen_waypoints = gen_waypoints

            best_wps = (
                min(NUM_WAYPOINTS, max(0, int(np.ceil(-best / 10))))
                if best < 0 else 0
            )

            wp_coords = "  ".join(
                f"({w[0]:.1f},{w[1]:.1f})" for w in gen_waypoints
            )
            console.rule(f"Gen {gen + 1}/{BUDGET}")
            if best_wps == NUM_WAYPOINTS:
                speed_pct = (-best - NUM_WAYPOINTS * 10.0) * 100.0
                console.log(
                    f"Best fitness: {best:.3f}  "
                    f"({best_wps}/{NUM_WAYPOINTS} waypoints — {speed_pct:.1f}% speed bonus)  "
                    f"wps: {wp_coords}"
                )
            else:
                console.log(
                    f"Best fitness: {best:.3f}  ({best_wps}/{NUM_WAYPOINTS} waypoints reached)  "
                    f"wps: {wp_coords}"
                )

    return optimizer.provide_recommendation().value, best_seen_weights, best_seen_waypoints, input_dim


# ── Viewer replay ─────────────────────────────────────────────────────────────

def replay_in_viewer(
    weights_path: Path,
    waypoints_path: Optional[Path] = None,
) -> None:
    """
    Load saved weights and run the best individual in an interactive MuJoCo viewer.

    Waypoint resolution order:
      1. ``waypoints_path`` if explicitly given.
      2. ``best_seen_waypoints.npy`` in the same directory as the weights file.
      3. Freshly sampled waypoints using BASE_SEED (fallback).
    """
    import mujoco.viewer  # local import — only needed for this path

    weights = np.load(weights_path)
    console.log(f"[cyan]Loaded weights from {weights_path}  ({len(weights)} params)[/cyan]")

    # Resolve waypoints.
    waypoints: list[np.ndarray]
    if waypoints_path is not None:
        raw = np.load(waypoints_path)
        waypoints = [raw[i] for i in range(len(raw))]
        console.log(f"[cyan]Loaded waypoints from {waypoints_path}[/cyan]")
    else:
        auto_wps = weights_path.parent / "best_seen_waypoints.npy"
        if auto_wps.exists():
            raw = np.load(auto_wps)
            waypoints = [raw[i] for i in range(len(raw))]
            console.log(f"[cyan]Auto-loaded waypoints from {auto_wps}[/cyan]")
        else:
            console.log("[yellow]No waypoints file found — sampling fresh waypoints with BASE_SEED.[/yellow]")
            rng = np.random.default_rng(BASE_SEED)
            waypoints = sample_waypoints(rng)

    console.log(f"Waypoints: {[w.tolist() for w in waypoints]}")

    ctx             = _build_context()
    model           = ctx["model"]
    data            = ctx["data"]
    cam_name        = ctx["cam_name"]
    target_mocap_id = ctx["target_mocap_id"]
    input_dim       = ctx["input_dim"]

    net = Network(input_size=input_dim, output_size=model.nu)
    fill_parameters(net, weights)

    mujoco.mj_resetData(model, data)
    data.mocap_pos[target_mocap_id] = waypoints[0]

    num_wps        = len(waypoints)
    wp_idx         = 0
    current_target = waypoints[0]
    control_step   = 0
    current_ctrl   = np.zeros(model.nu)

    control_renderer = mujoco.Renderer(model, height=96, width=128)

    console.rule("[bold green]MuJoCo Viewer — Replay[/bold green]")
    console.log("Close the viewer window (or press Escape) to stop.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_start = time.time()
        while viewer.is_running() and data.time < DURATION:
            # Vision + network control every 50 physics steps.
            if control_step % 50 == 0:
                control_renderer.update_scene(data, camera=cam_name)
                img    = control_renderer.render()
                vision = analyze_sections(isolate_green(img))
                rs     = get_robot_state(data)
                phase  = [
                    2.0 * np.sin(data.time * 2.0 * np.pi),
                    2.0 * np.cos(data.time * 2.0 * np.pi),
                ]
                prog   = [wp_idx / max(num_wps - 1, 1)]
                state  = np.concatenate([rs, vision, phase, prog]).astype(np.float32)
                current_ctrl = net.forward(model, data, state)

            np.copyto(data.ctrl, current_ctrl)
            mujoco.mj_step(model, data)
            control_step += 1

            # Waypoint advancement.
            if wp_idx < num_wps:
                dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
                if dist <= REACH_RADIUS:
                    wp_idx += 1
                    if wp_idx < num_wps:
                        current_target = waypoints[wp_idx]
                        data.mocap_pos[target_mocap_id] = current_target
                        console.log(
                            f"[green]Waypoint {wp_idx}/{num_wps} reached at "
                            f"t={data.time:.2f}s[/green]"
                        )
                    else:
                        console.log(
                            f"[bold green]All {num_wps} waypoints reached! "
                            f"t={data.time:.2f}s[/bold green]"
                        )

            viewer.sync()

            # Pace the loop to real-time.
            elapsed = time.time() - step_start
            remaining = model.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)
            step_start = time.time()

    control_renderer.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Replay mode: skip evolution, open interactive viewer ──────────────────
    if args.replay is not None:
        replay_in_viewer(
            weights_path=Path(args.replay),
            waypoints_path=Path(args.replay_waypoints) if args.replay_waypoints else None,
        )
        return

    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)

    start = time.time()
    recommendation_weights, best_seen_weights, best_seen_waypoints, input_dim = evolve()
    elapsed = time.time() - start

    # Save the CMA recommendation (distribution mean) — best for warm-starting
    # future runs since it's a smooth summary of the converged search region.
    weights_path = DATA / "best_weights.npy"
    np.save(weights_path, recommendation_weights)
    console.log(f"Evolution finished in {elapsed / 60:.1f} min. Weights → {weights_path}")

    # Save the best-seen individual's weights and waypoints so --replay can
    # reproduce the exact episode that achieved the best fitness.
    if best_seen_weights is not None:
        best_seen_path = DATA / "best_seen_weights.npy"
        np.save(best_seen_path, best_seen_weights)
        console.log(f"Best-seen weights → {best_seen_path}")
    if best_seen_waypoints is not None:
        wps_path = DATA / "best_seen_waypoints.npy"
        np.save(wps_path, np.array(best_seen_waypoints))
        console.log(f"Best-seen waypoints → {wps_path}")

    if args.no_video:
        return

    # ── Replay the best-evaluated individual on the exact waypoints it solved ──
    # best_seen_weights is the individual that achieved best_seen_fitness; it was
    # directly evaluated (unlike the CMA mean) and we know its waypoint layout.
    if best_seen_weights is None or best_seen_waypoints is None:
        console.log("[red]No evaluated individual found — skipping video.[/red]")
        return

    replay_waypoints = best_seen_waypoints
    best_weights     = best_seen_weights
    console.log(f"Video replay waypoints (best-seen): {[w.tolist() for w in replay_waypoints]}")

    ctx             = _build_context()
    model           = ctx["model"]
    data            = ctx["data"]
    cam_name        = ctx["cam_name"]
    target_mocap_id = ctx["target_mocap_id"]

    net = Network(input_size=input_dim, output_size=model.nu)
    fill_parameters(net, best_weights)

    mujoco.mj_resetData(model, data)
    data.mocap_pos[target_mocap_id] = replay_waypoints[0]

    videos_dir = DATA / "videos"
    videos_dir.mkdir(exist_ok=True)
    video_recorder = VideoRecorder(
        file_name="randomized_best", output_folder=str(videos_dir)
    )

    fps               = 30
    dt                = model.opt.timestep
    steps_per_frame   = max(1, int(round(1.0 / (fps * dt))))
    control_step_freq = 50
    current_ctrl      = np.zeros(model.nu)
    render_step       = 0
    wp_idx            = 0
    current_target    = replay_waypoints[0]
    num_wps           = len(replay_waypoints)

    control_renderer = mujoco.Renderer(model, height=24*4, width=32*4)

    def get_control(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
        nonlocal wp_idx, current_target
        control_renderer.update_scene(d, camera=cam_name)
        img    = control_renderer.render()
        vision = analyze_sections(isolate_green(img))
        rs     = get_robot_state(d)
        phase  = [2.0 * np.sin(d.time * 2.0 * np.pi), 2.0 * np.cos(d.time * 2.0 * np.pi)]
        prog   = [wp_idx / max(num_wps - 1, 1)]
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

                if wp_idx < num_wps:
                    dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
                    if dist <= REACH_RADIUS:
                        wp_idx += 1
                        if wp_idx < num_wps:
                            current_target = replay_waypoints[wp_idx]
                            data.mocap_pos[target_mocap_id] = current_target

            renderer.update_scene(data, camera=camera_id)
            video_recorder.write(renderer.render())

    video_recorder.release()
    control_renderer.close()
    console.log(f"[green]Video saved to {videos_dir}[/green]")


if __name__ == "__main__":
    main()
    gc.disable()
