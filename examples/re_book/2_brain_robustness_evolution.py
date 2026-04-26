"""
Robust brain evolution — worst-case multi-target training.

Each candidate is evaluated on three target positions spread ~120° apart.
Fitness is the *worst* per-target score (maximum distance to the hardest
target), not the average.  This forces the evolved brain to perform
acceptably in every direction rather than hiding a weak heading behind
strong ones.

Weights are saved to __data__/2_brain_robustness_evolution/best_weights.npy
and can be used directly as --warm-start for 5_randomized_waypoints.py.
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
from ariel.simulation.tasks.targeted_locomotion import distance_to_target
from ariel.utils.renderers import VideoRecorder

install()
console = Console()

warnings.filterwarnings(
    "ignore",
    message="TPA: apparent inconsistency",
    category=UserWarning,
    module="cma",
)

# ── CLI ───────────────────────────────────────────────────────────────────────

import argparse
parser = argparse.ArgumentParser(description="Robust multi-target brain evolution")
parser.add_argument("--budget",       type=int,   default=200,  help="CMA generations")
parser.add_argument("--population",   type=int,   default=48,   help="Requested population size")
parser.add_argument("--dur",          type=int,   default=10,   help="Episode duration (s)")
parser.add_argument("--reach-radius", type=float, default=0.25, help="Arrival threshold (m)")
parser.add_argument("--workers",      type=int,   default=max(1, os.cpu_count() or 1))
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--no-video",     action="store_true", help="Skip video recording")
args = parser.parse_args()

BUDGET       = args.budget
POP_SIZE     = args.population
DURATION     = args.dur
REACH_RADIUS = max(0.01, args.reach_radius)
NUM_WORKERS  = max(1, args.workers)
BASE_SEED    = args.seed

# Three targets spread ~120° apart at equal radius.
# The spider spawns at the origin; it must learn to walk in every direction.
#
#        (-2, 0)   LEFT
#            \
#     origin  ●
#            / \
#    (1, -1.7)   (1, 1.7)
#   FORWARD-RIGHT  BACK-RIGHT
#
TARGET_POSITIONS: list[np.ndarray] = [
    np.array([ 2.0,  0.0,  0.1]),   # right
    np.array([-1.0, -1.73, 0.1]),   # forward-left  (~120°)
    np.array([-1.0,  1.73, 0.1]),   # back-left     (~240°)
]

SCRIPT_NAME = Path(__file__).stem
DATA = Path.cwd() / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)


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
    sections = np.array_split(mask, 3, axis=1)
    return [cv2.countNonZero(s) / max(s.size, 1) for s in sections]


# ── Simulation runner ─────────────────────────────────────────────────────────

def run_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    network: Network,
    target_position: np.ndarray,
    renderer: mujoco.Renderer,
    cam_name: Optional[str],
    control_step_freq: int = 50,
) -> float:
    """Run one episode toward target_position; return min distance reached."""
    current_action      = np.zeros(model.nu)
    min_dist            = float("inf")
    step                = 0

    while data.time < DURATION:
        if step % control_step_freq == 0:
            renderer.update_scene(data, camera=cam_name)
            img    = renderer.render()
            vision = analyze_sections(isolate_green(img))

            robot_state = get_robot_state(data)
            phase = [
                2.0 * np.sin(data.time * 2.0 * np.pi),
                2.0 * np.cos(data.time * 2.0 * np.pi),
            ]
            state = np.concatenate([robot_state, vision, phase]).astype(np.float32)
            current_action = network.forward(model, data, state)

        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)
        step += 1

        pos  = np.array(data.qpos[:2])
        dist = float(np.linalg.norm(pos - target_position[:2]))
        min_dist = min(min_dist, dist)

    return min_dist


# ── Per-process context ───────────────────────────────────────────────────────

_RENDER_INIT_LOCK = threading.Lock()
_process_local_ctx: Optional[dict[str, Any]] = None


def _build_context() -> dict[str, Any]:
    world  = SimpleFlatWorld()
    spider = body_spider45()
    world.spawn(spider.spec, position=[0.0, 0.0, 0.1])

    # Single green marker — repositioned before each target episode.
    marker = world.spec.worldbody.add_body(
        name="green_target", mocap=True, pos=TARGET_POSITIONS[0].tolist()
    )
    marker.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.1, 0.1, 0.1],
        rgba=[0, 1, 0, 1],
    )

    world.spec.worldbody.add_camera(
        name="video_cam",
        pos=[0, -1, 3],
        xyaxes=[1, 0, 0, 0, 3, 0],
    )

    model = world.spec.compile()
    data  = mujoco.MjData(model)

    cam_name: Optional[str] = None
    for i in range(model.ncam):
        name = model.camera(i).name
        if ("camera" in name or "core" in name) and "video" not in name:
            cam_name = name
            break

    target_mocap_id  = model.body("green_target").mocapid[0]
    robot_state_size = len(get_robot_state(data))
    input_dim        = robot_state_size + 3 + 2   # state + vision + phase

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
    """
    Evaluate one brain on all targets; return the worst (highest) distance.

    Using max rather than mean forces the optimiser to improve on the
    hardest target instead of coasting on strong ones.
    """
    ctx             = _get_ctx()
    model           = cast(mujoco.MjModel,  ctx["model"])
    data            = cast(mujoco.MjData,   ctx["data"])
    network         = cast(Network,         ctx["network"])
    renderer        = cast(mujoco.Renderer, ctx["renderer"])
    cam_name        = cast(Optional[str],   ctx["cam_name"])
    target_mocap_id = cast(int,             ctx["target_mocap_id"])

    fill_parameters(network, weights)

    per_target: list[float] = []
    for target_pos in TARGET_POSITIONS:
        mujoco.mj_resetData(model, data)
        data.mocap_pos[target_mocap_id] = target_pos
        dist = run_episode(
            model=model,
            data=data,
            network=network,
            target_position=target_pos,
            renderer=renderer,
            cam_name=cam_name,
        )
        per_target.append(dist)

    return max(per_target)   # worst-case across all targets


# ── Evolution ─────────────────────────────────────────────────────────────────

def evolve() -> tuple[np.ndarray, int]:
    ctx       = _build_context()
    input_dim = ctx["input_dim"]
    model     = ctx["model"]

    dummy_net  = Network(input_size=input_dim, output_size=model.nu)
    num_params = sum(p.numel() for p in dummy_net.parameters())

    min_lambda = 4 + int(3 * np.log(max(num_params, 2)))
    pop_size   = max(POP_SIZE, min_lambda)
    if pop_size % 2 != 0:
        pop_size += 1

    initial_guess = np.random.uniform(-0.5, 0.5, size=num_params)
    param = ng.p.Array(init=initial_guess).set_mutation(sigma=0.3)

    cma_config = ng.optimizers.ParametrizedCMA(popsize=pop_size)
    optimizer  = cma_config(
        parametrization=param,
        budget=BUDGET * pop_size,
        num_workers=pop_size,
    )

    console.rule("[bold magenta]Robust Brain Evolution — worst-case multi-target[/bold magenta]")
    console.log(
        f"params={num_params}  pop_size={pop_size} (requested {POP_SIZE})  "
        f"budget={BUDGET} gens  workers={NUM_WORKERS}"
    )
    console.log(f"Targets: {[t.tolist() for t in TARGET_POSITIONS]}")
    console.log(f"Fitness: worst-case distance across {len(TARGET_POSITIONS)} targets")

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(BASE_SEED,),
    ) as executor:
        prev_best = float("inf")
        for gen in range(BUDGET):
            candidates = [optimizer.ask() for _ in range(pop_size)]
            fitnesses  = list(executor.map(_evaluate_candidate, [c.value for c in candidates]))

            for cand, fit in zip(candidates, fitnesses):
                optimizer.tell(cand, fit)

            best  = float(np.min(fitnesses))
            delta = best - prev_best
            prev_best = best
            console.rule(f"Gen {gen + 1}/{BUDGET}")
            console.log(
                f"Best (worst-case dist): {best:.4f}  "
                f"({'↓' if delta < 0 else '↑'}{abs(delta):.4f})"
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
    console.log(
        f"Evolution finished in {elapsed / 60:.1f} min. "
        f"Weights → {weights_path}"
    )

    if args.no_video:
        return

    # ── Replay: one video per target ─────────────────────────────────────────
    ctx             = _build_context()
    model           = ctx["model"]
    data            = ctx["data"]
    cam_name        = ctx["cam_name"]
    target_mocap_id = ctx["target_mocap_id"]

    net = Network(input_size=input_dim, output_size=model.nu)
    fill_parameters(net, best_weights)

    videos_dir = DATA / "videos"
    videos_dir.mkdir(exist_ok=True)

    try:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "video_cam")
    except Exception:
        camera_id = -1

    fps               = 30
    dt                = model.opt.timestep
    steps_per_frame   = max(1, int(round(1.0 / (fps * dt))))
    control_step_freq = 50

    control_renderer = mujoco.Renderer(model, height=24, width=32)

    for idx, target_pos in enumerate(TARGET_POSITIONS):
        mujoco.mj_resetData(model, data)
        data.mocap_pos[target_mocap_id] = target_pos

        recorder = VideoRecorder(
            file_name=f"robust_target_{idx}",
            output_folder=str(videos_dir),
        )

        current_ctrl = np.zeros(model.nu)
        render_step  = 0

        def get_control(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
            control_renderer.update_scene(d, camera=cam_name)
            img    = control_renderer.render()
            vision = analyze_sections(isolate_green(img))
            rs     = get_robot_state(d)
            phase  = [2.0 * np.sin(d.time * 2.0 * np.pi), 2.0 * np.cos(d.time * 2.0 * np.pi)]
            state  = np.concatenate([rs, vision, phase]).astype(np.float32)
            return net.forward(m, d, state)

        with mujoco.Renderer(model, height=480, width=640) as renderer:
            while data.time < DURATION:
                for _ in range(steps_per_frame):
                    if render_step % control_step_freq == 0:
                        current_ctrl = get_control(model, data)
                    np.copyto(data.ctrl, current_ctrl)
                    mujoco.mj_step(model, data)
                    render_step += 1

                renderer.update_scene(data, camera=camera_id)
                recorder.write(renderer.render())

        recorder.release()
        console.log(
            f"[green]Video {idx + 1}/{len(TARGET_POSITIONS)} saved "
            f"(target {target_pos[:2].tolist()})[/green]"
        )

    control_renderer.close()


if __name__ == "__main__":
    main()
    gc.disable()
