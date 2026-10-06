# Standard libraries
import gc
import random
import warnings
from typing import Any
from pathlib import Path
import time
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# Pretty little errors and progress bars
from rich.console import Console
from rich.traceback import install

install()
console = Console()

warnings.filterwarnings(
    "ignore",
    message="TPA: apparent inconsistency",
    category=UserWarning,
    module="cma",
)

# Third-party libraries
import numpy as np
import mujoco
import matplotlib.pyplot as plt

# Network
import torch
from torch import nn

# Learner
import nevergrad as ng

# Local libraries
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
from ariel.utils.renderers import VideoRecorder
from ariel.simulation.tasks.gait_learning import xy_displacement

# ============================================================================ #
#                          Command-line arguments                              #
# ============================================================================ #
import argparse

MORPHOLOGY_CHOICES = ["spider", "centipede", "gecko"]

parser = argparse.ArgumentParser(description="Undirected locomotion evolution")
parser.add_argument("--budget", type=int, default=200, help="Number of generations")
parser.add_argument("--dur", type=int, default=20, help="Evaluation duration (seconds)")
parser.add_argument("--population", type=int, default=30, help="Population size")
parser.add_argument(
    "--morphology",
    type=str,
    default="spider",
    choices=MORPHOLOGY_CHOICES,
    help="Robot morphology to evolve",
)
parser.add_argument(
    "--workers",
    type=int,
    default=max(1, os.cpu_count() or 1),
    help="Number of parallel worker processes",
)
parser.add_argument("--seed", type=int, default=42, help="Base random seed")
parser.add_argument("--view", action="store_true", help="Open MuJoCo viewer after evolution")
parser.add_argument(
    "--base-evals",
    type=int,
    default=None,
    dest="base_evals",
    help="Total function evaluations budget; overrides --budget. "
         "Actual generations = base_evals // actual_pop_size, giving each "
         "morphology a fair budget regardless of parameter count.",
)
parser.add_argument(
    "--n-pairs",
    type=int,
    default=2,
    dest="n_pairs",
    help="Number of limb-bearing spine segments (centipede morphology only)",
)
args = parser.parse_args()

BUDGET = args.budget
DURATION = args.dur
POP_SIZE = args.population
NUM_WORKERS = max(1, args.workers)
BASE_SEED = int(args.seed)

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
_morph_key = f"centipede_{args.n_pairs}pairs" if args.morphology == "centipede" else args.morphology
DATA = Path(CWD / "__data__" / SCRIPT_NAME / _morph_key)
DATA.mkdir(exist_ok=True, parents=True)


def _load_body():
    """Return a fresh body instance for the selected morphology."""
    if args.morphology == "spider":
        from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider import spider
        return spider()
    if args.morphology == "centipede":
        from ariel.body_phenotypes.robogen_lite.prebuilt_robots.centipede import body_centipede_n
        return body_centipede_n(args.n_pairs)
    if args.morphology == "gecko":
        from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
        return gecko()
    raise ValueError(f"Unknown morphology: {args.morphology}")


# ============================================================================ #
#                       Network and helper functions                           #
# ============================================================================ #

class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.hidden_act = nn.ELU()
        self.output_act = nn.Tanh()
        for p in self.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, x: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(x, dtype=torch.float32)
        t = self.hidden_act(self.fc1(t))
        t = self.hidden_act(self.fc2(t))
        t = self.output_act(self.fc3(t)) * (torch.pi / 2)
        return t.numpy()


@torch.no_grad()
def fill_parameters(net: nn.Module, vector: torch.Tensor) -> None:
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(vector[address: address + n], device=d.device)
        address += n
    if address != len(vector):
        raise IndexError("Parameter vector length mismatch")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _init_worker(base_seed: int) -> None:
    torch.set_num_threads(1)
    worker_seed = (base_seed + os.getpid()) % (2**32 - 1)
    _seed_everything(worker_seed)


# ============================================================================ #
#                         Simulation runner (no vision)                        #
# ============================================================================ #

def run_simulation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    network: Network,
    duration: int,
    control_step_freq: int = 50,
) -> dict[str, Any]:
    """Run one episode and return locomotion metrics."""
    current_action = np.zeros(model.nu)
    last_pos = data.qpos[0:3].copy()
    total_path_length = 0.0
    trajectory = []
    step = 0

    while data.time < duration:
        if step % control_step_freq == 0:
            robot_state = get_robot_state(data)
            phase = [
                2 * np.sin(data.time * 2.0 * np.pi),
                2 * np.cos(data.time * 2.0 * np.pi),
            ]
            state = np.concatenate([robot_state, phase]).astype(np.float32)
            current_action = network.forward(state)
            trajectory.append((float(data.qpos[0]), float(data.qpos[1])))

        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)
        step += 1

        current_pos = data.qpos[0:3].copy()
        total_path_length += float(np.linalg.norm(current_pos - last_pos))
        last_pos = current_pos

    final_xy = (float(data.qpos[0]), float(data.qpos[1]))
    displacement = xy_displacement((0.0, 0.0), final_xy)

    return {
        "displacement": displacement,
        "path_length": total_path_length,
        "trajectory": trajectory,
        "final_xy": final_xy,
    }


# ============================================================================ #
#                         Process-local simulation context                     #
# ============================================================================ #

_process_local_ctx: dict[str, Any] | None = None


def _build_simulation_context() -> dict[str, Any]:
    world = SimpleFlatWorld()
    body = _load_body()
    world.spawn(body.spec, position=[0, 0, 0.1])

    # Global overhead camera for video recording
    world.spec.worldbody.add_camera(
        name="video_cam",
        pos=[0, 0, 5],
        xyaxes=[1, 0, 0, 0, 1, 0],
    )

    model = world.spec.compile()
    data = mujoco.MjData(model)

    num_joints = model.nq - 7
    # inputs: proprioception (3 + num_joints) + phase (2)
    input_dim = 3 + num_joints + 2

    network = Network(input_size=input_dim, output_size=model.nu, hidden_size=32)

    return {
        "model": model,
        "data": data,
        "network": network,
        "input_dim": input_dim,
    }


def _get_process_context() -> dict[str, Any]:
    global _process_local_ctx
    if _process_local_ctx is None:
        _process_local_ctx = _build_simulation_context()
    return _process_local_ctx


def _evaluate_candidate(weights: np.ndarray) -> float:
    """Evaluate one candidate; called in worker processes."""
    ctx = _get_process_context()
    model: mujoco.MjModel = ctx["model"]
    data: mujoco.MjData = ctx["data"]
    network: Network = ctx["network"]

    fill_parameters(network, torch.as_tensor(weights, dtype=torch.float32))
    mujoco.mj_resetData(model, data)

    metrics = run_simulation(model, data, network, DURATION)
    # Nevergrad minimises — negate displacement so higher = better
    return -metrics["displacement"]


# ============================================================================ #
#                            Evolutionary loop                                 #
# ============================================================================ #

def evolve(model: mujoco.MjModel) -> tuple[np.ndarray, int]:
    num_joints = model.nq - 7
    input_dim = 3 + num_joints + 2

    dummy_net = Network(input_size=input_dim, output_size=model.nu, hidden_size=32)
    num_params = sum(p.numel() for p in dummy_net.parameters())

    min_lambda = 4 + int(3 * np.log(max(num_params, 2)))
    pop_size = max(POP_SIZE, min_lambda)
    if pop_size % 2 != 0:
        pop_size += 1

    if args.base_evals is not None:
        budget = max(1, args.base_evals // pop_size)
    else:
        budget = BUDGET

    initial_guess = np.random.uniform(low=-0.5, high=0.5, size=num_params)
    param = ng.p.Array(init=initial_guess)
    param.set_mutation(sigma=1.0)

    cma_config = ng.optimizers.ParametrizedCMA(popsize=pop_size)
    optimizer = cma_config(
        parametrization=param,
        budget=(budget * pop_size),
        num_workers=pop_size,
    )

    console.log(
        f"[bold]Morphology:[/bold] {_morph_key} | "
        f"[bold]Pop:[/bold] {pop_size} (requested {POP_SIZE}) | "
        f"[bold]Workers:[/bold] {NUM_WORKERS} | "
        f"[bold]Params:[/bold] {num_params} | "
        f"[bold]Budget:[/bold] {budget} gens ({budget * pop_size} evals)"
    )

    all_weights: list[np.ndarray] = []
    all_fitnesses: list[float] = []
    sigmas: list[float] = []

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(BASE_SEED,),
    ) as executor:
        for gen in range(budget):
            candidates = [optimizer.ask() for _ in range(pop_size)]
            fitnesses = list(
                executor.map(_evaluate_candidate, [c.value for c in candidates])
            )
            for candidate, fit in zip(candidates, fitnesses):
                optimizer.tell(candidate, fit)
                all_weights.append(candidate.value.copy())
                all_fitnesses.append(fit)

            sigmas.append(optimizer.es.sigma)
            best_displacement = -float(np.min(fitnesses))
            console.rule(f"Gen {gen + 1}/{budget}")
            console.log(f"Best displacement: {best_displacement:.4f} m | sigma: {sigmas[-1]:.4f}")

    best_weights = optimizer.provide_recommendation().value

    log_path = DATA / "search_log.npz"
    np.savez(
        log_path,
        weights=np.array(all_weights),
        fitnesses=np.array(all_fitnesses),
        pop_size=np.array(pop_size),
        sigmas=np.array(sigmas),
    )
    console.log(f"[green]Search log saved to {log_path}[/green]")

    return best_weights, input_dim


# ============================================================================ #
#                               Main entry point                               #
# ============================================================================ #

def main():
    _seed_everything(BASE_SEED)
    mujoco.set_mjcb_control(None)

    world = SimpleFlatWorld()
    body = _load_body()
    world.spawn(body.spec, position=[0, 0, 0.1])
    world.spec.worldbody.add_camera(
        name="video_cam",
        pos=[0, 0, 5],
        xyaxes=[1, 0, 0, 0, 1, 0],
    )

    model = world.spec.compile()
    data = mujoco.MjData(model)

    best_weights, input_dim = evolve(model)
    return model, data, best_weights, input_dim


if __name__ == "__main__":
    start = time.time()
    model, data, best_weights, input_dim = main()
    gc.disable()

    elapsed = time.time() - start
    console.log(f"Evolution took {elapsed / 60:.2f} minutes")

    weights_path = DATA / "best_weights.npy"
    np.save(weights_path, best_weights)
    console.log(f"[green]Best weights saved to {weights_path}[/green]")

# ============================================================================ #
#                          Replay best & record video                          #
# ============================================================================ #
    network = Network(input_size=input_dim, output_size=model.nu, hidden_size=32)
    fill_parameters(network, torch.as_tensor(best_weights, dtype=torch.float32))

    path_to_video_folder = str(DATA / "videos")
    os.makedirs(path_to_video_folder, exist_ok=True)

    mujoco.mj_resetData(model, data)

    fps = 30
    dt = model.opt.timestep
    steps_per_frame = max(1, int(round(1.0 / (fps * dt))))
    control_step_freq = 50
    current_ctrl = np.zeros(model.nu)
    render_step = 0

    # Tracking camera that follows the robot's root body.
    track_cam = mujoco.MjvCamera()
    track_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    track_cam.trackbodyid = 1  # root body (index 0 is world)
    track_cam.distance = 2.5
    track_cam.elevation = -30
    track_cam.azimuth = 45

    viz_options = mujoco.MjvOption()
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False

    VIDEO_W, VIDEO_H = 1280, 720
    video_recorder = VideoRecorder(
        file_name=f"{args.morphology}_locomotion_best",
        output_folder=path_to_video_folder,
        width=VIDEO_W,
        height=VIDEO_H,
    )

    console.log("[cyan]Rendering best video...[/cyan]")

    def get_control(d: mujoco.MjData) -> np.ndarray:
        robot_state = get_robot_state(d)
        phase = [
            2 * np.sin(d.time * 2.0 * np.pi),
            2 * np.cos(d.time * 2.0 * np.pi),
        ]
        state = np.concatenate([robot_state, phase]).astype(np.float32)
        return network.forward(state)

    with mujoco.Renderer(model, height=VIDEO_H, width=VIDEO_W) as renderer:
        while data.time < DURATION:
            for _ in range(steps_per_frame):
                if render_step % control_step_freq == 0:
                    current_ctrl = get_control(data)
                np.copyto(data.ctrl, current_ctrl)
                mujoco.mj_step(model, data)
                render_step += 1
            renderer.update_scene(data, scene_option=viz_options, camera=track_cam)
            video_recorder.write(frame=renderer.render())

    video_recorder.release()
    console.log(f"[green]Video saved to {path_to_video_folder}[/green]")

# ============================================================================ #
#                           Interactive MuJoCo viewer                         #
# ============================================================================ #
    if args.view:
        import sys
        import mujoco.viewer

        mujoco.mj_resetData(model, data)
        control_step_freq = 50
        current_ctrl = np.zeros(model.nu)
        viewer_step = 0

        console.rule("[bold green]MuJoCo Viewer — close window or press Esc to quit[/bold green]")

        if sys.platform == "darwin" or not hasattr(mujoco.viewer, "launch_passive"):
            # Blocking fallback (macOS or older mujoco)
            mujoco.viewer.launch(model=model, data=data)
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                step_start = time.time()
                while viewer.is_running() and data.time < DURATION:
                    if viewer_step % control_step_freq == 0:
                        current_ctrl = get_control(data)
                    np.copyto(data.ctrl, current_ctrl)
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    viewer_step += 1

                    # Pace to real-time so the integrator doesn't explode.
                    elapsed = time.time() - step_start
                    remaining = model.opt.timestep - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
                    step_start = time.time()

# ============================================================================ #
#                             Trajectory plot                                  #
# ============================================================================ #
    console.log("[cyan]Generating trajectory plot...[/cyan]")

    mujoco.mj_resetData(model, data)
    metrics = run_simulation(model, data, network, DURATION)

    path = metrics["trajectory"]
    x_coords = [p[0] for p in path]
    y_coords = [p[1] for p in path]

    plt.figure(figsize=(8, 8))
    plt.plot(x_coords[0], y_coords[0], "go", markersize=10, label="Start")
    plt.plot(x_coords[-1], y_coords[-1], "r*", markersize=15, label="End")
    plt.plot(x_coords, y_coords, "b-", linewidth=2, label="Robot Path")
    plt.title(f"Undirected Locomotion — {args.morphology} (displacement: {metrics['displacement']:.3f} m)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plot_path = os.path.join(path_to_video_folder, "trajectory.png")
    plt.savefig(plot_path)
    console.log(f"[green]Trajectory plot saved to {plot_path}[/green]")
    os._exit(0)
