# Standard libraries
import gc
import random
from typing import Literal, cast, List, Optional, Any
from pathlib import Path
import time
import os
import threading
import cv2 
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# Pretty little errors and progress bars
from rich.console import Console
from rich.traceback import install

# Initialize rich console and traceback handler
install()
console = Console()

# Third-party libraries
import numpy as np
import mujoco
import matplotlib.pyplot as plt

# Network imports
import torch
from torch import nn
from torch.nn import Tanh

# Learner
import nevergrad as ng

# Local libraries
from ariel.simulation.environments import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider_with_blocks import body_spider45
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
from ariel.utils.renderers import VideoRecorder, video_renderer
from ariel.simulation.tasks.targeted_locomotion import (
    fitness_delta_distance, 
    fitness_distance_and_efficiency, 
    fitness_survival_and_locomotion,
    fitness_direct_path,
    distance_to_target,
    fitness_speed_to_target,
)

# Set up command line argument parsing
# If none given, default values are used.
import argparse
parser = argparse.ArgumentParser(description='Evolution simulation with configurable budget')
parser.add_argument('--budget', type=int, default=10, help='Number of generations for learning')
parser.add_argument('--dur', type=int, default=10, help="Duration of an evaluation")
parser.add_argument('--population', type=int, default=10, help="Population size")
parser.add_argument('--fitness', type=str, default='distance', choices=['delta', 'efficiency', 'survival', 'direct', 'distance', 'speed'])
parser.add_argument('--reach-radius', type=float, default=0.25, help='Planar distance threshold for counting target arrival')
parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() or 1), help='Number of worker threads for parallel candidate evaluation')
args = parser.parse_args()

BUDGET = args.budget
DURATION = args.dur
POP_SIZE = args.population
REACH_RADIUS = max(0.01, args.reach_radius)
NUM_WORKERS = max(1, args.workers)

# 1. Defined 3 target positions to prevent overfitting
# TARGET_POSITIONS = [ 
#     [-0.5 , -2, 0.1],  # Left
#     [0.0, -2, 0.1],    # Center
#     [0.5, -2, 0.1]     # Right
# ]

TARGET_POSITIONS = [ 
    [0.5, -2, 0.1]  # Right
]

# Global constants
# Get file name and location to create data save folder.
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)



# ============================================================================ #
#                       Network and Helper function                            #
# ============================================================================ #
class Network(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, hidden_size: int
    ) -> None:
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        self.hidden_activation = nn.ELU()
        self.output_activation = nn.Tanh()

        self.input = input_size

        # Disable gradients for all parameters
        for param in self.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def forward(self, model, data, state):
        x = torch.Tensor(state)

        x = self.hidden_activation(self.fc1(x))
        x = self.hidden_activation(self.fc2(x))
        x = self.output_activation(self.fc4(x)) * (torch.pi / 2)

        return x.detach().numpy()

@torch.no_grad()
def fill_parameters(net: nn.Module, vector: torch.Tensor):
    """Fill the parameters of a torch module (net) from a vector.

    No gradient information is kept.

    The vector's length must be exactly the same with the number
    of parameters of the PyTorch module.

    Args:
        net: The torch module whose parameter values will be filled.
        vector: A 1-D torch tensor which stores the parameter values.
    """
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(vector[address : address + n], device=d.device)
        address += n

    if address != len(vector):
        raise IndexError("The parameter vector is larger than expected")



# ============================================================================ #
#                         Camera frame processing                              #
# ============================================================================ #
    
def isolate_green(frame):
    # Convert to HSV color space
    # Enhances colours, makes it easier to detect target colour
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # Define range for green color
    # At the moment hardcoding seems to work best
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green color
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    return green_mask

def analyze_sections(green_mask):
    sections = np.array_split(green_mask, 3, axis=1)
    left_section, middle_section, right_section = sections[0], sections[1], sections[2]
    
    # Calculate percentage of green pixels in each section
    def get_green_percentage(section):
        total_pixels = section.size
        if total_pixels == 0:
            return 0.0
        green_pixels = cv2.countNonZero(section)
        return (green_pixels / total_pixels) 
    
    left_percent = get_green_percentage(left_section)
    middle_percent = get_green_percentage(middle_section)
    right_percent = get_green_percentage(right_section)
    
    return [left_percent, middle_percent, right_percent]



# ============================================================================ #
#                  Custom simulation runner with camera                        #
# ============================================================================ #

def run_vision_simulation(model, 
                          data, 
                          network:Network, 
                          duration:int, 
                          target_position: Optional[np.ndarray] = None,
                          renderer=None, 
                          cam_name=None,
                          control_step_freq=50 
                          ):
    """Custom runner that processes vision."""
    
    # Setup Renderer if not passed (creates a new context)
    owns_renderer = renderer is None
    if renderer is None:
        renderer = mujoco.Renderer(model, height=24, width=32) 
    
    timestep = model.opt.timestep
    
    # Initialize control placeholder
    current_action = np.zeros(model.nu)

    last_pos = np.array(data.qpos[0:3].copy())
    total_path_length = 0.0
    min_distance_to_target = float("inf")
    time_to_target: Optional[float] = None
    
    trajectory = []
    
    try:
        while data.time < duration:
            # Calculate deduced step count (Optimization from controller.py)
            deduced_step = int(np.ceil(data.time / timestep))
            
            # --- CONTROL STEP ---
            # Only run expensive vision and network pass every N steps
            if deduced_step % control_step_freq == 0:
                with _RENDER_STEP_LOCK:
                    renderer.update_scene(data, camera=cam_name)
                    img = renderer.render()
                
                # 2. Process Vision
                mask = isolate_green(img)
                vision_inputs = analyze_sections(mask)

                # 3. Prepare Inputs
                robot_state = get_robot_state(data)
                
                # Using both sin and cos gives the network a smooth, circular sense of time
                phase_inputs = [
                    2*np.sin(data.time * 2.0 * np.pi), 
                    2*np.cos(data.time * 2.0 * np.pi)
                ]
                
                state_input = np.concatenate([
                    robot_state,
                    vision_inputs,
                    phase_inputs  # Add to the end
                ]).astype(np.float32)

                # 4. Network Forward Pass
                current_action = network.forward(model, data, state_input)
                trajectory.append((data.qpos[0], data.qpos[1]))
            
            # 5. Apply Control (Hold previous action if not a control step)
            data.ctrl[:] = current_action
            
            # 6. Step Physics
            mujoco.mj_step(model, data)

            current_pos = np.array(data.qpos[0:3].copy())
            total_path_length += np.linalg.norm(current_pos - last_pos)
            last_pos = current_pos

            if target_position is not None:
                planar_distance = float(np.linalg.norm(current_pos[:2] - target_position[:2]))
                min_distance_to_target = min(min_distance_to_target, planar_distance)
                if time_to_target is None and planar_distance <= REACH_RADIUS:
                    time_to_target = float(data.time)
    finally:
        if owns_renderer:
            renderer.close()

    if target_position is None:
        min_distance_to_target = float(np.linalg.norm(last_pos[:2]))

    return {
        "path_length": total_path_length,
        "trajectory" : trajectory,
        "min_distance_to_target": min_distance_to_target,
        "time_to_target": time_to_target,
        }
        
# ============================================================================ #
#                         Define evolutionary loop                             #
# ============================================================================ #

_THREAD_LOCAL = threading.local()
_RENDER_INIT_LOCK = threading.Lock()
_RENDER_STEP_LOCK = threading.Lock()


def _build_simulation_context() -> dict[str, Any]:
    """Build one isolated simulation context for a worker thread."""
    world = SimpleFlatWorld()

    target_pos = TARGET_POSITIONS[0]
    target_body = world.spec.worldbody.add_body(name="green_target", mocap=True, pos=target_pos)
    target_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.1, 0.1, 0.1],
        rgba=[0, 1, 0, 1],
    )

    world.spec.worldbody.add_camera(
        name="video_cam",
        pos=[0, -1, 3],
        xyaxes=[1, 0, 0, 0, 3, 0],
    )

    spider_core = body_spider45()
    world.spawn(spider_core.spec, position=[0, 0, 0.1])

    model = world.spec.compile()
    data = mujoco.MjData(model)

    robot_cam_name = None
    for i in range(model.ncam):
        name = model.camera(i).name
        if "camera" in name or "core" in name:
            robot_cam_name = name
            break
    if robot_cam_name is None and model.ncam > 0:
        robot_cam_name = model.camera(0).name

    target_mocap_id = model.body("green_target").mocapid[0]
    num_joints = len(data.qpos) - 7
    input_dim = 3 + num_joints + 3 + 2

    network = Network(
        input_size=input_dim,
        output_size=model.nu,
        hidden_size=32,
    )

    return {
        "model": model,
        "data": data,
        "network": network,
        "robot_cam_name": robot_cam_name,
        "target_mocap_id": target_mocap_id,
        "input_dim": input_dim,
    }


def _get_thread_context() -> dict[str, Any]:
    """Lazily initialize one Mujoco context per worker thread."""
    if not hasattr(_THREAD_LOCAL, "ctx"):
        _THREAD_LOCAL.ctx = _build_simulation_context()
    return cast(dict[str, Any], _THREAD_LOCAL.ctx)


def _fitness_from_metrics(
    initial_pos: np.ndarray,
    final_pos: np.ndarray,
    final_z_height: float,
    target_pos_arr: np.ndarray,
    metrics: dict[str, Any],
) -> float:
    if args.fitness == 'delta':
        return fitness_delta_distance(initial_pos, final_pos, target_pos_arr)
    if args.fitness == 'distance':
        return distance_to_target(final_pos, target_pos_arr)
    if args.fitness == 'survival':
        return fitness_survival_and_locomotion(initial_pos, final_pos, target_pos_arr, final_z_height)
    if args.fitness == 'efficiency':
        return fitness_distance_and_efficiency(initial_pos, final_pos, target_pos_arr, 0.0)
    if args.fitness == 'direct':
        return fitness_direct_path(initial_pos, final_pos, target_pos_arr, metrics["path_length"])
    if args.fitness == 'speed':
        return fitness_speed_to_target(
            time_to_target=metrics["time_to_target"],
            duration=DURATION,
            min_distance_to_target=metrics["min_distance_to_target"],
        )
    return distance_to_target(final_pos, target_pos_arr)


def _evaluate_candidate(weights: np.ndarray) -> float:
    """Evaluate one candidate controller in a thread-local simulation context."""
    ctx = _get_thread_context()
    model = cast(mujoco.MjModel, ctx["model"])
    data = cast(mujoco.MjData, ctx["data"])
    network = cast(Network, ctx["network"])
    robot_cam_name = cast(Optional[str], ctx["robot_cam_name"])
    target_mocap_id = cast(int, ctx["target_mocap_id"])

    fill_parameters(network, torch.as_tensor(weights, dtype=torch.float32))

    with _RENDER_INIT_LOCK:
        renderer = mujoco.Renderer(model, height=24, width=32)

    try:
        total_fitness = 0.0
        for target_pos in TARGET_POSITIONS:
            mujoco.mj_resetData(model, data)
            data.mocap_pos[target_mocap_id] = target_pos

            initial_pos = np.array(data.qpos[0:3].copy())
            target_pos_arr = np.array(target_pos)

            metrics = run_vision_simulation(
                model,
                data,
                network=network,
                duration=DURATION,
                target_position=target_pos_arr,
                renderer=renderer,
                cam_name=robot_cam_name,
                control_step_freq=50,
            )

            final_pos = np.array(data.qpos[0:3].copy())
            final_z_height = final_pos[2]
            total_fitness += _fitness_from_metrics(
                initial_pos=initial_pos,
                final_pos=final_pos,
                final_z_height=final_z_height,
                target_pos_arr=target_pos_arr,
                metrics=metrics,
            )

        return total_fitness / len(TARGET_POSITIONS)
    finally:
        renderer.close()

def evolve(world, model, data) -> tuple[np.ndarray, int]:
    """Evolve the robot's movement using multithreaded candidate evaluation."""
    del world, data
    console.log(f"Evolving for {BUDGET} generations with Vision Input")

    num_joints = model.nq - 7
    input_dim = 3 + num_joints + 3 + 2

    dummy_net = Network(input_size=input_dim, output_size=model.nu, hidden_size=32)
    num_params = sum(p.numel() for p in dummy_net.parameters())
    
    initial_guess = np.random.uniform(low=-0.5, high=0.5, size=num_params)
    param = ng.p.Array(init=initial_guess)
    param.set_mutation(sigma=0.075) 
    
    cma_config = ng.optimizers.ParametrizedCMA(popsize=POP_SIZE)
    optimizer = cma_config(
        parametrization=param,
        budget=(BUDGET * POP_SIZE),
        num_workers=POP_SIZE, 
    )

    console.log(f"Population size: {POP_SIZE} | Workers: {NUM_WORKERS}")

    eval_timeout_s = max(30, int(DURATION * 8))
    console.log(f"[cyan]Evaluator mode: generation-deadline ({eval_timeout_s}s) + non-blocking shutdown[/cyan]")

    executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)
    timed_out = False
    try:
        for bud in range(BUDGET + 1):
            candidates = [optimizer.ask() for _ in range(POP_SIZE)]
            futures = [executor.submit(_evaluate_candidate, c.value) for c in candidates]
            fitnesses: list[float] = [float("inf")] * len(candidates)
            future_to_idx = {future: idx for idx, future in enumerate(futures)}

            start_t = time.monotonic()
            deadline = start_t + eval_timeout_s
            pending = set(futures)
            last_progress_log_t = start_t

            while pending:
                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    break

                done, pending = wait(
                    pending,
                    timeout=min(2.0, max(0.1, remaining)),
                    return_when=FIRST_COMPLETED,
                )

                for future in done:
                    idx = future_to_idx[future]
                    try:
                        fitnesses[idx] = float(future.result())
                    except Exception as exc:
                        console.log(
                            f"[yellow]Candidate failed at generation {bud}, index {idx}: {exc}; assigning inf fitness.[/yellow]"
                        )
                        fitnesses[idx] = float("inf")

                if time.monotonic() - last_progress_log_t >= 10.0:
                    completed = len(candidates) - len(pending)
                    elapsed = int(time.monotonic() - start_t)
                    console.log(
                        f"[cyan]Gen {bud} eval progress:[/cyan] {completed}/{len(candidates)} completed | elapsed {elapsed}s"
                    )
                    last_progress_log_t = time.monotonic()

            if pending:
                console.log(
                    f"[yellow]Generation {bud} hit eval deadline ({eval_timeout_s}s). "
                    f"Marking {len(pending)} pending candidates as inf and stopping evolution early.[/yellow]"
                )
                for future in pending:
                    future.cancel()
                timed_out = True

            for candidate, fit in zip(candidates, fitnesses):
                optimizer.tell(candidate, fit)

            gen_best = float(np.min(fitnesses))
            console.rule(f"Budget: {bud}/{BUDGET}")
            console.log(f"Best Fit (Gen): {gen_best:.4f}")

            if timed_out:
                break
    finally:
        # Never block on thread joins if a worker gets stuck in native code.
        executor.shutdown(wait=False, cancel_futures=True)

    best_ind = optimizer.provide_recommendation().value
    return best_ind, input_dim

# ============================================================================ #
#                           Main entry function                                #
# ============================================================================ #

def main():
    mujoco.set_mjcb_control(None)

    # Initialise world
    world = SimpleFlatWorld()
    
    # Add Green Target Object
    target_pos = TARGET_POSITIONS[0] # left
    target_body = world.spec.worldbody.add_body(name="green_target", mocap=True, pos=target_pos)
    target_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, 
        size=[0.1, 0.1, 0.1], 
        rgba=[0, 1, 0, 1]
    ) 
    
    # Add Global Camera for Video Recording
    world.spec.worldbody.add_camera(
        name="video_cam", 
        pos=[0, -1, 3], 
        xyaxes=[1, 0, 0, 0, 3, 0]
    )

    # Spawn Spider
    spider_core = body_spider45()
    world.spawn(spider_core.spec, position=[0, 0, 0.1])
    
    model = world.spec.compile()
    data = mujoco.MjData(model)

    best_weights, final_input_dim = evolve(world, model, data)
    
    return model, data, best_weights, world, final_input_dim

if __name__ == "__main__":
    start = time.time()
    model, data, best_weights, world, input_dim = main()
    gc.disable()
    
    end = time.time()

    console.log(f"Evolution took {(end-start)/60:.2f} minutes")

    weights_path = "3_spider_vision_new.npy"
    # Unconditionally save the new weights, overwriting any old ones
    np.save(weights_path, best_weights)
    console.log(f"[green]Best weights saved to {weights_path}[/green]")

# ============================================================================ #
#                           Initialise world and                               #
#                           load best performer                                #
#                           for video recording                                #
# ============================================================================ #
    network = Network(
        input_size=input_dim, 
        output_size=model.nu, 
        hidden_size=32
    )
    fill_parameters(network, torch.Tensor(best_weights))

    # Identify robot camera
    robot_cam_name = None
    for i in range(model.ncam):
        cam_name = model.camera(i).name
        # Check for 'core' to match the spider's camera naming convention
        if ("camera" in cam_name or "core" in cam_name) and "video" not in cam_name:
            robot_cam_name = cam_name
            break

    path_to_video_folder = str(DATA / "videos")
    os.makedirs(path_to_video_folder, exist_ok=True) # Ensure folder exists
    
    # Reset Simulation & Target
    mujoco.mj_resetData(model, data)
    target_mocap_id = model.body("green_target").mocapid[0]
    data.mocap_pos[target_mocap_id] = TARGET_POSITIONS[0]
    
    # 1. Setup separate renderer for the Robot's Vision (Low Res)
    control_renderer = mujoco.Renderer(model, height=24, width=32)
    
    def get_vision_control_signal(m, d):
        if robot_cam_name:
            control_renderer.update_scene(d, camera=robot_cam_name)
            img = control_renderer.render()
            mask = isolate_green(img)
            vision_inputs = analyze_sections(mask)
        else:
            vision_inputs = [0,0,0]
            
        robot_state = get_robot_state(d)
        
        phase_inputs = [
            2*np.sin(d.time * 2.0 * np.pi), 
            2*np.cos(d.time * 2.0 * np.pi)
        ]
        
        state = np.concatenate([
            robot_state,
            vision_inputs,
            phase_inputs
        ]).astype(np.float32)
        
        return network.forward(m, d, state)

# --- REPLAY BEST & RECORD VIDEO ---
    console.log("[cyan]Rendering Best Video...[/cyan]")
    
    # Setup VideoRecorder
    video_recorder = VideoRecorder(
        file_name="spider_vision_best", 
        output_folder=path_to_video_folder
    )

    # Setup Visualization Options
    viz_options = mujoco.MjvOption()
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = False

    # Get Camera ID ("video_cam")
    try:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "video_cam")
    except Exception:
        camera_id = -1 # Fallback to default free camera if not found

    # Timing Variables
    fps = 30
    dt = model.opt.timestep
    # Prevent infinite loops when dt is large enough that 1/(fps*dt) < 1.
    steps_per_frame = max(1, int(round(1.0 / (fps * dt))))
    control_step_freq = 50
    current_ctrl = np.zeros(model.nu)

    # Main Rendering Loop (Using Context Manager for safe memory handling)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        
        while data.time < DURATION:
            # INNER LOOP: Step physics N times to match Video FPS
            for _ in range(steps_per_frame):
                deduced_step = int(np.ceil(data.time / dt))

                if deduced_step % control_step_freq == 0:
                    current_ctrl = get_vision_control_signal(model, data)

                # Safely copy control array
                np.copyto(data.ctrl, current_ctrl)
                mujoco.mj_step(model, data)

            # OUTER LOOP: Render Frame (Once per 1/30th second)
            renderer.update_scene(
                data, 
                scene_option=viz_options, 
                camera=camera_id
            )
            video_recorder.write(frame=renderer.render())

        # Finish Video
        video_recorder.release()
        console.log(f"[green]Video rendering complete. Saved to {path_to_video_folder}[/green]")

# ============================================================================ #
#                           Plotting the Trajectory                            #
# ============================================================================ #
    console.log("[cyan]Generating Trajectory Plot...[/cyan]")
    
    # Pick one target position to test on (e.g., the first one)
    test_target = TARGET_POSITIONS[0]
    mujoco.mj_resetData(model, data)
    data.mocap_pos[target_mocap_id] = test_target
    
    # Run the simulation once more to get the path
    metrics = run_vision_simulation(
        model, 
        data, 
        network=network, 
        duration=DURATION, 
        target_position=np.asarray(test_target),
        renderer=None, # No need to render video for this
        cam_name=robot_cam_name, 
        control_step_freq=50
    )
    
    # Extract X and Y coordinates
    path = metrics["trajectory"]
    x_coords = [p[0] for p in path]
    y_coords = [p[1] for p in path]
    
    # Create the plot
    plt.figure(figsize=(8, 8))
    
    # Plot the robot's starting position
    plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
    
    # Plot the Target position
    plt.plot(test_target[0], test_target[1], 'r*', markersize=15, label='Target')
    
    # Plot the actual path
    plt.plot(x_coords, y_coords, 'b-', linewidth=2, label='Robot Path')
    
    plt.title(f"Robot Trajectory Map (Fitness: {args.fitness})")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.legend()
    plt.grid(True)
    
    # Save the plot next to your videos
    plot_path = os.path.join(path_to_video_folder, f"trajectory_{args.fitness}.png")
    plt.savefig(plot_path)
    console.log(f"[green]Trajectory map saved to {plot_path}[/green]")
    os._exit(0)