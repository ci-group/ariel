# Standard libraries
import gc
import random
from typing import Literal, cast, List, Optional, Any
from pathlib import Path
import time
import os
import threading
import cv2 
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

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

# Learner
import nevergrad as ng

# Local libraries
from ariel.simulation.environments import SimpleFlatWorld
# from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider_with_blocks import body_spider45
# from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko as body_spider45
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.random_spider20 import random_spider20 as body_spider45

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
parser.add_argument('--seed', type=int, default=42, help='Base random seed for reproducibility')
parser.add_argument('--vision-decimation', type=int, default=2, help='Update vision every N control updates (1-4) and reuse last vision input in between')
args = parser.parse_args()

BUDGET = args.budget
DURATION = args.dur
POP_SIZE = args.population
REACH_RADIUS = max(0.01, args.reach_radius)
NUM_WORKERS = max(1, args.workers)
BASE_SEED = int(args.seed)
VISION_DECIMATION = max(1, min(4, int(args.vision_decimation)))


def _seed_everything(seed: int) -> None:
    """Seed all RNGs used by this script."""
    random.seed(seed)
    np.random.seed(seed)


def _init_worker(base_seed: int) -> None:
    """Initializer called once in each worker process."""
    worker_seed = (base_seed + os.getpid()) % (2**32 - 1)
    _seed_everything(worker_seed)

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
class FastNumpyNetwork:
    """NumPy MLP matching the PyTorch architecture used in the baseline script."""
    def __init__(self, input_size, hidden_size, output_size, weights):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Layer 1: Input -> Hidden
        w1_end = hidden_size * input_size
        b1_end = w1_end + hidden_size
        
        # Layer 2: Hidden -> Hidden
        w2_end = b1_end + (hidden_size * hidden_size)
        b2_end = w2_end + hidden_size

        # Layer 3: Hidden -> Output
        w3_end = b2_end + (output_size * hidden_size)
        b3_end = w3_end + output_size
        
        # Slice and reshape weights
        self.w1 = weights[0:w1_end].reshape(hidden_size, input_size)
        self.b1 = weights[w1_end:b1_end]
        
        self.w2 = weights[b1_end:w2_end].reshape(hidden_size, hidden_size)
        self.b2 = weights[w2_end:b2_end]

        self.w3 = weights[b2_end:w3_end].reshape(output_size, hidden_size)
        self.b3 = weights[w3_end:b3_end]

    @staticmethod
    def _elu(x):
        return np.where(x > 0.0, x, np.exp(x) - 1.0)

    def forward(self, x):
        # Hidden Layer 1
        x = np.dot(self.w1, x) + self.b1
        x = self._elu(x)
        
        # Hidden Layer 2
        x = np.dot(self.w2, x) + self.b2
        x = self._elu(x)

        # Output Layer
        x = np.dot(self.w3, x) + self.b3
        return np.tanh(x) * (np.pi / 2.0)

# ============================================================================ #
#                         Camera frame processing                              #
# ============================================================================ #
    
# PRE-ALLOCATE GLOBALLY: Create these once, never again.
LOWER_GREEN = np.array([35, 40, 40], dtype=np.uint8)
UPPER_GREEN = np.array([85, 255, 255], dtype=np.uint8)

# Hardcoded slice sizes for a 32x24 image (32 / 3 = ~10.6)
# Left: 11px, Middle: 11px, Right: 10px
TOTAL_P1 = 24 * 11
TOTAL_P2 = 24 * 11
TOTAL_P3 = 24 * 10

def get_vision_inputs(frame: np.ndarray) -> tuple[float, float, float]:
    """Highly optimized, zero-allocation vision processor."""
    # 1. Convert and Mask (OpenCV handles this purely in C)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    
    # 2. Slice and Count directly (Bypasses np.array_split entirely)
    p1 = cv2.countNonZero(mask[:, 0:11]) / TOTAL_P1
    p2 = cv2.countNonZero(mask[:, 11:22]) / TOTAL_P2
    p3 = cv2.countNonZero(mask[:, 22:32]) / TOTAL_P3
    
    return p1, p2, p3

# ============================================================================ #
#                  Custom simulation runner with camera                        #
# ============================================================================ #

def run_vision_simulation(
    model, data, network, duration: float, 
    target_position: Optional[np.ndarray] = None,
    renderer=None, cam_name=None,
    control_step_freq=50 
):
    timestep = model.opt.timestep
    
    last_pos = np.array(data.qpos[0:3].copy())
    start_z = last_pos[2]
    total_path_length = 0.0
    min_distance_to_target = float("inf")
    time_to_target: Optional[float] = None
    trajectory = []
    
    # 1. DYNAMIC PRE-ALLOCATION
    # Get the exact length of the robot state to build our reusable input array
    dummy_robot_state = get_robot_state(data)
    r_len = len(dummy_robot_state)
    input_dim = r_len + 3 + 2 # Proprioception + 3 Vision + 2 Phase
    
    # This array is created ONCE per simulation, eliminating massive memory overhead
    state_input = np.zeros(input_dim, dtype=np.float32)
    current_action = np.zeros(model.nu, dtype=np.float32)
    
    # Default vision inputs in case renderer is disabled
    v1, v2, v3 = 0.0, 0.0, 0.0
    control_updates = 0

    while data.time < duration:
        step = int(np.ceil(data.time / timestep))
        
        # --- CONTROL STEP ---
        if step % control_step_freq == 0:
            control_updates += 1

            # 1. Vision Update
            if renderer is not None:
                # Decimate expensive camera processing but keep control updates frequent.
                should_update_vision = (control_updates == 1) or (control_updates % VISION_DECIMATION == 0)
                if should_update_vision:
                    renderer.update_scene(data, camera=cam_name)
                    v1, v2, v3 = get_vision_inputs(renderer.render())

            # 2. Proprioception Update
            robot_state = get_robot_state(data)
            
            # 3. Update state array IN-PLACE (Zero memory allocation!)
            state_input[0:r_len] = robot_state
            state_input[r_len] = v1
            state_input[r_len+1] = v2
            state_input[r_len+2] = v3
            state_input[-2] = 2.0 * np.sin(data.time * 2.0 * np.pi)
            state_input[-1] = 2.0 * np.cos(data.time * 2.0 * np.pi)

            # 4. Network Forward Pass
            current_action = network.forward(state_input)
            trajectory.append((float(data.qpos[0]), float(data.qpos[1])))
        
        # Apply Control
        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)

        # Distance Tracking (Using slices directly instead of deep copying)
        current_pos = data.qpos[0:3] 
        total_path_length += float(np.linalg.norm(current_pos - last_pos))
        last_pos = np.array(current_pos) # Copy only when saving old state

        if target_position is not None:
            planar_distance = float(np.linalg.norm(current_pos[:2] - target_position[:2]))
            min_distance_to_target = min(min_distance_to_target, planar_distance)
            if time_to_target is None and planar_distance <= REACH_RADIUS:
                time_to_target = float(data.time)

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

def _build_simulation_context() -> dict[str, Any]:
    """Build one isolated simulation context for a worker thread."""
    world = SimpleFlatWorld()

    # Initialize the target at the origin (it will be teleported during eval)
    target_body = world.spec.worldbody.add_body(name="green_target", mocap=True, pos=[0, 0, 0.1])
    target_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.1, 0.1, 0.1],
        rgba=[0, 1, 0, 1],
    )

    # Satellite View Camera (Look straight down from 10m up)
    world.spec.worldbody.add_camera(
        name="video_cam",
        pos=[0, -1.0, 10.0],         # Centered between start and targets
        xyaxes=[1, 0, 0, 0, 1, 0],   # Perfect top-down orientation
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

    with _RENDER_INIT_LOCK:
        renderer = mujoco.Renderer(model, height=24, width=32)

    return {
        "model": model,
        "data": data,
        "renderer": renderer,
        "robot_cam_name": robot_cam_name,
        "target_mocap_id": target_mocap_id,
        "input_dim": input_dim,
    }

_process_local_ctx = None

def _get_process_context() -> dict[str, Any]:
    """Lazily initialize one isolated Mujoco context per worker process."""
    global _process_local_ctx
    if _process_local_ctx is None:
        _process_local_ctx = _build_simulation_context()
        
    return _process_local_ctx

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
    """Evaluate one candidate controller in a process-local simulation context."""
    ctx = _get_process_context()
    model = ctx["model"]
    data = ctx["data"]
    target_mocap_id = ctx["target_mocap_id"]

    network = FastNumpyNetwork(
        input_size=ctx["input_dim"],
        hidden_size=16,
        output_size=model.nu,
        weights=weights
    )

    total_fitness = 0.0
    for target_pos in TARGET_POSITIONS:
        mujoco.mj_resetData(model, data)
        
        # Teleport and Sync Physics
        data.mocap_pos[target_mocap_id] = target_pos
        mujoco.mj_forward(model, data) 

        initial_pos = np.array(data.qpos[0:3].copy())
        target_pos_arr = np.array(target_pos)

        metrics = run_vision_simulation(
            model=model, data=data, network=network,
            duration=DURATION, target_position=target_pos_arr,
            renderer=ctx["renderer"], # Eyes are open
            cam_name=ctx["robot_cam_name"],
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

    average_fitness = total_fitness / len(TARGET_POSITIONS)
    return float(average_fitness)


def evolve(world: Any, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, int]:
    del world, data
    console.log(f"Evolving for {BUDGET} generations with Vision Input")
    
    # 1. Calculate exactly how many inputs the brain needs
    num_joints = model.nq - 7
    input_dim = 3 + num_joints + 3 + 2 

    # 2. Calculate parameter count for a 3-layer MLP matching the baseline PyTorch net
    hidden_size = 16
    layer1_size = (hidden_size * input_dim) + hidden_size
    layer2_size = (hidden_size * hidden_size) + hidden_size
    layer3_size = (model.nu * hidden_size) + model.nu
    total_weights = layer1_size + layer2_size + layer3_size
    
    # 3. Provide the initial random starting point
    initial_weights = np.random.uniform(-0.5, 0.5, size=(total_weights,))
    param = ng.p.Array(init=initial_weights)
    param.set_mutation(sigma=0.075)

    cma_config = ng.optimizers.ParametrizedCMA(popsize=POP_SIZE)
    optimizer = cma_config(
        parametrization=param,
        budget=(BUDGET * POP_SIZE),
        num_workers=POP_SIZE,
    )

    console.log(f"Population size: {POP_SIZE} | Workers: {NUM_WORKERS}")

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(BASE_SEED,),
    ) as executor:
        for bud in range(BUDGET + 1):
            candidates = [optimizer.ask() for _ in range(POP_SIZE)]
            fitnesses = list(executor.map(_evaluate_candidate, [c.value for c in candidates]))

            for candidate, fit in zip(candidates, fitnesses):
                optimizer.tell(candidate, fit)

            gen_best = float(np.min(fitnesses))
            console.rule(f"Budget: {bud}/{BUDGET}")
            console.log(f"Best Fit (Gen): {gen_best:.4f}")

    best_candidate = optimizer.provide_recommendation()
    return best_candidate.value, input_dim

# ============================================================================ #
#                           Main entry function                                #
# ============================================================================ #

def main():
    _seed_everything(BASE_SEED)
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
    # network = Network(
    #     input_size=input_dim, 
    #     output_size=model.nu, 
    #     hidden_size=16
    # )
    # fill_parameters(network, torch.Tensor(best_weights))

    network = FastNumpyNetwork(
        input_size=input_dim,
        hidden_size=16,
        output_size=model.nu,
        weights=best_weights
    )

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
            
            # Use our new optimized vision function!
            v1, v2, v3 = get_vision_inputs(img)
            vision_inputs = [v1, v2, v3]
        else:
            vision_inputs = [0, 0, 0]
            
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
        
        # Call the network normally
        return network.forward(state)

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