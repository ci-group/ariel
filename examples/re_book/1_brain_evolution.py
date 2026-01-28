# Standard libraries
import random
from typing import Literal, cast, List, Optional, Any
from pathlib import Path
import time
import os
import cv2 

# Pretty little errors and progress bars
from rich.console import Console
from rich.traceback import install

# Initialize rich console and traceback handler
install()
console = Console()

# Third-party libraries
import numpy as np
import mujoco
import keyboard as kb

# Network imports
import torch
from torch import nn
from torch.nn import Tanh

# Learner
from evotorch.algorithms import CMAES
from evotorch.neuroevolution import NEProblem

# Local libraries
from ariel.simulation.environments import SimpleFlatWorld
from new_bodies import body_spider45
from ariel.utils.tracker import Tracker
from ariel.utils.renderers import VideoRecorder, video_renderer

# Set up command line argument parsing
# If none given, default values are used.
import argparse
parser = argparse.ArgumentParser(description='Evolution simulation with configurable budget')
parser.add_argument('--budget', type=int, default=600, help='Number of generations for learning')
parser.add_argument('--dur', type=int, default=15, help="Duration of an evaluation")
parser.add_argument('--population', type=int, default=26, help="Population size")
args = parser.parse_args()

BUDGET = args.budget
DURATION = args.dur
POP_SIZE = args.population

# 1. Defined 3 target positions to prevent overfitting
TARGET_POSITIONS = [ 
    [-0.5 , -2, 0.1],  # Left
    [0.0, -2, 0.1],    # Center
    [0.5, -2, 0.1]     # Right
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
        self.tanh = Tanh()  

        self.input = input_size

        # Disable gradients for all parameters
        for param in self.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def forward(self,model, data, state):
        x = torch.Tensor(state)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc4(x)) * (torch.pi / 2)
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
    _, width = green_mask.shape
    section_width = width // 3
    
    # Split into three sections
    left_section = green_mask[:, :section_width]
    middle_section = green_mask[:, section_width:2*section_width]
    right_section = green_mask[:, 2*section_width:]
    
    # Calculate percentage of green pixels in each section
    def get_green_percentage(section):
        total_pixels = section.size
        if total_pixels == 0: return 0.0
        green_pixels = np.count_nonzero(section)
        return (green_pixels / total_pixels) 
    
    left_percent = get_green_percentage(left_section)
    middle_percent = get_green_percentage(middle_section)
    right_percent = get_green_percentage(right_section)
    
    return [left_percent, middle_percent, right_percent]

# ============================================================================ #
#                           Define observations                                #
# ============================================================================ #

def get_robot_state(data):
    """
    Extracts the robot state EXCLUDING global position.
    Processes quaternion to be consistent (scaled by sign of w).
    """
    # 1. Get Quaternion (w, x, y, z) - Index 3 to 7
    quat = data.qpos[3:7].copy()
    
    # 2. Scale/Normalize Quaternion
    # If w is negative, negate the whole quaternion.
    if quat[0] < 0:
        quat = -quat
        
    # 3. Use only the Imaginary parts (x, y, z)
    quat_imag = quat[1:] 
    
    # 4. Get Hinge Joints (Index 7 onwards)
    joints = data.qpos[7:]
    
    return np.concatenate([quat_imag, joints])


# ============================================================================ #
#                  Custom simulation runner with camera                        #
# ============================================================================ #

def run_vision_simulation(model, 
                          data, 
                          network:Network, 
                          duration:int, 
                          renderer=None, 
                          cam_name=None,
                          control_step_freq=50 
                          ):
    """Custom runner that processes vision."""
    
    # Setup Renderer if not passed (creates a new context)
    if renderer is None:
        renderer = mujoco.Renderer(model, height=48, width=64) 
    
    timestep = model.opt.timestep
    
    # Initialize control placeholder
    current_action = np.zeros(model.nu)
    
    while data.time < duration:
        
        # Calculate deduced step count (Optimization from controller.py)
        deduced_step = int(np.ceil(data.time / timestep))
        
        # --- CONTROL STEP ---
        # Only run expensive vision and network pass every N steps
        if deduced_step % control_step_freq == 0:
            

            renderer.update_scene(data, camera=cam_name)
            img = renderer.render()
            
            # 2. Process Vision
            mask = isolate_green(img)
            vision_inputs = analyze_sections(mask)

            # 3. Prepare Inputs
            robot_state = get_robot_state(data)
            
            state_input = np.concatenate([
                robot_state,
                vision_inputs
            ]).astype(np.float32)


            console.log(len(state_input))
            exit(1)

            # 4. Network Forward Pass
            current_action = network.forward(model, data, state_input)
        
        # 5. Apply Control (Hold previous action if not a control step)
        data.ctrl[:] = current_action
        
        
        # 6. Step Physics
        mujoco.mj_step(model, data)
         

# ============================================================================ #
#                         Define evolutionary loop                             #
# ============================================================================ #
  
def evolve(world, model, data) -> List[float]:
    """Evolve the robot's movement using an evolutionary algorithm."""

    tracker = Tracker(mujoco_obj_to_find=data, observable_attributes=["xpos"])
    tracker.setup(world.spec, data)

    # Identify Camera for ROBOT VISION (on the spider)
    robot_cam_name = None
    for i in range(model.ncam):
        name = model.camera(i).name
        if "camera" in name or "core" in name:
            robot_cam_name = name
            break
    if robot_cam_name is None and model.ncam > 0:
        robot_cam_name = model.camera(0).name

    # Pre-initialize renderer for the evolution loop
    renderer = mujoco.Renderer(model, height=48, width=64)

    # Get Mocap ID for the green target
    try:
        target_mocap_id = model.body("green_target").mocapid[0]
    except:
        console.print("[red]Error: Green target mocap body not found![/red]")
        target_mocap_id = 0

    # Define the fitness function
    def fitness_function(x: Network) -> float:
        total_fitness = 0.0
        
        # 2. Evaluate on ALL 3 target positions
        for target_pos in TARGET_POSITIONS:
            mujoco.mj_resetData(model, data)
            tracker.reset()
            
            # Move the Green Target Object
            # Multiple target positions enhance robustness
            data.mocap_pos[target_mocap_id] = target_pos

            # Run Simulation
            run_vision_simulation(
                model, 
                data, 
                network=x, 
                duration=DURATION, 
                renderer=renderer,
                cam_name=robot_cam_name,
                control_step_freq=50 
            )
            
            # Calculate Fitness (Distance to CURRENT target)
            xc, yc, zc = data.qpos[0:3] 
            xt, yt, zt = target_pos

            dist = np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)
            total_fitness += dist
            
        # Return average fitness across the 3 scenarios
        return total_fitness / len(TARGET_POSITIONS)

    console.log(f"Evolving for {BUDGET} generations with Vision Input")
    
    # --- CALCULATE NEW INPUT SIZE ---
    num_joints = len(data.qpos) - 7

    # Inputs: Quat(3) + Joints(N) + Vision(3)
    # In case of the spider: 3 + 8 + 3
    input_dim = 3 + num_joints + 3

    # Initialise Neural Network Controller
    # 14 input neurons
    # 32 hidden layer neurons
    # 8 output neurons
    network = Network(
        input_size=input_dim, 
        output_size=model.nu, 
        hidden_size=32
    )
    
    # Initialise Problem for the solver/learner
    problem = NEProblem(
            objective_sense="min",
            network_eval_func=fitness_function,
            network=network.eval(),
            initial_bounds=(-0.1, 0.1),
            device="cpu"
    )
    
    # Initialise CMA-ES learner 
    searcher = CMAES(problem=problem,
                     stdev_init=0.1
                     )
    
    console.log(f"Population size: {searcher.popsize}")
    
    for bud in range(BUDGET + 1):
        searcher.step()
        gen_best = searcher.status["pop_best_eval"]
        
        console.rule(f"Budget: {bud}/{BUDGET}")
        console.log(f"Best Fit (Avg): {gen_best:.4f}")

        if kb.is_pressed('q'):
            break

    best_ind = searcher.status["best"].values
    return best_ind, input_dim


# ============================================================================ #
#                           Main entry function                                #
# ============================================================================ #

def main():
    mujoco.set_mjcb_control(None)

    # Initialise world
    world = SimpleFlatWorld()
    
    # Add Green Target Object
    target_body = world.spec.worldbody.add_body(name="green_target", mocap=True, pos=[0, -2, 0.1])
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
    end = time.time()

    console.log(f"Evolution took {(end-start)/60:.2f} minutes")

    weights_path = "3_spider_vision_new.npy"
    if os.path.exists(weights_path):
        console.log(f"Weights file found at {weights_path}. Loading weights...")
        best_weights = np.load(weights_path)
    else:
        np.save(weights_path, best_weights)
        console.log(f"Best weights saved to {weights_path}")

# ============================================================================ #
#                           Initialise world and                               #
#                           load best  performer                               #
#                           for  video recording                               #
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
        if "camera" in model.camera(i).name and "video" not in model.camera(i).name:
            robot_cam_name = model.camera(i).name
            break
            
    console.log("Rendering Best Video...")
    path_to_video_folder = str(DATA / "videos")
    
    mujoco.mj_resetData(model, data)
    
    # Set target to middle position for the video demo
    target_mocap_id = model.body("green_target").mocapid[0]
    data.mocap_pos[target_mocap_id] = [0.0, -2, 0.1]
    
    # 1. Renderer for Robot Vision (Low Res)
    control_renderer = mujoco.Renderer(model, height=48, width=64)
    
    # 2. Renderer for Video Output (High Res)
    video_capture_renderer = mujoco.Renderer(model, height=480, width=640)
    
    def get_vision_control_signal(m, d):
        if robot_cam_name:
            control_renderer.update_scene(d, camera=robot_cam_name)
            img = control_renderer.render()
            mask = isolate_green(img)
            vision_inputs = analyze_sections(mask)
        else:
            vision_inputs = [0,0,0]
            
        robot_state = get_robot_state(d)
        
        state = np.concatenate([
            robot_state,
            vision_inputs
        ]).astype(np.float32)
        
        return network.forward(m, d, state)

    # Video Timing Variables
    fps = 30
    video_dt = 1.0 / fps
    next_video_time = 0.0
    frames = []

    # Simulation Timing Variables (must match evolve/run_vision_simulation exactly)
    timestep = model.opt.timestep
    control_step_freq = 50 
    
    current_ctrl = np.zeros(model.nu)

# --- REPLAY BEST & RECORD VIDEO ---
    console.log("Rendering Best Video...")
    path_to_video_folder = str(DATA / "videos")
    
    # 1. Setup VideoRecorder (using your Ariel library class)
    video_recorder = VideoRecorder(
        file_name="spider_vision_best", 
        output_folder=path_to_video_folder
    )

    # 2. Reset Simulation & Target
    mujoco.mj_resetData(model, data)
    target_mocap_id = model.body("green_target").mocapid[0]
    data.mocap_pos[target_mocap_id] = [0.0, -2, 0.1]

    # 3. Setup Visualization Options (from your snippet)
    viz_options = mujoco.MjvOption()
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = False

    # 4. Get Camera ID ("video_cam" is the one we created earlier)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "video_cam")

    # 5. Timing Variables
    fps = 30
    dt = model.opt.timestep
    steps_per_frame = int(1.0 / (fps * dt))
    control_step_freq = 50
    current_ctrl = np.zeros(model.nu)

    # 6. Setup separate renderer for the Robot's Vision (Low Res)
    # We keep this outside the video loop so we don't recreate it every frame
    control_renderer = mujoco.Renderer(model, height=48, width=64)

    # 7. Main Rendering Loop (Using Context Manager as requested)
    # We use the video_recorder width/height for the output video
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        
        while data.time < DURATION:
            
            # INNER LOOP: Step physics N times to match Video FPS
            # We must loop manually here to inject the Control Logic
            for _ in range(steps_per_frame):
                
                # A. Calculate deduced step (Exact same timing as training)
                deduced_step = int(np.ceil(data.time / dt))

                # B. Run Network if needed
                if deduced_step % control_step_freq == 0:
                    current_ctrl = get_vision_control_signal(model, data)

                # C. Apply Control & Step
                data.ctrl[:] = current_ctrl
                mujoco.mj_step(model, data)

            # OUTER LOOP: Render Frame (Once per 1/30th second)
            renderer.update_scene(
                data, 
                scene_option=viz_options, 
                camera=camera_id
            )
            
            # Use the VideoRecorder's write method (handles cv2/saving internally)
            video_recorder.write(frame=renderer.render())

    # 8. Finish
    video_recorder.release()
    console.log(f"[green]Video rendering complete. Saved to {path_to_video_folder}[/green]")
