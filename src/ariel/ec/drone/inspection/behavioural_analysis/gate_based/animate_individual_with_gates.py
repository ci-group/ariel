from ariel.visualisation.drone.animation import view, animate
from ariel.simulation.tasks.drone_gate_env import DroneGateEnv
from ariel.ec.drone.evaluators.gate_train import backandforth, circle, slalom, figure8, animate_policy
from ariel.ec.drone.inspection.morphological_descriptors.hovering_info import get_sim

import os
from stable_baselines3 import PPO
import torch
import numpy as np


def animate_individual(
    gate_cfg,
    individual_dir,
    save_dir,
    file_name,
    device=None,
    view_type='top', # Options: 'top', 'iso'
    follow=False,
    draw_forces=False,
    draw_path=False,
    auto_play=True,
    record=False,
    motor_colors=['red', 'blue', 'green', 'orange', 'purple', 'brown']
):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ind_policy_file = individual_dir + "/policy.zip"
    individual_body = individual_dir + "/individual.npy"
    if not os.path.exists(individual_body):
        individual_body = individual_dir + "/genome.npy"
    individual = np.load(individual_body, allow_pickle=True)
    if hasattr(individual, "arms"):
        individual = individual.arms
    elif isinstance(individual, np.ndarray) and individual.dtype == object:
        individual = individual.item().arms

    # Define the environment
    if gate_cfg == "backandforth":
        gate_pos = backandforth.gate_pos
        gate_yaw = backandforth.gate_yaw
        start_pos = backandforth.starting_pos
        x_bounds = backandforth.x_bounds
        y_bounds = backandforth.y_bounds
        z_bounds = backandforth.z_bounds
    elif gate_cfg == "figure8":
        gate_pos = figure8.gate_pos
        gate_yaw = figure8.gate_yaw
        start_pos = figure8.starting_pos
        x_bounds = figure8.x_bounds
        y_bounds = figure8.y_bounds
        z_bounds = figure8.z_bounds
    elif gate_cfg == "circle":
        gate_pos = circle.gate_pos
        gate_yaw = circle.gate_yaw
        start_pos = circle.starting_pos
        x_bounds = circle.x_bounds
        y_bounds = circle.y_bounds
        z_bounds = circle.z_bounds
    elif gate_cfg == "slalom":
        gate_pos = slalom.gate_pos
        gate_yaw = slalom.gate_yaw
        start_pos = slalom.starting_pos
        x_bounds = slalom.x_bounds
        y_bounds = slalom.y_bounds
        z_bounds = slalom.z_bounds
    else:
        raise ValueError("Invalid gate configuration")

    env = DroneGateEnv(
        num_envs=1,
        individual=individual,
        gates_pos=gate_pos,
        gate_yaw=gate_yaw,
        start_pos=start_pos,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        initialize_at_random_gates=False,
        gates_ahead=1,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
    )

    # Load the policy
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64]),
        log_std_init=0,
    )
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        n_steps=1000,
        batch_size=1000,
        n_epochs=10,
        gamma=0.999,
        device=device,
    )
    model = PPO.load(ind_policy_file)

    animate_policy(
        individual,
        model,
        env,
        deterministic=True,
        log_times=False,
        print_vel=False,
        log=None,
        record_steps=1200,
        record_file=save_dir + file_name,
        show_window=False,
        follow=follow,
        draw_forces=draw_forces,
        draw_path=draw_path,
        auto_play=auto_play,
        record=record,
        motor_colors=motor_colors,
    )

# Example usage
if __name__ == "__main__":
    gate_cfg="slalom"
    individual_dir="/home/jed/workspaces/airevolve/data_backup/asym_slalom/asym_slalom4evo_logs_20250320_095329/gen40/ind964/"
    save_dir="./plots/example_comparison/"
    animate_individual(gate_cfg, individual_dir, save_dir, 
                       file_name="top_view.mp4", 
                       device=None, view_type='top', follow=True, 
                       draw_forces=False, draw_path=False, 
                       auto_play=True, record=False)
    
    animate_individual(gate_cfg, individual_dir, save_dir,
                          file_name="iso_view.mp4", 
                          device=None, view_type='iso', follow=True, 
                          draw_forces=False, draw_path=False, 
                          auto_play=True, record=False)