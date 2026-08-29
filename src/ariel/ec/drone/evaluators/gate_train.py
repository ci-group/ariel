# library imports
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
from stable_baselines3 import PPO
from datetime import datetime
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

# Suppress the render_mode warning from stable_baselines3
warnings.filterwarnings("ignore", message="The `render_mode` attribute is not defined in your environment")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ariel.simulation.tasks.drone_gate_env import DroneGateEnv
from ariel.visualisation.drone.animation import view as animation_view
from ariel.ec.drone.inspection.morphological_descriptors.hovering_info import get_sim
from ariel.ec.drone.inspection.drone_visualizer import DroneVisualizer 

import argparse

# Track configurations.
# Conventions match optimal_quad_control_RL: NED frame (z+ down), gates set
# at GATE_ALT m above the ground plane (z=-1.5). Vertical bounds give the
# policy enough exploration room: ground at z=0, ceiling at z=-7. Horizontal
# bounds are widened to ±5 m where sensible. The earlier z=0 / z_bounds=[-1,1]
# layout was too tight for any drone with TWR > ~3 to learn from.
GATE_ALT = -1.5

class backandforth():
    gate_pos = np.array([
        [  2.0,  0.0,  GATE_ALT],
        [  8.0,  0.0,  GATE_ALT],
        [  8.0,  0.0,  GATE_ALT],
        [  2.0,  0.0,  GATE_ALT],
    ], dtype=np.float32)
    gate_yaw = np.array([0,0,2,2], dtype=np.float32) * np.pi / 2
    x_bounds = np.array([-2, 12], dtype=np.float32)
    y_bounds = np.array([-5, 5], dtype=np.float32)
    z_bounds = np.array([-7, 0], dtype=np.float32)
    starting_pos = np.array([0.0, 0.0, GATE_ALT])

class figure8():
    gate_pos = np.array([
        [  1.5, -1.5,  GATE_ALT],
        [  3.0,  0.0,  GATE_ALT],
        [  1.5,  1.5,  GATE_ALT],
        [  0.0,  0.0,  GATE_ALT],
        [ -1.5, -1.5,  GATE_ALT],
        [ -3.0,  0.0,  GATE_ALT],
        [ -1.5,  1.5,  GATE_ALT],
        [  0.0,  0.0,  GATE_ALT],
    ], dtype=np.float32)
    gate_yaw = np.array([0,-1,0,1,2,-1,2,1], dtype=np.float32) * np.pi / 2
    x_bounds = np.array([-5, 5], dtype=np.float32)
    y_bounds = np.array([-5, 5], dtype=np.float32)
    z_bounds = np.array([-7, 0], dtype=np.float32)
    starting_pos = np.array([0.0, -1.5, GATE_ALT])

class circle():
    gate_pos = np.array([
        [  0.0, -1.5,  GATE_ALT],
        [  1.5,  0.0,  GATE_ALT],
        [  0.0,  1.5,  GATE_ALT],
        [ -1.5,  0.0,  GATE_ALT]
    ], dtype=np.float32)
    gate_yaw = np.array([0,1,2,3], dtype=np.float32) * np.pi / 2
    x_bounds = np.array([-5, 5], dtype=np.float32)
    y_bounds = np.array([-5, 5], dtype=np.float32)
    z_bounds = np.array([-7, 0], dtype=np.float32)
    starting_pos = np.array([-1.5, -1.5, GATE_ALT])

class slalom():
    gate_pos = np.array(
        [[x, (i % 2) * (1 if i % 4 == 1 else -1), GATE_ALT] for i, x in enumerate(range(0, 82, 2))],
        dtype=np.float32,
    )
    ng = len(gate_pos)
    gate_yaw = np.tile([1, 0, -1, 0], ng) * np.pi / 2
    x_bounds = np.array([-2, 82+1], dtype=np.float32)
    y_bounds = np.array([-5, 5], dtype=np.float32)
    z_bounds = np.array([-7, 0], dtype=np.float32)
    starting_pos = np.array([0, -1, GATE_ALT])
    
# ANIMATION FUNCTION
def animate_policy(individual, model, env, deterministic=False, log_times=False, print_vel=False, log=None, view_type="top",
                    motor_colors=['red', 'blue', 'green', 'orange', 'purple', 'brown'], **kwargs):
    env.reset()
    
    # Convert individual to propellers configuration
    propellers, _ = env._convert_individual_to_propellers(individual)
    
    def get_drone_state():
        actions, _ = model.predict(env.states, deterministic=deterministic)

        states, rewards, dones, infos = env.step(actions)
        if log != None:
            log(states)
        if print_vel:
            # compute mean velocity
            vels = env.world_states[:,3:6]
            mean_vel = np.linalg.norm(vels, axis=1).mean()
            print(mean_vel)
        if log_times:
            if rewards[0] == 10:
                print(env.step_counts[0]*env.dt)
        
        # Return drone state in the format expected by the view function
        world_state = env.world_states[0]  # Get first environment
        num_motors = len(propellers)
        drone_state = {
            'x': world_state[0],
            'y': world_state[1], 
            'z': world_state[2],
            'phi': world_state[6],
            'theta': world_state[7],
            'psi': world_state[8]
        }
        
        # Add motor thrust values
        for i in range(num_motors):
            if i < env.prev_actions.shape[1]:
                drone_state[f'u{i+1}'] = env.prev_actions[0][i]
            else:
                drone_state[f'u{i+1}'] = 0
        
        return drone_state
    
    animation_view(propellers, get_drone_state, gate_pos=env.gate_pos, gate_yaw=env.gate_yaw, view_type=view_type, motor_colors=motor_colors, **kwargs)

class FullStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.tags = [
            'rollout/ep_len_mean', 'rollout/ep_rew_mean', 'time/fps',
            'train/approx_kl', 'train/clip_fraction', 'train/clip_range',
            'train/entropy_loss', 'train/explained_variance', 'train/learning_rate',
            'train/loss', 'train/policy_gradient_loss', 'train/std', 'train/value_loss'
        ]

    def _on_step(self) -> bool:
        for tag in self.tags:
            if tag in self.logger.name_to_value:
                self.logger.record(f"monitor/{tag}", self.logger.name_to_value[tag])
        return True

    def _on_rollout_end(self) -> None:
        # for tag in self.tags:
        #     if tag in self.logger.name_to_value:
        #         self.logger.record(f"monitor/{tag}", self.logger.name_to_value[tag])
        # Force flush for debugging; can remove later
        self.logger.dump(self.num_timesteps)

def train(individual, gate_cfg, total_timesteps=int(1E8), save_dir="./logs", num_envs=100, device="cuda:0", num=None, max_steps=1200):

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

    # SETUP LOGGING
    save_dir = save_dir+"/"
    # models_dir = save_dir+'/models/'
    # log_dir = save_dir+'/logs/'
    # video_dir = save_dir+'/videos/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # if not os.path.exists(models_dir):
    #     os.makedirs(models_dir)
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)    

    env = DroneGateEnv(
        num_envs=num_envs,
        individual=individual,
        gates_pos=gate_pos,
        gate_yaw=gate_yaw,
        start_pos=start_pos,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        gates_ahead=1,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
        device=device,
        max_steps=max_steps,
    )
    test_env = DroneGateEnv(
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
        device=device,
        max_steps=max_steps,
    )

    # Wrap the environment in a Monitor wrapper
    # Make monitor1 folder
    if num is not None:
        monitor_file = save_dir+f"m{num}"
    else: 
        monitor_file = save_dir
    
    env = VecMonitor(env, filename=monitor_file)
    # custom_logger = configure(save_dir, ["stdout", "csv", "tensorboard"])

    # MODEL DEFINITION (matches optimal_quad_control_RL/train.py:149-161).
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        log_std_init=0.0,
    )
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=save_dir,
        n_steps=1000,
        batch_size=5000,
        n_epochs=10,
        gamma=0.999,
        device=device,
    )
    # model.set_logger(custom_logger)

    # TRAINING
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, log_interval=100, callback=FullStatsCallback())
    if num is None:
        model.save(save_dir + '/' + "policy")
    else:
        model.save(save_dir + '/' + f"policy{num}")
    # model_path = save_dir + "/"+"best_model.zip"
    # PPO_model = PPO.load(model_path)

    # Plotting the training curve
    try:
        data = pd.read_csv(monitor_file+".monitor.csv", skiprows=1)  # Skip the first row (comments)
    except:
        data = pd.read_csv(monitor_file+"monitor.csv", skiprows=1)
    episode_rewards = data["r"]  # Rewards per episode
    time_steps = data["t"]  # Timesteps at each episode
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, episode_rewards, label="Episode Reward")
    # plt.fill_between(time_steps[:,0], episode_rewards_mean - episode_rewards_std, episode_rewards_mean + episode_rewards_std, alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    plt.legend()
    if num is None:
        plt.savefig(save_dir+"/figure.png")
    else:
        plt.savefig(save_dir+f"/figure{num}.png")
    plt.close()
    # plt.show()
    # TESTING
    test_env.reset()
    # if num is None:
    #     animate_policy(individual, model, test_env, deterministic=False, log_times=False, print_vel=False, log=None, 
    #                 record_steps=1200, record_file=save_dir + f'v.mp4',
    #                 show_window=False)
    # else:
    #     animate_policy(individual, model, test_env, deterministic=False, log_times=False, print_vel=False, log=None, 
    #                 record_steps=1200, record_file=save_dir + f'v{num}.mp4',
    #                 show_window=False)

    test_env.reset()
    # do 1200 steps and print state and action
    for i in range(1000):
        num = test_env.num_state_history+1
        state_len = int(len(test_env.states[0])/num)
        actions, _ = model.predict(test_env.states, deterministic=True)
        states, rewards, dones, infos = test_env.step(actions)
    
    return infos[0]["num_gates_passed"][0]

def evaluate_individual(individual, ind_save_dir, training_ts, num_envs, gate_cfg, device="cuda:0", num=None, max_steps=1200) -> list:
    start_time = time.time()
    sim = get_sim(individual)
    sim.compute_hover(verbose=False)
    if sim.static_success == False:
        spinning_success = sim.spinning_success
    else:  
        spinning_success = False

    success = sim.static_success# or spinning_success
    if not success:
        try:
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(111, projection='3d')
            visualizer = DroneVisualizer()
            visualizer.plot_3d(individual, ax=ax, title=f"Failed Individual (Gen {num})", fitness=0, generation=num)
            if num is not None:
                plt.savefig(ind_save_dir + f"/morphology{num}.png")
            else:
                plt.savefig(ind_save_dir + "/morphology.png")
            plt.close()
        except:
            print(f"Failed to plot:\n {individual}")
        return 0
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    visualizer = DroneVisualizer()
    visualizer.plot_3d(individual, ax=ax, title=f"Pre-training (Gen {num})", fitness=np.nan, generation=num)
    if num is not None:
        plt.savefig(ind_save_dir + f"/morphology{num}.png")
    else:
        plt.savefig(ind_save_dir + "/morphology.png")
    plt.close()

    num_gates_passed = train(individual, gate_cfg, total_timesteps=int(float(training_ts)), save_dir=ind_save_dir, num_envs=int(num_envs), device=device, num=num, max_steps=max_steps)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    visualizer = DroneVisualizer()
    visualizer.plot_3d(individual, ax=ax, title=f"Post-training (Gen {num})", fitness=num_gates_passed, generation=num)
    if num is not None:
        plt.savefig(ind_save_dir + f"/morphology{num}.png")
    else:
        plt.savefig(ind_save_dir + "/morphology.png")
    plt.close()

    end_time = time.time()
    # print(f"{end_time-start_time} seconds to evaluate, fitness={num_gates_passed}")
    return num_gates_passed

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--training_timesteps', default=1E8) 
    parser.add_argument('--num_envs', default=100)
    parser.add_argument('--gate_cfg', default='figure8')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num', default=None)
    args = parser.parse_args()

    # Load Bf and Bm from directory
    individual = np.load(args.filename + "/individual.npy", allow_pickle=True)
    individual = individual.astype(np.float32)

    if args.num is None:
        num = None
    else:
        num = int(args.num)
    num_gates_passed = evaluate_individual(individual, args.filename, args.training_timesteps, args.num_envs, args.gate_cfg, args.device, num=num)

    print(num_gates_passed)
