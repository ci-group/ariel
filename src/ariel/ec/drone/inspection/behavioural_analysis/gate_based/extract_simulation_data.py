from stable_baselines3 import PPO
from ariel.ec.drone.inspection.morphological_descriptors.hovering_info import get_sim
from ariel.simulation.tasks.drone_gate_env import DroneGateEnv
from ariel.ec.drone.evaluators.gate_train import backandforth, circle, slalom, figure8
import torch
import numpy as np

def extract_simulation_data(individual, policy_file, gate_cfg, device):
    """
    Extract simulation data for a given individual and policy file.

    Args:
        individual (np.ndarray): The individual configuration.
        policy_file (str): Path to the policy file.
        gate_cfg (str): Gate configuration (e.g., "circle", "slalom").
        device (str): Device to run the simulation on (e.g., "cuda:0" or "cpu").

    Returns:
        dict: A dictionary containing positions, velocities, angular velocities, gate passes, and actions.
    """
    # Unwrap genome objects (e.g. SphericalNeatGenome) to the raw arms array
    if hasattr(individual, "arms"):
        individual = individual.arms
    elif isinstance(individual, np.ndarray) and individual.dtype == object:
        individual = individual.item().arms

    # Define the environment based on the gate configuration
    gate_configs = {
        "backandforth": backandforth,
        "circle": circle,
        "slalom": slalom,
        "figure8": figure8
    }
    if gate_cfg not in gate_configs:
        raise ValueError("Invalid gate configuration")

    gate_config = gate_configs[gate_cfg]

    env = DroneGateEnv(
        num_envs=1,
        individual=individual,
        gates_pos=gate_config.gate_pos,
        gate_yaw=gate_config.gate_yaw,
        start_pos=gate_config.starting_pos,
        x_bounds=gate_config.x_bounds,
        y_bounds=gate_config.y_bounds,
        z_bounds=gate_config.z_bounds,
        initialize_at_random_gates=False,
        gates_ahead=1,
        num_state_history=0,
        num_action_history=0,
        history_step_size=1,
        render_mode=None,
    )
    
    # Load the policy
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64]), log_std_init=0)
    model = PPO("MlpPolicy", env,
                policy_kwargs=policy_kwargs,
                verbose=0,
                n_steps=1000,
                batch_size=1000,
                n_epochs=10,
                gamma=0.999,
                device=device)
    
    try:
        model = PPO.load(policy_file)
    except:
        model = PPO.load(policy_file[:-4])

    # Run the simulation
    env.reset()
    positions, velocities, angular_velocities, gate_passes, actions = [], [], [], [], []

    for _ in range(1200):
        action, _ = model.predict(env.states, deterministic=True)
        states, rewards, dones, infos = env.step(action)

        positions.append(env.world_states[0, 0:3])
        velocities.append(env.world_states[0, 3:6])
        angular_velocities.append(env.world_states[0, 9:12])
        gate_passes.append(infos[0]["gate_passed"])
        # Normalize actions to be between 0 and 1
        a = action[0] * 0.5 + 0.5
        actions.append(a)

    # Convert lists to numpy arrays
    positions = np.array(positions[:-1])
    velocities = np.array(velocities[:-1])
    angular_velocities = np.array(angular_velocities[:-1])
    gate_passes = np.array(gate_passes[:-1])
    actions = np.array(actions[:-1])

    return {
        "positions": positions,
        "velocities": velocities,
        "angular_velocities": angular_velocities,
        "gate_passes": gate_passes,
        "actions": actions
    }
