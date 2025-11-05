"""Simulation and evaluation utilities."""

import mujoco
import numpy as np
from tqdm import tqdm
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


def spatial_controller(model: mujoco.MjModel, data: mujoco.MjData, 
                      population, tracked_geoms, num_joints):
    """Controller using joint control parameters from genotype."""
    num_joints_per_robot = num_joints
    num_spawned_robots = len(tracked_geoms)
    
    for robot_idx in range(min(num_spawned_robots, len(population))):
        individual = population[robot_idx]
        genotype = individual.genotype
        
        for j in range(num_joints_per_robot):
            ctrl_idx = robot_idx * num_joints_per_robot + j
            if ctrl_idx < model.nu and j * 3 + 2 < len(genotype):
                amplitude = genotype[j * 3]
                frequency = genotype[j * 3 + 1]
                phase = genotype[j * 3 + 2]
                
                control_value = amplitude * np.sin(frequency * data.time + phase)
                data.ctrl[ctrl_idx] = np.clip(
                    control_value, 
                    -1.0,  # config.control_clip_min
                    1.0    # config.control_clip_max
                )


def evaluate_population(population, config):
    """Evaluate fitness (distance traveled) for each individual."""
    print(f"  Evaluating generation (population size: {len(population)})")
    
    # Create isolated environment
    isolated_world = SimpleFlatWorld(config.world_size)
    isolated_robot = gecko()
    isolated_world.spawn(
        isolated_robot.spec, 
        spawn_position=[0, 0, 0.5],
        correct_for_bounding_box=False
    )
    isolated_model = isolated_world.spec.compile()
    
    # Get core geom ID
    core_geom_id = mujoco.mj_name2id(isolated_model, mujoco.mjtObj.mjOBJ_GEOM, "robot-0core")
    
    fitness_values = []
    
    for i, individual in enumerate(tqdm(population, desc="Evaluating robots", unit="robot")):
        isolated_data = mujoco.MjData(isolated_model)
        
        mujoco.mj_forward(isolated_model, isolated_data)
        start_position = isolated_data.geom_xpos[core_geom_id].copy()
        
        def single_robot_controller(model, data):
            genotype = individual.genotype
            for j in range(min(model.nu, len(genotype) // 3)):
                if j * 3 + 2 < len(genotype):
                    amplitude = genotype[j * 3]
                    frequency = genotype[j * 3 + 1]
                    phase = genotype[j * 3 + 2]
                    control_value = amplitude * np.sin(frequency * data.time + phase)
                    data.ctrl[j] = np.clip(
                        control_value,
                        config.control_clip_min,
                        config.control_clip_max
                    )
        
        mujoco.set_mjcb_control(single_robot_controller)
        sim_steps = int(config.simulation_time / isolated_model.opt.timestep)
        for _ in range(sim_steps):
            mujoco.mj_step(isolated_model, isolated_data)
        
        end_position = isolated_data.geom_xpos[core_geom_id].copy()
        distance = np.linalg.norm(end_position - start_position)
        
        individual.fitness = distance
        individual.start_position = start_position
        individual.end_position = end_position
        fitness_values.append(distance)
    
    print(f"  Fitness evaluation complete!")
    return fitness_values
