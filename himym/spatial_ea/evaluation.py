"""Fitness evaluation functions for evolutionary algorithm."""

import mujoco
import numpy as np
from spatial_individual import SpatialIndividual
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from hyperneat import CPPN, SubstrateNetwork, create_substrate_for_gecko

def evaluate_population(
    population: list[SpatialIndividual],
    world_size: list[float],
    simulation_time: float,
    control_clip_min: float,
    control_clip_max: float,
    use_directional_fitness: bool = False,
    target_distance_min: float = 5.0,
    target_distance_max: float = 10.0,
    progress_weight: float = 0.5
) -> list[float]:
    """Evaluate individuals in isolation using HyperNEAT controllers."""
    
    # Create isolated environment for reuse
    isolated_world = SimpleFlatWorld(world_size)
    isolated_robot = gecko()
    isolated_world.spawn(
        isolated_robot.spec, 
        spawn_position=[0, 0, 0.5],
        correct_for_bounding_box=False
    )
    isolated_model = isolated_world.spec.compile()
    num_joints = isolated_model.nu
    
    # Get core geom ID
    all_geoms = isolated_world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    core_geom_id = None
    for geom in all_geoms:
        if geom.name == "robot-0core":
            core_geom_id = mujoco.mj_name2id(
                isolated_model, mujoco.mjtObj.mjOBJ_GEOM, "robot-0core"
            )
            break
    
    fitness_values = []
    
    for i, individual in enumerate(population):
        isolated_data = mujoco.MjData(isolated_model)
        
        # Set random robot orientation
        joint_id = mujoco.mj_name2id(isolated_model, mujoco.mjtObj.mjOBJ_JOINT, "robot-0")
        robot_yaw = 0.0
        if joint_id >= 0:
            qpos_addr = isolated_model.jnt_qposadr[joint_id]
            robot_yaw = np.random.uniform(0, 2 * np.pi)
            isolated_data.qpos[qpos_addr + 3] = np.cos(robot_yaw / 2)
            isolated_data.qpos[qpos_addr + 4] = 0.0
            isolated_data.qpos[qpos_addr + 5] = 0.0
            isolated_data.qpos[qpos_addr + 6] = np.sin(robot_yaw / 2)
        
        mujoco.mj_forward(isolated_model, isolated_data)
        start_position = isolated_data.geom_xpos[core_geom_id].copy()
        
        # Generate random target if using directional fitness
        if use_directional_fitness:
            target_distance = np.random.uniform(target_distance_min, target_distance_max)
            relative_angle = np.random.uniform(-np.pi, np.pi)
            target_angle = robot_yaw + relative_angle
            
            target_position = start_position + np.array([
                target_distance * np.cos(target_angle),
                target_distance * np.sin(target_angle),
                0.0
            ])
            individual.target_position = target_position.copy()
        else:
            target_position = None
        
        # Create HyperNEAT controller
        cppn = CPPN(individual.genotype)
        input_coords, hidden_coords, output_coords = create_substrate_for_gecko(
            num_joints=num_joints,
            use_hidden_layer=True,
            hidden_layer_size=num_joints
        )
        substrate = SubstrateNetwork(
            input_coords=input_coords,
            hidden_coords=hidden_coords,
            output_coords=output_coords,
            cppn=cppn,
            weight_threshold=0.2
        )
        
        def single_robot_controller(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            joint_angles = data.qpos[7:7+num_joints].copy()
            
            # CPG oscillator inputs
            osc1 = np.sin(data.time * 1.0)
            osc2 = np.cos(data.time * 1.0)
            osc3 = np.sin(data.time * 2.0)
            osc4 = np.cos(data.time * 2.0)
            cpg_inputs = np.array([osc1, osc2, osc3, osc4])
            bias_input = np.array([1.0])
            
            # Directional input to target
            if use_directional_fitness and target_position is not None:
                current_position = data.geom_xpos[core_geom_id][:2]
                target_vector = target_position[:2] - current_position
                distance = np.linalg.norm(target_vector)
                directional_inputs = target_vector / distance if distance > 0.01 else np.array([0.0, 0.0])
            else:
                directional_inputs = np.array([0.0, 0.0])
            
            sensor_inputs = np.concatenate([joint_angles, cpg_inputs, directional_inputs, bias_input])
            motor_outputs = substrate.activate(sensor_inputs)
            data.ctrl[:] = np.clip(motor_outputs, control_clip_min, control_clip_max)
        
        # Set controller and run simulation
        mujoco.set_mjcb_control(single_robot_controller)
        sim_steps = int(simulation_time / isolated_model.opt.timestep)
        for _ in range(sim_steps):
            mujoco.mj_step(isolated_model, isolated_data)
        
        # Record end position and calculate fitness
        end_position = isolated_data.geom_xpos[core_geom_id].copy()
        
        # Calculate fitness
        if use_directional_fitness and target_position is not None:
            start_to_target = np.linalg.norm(target_position - start_position)
            end_to_target = np.linalg.norm(target_position - end_position)
            progress_toward_target = start_to_target - end_to_target
            total_distance = np.linalg.norm(end_position - start_position)
            
            if total_distance > 0.01:
                direction_quality = progress_toward_target / total_distance
                directional_bonus = progress_weight * direction_quality
                fitness = total_distance * (1.0 + directional_bonus)
            else:
                fitness = 0.0
            
            individual.fitness = fitness
            individual.start_position = start_position
            individual.end_position = end_position
            individual.progress_toward_target = progress_toward_target
            individual.total_distance = total_distance
        else:
            distance = np.linalg.norm(end_position - start_position)
            individual.fitness = distance
            individual.start_position = start_position
            individual.end_position = end_position
            individual.progress_toward_target = 0.0
            individual.total_distance = distance
        
        fitness_values.append(individual.fitness)
        
        if (i + 1) % 5 == 0:
            print(f"    Evaluated {i + 1}/{len(population)} robots")
    
    return fitness_values
