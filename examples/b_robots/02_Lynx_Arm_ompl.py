import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath('.'))

from ompl import util as ou
# Method 1: Set Log Level to suppress INFO and DEBUG
# ou.setLogLevel(ou.LOG_WARN)
ou.noOutputHandler()

from omegaconf import OmegaConf

from ariel.body_phenotypes.Lynx_Arm.lynx.utils.env_sim_ompl import LynxPlanner


def main():
    # Load configuration
    cfg = OmegaConf.load("src/ariel/body_phenotypes/Lynx_Arm/lynx/configs/sim.yaml")

    custom_dict = {
        "genotype_tube": [1, 1, 1, 1, 1],
        "l1_end_point_pos": [0.0, 0.0, 0.4],
        "l2_end_point_pos": [0.0, 0.0, 0.2],
        "l3_end_point_pos": [0.0, 0.0, 0.4],
        "l4_end_point_pos": [0.0, 0.0, 0.35],
        "l5_end_point_pos": [0.0, 0.0, 0.4],
        "l6_end_point_pos": [0.0, 0.0, 0.2],
        }

    cfg.MorphConfig.robot_description_dict.l1_end_point_pos = custom_dict["l1_end_point_pos"] 
    cfg.MorphConfig.robot_description_dict.l2_end_point_pos = custom_dict["l2_end_point_pos"] 
    cfg.MorphConfig.robot_description_dict.l3_end_point_pos = custom_dict["l3_end_point_pos"] 
    cfg.MorphConfig.robot_description_dict.l4_end_point_pos = custom_dict["l4_end_point_pos"] 
    cfg.MorphConfig.robot_description_dict.l5_end_point_pos = custom_dict["l5_end_point_pos"] 
    cfg.MorphConfig.robot_description_dict.l6_end_point_pos = custom_dict["l6_end_point_pos"] 
    omega_conf = OmegaConf.create(cfg)

    # Initialize the OMPL planner
    planner = LynxPlanner(omega_conf)

    # Define a goal position relative to the body position
    # [x, y, z]
    goal = [0.3, 0.2, 0.4] 

    # print(f"Planning path to goal: {goal}")

    # Run the planner with visualization
    # This will print the segment durations we just added
    start = time.time()
    # path, error = planner.run_multiple(goal, start_arr=None, num_points = 2)
    path, error = planner.run(goal, visualize=False)
    # print("Total time for all points: ", time.time() - start)
    # print(f"Final Cartesian error: {error:.4f}m")


    # if path:
    #     print(f"\nSuccess!")
    #     print(f"Number of waypoints: {len(path)}")
    #     print(f"Final Cartesian error: {error:.4f}m")
    # else:
    #     print("\nFailed to find a valid path.")


if __name__ == "__main__":
    main()
