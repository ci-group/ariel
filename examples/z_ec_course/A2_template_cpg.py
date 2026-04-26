"""A2: Template for creating a custom controller using NaCPG."""

# Standard libraries
from pathlib import Path

# Third-party libraries
import matplotlib.pyplot as plt
import mujoco
import nevergrad as ng
import numpy as np
from mujoco import viewer

# import prebuilt robot phenotypes
from ariel import console
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.controllers import NaCPG
from ariel.simulation.controllers.controller import Controller, Tracker
from ariel.simulation.controllers.na_cpg import create_fully_connected_adjacency
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.runners import simple_runner

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]


def fitness_function(history: list[list[float]]) -> float:
    return history[-1][1]


def show_xpos_history(history: list[list[float]]) -> None:
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Create figure and axis
    plt.figure(figsize=(10, 6))

    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], "b-", label="Path")
    plt.plot(pos_data[0, 0], pos_data[0, 1], "go", label="Start")
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], "ro", label="End")

    # Add labels and title
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Robot Path in XY Plane")
    plt.legend()
    plt.grid(visible=True)

    # Set equal aspect ratio and center at (0,0)
    plt.axis("equal")
    plt.show()


def main() -> None:
    """Entry function to run the simulation with random movements."""
    # Initialise controller to controller to None, always in the beginning.
    mujoco.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SimpleFlatWorld()

    # Initialise robot body
    # YOU MUST USE THE GECKO BODY
    gecko_core = gecko()  # DO NOT CHANGE

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(gecko_core.spec, position=[0, 0, 0])

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Initialise data tracking
    # to_track is automatically updated every time step
    # You do not need to touch it.
    mujoco_type_to_find = mujoco.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    # Setup the NaCPG controller
    adj_dict = create_fully_connected_adjacency(len(data.ctrl.copy()))
    na_cpg_mat = NaCPG(adj_dict, angle_tracking=True)

    # Setup Nevergrad optimizer
    params = ng.p.Instrumentation(
        phase=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(
            -2 * np.pi,
            2 * np.pi,
        ),
        w=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(-2 * np.pi, 2 * np.pi),
        amplitudes=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(
            -2 * np.pi,
            2 * np.pi,
        ),
        ha=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(-10, 10),
        b=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(-100, 100),
    )
    num_of_workers = 50
    budget = 500
    optim = ng.optimizers.PSO
    optimizer = optim(
        parametrization=params,
        budget=budget,
        num_workers=num_of_workers,
    )

    # Simulate the robot
    ctrl = Controller(
        controller_callback_function=lambda _, d: na_cpg_mat.forward(d.time),
        tracker=tracker,
    )

    # Set the control callback function
    # This is called every time step to get the next action.
    # Pass the model and data to the tracker
    ctrl.tracker.setup(world.spec, data)

    # Set the control callback function
    mujoco.set_mjcb_control(
        ctrl.set_control,
    )

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # Run optimization loop
    best_fitness = float("inf")
    best_params = None
    for idx in range(optimizer.budget):
        ctrl.tracker.reset()
        x = optimizer.ask()
        na_cpg_mat.set_param_with_dict(x.kwargs)
        simple_runner(
            model,
            data,
            duration=10,
        )
        loss = fitness_function(tracker.history["xpos"][0])
        optimizer.tell(x, loss)
        if loss < best_fitness:
            best_fitness = loss
            best_params = x.kwargs
            console.log(
                f"({idx}) Current loss: {loss}, Best loss: {best_fitness}",
            )

    # Rerun best parameters
    na_cpg_mat.set_param_with_dict(best_params)
    ctrl.tracker.reset()
    mujoco.mj_resetData(model, data)

    # This opens a viewer window and runs the simulation with your controller
    # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.
    viewer.launch(
        model=model,
        data=data,
    )

    show_xpos_history(tracker.history["xpos"][0])


if __name__ == "__main__":
    main()
