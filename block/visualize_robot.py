"""Visualize the baby robot.

show the baby robot in the interactive viewer,
with a CPG controller
and a tracker to record the position of the charging station.
"""
# Import baby robot
from baby_robot import baby_robot

# Third-party libraries
import mujoco
import numpy as np
from mujoco import viewer

# ARIEL — controller
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import (
    NaCPG,
    create_fully_connected_adjacency,
)

# ARIEL — simulation environments
from ariel.simulation.environments import SimpleFlatWorld, OlympicArena, SimpleTiltedWorld

# ARIEL — utilities
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker

# Always reset the control callback before building a new simulation.
mujoco.set_mjcb_control(None)




# Create the world
world = SimpleFlatWorld()
# Create the robot body
baby_robot = baby_robot()
# Spawn the robot at the origin
world.spawn(baby_robot.spec,position=[0,0,0])
# Charging station
target_pos=[0.5, 0.5, 0]
target_body = world.spec.worldbody.add_body(name="charging_station",
                                                mocap=True,
                                                pos=target_pos)

# Traffic cone
#  base (wide cylinder)
target_body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=[0.2, 0.002, 0],           # radius=0.2, half-height=0.005
    pos=[0, 0, 0],            # x, y, z
    rgba=[1.0, 0.4, 0.0, 1.0],   # orange
)

target_body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=[0.1, 0.3, 0],
    pos=[0, 0, 0.3],
    rgba=[1.0, 0.4, 0.0, 1.0],
)

# White reflective stripe
target_body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=[0.1, 0.008, 0],
    pos=[0, 0, 0.5],
    rgba=[1.0, 1.0, 1.0, 0.9],   # white
)


# Compile into a MuJoCo model and initialization data
model = world.spec.compile()
data = mujoco.MjData(model)

print(f"Number of actuators (joints): {model.nu}")
print(f"Number of dof: {model.nv}")


# Build the CPG — one node per actuator, fully connected.
adj_dict = create_fully_connected_adjacency(model.nu)
cpg = NaCPG(
    adjacency_dict=adj_dict,
    hard_bounds=(-np.pi / 2, np.pi / 2),  # keep angles within hinge limits
)

print(f"CPG nodes (= actuators): {cpg.n}")
print(f"Total learnable parameters: {cpg.num_of_parameters}")

# Set up the tracker — Controller will call tracker.update() automatically.
tracker = Tracker(
    mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM,
    name_to_bind="core",
    observable_attributes=["xpos"],
)
tracker.setup(world.spec, data)

# Wrap the CPG: the callback receives (model, data) and returns joint angles.
def cpg_callback(model: mujoco.MjModel, data: mujoco.MjData):
    return cpg.forward(time=data.time)

# Create the Controller.
controller = Controller(
    controller_callback_function=cpg_callback,
    time_steps_per_ctrl_step=50,   # call CPG every 50 physics steps
    time_steps_per_save=500,        # record tracker data every 500 steps
    alpha=0.5,                      # smoothing: 0 = never update, 1 = immediate update
    tracker=tracker,
)

# Register with MuJoCo — controller.set_control IS the callback.
mujoco.set_mjcb_control(controller.set_control)

print("Controller registered.")

# Uncomment to open the interactive viewer
mujoco.set_mjcb_control(controller.set_control)
viewer.launch(model=model, data=data)