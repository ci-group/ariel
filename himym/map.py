import mujoco
import numpy as np
import matplotlib.pyplot as plt

from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


def main() -> None:
    # Create the world (1m x 1m floor, 0.1m thick)
    world = SimpleFlatWorld([5, 5, 0.1])

    # Create a single brick
    # brick = BrickModule(index=0)
    # core = CoreModule(index=0)
    gecko_robot1 = gecko()
    gecko_robot2 = gecko()
    gecko_robot3 = gecko()

    # Spawn it slightly above the floor so it falls naturally
    # world.spawn(brick.spec, spawn_position=[-0.05, 0, 0.05])
    world.spawn(gecko_robot1.spec, spawn_position=[0.1, 0, 0.05], prefix_id=0)
    world.spawn(gecko_robot2.spec, spawn_position=[0.2+(0.05*16), 0, 0.05], prefix_id=1)
    world.spawn(gecko_robot3.spec, spawn_position=[0.2+(0.05*16), 5, 0.05], prefix_id=2)

    # Compile and create simulation data
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Forward simulate once (so positions are valid)
    mujoco.mj_forward(model, data)

    # Get positions of all bodies
    xpos = data.xpos  # shape (nbodies, 3)

    # Make a simple 2D plot (top-down, x vs y)
    plt.figure(figsize=(6, 6))
    plt.scatter(xpos[:, 0], xpos[:, 1], c="blue", s=30, label="Bodies")

    # Annotate with body names
    # for i, name in enumerate(model.body_names):
    #     plt.text(xpos[i, 0], xpos[i, 1], name, fontsize=6)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Top-down map of MuJoCo world")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
