import mujoco
import numpy as np
import matplotlib.pyplot as plt

from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


def main() -> None:
    # Create the world (1m x 1m floor, 0.1m thick)
    world = SimpleFlatWorld([1, 1, 0.1])

    # Create gecko robots
    gecko_robot1 = gecko()
    gecko_robot2 = gecko()
    gecko_robot3 = gecko()

    # Spawn them at different positions
    world.spawn(gecko_robot1.spec, spawn_position=[0.1, 0, 0.05], prefix_id=0)
    world.spawn(gecko_robot2.spec, spawn_position=[0.2 + (0.05 * 16), 0, 0.05], prefix_id=1)
    # world.spawn(gecko_robot3.spec, spawn_position=[0.2 + (0.05 * 16), 5, 0.05], prefix_id=2)

    # Compile and create simulation data
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Forward simulate once (so positions are valid)
    mujoco.mj_forward(model, data)

    # Get positions of all bodies
    xpos = data.xpos  # shape (nbodies, 3)

    # Make a simple 2D plot (top-down, x vs y)
    plt.figure(figsize=(8, 8))
    plt.scatter(xpos[:, 0], xpos[:, 1], c="blue", s=30, label="Bodies")

    # Add grid lines
    grid_step = 0.025  # (5+5)/0.05*2 => 200 divisions => spacing = 0.025
    x_ticks = np.arange(-5, 5 + grid_step, grid_step)
    y_ticks = np.arange(-5, 5 + grid_step, grid_step)

    # Set ticks for readability (every 0.5m shown)
    plt.xticks(np.arange(-5, 5.5, 0.5))
    plt.yticks(np.arange(-5, 5.5, 0.5))

    # Draw grid
    plt.grid(True, which="both", color="gray", linewidth=0.5, alpha=0.5)

    # Force square aspect ratio and limits
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Top-down map of MuJoCo world with grid")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
