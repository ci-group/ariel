"""Assignment 3 template code."""

# Standard library
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np

# Local libraries
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)


def show_xpos_history(
    history: list[float],
    spawn_position: list[float],
    target_position: list[float],
    *,
    save: bool = True,
    show: bool = True,
) -> None:
    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena(
        load_precompiled=False,
    )

    # Add some objects to the world
    start_sphere = r"""
    <mujoco>
        <worldbody>
            <geom name="green_sphere"
            size=".1"
            rgba="0 1 0 1"/>
        </worldbody>
    </mujoco>
    """
    end_sphere = r"""
    <mujoco>
        <worldbody>
            <geom name="red_sphere"
            size=".1"
            rgba="1 0 0 1"/>
        </worldbody>
    </mujoco>
    """
    target_box = r"""
    <mujoco>
        <worldbody>
            <geom name="magenta_box"
                size=".1 .1 .1"
                type="box"
                rgba="1 0 1 0.75"/>
        </worldbody>
    </mujoco>
    """
    spawn_box = r"""
    <mujoco>
        <worldbody>
            <geom name="gray_box"
            size=".1 .1 .1"
            type="box"
            rgba="0.5 0.5 0.5 0.5"/>
        </worldbody>
    </mujoco>
    """
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Starting point of robot
    adjustment = np.array((0, 0, target_position[2] + 1))
    world.spawn(
        mj.MjSpec.from_string(start_sphere),
        position=pos_data[0] + (adjustment * 1.5),
        correct_collision_with_floor=False,
    )

    # End point of robot
    world.spawn(
        mj.MjSpec.from_string(end_sphere),
        position=pos_data[-1] + (adjustment * 2),
        correct_collision_with_floor=False,
    )

    # Target position
    world.spawn(
        mj.MjSpec.from_string(target_box),
        position=target_position + adjustment,
        correct_collision_with_floor=False,
    )

    # Spawn position of robot
    world.spawn(
        mj.MjSpec.from_string(spawn_box),
        position=spawn_position,
        correct_collision_with_floor=False,
    )

    # Draw the path of the robot
    smooth = np.linspace(0, 1, len(pos_data))
    inv_smooth = 1 - smooth
    smooth_rise = np.linspace(1.25, 1.95, len(pos_data))
    for i in range(1, len(pos_data)):
        # Get the two points to draw the distance between
        pos_i = pos_data[i]
        pos_j = pos_data[i - 1]

        # Size of the box to represent the distance
        distance = pos_i - pos_j
        minimum_size = 0.05
        geom_size = np.array([
            max(abs(distance[0]) / 2, minimum_size),
            max(abs(distance[1]) / 2, minimum_size),
            max(abs(distance[2]) / 2, minimum_size),
        ])
        geom_size_str: str = f"{geom_size[0]} {geom_size[1]} {geom_size[2]}"

        # Position the box in the middle of the two points
        half_way_point = (pos_i + pos_j) / 2
        geom_pos_str = (
            f"{half_way_point[0]} {half_way_point[1]} {half_way_point[2]}"
        )

        # Smooth color transition from green to red
        geom_rgba = f"{smooth[i]} {inv_smooth[i]} 0 0.75"
        path_box = rf"""
        <mujoco>
            <worldbody>
                <geom name="yellow_sphere"
                    type="box"
                    pos="{geom_pos_str}"
                    size="{geom_size_str}"
                    rgba="{geom_rgba}"
                />
            </worldbody>
        </mujoco>
        """
        world.spawn(
            mj.MjSpec.from_string(path_box),
            position=(adjustment * smooth_rise[i]),
            correct_collision_with_floor=False,
        )

    # Setup the plot
    _, ax = plt.subplots()

    # Add legend to the plot
    plt.rc("legend", fontsize="small")
    red_patch = mpatches.Patch(color="red", label="End Position")
    gray_patch = mpatches.Patch(color="gray", label="Spawn Position")
    green_patch = mpatches.Patch(color="green", label="Start Position")
    magenta_patch = mpatches.Patch(color="magenta", label="Target Position")
    yellow_patch = mpatches.Patch(color="yellow", label="Robot Path")
    ax.legend(
        handles=[
            green_patch,
            red_patch,
            magenta_patch,
            gray_patch,
            yellow_patch,
        ],
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
    )

    # Add labels and title
    ax.set_xlabel("Y Position")
    ax.set_ylabel("X Position")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    # Title
    plt.title("Robot Path in XY Plane")

    # Render the background image
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(
        model,
        data,
        save_path=save_path,
        save=True,
        width=200,
        height=600,
        cam_fovy=8,
        cam_pos=[2.1, 0, 50],
        cam_quat=[-0.7071, 0, 0, 0.7071],
    )

    # Setup background image
    img = plt.imread(save_path)
    ax.imshow(img)

    # Save the figure
    if save:
        fig_path = DATA / "robot_path.png"
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)

    # Show results
    if show:
        plt.show()
