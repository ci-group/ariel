"""EC A3 Template Code (Jack)."""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
from rich.progress import track

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    CoreModule,
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import (
    single_frame_renderer,
    tracking_video_renderer,
    video_renderer,
)
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal[
    "launcher",
    "video",
    "simple",
    "tracking",
    "no_control",
    "frame",
]
type Vector = npt.NDArray[np.float64]

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]
GENOTYPE_SIZE = 64


def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return -cartesian_distance


def show_xpos_history(history: list[float]) -> None:
    # Create a tracking camera
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(
        model,
        data,
        save_path=save_path,
        save=True,
    )

    # Setup background image
    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Calculate initial position
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    # Plot x,y trajectory
    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

    # Add labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()

    # Title
    plt.title("Robot Path in XY Plane")

    # Show results
    plt.savefig(DATA / "robot_path.png")


def create_robot_body(
    genotype: list[np.ndarray] | None = None,
    *,
    save_graph: bool = False,
) -> CoreModule:
    # Create random genotype if None is provided
    if genotype is None:
        type_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        conn_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        rot_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        genotype = [
            type_p_genes,
            conn_p_genes,
            rot_p_genes,
        ]

    # Decode the genotype into probability matrices
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward(genotype)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    # Save the graph to a file
    if save_graph is True:
        save_graph_as_json(
            robot_graph,
            DATA / "robot_graph.json",
        )

    # Print all nodes
    return construct_mjspec_from_graph(robot_graph)


def quick_spawn(
    robot: CoreModule,
) -> tuple[mj.MjModel, mj.MjData, OlympicArena]:
    mj.set_mjcb_control(None)
    world = OlympicArena()
    temp_robot_xml = robot.spec.to_xml()
    temp_robot = mj.MjSpec.from_string(temp_robot_xml)
    world.spawn(
        temp_robot,
        position=SPAWN_POS,
    )
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    return (cast("mj.MjModel", model), data, world)


class NN:
    def __init__(self, robot: CoreModule) -> None:
        _, data, _ = quick_spawn(robot)

        # Get relevant info
        self.input_size = len(data.qpos.copy())
        self.hidden_size = 8
        self.output_size = len(data.ctrl)

        # Clear cache
        del data

    def random_controller(
        self,
    ) -> None:
        # Initialize the networks weights randomly
        # Normally, you would use the genes of an individual as the weights,
        # Here we set them randomly for simplicity.
        w1 = RNG.normal(
            loc=0.0138,
            scale=0.5,
            size=(self.input_size, self.hidden_size),
        )
        w2 = RNG.normal(
            loc=0.0138,
            scale=0.5,
            size=(self.hidden_size, self.hidden_size),
        )
        w3 = RNG.normal(
            loc=0.0138,
            scale=0.5,
            size=(self.hidden_size, self.output_size),
        )
        self.weights = (w1, w2, w3)

    def set_controller_weights(
        self,
        weights: tuple[Vector, Vector, Vector],
    ) -> None:
        self.weights = weights

    def forward(
        self,
        model: mj.MjModel,
        data: mj.MjData,
    ) -> npt.NDArray[np.float64]:
        # Get inputs, in this case the positions of the actuator motors (hinges)
        inputs = data.qpos

        # Run the inputs through the lays of the network.
        layer1 = np.tanh(np.dot(inputs, self.weights[0]))
        layer2 = np.tanh(np.dot(layer1, self.weights[1]))
        outputs = np.tanh(np.dot(layer2, self.weights[2]))

        # Scale the outputs
        return outputs * np.pi


def run(
    model: mj.MjModel,
    data: mj.MjData,
    duration: int = 15,
    mode: ViewerTypes = "viewer",
) -> None:
    match mode:
        case "simple":
            # This disables visualisation (fastest option)
            simple_runner(
                model,
                data,
                duration=duration,
            )
        case "frame":
            # Render a single frame (for debugging)
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            # This records a video of the simulation
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
            )
        case "tracking":
            # This records a video of the simulation
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            tracking_video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
            )
        case "launcher":
            # This opens a liver viewer of the simulation
            viewer.launch(
                model=model,
                data=data,
            )
        case "no_control":
            # If mj.set_mjcb_control(None), you can control the limbs manually.
            mj.set_mjcb_control(None)
            viewer.launch(
                model=model,
                data=data,
            )


def evaluate(
    robot: CoreModule,
    nn: NN,
    *,
    plot_and_record: bool = False,
) -> float:
    # Define what to track
    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    # Create the controller
    controller = Controller(
        controller_callback_function=nn.forward,
        tracker=tracker,
    )

    # Create the robot in the world
    model, data, world = quick_spawn(robot)

    # Pass the model and data to the tracker
    controller.tracker.setup(world.spec, data)

    # Set the control callback function
    mj.set_mjcb_control(controller.set_control)

    # Choose the mode
    duration = 30
    if plot_and_record is True:
        # Run the simulation
        run(
            model,
            data,
            duration=duration,
            mode="video",
        )

        # Show the tracked history
        show_xpos_history(tracker.history["xpos"][0])
    else:
        # Run the simulation
        run(
            model,
            data,
            duration=duration,
            mode="simple",
        )

    # Calculate and print the fitness
    return fitness_function(tracker.history["xpos"][0])


def main() -> None:
    """Entry point."""
    best_fitness = -np.inf
    for _ in track(range(200)):
        # Create a robot
        robot = create_robot_body()
        nn = NN(robot)
        nn.random_controller()
        fitness = evaluate(robot, nn, plot_and_record=False)
        if fitness > best_fitness:
            fitness_other = evaluate(robot, nn, plot_and_record=True)
            console.log(
                "\n",
                f"Best fitness so far: {best_fitness} vs {fitness_other}",
                f"Are they the same? Δ {(fitness - best_fitness):.5f}",
                "\n",
            )
            best_fitness = fitness
    console.log(f"Best fitness achieved: {best_fitness}")


if __name__ == "__main__":
    main()
