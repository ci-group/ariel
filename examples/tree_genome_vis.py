"""Assignment 3 template code."""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec.genotypes.tree.tree_genome import TreeGenome, TreeNode
from ariel.body_phenotypes.robogen_lite import config
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.morphological_descriptor import MorphologicalMeasures
from ariel.utils.graph_ops import robot_json_to_digraph, load_robot_json_file


# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]


def fitness_function(history: list[float]) -> float:
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
        camera=camera,
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
    plt.show()

def create_custom_genome() -> TreeGenome:
    """Create your custom tree genome here.

    Modify this function to define the robot structure you want to visualize.
    You can use the TreeGenerator methods or manually build the tree.
    """
    # Option 1: Use predefined generators
    # return TreeGenerator.star_shape(num_arms=4)
    # return TreeGenerator.binary_tree(depth=3)
    # return TreeGenerator.random_tree(max_depth=3, branching_prob=0.6)

    return TreeGenerator.random_tree(max_depth=10)

    #return TreeGenome.default_init()
    # Option 2: Build manually
    genome = TreeGenome.default_init()  # Starts with CORE module

    # Add a brick to the front
    front_brick = TreeNode(
        config.ModuleInstance(
            type=config.ModuleType.BRICK,
            rotation=config.ModuleRotationsIdx.DEG_0,
            links={}
        )
    )
    genome.root.front = front_brick

    # Add a hinge to the right of the core
    right_hinge = TreeNode(
        config.ModuleInstance(
            type=config.ModuleType.HINGE,
            rotation=config.ModuleRotationsIdx.DEG_90,
            links={}
        )
    )
    genome.root.right = right_hinge

    # Add another brick to the front of the right hinge
    hinge_front_brick = TreeNode(
        config.ModuleInstance(
            type=config.ModuleType.BRICK,
            rotation=config.ModuleRotationsIdx.DEG_0,
            links={}
        )
    )
    right_hinge.front = hinge_front_brick

    return genome

def create_multi_limb_robot():
    # Build a complex robot manually
    genome = TreeGenome.default_init()  # Core

    # Add main branches from core
    for face in [config.ModuleFaces.FRONT, config.ModuleFaces.BACK,
                 config.ModuleFaces.LEFT, config.ModuleFaces.RIGHT]:
        # Add brick to each main direction
        main_brick = TreeNode(config.ModuleInstance(
            type=config.ModuleType.BRICK,
            rotation=config.ModuleRotationsIdx.DEG_0,
            links={}
        ))
        setattr(genome.root, face.name.lower(), main_brick)

        # Add sub-branches from each main brick
        for sub_face in [config.ModuleFaces.FRONT, config.ModuleFaces.LEFT, config.ModuleFaces.RIGHT]:
            if sub_face in config.ALLOWED_FACES[config.ModuleType.BRICK]:
                # Add hinge for articulation
                hinge = TreeNode(config.ModuleInstance(
                    type=config.ModuleType.HINGE,
                    rotation=config.ModuleRotationsIdx.DEG_0,
                    links={}
                ))
                try:
                    setattr(main_brick, sub_face.name.lower(), hinge)

                    # Add end effector brick
                    end_brick = TreeNode(config.ModuleInstance(
                        type=config.ModuleType.BRICK,
                        rotation=config.ModuleRotationsIdx.DEG_0,
                        links={}
                    ))
                    hinge.front = end_brick
                except ValueError:
                    # Face already occupied, skip
                    pass
    return genome

def create_max_limb_robot():
    print("\n" + "=" * 50)
    print("Testing MAXIMUM LIMBS robot:")

    # Build a robot that maximizes the number of limbs
    genome = TreeGenome.default_init()  # Core

    # Core has 6 faces: FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM
    # Each can have a brick with 5 faces: FRONT, LEFT, RIGHT, TOP, BOTTOM
    # Each brick face can have a hinge with 1 face: FRONT
    # Each hinge can have a final brick (limb endpoint)

    core_faces = [config.ModuleFaces.FRONT, config.ModuleFaces.BACK,
                  config.ModuleFaces.LEFT, config.ModuleFaces.RIGHT,
                  config.ModuleFaces.TOP, config.ModuleFaces.BOTTOM]

    limb_count = 0

    for core_face in core_faces:
        # Add brick to each core face
        main_brick = TreeNode(config.ModuleInstance(
            type=config.ModuleType.BRICK,
            rotation=config.ModuleRotationsIdx.DEG_0,
            links={}
        ))
        setattr(genome.root, core_face.name.lower(), main_brick)

        # Each brick can have limbs on all its available faces
        brick_faces = config.ALLOWED_FACES[config.ModuleType.BRICK]

        for brick_face in brick_faces:
            # Add hinge for articulation (limb joint)
            hinge = TreeNode(config.ModuleInstance(
                type=config.ModuleType.HINGE,
                rotation=config.ModuleRotationsIdx.DEG_0,
                links={}
            ))

            try:
                setattr(main_brick, brick_face.name.lower(), hinge)

                # Add end effector brick (limb endpoint)
                end_brick = TreeNode(config.ModuleInstance(
                    type=config.ModuleType.BRICK,
                    rotation=config.ModuleRotationsIdx.DEG_0,
                    links={}
                ))
                hinge.front = end_brick  # Hinge only has FRONT face
                limb_count += 1

            except ValueError:
                # Face already occupied, skip
                pass

    return genome

def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
) -> npt.NDArray[np.float64]:
    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu

    # Initialize the networks weights randomly
    # Normally, you would use the genes of an individual as the weights,
    # Here we set them randomly for simplicity.
    w1 = RNG.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
    w2 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
    w3 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))

    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data.qpos

    # Run the inputs through the lays of the network.
    layer1 = np.tanh(np.dot(inputs, w1))
    layer2 = np.tanh(np.dot(layer1, w2))
    outputs = np.tanh(np.dot(layer2, w3))

    # Scale the outputs
    return outputs * np.pi


def morph_descriptors(robot_graph: Any):
    # Analyze morphology using the phenotype graph
    measures = MorphologicalMeasures(robot_graph)

    print(f"\nMorphological measures:")
    print(f"  Number of modules: {measures.num_modules}")
    print(f"  Number of bricks: {measures.num_bricks}")
    print(f"  Number of active hinges: {measures.num_active_hinges}")
    print(f"  Bounding box: {measures.bounding_box_depth}x{measures.bounding_box_width}x{measures.bounding_box_height}")
    print(f"  Coverage: {measures.coverage:.3f}")
    print(f"  Branching: {measures.branching:.3f}")
    print(f"  Limbs: {measures.limbs:.3f}")
    print(f"  Length of limbs: {measures.length_of_limbs:.3f}")
    print(f"  Symmetry: {measures.symmetry:.3f}")
    print(f"  Joints: {measures.J: 3f}")
    print(f"  Is 2D: {measures.is_2d}")
    print(f" Size: {measures.size:.3f}")

    return np.array([measures.B, measures.L, measures.E, measures.S, measures.P, measures.J])


def simple_controller(model: mj.MjModel, data: mj.MjData) -> np.ndarray:
    """Simple oscillating controller for robot movement."""
    time = data.time
    frequency = 2.0
    amplitude = 0.8

    controls = np.zeros(model.nu)
    for i in range(model.nu):
        phase_offset = i * np.pi / 4  # Different phase for each joint
        controls[i] = amplitude * np.sin(frequency * time + phase_offset)

    return controls


def experiment(
    robot: Any,
    controller: Controller,
    duration: int = 15,
    mode: ViewerTypes = "viewer",
    camera_view: str = "default"  # Add camera parameter
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(robot.spec, spawn_position=SPAWN_POS)

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),
    )

    # ------------------------------------------------------------------ #
    match mode:
        case "simple":
            # This disables visualisation (fastest option)
            simple_runner(
                model,
                data,
                duration=duration,
            )
        case "frame":
            # Create custom camera based on view type
            camera = mj.MjvCamera()
            camera.type = mj.mjtCamera.mjCAMERA_FREE

            if camera_view == "front":
                camera.lookat = [0, 0, 0.5]
                camera.distance = 3.0
                camera.azimuth = 0
                camera.elevation = -20
            elif camera_view == "side":
                camera.lookat = [0, 0, 0.5]
                camera.distance = 3.0
                camera.azimuth = 90
                camera.elevation = -20
            elif camera_view == "isometric":
                camera.lookat = [0, 0, 0.5]
                camera.distance = 4.0
                camera.azimuth = 45
                camera.elevation = -30
            else:  # default
                camera = None

            save_path = str(DATA / f"robot_{camera_view}.png")
            single_frame_renderer(
                model,
                data,
                save=True,
                save_path=save_path,
                camera=camera
            )
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
    # ==================================================================== #


def main() -> None:
    """Entry point."""
    # ? ------------------------------------------------------------------ #

    tree_genome = create_max_limb_robot()

    robot_graph = tree_genome.to_digraph(tree_genome)
    robot_graph = load_robot_json_file("examples/target_robots/large_robot_25.json")
    morph_descriptors(robot_graph)

    # ? ------------------------------------------------------------------ #
    # Save the graph to a file
    #save_graph_as_json(
    #    robot_graph,
    #    DATA / "robot_graph.json",
    #)

    # ? ------------------------------------------------------------------ #
    # Print all nodes
    core = construct_mjspec_from_graph(robot_graph)

    # ? ------------------------------------------------------------------ #
    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    # ? ------------------------------------------------------------------ #
    # Simulate the robot
    ctrl = Controller(
        controller_callback_function=simple_controller,
        # controller_callback_function=random_move,
        tracker=tracker,
    )

    experiment(robot=core, controller=ctrl, mode="frame",
               camera_view="isometric")

    show_xpos_history(tracker.history["xpos"][0])

    fitness = fitness_function(tracker.history["xpos"][0])
    msg = f"Fitness of generated robot: {fitness}"
    console.log(msg)


if __name__ == "__main__":
    main()
