"""Test: initialization of environment classes."""

# Standard library
import inspect

# Third-party libraries
import mujoco

# Local libraries
from ariel.simulation import environments
from ariel.simulation.environments import (
    AmphitheatreTerrainWorld,
    BaseWorld,
    CompoundWorld,
    CraterTerrainWorld,
    OlympicArena,
    RuggedTerrainWorld,
    RuggedTiltedWorld,
    SimpleFlatWorld,
    SimpleTiltedWorld,
)


def test_amphitheatre_terrain_world_initialization() -> None:
    """Simply instantiate the AmphitheatreTerrainWorld class."""
    world = AmphitheatreTerrainWorld()
    model = world.spec.compile()
    data = mujoco.MjData(model)
    del model, data, world


def test_base_world_initialization() -> None:
    """Simply instantiate the BaseWorld class."""
    world = BaseWorld()
    model = world.spec.compile()
    data = mujoco.MjData(model)
    del model, data, world


def test_compound_world_initialization() -> None:
    """Simply instantiate the CompoundWorld class."""
    world = CompoundWorld()
    model = world.spec.compile()
    data = mujoco.MjData(model)
    del model, data, world


def test_crater_terrain_world_initialization() -> None:
    """Simply instantiate the CraterTerrainWorld class."""
    world = CraterTerrainWorld()
    model = world.spec.compile()
    data = mujoco.MjData(model)
    del model, data, world


def test_olympic_arena_initialization() -> None:
    """Simply instantiate the OlympicArena class."""
    world = OlympicArena()
    model = world.spec.compile()
    data = mujoco.MjData(model)
    del model, data, world


def test_rugged_terrain_world_initialization() -> None:
    """Simply instantiate the RuggedTerrainWorld class."""
    world = RuggedTerrainWorld()
    model = world.spec.compile()
    data = mujoco.MjData(model)
    del model, data, world


def test_rugged_tilted_world_initialization() -> None:
    """Simply instantiate the RuggedTiltedWorld class."""
    world = RuggedTiltedWorld()
    model = world.spec.compile()
    data = mujoco.MjData(model)
    del model, data, world


def test_simple_flat_world_initialization() -> None:
    """Simply instantiate the SimpleFlatWorld class."""
    world = SimpleFlatWorld()
    model = world.spec.compile()
    data = mujoco.MjData(model)
    del model, data, world


def test_simple_tilted_world_initialization() -> None:
    """Simply instantiate the SimpleTiltedWorld class."""
    world = SimpleTiltedWorld()
    model = world.spec.compile()
    data = mujoco.MjData(model)
    del model, data, world


def test_instantiation_of_all_environments() -> None:
    """Instantiate all ARIEL environments."""
    for _, (_, cls) in enumerate(
        inspect.getmembers(environments, inspect.isclass),
    ):
        xml = r"""
        <mujoco>
        <worldbody>
            <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
            <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
        </worldbody>
        </mujoco>
        """

        # Instantiate the environment
        world: BaseWorld = cls(
            load_precompiled=False,
        )

        # Ensure the environment
        assert issubclass(type(world), BaseWorld)

        # Spawn a test object to validate the environment
        test_object = mujoco.MjSpec.from_string(xml)
        world.spawn(
            test_object,
            position=(0, 0, 0),
            rotation=(0, 0, 0),
            correct_collision_with_floor=True,
        )

        # Compile the model and create data
        model = world.spec.compile()
        data = mujoco.MjData(model)

        # Step the simulation to ensure no errors
        mujoco.mj_step(model, data)

        # Clear memory
        del xml, world, test_object, model, data


def test_all_heightmap_functions() -> None:
    """Test all heightmap functions to ensure they run without error."""
    arguments = {
        "flat_heightmap": {
            "dims": (75, 100),
        },
        "rugged_heightmap": {
            "dims": (75, 100),
            "scale_of_noise": 4,
            "normalize": None,
        },
        "amphitheater_heightmap": {
            "dims": (75, 100),
            "ring_inner_radius": 0.2,
            "ring_outer_radius": 0.45,
            "cone_height": 1.0,
        },
        "crater_heightmap": {
            "dims": (75, 100),
            "crater_depth": 1.0,
            "crater_radius": 0.3,
        },
        "smooth_edges": {
            "dims": (100, 100),
            "edge_width": 25,
        },
    }
    # Loop through all functions in heightmap_functions.py
    for _, (_, func) in enumerate(
        inspect.getmembers(
            environments.heightmap_functions,
            inspect.isfunction,
        ),
    ):
        # Call the function with default parameters
        _ = func(**arguments[func.__name__])
