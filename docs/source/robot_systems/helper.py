"""Local helper file."""

# Third-party libraries
import mujoco as mj

from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

# Local libraries
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer


def show_module(module: str):
    """Entry point."""
    # World
    world = SimpleFlatWorld()

    # Object
    if module == "core":
        module_geom = CoreModule(0).spec
        zoom = 1
    elif module == "brick":
        module_geom = BrickModule(0).spec
        zoom = 0.5
    elif module == "hinge":
        module_geom = HingeModule(0).spec
        zoom = 0.5
    else:
        return Non

    # Add object to world
    world.spawn(
        module_geom,
        correct_collision_with_floor=True,
    )

    # Generate the model and data
    model = world.spec.compile()
    data = mj.MjData(model)

    # Render a single frame
    return single_frame_renderer(model, data, show=False, cam_fovy=zoom)


if __name__ == "__main__":
    show_module("core")
