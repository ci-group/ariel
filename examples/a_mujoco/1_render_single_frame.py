"""Example usage of `render_single_frame` function."""

# Third-party libraries
import mujoco

# Local libraries
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer


def main() -> None:
    """Entry point."""
    # World
    world = SimpleFlatWorld()

    # Object
    body = mujoco.MjSpec()
    cube = body.worldbody.add_body()
    cube.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.1, 0.1, 0.1),
        rgba=(0.8, 0.2, 0.2, 1.0),
    )

    # Add object to world
    world.spawn(
        body,
        correct_collision_with_floor=True,
    )

    # Generate the model and data
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Render a single frame
    single_frame_renderer(model, data, show=True)


if __name__ == "__main__":
    main()
