import numpy as np
import mujoco
from mujoco import viewer

from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


def main() -> None:
    # Create the world (small square floor, ~0.083 m wide)
    world = SimpleFlatWorld([1, 1, 0.1])

    # # Create a single brick
    # # brick = BrickModule(index=0)
    # # core = CoreModule(index=0)
    gecko_robot1 = gecko()
    # gecko_robot2 = gecko()
    # gecko_robot3 = gecko()

    # # Spawn it slightly above the floor so it falls naturally
    # # world.spawn(brick.spec, spawn_position=[-0.05, 0, 0.05])
    world.spawn(gecko_robot1.spec, spawn_position=[0, 0, 0.05], prefix_id=0)
    # # world.spawn(gecko_robot2.spec, spawn_position=[0.2+(0.05*16), 0, 0.05], prefix_id=1)
    # # world.spawn(gecko_robot3.spec, spawn_position=[0.2+(0.05*16), 5, 0.05], prefix_id=2)
    # world.spawn(gecko().spec, spawn_position=[1, 0, 0.05], prefix_id=1)

    # for i in range(15):
    #     world.spawn(gecko().spec, spawn_position=[np.random.uniform(-10+0.05, 10-0.05), np.random.uniform(-10+0.05, 10-0.05), 0.05], prefix_id=i)
    
    # Compile and create the model
    
    model = world.spec.compile()
    data = mujoco.MjData(model)

    print(data.xpos)

    # Launch viewer
    viewer.launch(model=model, data=data)


if __name__ == "__main__":
    main()
