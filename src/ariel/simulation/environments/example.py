from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.environments.obstacles import (ObstacleConfig, ObstacleType, generate_obstacles, attach_obstacles,)
world = SimpleFlatWorld()
cfg = ObstacleConfig(obstacle_type=ObstacleType.FOREST, num_obstacles=20, seed=0)
attach_obstacles(world.spec, generate_obstacles(cfg))
model = world.spec.compile()