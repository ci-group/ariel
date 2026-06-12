from envs.simple_env import SimpleWalkerEnvClass
import gymnasium as gym

gym.register(
    id = 'SimpleWalkingEnv-v0',
    entry_point = 'envs.simple_env:SimpleWalkerEnvClass',
    max_episode_steps = 500
)