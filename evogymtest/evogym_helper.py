import numpy as np
import gymnasium as gym
import warnings
from typing import Any

class EvoGymWrapper(gym.Wrapper):
    """
    A wrapper for EvoGym environments to handle API changes and suppress warnings.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
    
    def step(self, action: Any) -> Any:
        # Suppress potential warnings from underlying env
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            result = self.env.step(action)
        return result

    def reset(self, **kwargs: Any) -> Any:
        # Suppress potential warnings from underlying env
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            result = self.env.reset(**kwargs)
        return result

    def get_action(self, controller: Any) -> Any:
        """
        Expose get_action from the underlying environment.
        """
        # Use getattr to satisfy IDE inspections for methods not in the base gym.Env
        return getattr(self.unwrapped, 'get_action')(controller)

class EvoGymHelper:
    """
    EvoGym specific individual for evolutionary algorithms.
    """
    @staticmethod
    def grid_is_ok(grid, max_size):
        contains_action = False
        for i in range(max_size):
            for j in range(max_size):
                if grid[i][j] == 3.0 or grid[i][j] == 4.0:
                    contains_action = True
        return EvoGymHelper.is_fully_connected(grid) and contains_action

    @staticmethod
    def is_fully_connected(grid):
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)

        # Find a starting point with value between 1 and 4
        start = None
        for i in range(rows):
            for j in range(cols):
                if 1 <= grid[i, j] <= 4:
                    start = (i, j)
                    break
            if start:
                break

        if not start:
            return False  # No valid starting point (empty grid or no values between 1 and 4)

        # Flood fill (DFS)
        stack = [start]
        visited[start] = True

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        while stack:
            x, y = stack.pop()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:  # Within bounds
                    if not visited[nx, ny] and 1 <= grid[nx, ny] <= 4:  # Valid connection
                        visited[nx, ny] = True
                        stack.append((nx, ny))

        # Check if all 1-4 values were visited
        return np.all((grid < 1) | (grid > 4) | visited)