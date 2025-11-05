"""Grid management for spatial evolutionary algorithm."""

import numpy as np

from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld


class Grid:
    """Manages 3×3 grid of sectors with adjacency relationships."""
    
    def __init__(self, world_size, grid_size=3):
        self.grid_size = grid_size
        self.world_size = world_size
        
        # Calculate sector dimensions
        self.sector_width = world_size[0] / grid_size
        self.sector_height = world_size[1] / grid_size
        
        # Initialize sectors (dict: sector_id -> list of individual indices)
        self.sectors = {i: [] for i in range(grid_size * grid_size)}

        self.sector_distribution = np.zeros((grid_size * grid_size,), dtype=int)
        
    def get_sector_id(self, position):
        """Get sector ID from continuous position."""
        x, y = position[0], position[1]
        
        # Clamp to grid boundaries
        x = np.clip(x, 0, self.world_size[0] - 0.001)
        y = np.clip(y, 0, self.world_size[1] - 0.001)
        
        col = int(x / self.sector_width)
        row = int(y / self.sector_height)
        
        # Clamp to valid indices
        col = np.clip(col, 0, self.grid_size - 1)
        row = np.clip(row, 0, self.grid_size - 1)
        
        return row * self.grid_size + col
    
    def get_sector_center(self, sector_id, spawn_z):
        """Get center position of a sector."""
        row = sector_id // self.grid_size
        col = sector_id % self.grid_size
        
        x = (col + 0.5) * self.sector_width
        y = (row + 0.5) * self.sector_height
        z = spawn_z
        
        return np.array([x, y, z])
    
    def get_adjacent_sectors(self, sector_id):
        """Get list of adjacent sector IDs (8-way adjacency)."""
        row = sector_id // self.grid_size
        col = sector_id % self.grid_size
        
        adjacent = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip self
                
                new_row = row + dr
                new_col = col + dc
                
                # Check bounds
                if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                    adjacent.append(new_row * self.grid_size + new_col)
        
        return adjacent
    
    def update_sectors(self, population, positions):
        """Update sector membership based on current positions."""
        # Clear sectors
        self.sectors = {i: [] for i in range(self.grid_size * self.grid_size)}
        
        # Assign individuals to sectors
        for i, pos in enumerate(positions):
            sector_id = self.get_sector_id(pos)
            self.sectors[sector_id].append(i)
    
    def get_sector_bounds(self, sector_id):
        """Get (x_min, x_max, y_min, y_max) bounds of a sector."""
        row = sector_id // self.grid_size
        col = sector_id % self.grid_size
        
        x_min = col * self.sector_width
        x_max = (col + 1) * self.sector_width
        y_min = row * self.sector_height
        y_max = (row + 1) * self.sector_height
        
        return x_min, x_max, y_min, y_max
