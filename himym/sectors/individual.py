"""Individual representation for spatial evolutionary algorithm."""

import numpy as np


class SpatialIndividual:
    """Represents an individual in the spatial evolutionary algorithm."""
    
    def __init__(self, unique_id: int = None):
        self.genotype = []  # Includes joint control parameters + p_local at the end
        self.fitness = 0.0
        self.start_position = None
        self.end_position = None
        self.spawn_position = None
        self.robot_index = None
        self.unique_id = unique_id  
        self.parent_ids = []
        self.sector_id = None  # Current sector
        self.p_local = 0.5  # Mating preference (0=distant, 1=local)


    def __repr__(self) -> str:
        # genotype_str = ", ".join(f"{x:.4f}" for x in self.genotype)
        spawn_pos_str = ", ".join(f"{x:.2f}" for x in self.spawn_position[:2]) if self.spawn_position is not None else "None"
        return (f"SpatialIndividual(id={self.unique_id}, "
                # f"genotype=[{genotype_str}], "
                f"fitness={self.fitness:.2f}, "
                f"p_local={self.p_local:.2f}, "
                f"spawn_position={spawn_pos_str}, "
                f"sector_id={self.sector_id})")
