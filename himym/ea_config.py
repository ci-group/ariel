"""
Configuration loader for Evolutionary Algorithm parameters.

This module loads configuration from ea_config.yaml and provides
easy access to all EA parameters.
"""

import yaml
from pathlib import Path
from typing import Any


class EAConfig:
    """Configuration class for Evolutionary Algorithm parameters."""
    
    def __init__(self, config_path: str | None = None):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file.
                        If None, looks for ea_config.yaml in the same directory.
        """
        if config_path is None:
            # Look for config file in the same directory as this module
            config_path = str(Path(__file__).parent / "ea_config.yaml")
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    # Population Parameters
    @property
    def population_size(self) -> int:
        return self._config['population']['size']
    
    @property
    def num_generations(self) -> int:
        return self._config['population']['num_generations']
    
    @property
    def maintain_positions(self) -> bool:
        return self._config['population'].get('maintain_positions', True)
    
    # Selection Parameters
    @property
    def tournament_size(self) -> int:
        return self._config['selection']['tournament_size']
    
    @property
    def elitism(self) -> bool:
        return self._config['selection']['elitism']
    
    @property
    def pairing_radius(self) -> float:
        return self._config['selection'].get('pairing_radius', 2.0)
    
    @property
    def offspring_radius(self) -> float:
        return self._config['selection'].get('offspring_radius', 0.3)
    
    # Crossover Parameters
    @property
    def crossover_rate(self) -> float:
        return self._config['crossover']['rate']
    
    @property
    def crossover_type(self) -> str:
        return self._config['crossover']['type']
    
    # Mutation Parameters
    @property
    def mutation_rate(self) -> float:
        return self._config['mutation']['rate']
    
    @property
    def mutation_strength(self) -> float:
        return self._config['mutation']['strength']
    
    # Genotype Parameters - Amplitude
    @property
    def amplitude_min(self) -> float:
        return self._config['genotype']['amplitude']['min']
    
    @property
    def amplitude_max(self) -> float:
        return self._config['genotype']['amplitude']['max']
    
    @property
    def amplitude_init_min(self) -> float:
        return self._config['genotype']['amplitude']['init_min']
    
    @property
    def amplitude_init_max(self) -> float:
        return self._config['genotype']['amplitude']['init_max']
    
    # Genotype Parameters - Frequency
    @property
    def frequency_min(self) -> float:
        return self._config['genotype']['frequency']['min']
    
    @property
    def frequency_max(self) -> float:
        return self._config['genotype']['frequency']['max']
    
    @property
    def frequency_init_min(self) -> float:
        return self._config['genotype']['frequency']['init_min']
    
    @property
    def frequency_init_max(self) -> float:
        return self._config['genotype']['frequency']['init_max']
    
    # Genotype Parameters - Phase
    @property
    def phase_min(self) -> float:
        return self._config['genotype']['phase']['min']
    
    @property
    def phase_max(self) -> float:
        return self._config['genotype']['phase']['max']
    
    # Simulation Parameters
    @property
    def simulation_time(self) -> float:
        return self._config['simulation']['time']
    
    @property
    def final_demo_time(self) -> float:
        return self._config['simulation']['final_demo_time']
    
    @property
    def multi_robot_demo_time(self) -> float:
        return self._config['simulation']['multi_robot_demo_time']
    
    @property
    def control_clip_min(self) -> float:
        return self._config['simulation']['control_clip_min']
    
    @property
    def control_clip_max(self) -> float:
        return self._config['simulation']['control_clip_max']
    
    # Multi-Robot Parameters
    @property
    def num_demo_robots(self) -> int:
        return self._config['multi_robot']['num_robots']
    
    @property
    def world_size(self) -> list[float]:
        return self._config['multi_robot']['world_size']
    
    @property
    def spawn_x_min(self) -> float:
        return self._config['multi_robot']['spawn_area']['x_min']
    
    @property
    def spawn_x_max(self) -> float:
        return self._config['multi_robot']['spawn_area']['x_max']
    
    @property
    def spawn_y_min(self) -> float:
        return self._config['multi_robot']['spawn_area']['y_min']
    
    @property
    def spawn_y_max(self) -> float:
        return self._config['multi_robot']['spawn_area']['y_max']
    
    @property
    def spawn_z(self) -> float:
        return self._config['multi_robot']['spawn_area']['z']
    
    @property
    def min_spawn_distance(self) -> float:
        return self._config['multi_robot'].get('min_spawn_distance', 0.6)
    
    @property
    def robot_size(self) -> float:
        """Approximate robot size/diameter for visualization purposes (meters)."""
        return self._config['multi_robot'].get('robot_size', 0.4)
    
    # Output Paths
    @property
    def video_folder(self) -> str:
        return self._config['output']['video_folder']
    
    @property
    def figures_folder(self) -> str:
        return self._config['output']['figures_folder']
    
    @property
    def results_folder(self) -> str:
        return self._config['output']['results_folder']
    
    # Logging
    @property
    def print_individual_fitness(self) -> bool:
        return self._config['logging']['print_individual_fitness']
    
    @property
    def print_generation_stats(self) -> bool:
        return self._config['logging']['print_generation_stats']
    
    @property
    def print_final_genotype(self) -> bool:
        return self._config['logging']['print_final_genotype']
    
    def get_raw_config(self) -> dict[str, Any]:
        """Return the raw configuration dictionary."""
        return self._config
    
    def __repr__(self) -> str:
        return f"EAConfig(population_size={self.population_size}, num_generations={self.num_generations})"


# Create a default configuration instance
config = EAConfig()
