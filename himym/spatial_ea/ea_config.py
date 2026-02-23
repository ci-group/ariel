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
    
    # Incubation Parameters
    @property
    def incubation_enabled(self) -> bool:
        """Enable incubation phase before spatial evolution."""
        return self._config.get('incubation', {}).get('enabled', False)
    
    @property
    def incubation_population_size(self) -> int:
        """Population size during incubation."""
        return self._config.get('incubation', {}).get('population_size', 20)
    
    @property
    def incubation_num_generations(self) -> int:
        """Number of generations in incubation phase."""
        return self._config.get('incubation', {}).get('num_generations', 50)
    
    @property
    def incubation_mutation_rate(self) -> float:
        """Weight mutation probability during incubation."""
        return self._config.get('incubation', {}).get('mutation_rate', 0.8)
    
    @property
    def incubation_mutation_power(self) -> float:
        """Weight mutation strength during incubation."""
        return self._config.get('incubation', {}).get('mutation_power', 0.5)
    
    @property
    def incubation_add_connection_rate(self) -> float:
        """Probability of adding new connection during incubation."""
        return self._config.get('incubation', {}).get('add_connection_rate', 0.05)
    
    @property
    def incubation_add_node_rate(self) -> float:
        """Probability of adding new node during incubation."""
        return self._config.get('incubation', {}).get('add_node_rate', 0.03)
    
    @property
    def incubation_crossover_rate(self) -> float:
        """Probability of crossover vs cloning during incubation."""
        return self._config.get('incubation', {}).get('crossover_rate', 0.9)
    
    @property
    def incubation_tournament_size(self) -> int:
        """Number of individuals in tournament selection during incubation."""
        return self._config.get('incubation', {}).get('tournament_size', 3)
    
    @property
    def incubation_elitism_count(self) -> int:
        """Number of best individuals to preserve unchanged during incubation."""
        return self._config.get('incubation', {}).get('elitism_count', 2)
    
    @property
    def incubation_use_directional_fitness(self) -> bool:
        """Use directional fitness (movement toward target) instead of distance-only."""
        return self._config.get('incubation', {}).get('use_directional_fitness', True)
    
    @property
    def incubation_target_distance_min(self) -> float:
        """Minimum distance to randomly placed target (meters)."""
        return self._config.get('incubation', {}).get('target_distance_min', 5.0)
    
    @property
    def incubation_target_distance_max(self) -> float:
        """Maximum distance to randomly placed target (meters)."""
        return self._config.get('incubation', {}).get('target_distance_max', 10.0)
    
    @property
    def incubation_progress_weight(self) -> float:
        """Weight for progress toward target component in directional fitness."""
        return self._config.get('incubation', {}).get('progress_weight', 2.0)
    
    @property
    def incubation_distance_weight(self) -> float:
        """Weight for total distance traveled component in directional fitness."""
        return self._config.get('incubation', {}).get('distance_weight', 0.2)
    
    # Spatial EA Directional Fitness Parameters
    @property
    def use_directional_fitness(self) -> bool:
        """Use directional fitness for spatial EA (movement toward target) instead of distance-only."""
        return self._config.get('incubation', {}).get('use_directional_fitness', True)
    
    @property
    def target_distance_min(self) -> float:
        """Minimum distance to randomly placed target for spatial EA (meters)."""
        return self._config.get('incubation', {}).get('target_distance_min', 5.0)
    
    @property
    def target_distance_max(self) -> float:
        """Maximum distance to randomly placed target for spatial EA (meters)."""
        return self._config.get('incubation', {}).get('target_distance_max', 10.0)
    
    @property
    def progress_weight(self) -> float:
        """Weight for progress toward target component for spatial EA."""
        return self._config.get('incubation', {}).get('progress_weight', 0.5)
    
    # Population Parameters
    @property
    def population_size(self) -> int:
        return self._config['population']['size']
    
    @property
    def num_generations(self) -> int:
        return self._config['population']['num_generations']
    
    @property
    def max_population_limit(self) -> int:
        """Absolute maximum population (simulation stops if reached)."""
        return self._config['population'].get('max_population_limit', 100)
    
    @property
    def min_population_limit(self) -> int:
        """Minimum population (simulation stops if population drops below this)."""
        return self._config['population'].get('min_population_limit', 1)
    
    @property
    def stop_on_limits(self) -> bool:
        """Whether to stop simulation when population limits are reached."""
        return self._config['population'].get('stop_on_limits', True)
    
    # Selection Parameters
    @property
    def pairing_radius(self) -> float:
        return self._config['selection'].get('pairing_radius', 2.0)
    
    @property
    def offspring_radius(self) -> float:
        return self._config['selection'].get('offspring_radius', 0.3)
    
    @property
    def target_population_size(self) -> int:
        return self._config['selection'].get('target_population_size', 20)
    
    @property
    def selection_method(self) -> str:
        return self._config['selection'].get('selection_method', 'parents_die')
    
    @property
    def pairing_method(self) -> str:
        return self._config['selection'].get('pairing_method', 'proximity_fitness')
    
    @property
    def mating_zone_center(self) -> list[float]:
        """Center coordinates (x, y) of the mating zone."""
        return self._config['selection'].get('mating_zone_center', [0.0, 0.0])
    
    @property
    def mating_zone_radius(self) -> float:
        """Radius of the mating zone circle."""
        return self._config['selection'].get('mating_zone_radius', 3.0)
    
    @property
    def num_mating_zones(self) -> int:
        """Number of randomly placed mating zones."""
        return self._config['selection'].get('num_mating_zones', 1)
    
    @property
    def zone_relocation_strategy(self) -> str:
        """
        Strategy for relocating mating zones.
        
        Returns:
            One of: "static", "generation_interval", or "event_driven"
            
        Note: Provides backward compatibility with old boolean 'dynamic_mating_zones' config.
        If dynamic_mating_zones is found, converts: True -> "generation_interval", False -> "static"
        """
        # Check for new string-based parameter first
        strategy = self._config['selection'].get('zone_relocation_strategy', None)
        if strategy is not None:
            valid_strategies = ["static", "generation_interval", "event_driven"]
            if strategy not in valid_strategies:
                print(f"Warning: Invalid zone_relocation_strategy '{strategy}', defaulting to 'static'")
                return "static"
            return strategy
        
        # Backward compatibility: convert old boolean to string
        old_dynamic = self._config['selection'].get('dynamic_mating_zones', None)
        if old_dynamic is not None:
            return "generation_interval" if old_dynamic else "static"
        
        # Default to static if neither parameter exists
        return "static"
    
    @property
    def dynamic_mating_zones(self) -> bool:
        """
        DEPRECATED: Use zone_relocation_strategy instead.
        
        Maintained for backward compatibility only.
        Returns True if zone_relocation_strategy is "generation_interval" or "event_driven".
        """
        strategy = self.zone_relocation_strategy
        return strategy in ["generation_interval", "event_driven"]
    
    @property
    def zone_change_interval(self) -> int:
        """Generations between zone position changes (for dynamic zones)."""
        return self._config['selection'].get('zone_change_interval', 5)
    
    @property
    def min_zone_distance(self) -> float:
        """Minimum distance between zone centers (in zone radii units)."""
        return self._config['selection'].get('min_zone_distance', 2.0)
    
    @property
    def movement_bias(self) -> str:
        """Movement bias during mating phase: 'nearest_neighbor', 'nearest_zone', or 'none'."""
        return self._config['selection'].get('movement_bias', 'nearest_neighbor')
    
    @property
    def max_age(self) -> int:
        """Maximum age for probabilistic_age selection (death probability = age/max_age)."""
        return self._config['selection'].get('max_age', 10)
    
    # Energy-Based Selection Parameters
    @property
    def enable_energy(self) -> bool:
        """Enable energy-based survival mechanics."""
        return self._config['selection'].get('enable_energy', False)
    
    @property
    def initial_energy(self) -> float:
        """Starting energy for each individual."""
        return self._config['selection'].get('initial_energy', 100.0)
    
    @property
    def energy_depletion_rate(self) -> float:
        """Energy lost per generation (time-based passive depletion)."""
        return self._config['selection'].get('energy_depletion_rate', 1.0)
    
    @property
    def mating_energy_effect(self) -> str:
        """Effect of mating on energy: 'restore' (full reset), 'cost' (energy penalty), 'none' (no effect)."""
        return self._config['selection'].get('mating_energy_effect', 'restore')
    
    @property
    def mating_energy_amount(self) -> float:
        """Energy restored (if 'restore') or cost deducted (if 'cost') when mating occurs."""
        return self._config['selection'].get('mating_energy_amount', 50.0)
    
    # Density-Based Selection Parameters
    @property
    def locality_radius(self) -> float:
        """Gaussian kernel σ for local density calculation (meters)."""
        return self._config['selection'].get('locality_radius', 3.0)
    
    @property
    def critical_density(self) -> float:
        """ρ_c threshold where density death probability ≈ 0.63 × P_max."""
        return self._config['selection'].get('critical_density', 5.0)
    
    @property
    def base_death_prob(self) -> float:
        """P_base - baseline death probability for isolated individuals."""
        return self._config['selection'].get('base_death_prob', 0.05)
    
    @property
    def max_density_death_prob(self) -> float:
        """P_max - maximum additional death probability from crowding."""
        return self._config['selection'].get('max_density_death_prob', 0.8)
    
    @property
    def density_fitness_protection(self) -> float:
        """Reduction in death probability for high-fitness individuals (0-1)."""
        return self._config['selection'].get('density_fitness_protection', 0.0)
    
    # Crossover Parameters
    @property
    def crossover_rate(self) -> float:
        return self._config['crossover']['rate']
    
    # Mutation Parameters
    @property
    def mutation_rate(self) -> float:
        return self._config['mutation']['rate']
    
    @property
    def mutation_strength(self) -> float:
        return self._config['mutation']['strength']

    @property
    def mutation_add_connection_rate(self) -> float:
        return self._config['mutation']['add_connection_rate']
    
    @property
    def mutation_add_node_rate(self) -> float:
        return self._config['mutation']['add_node_rate']
    
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
    def control_clip_min(self) -> float:
        return self._config['simulation']['control_clip_min']
    
    @property
    def control_clip_max(self) -> float:
        return self._config['simulation']['control_clip_max']
    
    @property
    def use_periodic_boundaries(self) -> bool:
        return self._config['simulation'].get('use_periodic_boundaries', False)
    
    # Multi-Robot Parameters
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
    def print_generation_stats(self) -> bool:
        return self._config['logging']['print_generation_stats']
    
    @property
    def print_final_genotype(self) -> bool:
        return self._config['logging']['print_final_genotype']
    
    # Video Recording
    @property
    def record_generation_videos(self) -> bool:
        return self._config['video'].get('record_generation_videos', False)
    
    @property
    def save_generation_snapshots(self) -> bool:
        return self._config['video'].get('save_generation_snapshots', True)
    
    def get_raw_config(self) -> dict[str, Any]:
        """Return the raw configuration dictionary."""
        return self._config
    
    def __repr__(self) -> str:
        return f"EAConfig(population_size={self.population_size}, num_generations={self.num_generations})"


# Create a default configuration instance
config = EAConfig()
