"""
Experiment runner for conducting multiple evolutionary runs with parameter sweeps.

This module provides functionality to:
1. Run multiple trials with the same or different parameters
2. Handle early stopping (population extinction or explosion)
3. Aggregate statistics across runs (mean, std, min, max)
4. Save individual and aggregated results
5. Generate comparative visualizations
6. Parallel execution of independent trials using multiprocessing
"""

import gc
import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import itertools
import copy

from ea_config import EAConfig, config


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment (potentially multiple runs)."""
    
    # Experiment identification
    experiment_name: str
    num_runs: int = 5
    
    # Incubation parameters
    incubation_enabled: bool | None = None
    incubation_num_generations: int | None = None
    
    # Population parameters
    population_size: int | None = None
    num_generations: int | None = None
    max_population_limit: int | None = None
    min_population_limit: int | None = None
    stop_on_limits: bool | None = None
    
    # Selection parameters
    pairing_method: str | None = None
    movement_bias: str | None = None
    selection_method: str | None = None
    target_population_size: int | None = None
    pairing_radius: float | None = None
    offspring_radius: float | None = None
    max_age: int | None = None
    
    # Mating zone parameters (excluding mating_zone_center)
    mating_zone_radius: float | None = None
    num_mating_zones: int | None = None
    dynamic_mating_zones: bool | None = None  # DEPRECATED: use zone_relocation_strategy
    zone_relocation_strategy: str | None = None  # "static", "generation_interval", or "event_driven"
    zone_change_interval: int | None = None
    min_zone_distance: float | None = None
    
    # Energy parameters
    enable_energy: bool | None = None
    initial_energy: float | None = None
    energy_depletion_rate: float | None = None
    mating_energy_effect: str | None = None
    mating_energy_amount: float | None = None
    
    # Density-based selection parameters
    locality_radius: float | None = None  # Gaussian kernel σ for local density calculation
    critical_density: float | None = None  # ρ_c threshold where P_density ≈ 0.63 × P_max
    base_death_prob: float | None = None  # P_base - baseline death probability for isolated individuals
    max_density_death_prob: float | None = None  # P_max - max additional death prob from crowding
    density_fitness_protection: float | None = None  # Reduction in death prob for high-fitness individuals (0-1)
    
    # Mutation parameters (shared between incubation and spatial)
    mutation_rate: float | None = None
    mutation_strength: float | None = None
    add_connection_rate: float | None = None
    add_node_rate: float | None = None
    
    # Crossover parameters (shared between incubation and spatial)
    crossover_rate: float | None = None
    
    # Incubation-specific parameters (only used if different from spatial)
    incubation_tournament_size: int | None = None
    incubation_elitism_count: int | None = None
    incubation_use_directional_fitness: bool | None = None
    incubation_target_distance_min: float | None = None
    incubation_target_distance_max: float | None = None
    incubation_progress_weight: float | None = None
    
    # Simulation parameters
    simulation_time: float | None = None
    use_periodic_boundaries: bool | None = None
    
    # Output settings
    save_snapshots: bool = False
    save_trajectories: bool = True
    record_videos: bool = False
    save_individual_runs: bool = False  # Save detailed data for each run (trajectories, figures, etc.)
    
    # Random seed (None = random)
    random_seed: int | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def apply_to_config(self, base_config: EAConfig) -> dict[str, Any]:
        """
        Create a config dictionary with experiment parameters applied.
        
        Returns:
            Dictionary of parameter overrides
        """
        overrides = {}
        
        # Incubation parameters
        if self.incubation_enabled is not None:
            overrides['incubation.enabled'] = self.incubation_enabled
        if self.incubation_num_generations is not None:
            overrides['incubation.num_generations'] = self.incubation_num_generations
        
        # Population parameters
        if self.population_size is not None:
            overrides['population.size'] = self.population_size
            # Share with incubation if enabled
            if self.incubation_enabled:
                overrides['incubation.population_size'] = self.population_size
        if self.num_generations is not None:
            overrides['population.num_generations'] = self.num_generations
        if self.max_population_limit is not None:
            overrides['population.max_population_limit'] = self.max_population_limit
        if self.min_population_limit is not None:
            overrides['population.min_population_limit'] = self.min_population_limit
        if self.stop_on_limits is not None:
            overrides['population.stop_on_limits'] = self.stop_on_limits
        
        # Selection parameters
        if self.pairing_method is not None:
            overrides['selection.pairing_method'] = self.pairing_method
        if self.movement_bias is not None:
            overrides['selection.movement_bias'] = self.movement_bias
        if self.selection_method is not None:
            overrides['selection.selection_method'] = self.selection_method
        if self.target_population_size is not None:
            overrides['selection.target_population_size'] = self.target_population_size
        if self.pairing_radius is not None:
            overrides['selection.pairing_radius'] = self.pairing_radius
        if self.offspring_radius is not None:
            overrides['selection.offspring_radius'] = self.offspring_radius
        if self.max_age is not None:
            overrides['selection.max_age'] = self.max_age
        
        # Mating zone parameters
        if self.mating_zone_radius is not None:
            overrides['selection.mating_zone_radius'] = self.mating_zone_radius
        if self.num_mating_zones is not None:
            overrides['selection.num_mating_zones'] = self.num_mating_zones
        
        # Handle zone relocation strategy (new parameter takes precedence over old)
        if self.zone_relocation_strategy is not None:
            overrides['selection.zone_relocation_strategy'] = self.zone_relocation_strategy
        elif self.dynamic_mating_zones is not None:
            # Backward compatibility: convert old boolean to new string
            strategy = "generation_interval" if self.dynamic_mating_zones else "static"
            overrides['selection.zone_relocation_strategy'] = strategy
        
        if self.zone_change_interval is not None:
            overrides['selection.zone_change_interval'] = self.zone_change_interval
        if self.min_zone_distance is not None:
            overrides['selection.min_zone_distance'] = self.min_zone_distance
        
        # Energy parameters
        if self.enable_energy is not None:
            overrides['selection.enable_energy'] = self.enable_energy
        if self.initial_energy is not None:
            overrides['selection.initial_energy'] = self.initial_energy
        if self.energy_depletion_rate is not None:
            overrides['selection.energy_depletion_rate'] = self.energy_depletion_rate
        if self.mating_energy_effect is not None:
            overrides['selection.mating_energy_effect'] = self.mating_energy_effect
        if self.mating_energy_amount is not None:
            overrides['selection.mating_energy_amount'] = self.mating_energy_amount
        
        # Density-based selection parameters
        if self.locality_radius is not None:
            overrides['selection.locality_radius'] = self.locality_radius
        if self.critical_density is not None:
            overrides['selection.critical_density'] = self.critical_density
        if self.base_death_prob is not None:
            overrides['selection.base_death_prob'] = self.base_death_prob
        if self.max_density_death_prob is not None:
            overrides['selection.max_density_death_prob'] = self.max_density_death_prob
        if self.density_fitness_protection is not None:
            overrides['selection.density_fitness_protection'] = self.density_fitness_protection
        
        # Mutation parameters (shared between incubation and spatial)
        if self.mutation_rate is not None:
            overrides['mutation.rate'] = self.mutation_rate
            if self.incubation_enabled:
                overrides['incubation.mutation_rate'] = self.mutation_rate
        if self.mutation_strength is not None:
            overrides['mutation.strength'] = self.mutation_strength
            if self.incubation_enabled:
                overrides['incubation.mutation_power'] = self.mutation_strength
        if self.add_connection_rate is not None:
            overrides['mutation.add_connection_rate'] = self.add_connection_rate
            if self.incubation_enabled:
                overrides['incubation.add_connection_rate'] = self.add_connection_rate
        if self.add_node_rate is not None:
            overrides['mutation.add_node_rate'] = self.add_node_rate
            if self.incubation_enabled:
                overrides['incubation.add_node_rate'] = self.add_node_rate
        
        # Crossover parameters (shared between incubation and spatial)
        if self.crossover_rate is not None:
            overrides['crossover.rate'] = self.crossover_rate
            if self.incubation_enabled:
                overrides['incubation.crossover_rate'] = self.crossover_rate
        
        # Incubation-specific parameters (only if different from spatial)
        if self.incubation_tournament_size is not None:
            overrides['incubation.tournament_size'] = self.incubation_tournament_size
        if self.incubation_elitism_count is not None:
            overrides['incubation.elitism_count'] = self.incubation_elitism_count
        if self.incubation_use_directional_fitness is not None:
            overrides['incubation.use_directional_fitness'] = self.incubation_use_directional_fitness
        if self.incubation_target_distance_min is not None:
            overrides['incubation.target_distance_min'] = self.incubation_target_distance_min
        if self.incubation_target_distance_max is not None:
            overrides['incubation.target_distance_max'] = self.incubation_target_distance_max
        if self.incubation_progress_weight is not None:
            overrides['incubation.progress_weight'] = self.incubation_progress_weight
        
        # Simulation parameters
        if self.simulation_time is not None:
            overrides['simulation.time'] = self.simulation_time
        if self.use_periodic_boundaries is not None:
            overrides['simulation.use_periodic_boundaries'] = self.use_periodic_boundaries
        
        # Video/output parameters
        if self.save_snapshots is not None:
            overrides['video.save_generation_snapshots'] = self.save_snapshots
        if self.record_videos is not None:
            overrides['video.record_generation_videos'] = self.record_videos
        
        return overrides


@dataclass
class RunResult:
    """Results from a single evolutionary run."""
    
    run_id: int
    experiment_name: str
    
    # Run outcome
    completed_generations: int
    stopped_early: bool
    stop_reason: str
    
    # Statistics arrays (indexed by generation)
    generations: list[int] = field(default_factory=list)
    population_size: list[int] = field(default_factory=list)
    fitness_best: list[float] = field(default_factory=list)
    fitness_avg: list[float] = field(default_factory=list)
    fitness_worst: list[float] = field(default_factory=list)
    fitness_std: list[float] = field(default_factory=list)
    
    # Optional statistics
    energy_avg: list[float] = field(default_factory=list)
    age_avg: list[float] = field(default_factory=list)
    mating_pairs: list[int] = field(default_factory=list)
    
    # Timing
    duration_seconds: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = {
            'generation': self.generations,
            'population_size': self.population_size,
            'fitness_best': self.fitness_best,
            'fitness_avg': self.fitness_avg,
            'fitness_worst': self.fitness_worst,
            'fitness_std': self.fitness_std,
        }
        
        # Only add optional columns if they match the length of generations
        num_gens = len(self.generations)
        
        if self.energy_avg and len(self.energy_avg) == num_gens:
            data['energy_avg'] = self.energy_avg
        if self.age_avg and len(self.age_avg) == num_gens:
            data['age_avg'] = self.age_avg
        if self.mating_pairs and len(self.mating_pairs) == num_gens:
            data['mating_pairs'] = self.mating_pairs
        
        return pd.DataFrame(data)


@dataclass
class AggregatedResults:
    """Aggregated results across multiple runs."""
    
    experiment_name: str
    num_runs: int
    
    # Common generation axis (aligned across all runs)
    generations: np.ndarray
    
    # Population size statistics
    population_mean: np.ndarray
    population_std: np.ndarray
    population_min: np.ndarray
    population_max: np.ndarray
    
    # Fitness statistics
    fitness_best_mean: np.ndarray
    fitness_best_std: np.ndarray
    fitness_avg_mean: np.ndarray
    fitness_avg_std: np.ndarray
    
    # Run completion statistics
    runs_active: np.ndarray  # Number of runs still active at each generation
    completion_rate: np.ndarray  # Fraction of runs that reached this generation
    
    # Early stopping analysis
    num_extinctions: int = 0
    num_explosions: int = 0
    num_completed: int = 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = {
            'generation': self.generations,
            'population_mean': self.population_mean,
            'population_std': self.population_std,
            'population_min': self.population_min,
            'population_max': self.population_max,
            'fitness_best_mean': self.fitness_best_mean,
            'fitness_best_std': self.fitness_best_std,
            'fitness_avg_mean': self.fitness_avg_mean,
            'fitness_avg_std': self.fitness_avg_std,
            'runs_active': self.runs_active,
            'completion_rate': self.completion_rate,
        }
        return pd.DataFrame(data)


# ============================================================================
# Standalone Trial Runner for Multiprocessing
# ============================================================================

def _run_trial_subprocess(args: tuple) -> RunResult | None:
    """
    Standalone function to run a single trial in a separate process.
    
    This function is designed to be pickled and run in a subprocess pool.
    It accepts all necessary configuration as serializable arguments and
    does NOT rely on global config state.
    
    Args:
        args: Tuple containing (experiment_config_dict, run_id, output_dir_str, 
              base_config_path, verbose)
    
    Returns:
        RunResult or None if error
    """
    # Ensure MUJOCO_GL is set BEFORE any mujoco import (headless HPC nodes).
    # Prefer EGL then fall back to OSMesa. Respect externally-set value.
    import os
    import importlib
    if not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "egl"
        try:
            # If mujoco is importable, try a lightweight import to validate backend.
            # If it fails, we will fall back to osmesa.
            mujoco_spec = importlib.util.find_spec("mujoco")
            if mujoco_spec is not None:
                import mujoco  # type: ignore
        except Exception:
            os.environ["MUJOCO_GL"] = "osmesa"

    import time
    import traceback
    from pathlib import Path
    import numpy as np
    import ea_config
    from main_spatial_ea import SpatialEA
    
    # Unpack arguments
    exp_config_dict, run_id, output_dir_str, base_config_path, verbose = args
    
    # Reconstruct ExperimentConfig
    experiment_config = ExperimentConfig(**exp_config_dict)
    output_dir = Path(output_dir_str)
    
    try:
        start_time = time.time()
        
        # Create run-specific output directories
        run_output = output_dir / f"run_{run_id:03d}"
        run_output.mkdir(parents=True, exist_ok=True)
        
        # Override config paths for this run
        figures_dir = run_output / "figures"
        results_dir = run_output / "results"
        video_dir = run_output / "videos"
        
        figures_dir.mkdir(exist_ok=True)
        results_dir.mkdir(exist_ok=True)
        video_dir.mkdir(exist_ok=True)
        
        # IMPORTANT: We cannot replace ea_config.config because other modules
        # have already imported it with "from ea_config import config"
        # Instead, we must modify the existing config object's internal state
        import ea_config
        config = ea_config.config
        
        # Reload the config from the base config file
        with open(base_config_path, 'r') as f:
            import yaml
            config._config = yaml.safe_load(f)
        
        # Apply configuration overrides
        overrides = experiment_config.apply_to_config(config)
        
        # Apply overrides to this process's config copy
        original_values = {}
        for key, value in overrides.items():
            parts = key.split('.')
            obj = config._config
            for part in parts[:-1]:
                obj = obj[part]
            original_values[key] = obj[parts[-1]]
            obj[parts[-1]] = value
        
        # Override output directories
        original_figures = config._config['output']['figures_folder']
        original_results = config._config['output']['results_folder']
        original_videos = config._config['output']['video_folder']
        
        config._config['output']['figures_folder'] = str(figures_dir)
        config._config['output']['results_folder'] = str(results_dir)
        config._config['output']['video_folder'] = str(video_dir)
        config._config['video']['save_generation_snapshots'] = experiment_config.save_snapshots
        config._config['video']['record_generation_videos'] = experiment_config.record_videos
        
        # Set random seed to ensure different runs have different random states
        # Use time-based seed if not specified to ensure variability across runs
        if experiment_config.random_seed is not None:
            seed = experiment_config.random_seed + run_id
        else:
            # Use a combination of time and run_id for a unique seed
            import time
            seed = int((time.time() * 1000000) % (2**31)) + run_id
        np.random.seed(seed)
        
        if verbose:
            print(f"\n[Worker {run_id}] Starting run {run_id + 1}")
            print(f"[Worker {run_id}] Config - save_snapshots: {config.save_generation_snapshots}")
            print(f"[Worker {run_id}] Config - record_videos: {config.record_generation_videos}")
            print(f"[Worker {run_id}] Config - figures_folder: {config.figures_folder}")
        
        # NOTE: We skip compiling the robot spec here to avoid MuJoCo state
        # corruption. The SpatialEA will create and compile its own robot specs
        # during the simulation. We hardcode num_joints=8 for the gecko robot.
        num_joints = 8  # gecko robot has 8 actuated joints
        
        # Create SpatialEA instance
        spatial_ea = SpatialEA(num_joints=num_joints)
        
        # Run evolution
        spatial_ea.run_evolution()
        
        # Collect results from data collector
        dc = spatial_ea.data_collector
        
        # Create result
        result = RunResult(
            experiment_name=experiment_config.experiment_name,
            run_id=run_id,
            completed_generations=len(dc.generations),
            stopped_early=dc.stopped_early,
            stop_reason=dc.stop_reason if dc.stopped_early else "Completed",
            generations=dc.generations.copy(),
            population_size=dc.population_size.copy(),
            fitness_best=dc.fitness_best.copy(),
            fitness_avg=dc.fitness_avg.copy(),
            fitness_worst=dc.fitness_worst.copy(),
            fitness_std=dc.fitness_std.copy(),
            energy_avg=dc.energy_avg.copy() if dc.energy_avg else [],
            age_avg=dc.age_avg.copy() if dc.age_avg else [],
            mating_pairs=dc.mating_pairs.copy() if dc.mating_pairs else [],
            duration_seconds=time.time() - start_time
        )
        
        # Save individual run results
        import json
        with open(run_output / "results.json", 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        result.to_dataframe().to_csv(run_output / "statistics.csv", index=False)
        
        # Clean up detailed outputs if not saving individual runs
        if not experiment_config.save_individual_runs:
            # Remove trajectory plots and other detailed outputs
            if figures_dir.exists():
                for file in figures_dir.glob('*'):
                    file.unlink()
            
            # Remove genotype/controller saves
            if results_dir.exists():
                for file in results_dir.glob('*'):
                    file.unlink()
            
            # Remove videos
            if video_dir.exists():
                for file in video_dir.glob('*'):
                    file.unlink()
        
        if verbose:
            print(f"\n[Worker {run_id}] {'='*60}")
            print(f"[Worker {run_id}] Run {run_id + 1} completed:")
            print(f"[Worker {run_id}]   Generations: {result.completed_generations}/{config.num_generations}")
            print(f"[Worker {run_id}]   Status: {result.stop_reason}")
            print(f"[Worker {run_id}]   Duration: {result.duration_seconds:.1f}s")
            print(f"[Worker {run_id}]   Final population: {result.population_size[-1] if result.population_size else 0}")
            print(f"[Worker {run_id}]   Best fitness: {max(result.fitness_best) if result.fitness_best else 0:.4f}")
            if not experiment_config.save_individual_runs:
                print(f"[Worker {run_id}]   Note: Detailed outputs not saved")
            print(f"[Worker {run_id}] {'='*60}\n")
        
        # Cleanup
        del spatial_ea
        gc.collect()
        
        return result
        
    except Exception as e:
        print(f"\n[Worker {run_id}] ERROR in run {run_id + 1}: {e}")
        traceback.print_exc()
        return None


# ============================================================================
# ExperimentRunner Class
# ============================================================================


class ExperimentRunner:
    """
    Manages multiple evolutionary algorithm runs with parameter variations.
    """
    
    def __init__(
        self,
        base_output_dir: str = "__experiments__",
        base_config_path: str | None = None
    ):
        """
        Initialize the experiment runner.
        
        Args:
            base_output_dir: Root directory for all experiment outputs
            base_config_path: Path to base configuration file (default: ea_config.yaml)
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Use absolute path based on module location if no path specified
        if base_config_path is None:
            module_dir = Path(__file__).parent
            self.base_config_path = str(module_dir / "ea_config.yaml")
        else:
            self.base_config_path = base_config_path
        self.experiments: dict[str, list[RunResult]] = {}
        self._current_max_pop_limit: int | None = None  # Track max_population_limit for current experiment/grid
    
    def _cap_population_value(self, pop: float, max_limit: int | None) -> float:
        """
        Cap a population value to the maximum population limit.
        
        This ensures visualizations don't show populations exceeding the configured limit,
        even if the raw data contains slightly higher values due to timing/recording.
        
        Args:
            pop: The population value to cap
            max_limit: The maximum population limit (None means no cap)
            
        Returns:
            The capped population value
        """
        if max_limit is None:
            return pop
        return min(pop, max_limit)
        
    def run_single_trial(
        self,
        experiment_config: ExperimentConfig,
        run_id: int,
        output_dir: Path,
        verbose: bool = True
    ) -> RunResult:
        """
        Run a single evolutionary trial.
        
        Args:
            experiment_config: Experiment configuration
            run_id: Run identifier
            output_dir: Output directory for this specific run
            verbose: Whether to print progress
            
        Returns:
            RunResult with statistics
        """
        import time
        from main_spatial_ea import SpatialEA
        
        start_time = time.time()
        
        # Create run-specific output directories
        run_output = output_dir / f"run_{run_id:03d}"
        run_output.mkdir(parents=True, exist_ok=True)
        
        # Override config paths for this run
        figures_dir = run_output / "figures"
        results_dir = run_output / "results"
        video_dir = run_output / "videos"
        
        figures_dir.mkdir(exist_ok=True)
        results_dir.mkdir(exist_ok=True)
        video_dir.mkdir(exist_ok=True)
        
        # Apply configuration overrides
        overrides = experiment_config.apply_to_config(config)
        
        # Temporarily modify config (save and restore)
        original_values = {}
        for key, value in overrides.items():
            parts = key.split('.')
            obj = config._config
            for part in parts[:-1]:
                obj = obj[part]
            original_values[key] = obj[parts[-1]]
            obj[parts[-1]] = value
        
        # Override output directories
        original_figures = config._config['output']['figures_folder']
        original_results = config._config['output']['results_folder']
        original_videos = config._config['output']['video_folder']
        
        config._config['output']['figures_folder'] = str(figures_dir)
        config._config['output']['results_folder'] = str(results_dir)
        config._config['output']['video_folder'] = str(video_dir)
        config._config['video']['save_generation_snapshots'] = experiment_config.save_snapshots
        config._config['video']['record_generation_videos'] = experiment_config.record_videos
        
        # Set random seed to ensure different runs have different random states
        # Use time-based seed if not specified to ensure variability across runs
        if experiment_config.random_seed is not None:
            seed = experiment_config.random_seed + run_id
        else:
            # Use a combination of time and run_id for a unique seed
            import time
            seed = int((time.time() * 1000000) % (2**31)) + run_id
        np.random.seed(seed)
        
        try:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Run {run_id + 1}/{experiment_config.num_runs}: {experiment_config.experiment_name}")
                print(f"{'='*60}")
            
            # Setup robot
            # NOTE: We skip compiling the robot spec here to avoid MuJoCo state
            # corruption. The SpatialEA will create and compile its own robot specs
            # during the simulation. We hardcode num_joints=8 for the gecko robot.
            num_joints = 8  # gecko robot has 8 actuated joints
            
            # Create and run EA
            spatial_ea = SpatialEA(
                population_size=config.population_size,
                num_generations=config.num_generations,
                num_joints=num_joints
            )
            
            spatial_ea.run_evolution()
            
            # Collect results from data collector
            dc = spatial_ea.data_collector
            
            result = RunResult(
                run_id=run_id,
                experiment_name=experiment_config.experiment_name,
                completed_generations=len(dc.generations),
                stopped_early=dc.stopped_early,
                stop_reason=dc.stop_reason if dc.stopped_early else "Completed",
                generations=dc.generations.copy(),
                population_size=dc.population_size.copy(),
                fitness_best=dc.fitness_best.copy(),
                fitness_avg=dc.fitness_avg.copy(),
                fitness_worst=dc.fitness_worst.copy(),
                fitness_std=dc.fitness_std.copy(),
                energy_avg=dc.energy_avg.copy() if dc.energy_avg else [],
                age_avg=dc.age_avg.copy() if dc.age_avg else [],
                mating_pairs=dc.mating_pairs.copy() if dc.mating_pairs else [],
                duration_seconds=time.time() - start_time
            )
            
            # Save individual run results (always save minimal stats)
            with open(run_output / "results.json", 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            
            result.to_dataframe().to_csv(run_output / "statistics.csv", index=False)
            
            # Clean up detailed outputs if not saving individual runs
            if not experiment_config.save_individual_runs:
                # Remove trajectory plots and other detailed outputs
                if figures_dir.exists():
                    for file in figures_dir.glob('*'):
                        file.unlink()
                    # Keep the directory but empty
                
                # Remove genotype/controller saves
                if results_dir.exists():
                    for file in results_dir.glob('*'):
                        file.unlink()
                
                # Remove videos
                if video_dir.exists():
                    for file in video_dir.glob('*'):
                        file.unlink()
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Run {run_id + 1} completed:")
                print(f"  Generations: {result.completed_generations}/{config.num_generations}")
                print(f"  Status: {result.stop_reason}")
                print(f"  Duration: {result.duration_seconds:.1f}s")
                print(f"  Final population: {result.population_size[-1] if result.population_size else 0}")
                print(f"  Best fitness: {max(result.fitness_best) if result.fitness_best else 0:.4f}")
                if not experiment_config.save_individual_runs:
                    print(f"  Note: Detailed outputs not saved (save_individual_runs=False)")
                print(f"{'='*60}\n")
            
            # Cleanup spatial_ea to release MuJoCo resources
            del spatial_ea
            
            # Force garbage collection to ensure cleanup
            gc.collect()
            
            return result
            
        finally:
            # Restore original config values
            for key, value in original_values.items():
                parts = key.split('.')
                obj = config._config
                for part in parts[:-1]:
                    obj = obj[part]
                obj[parts[-1]] = value
            
            config._config['output']['figures_folder'] = original_figures
            config._config['output']['results_folder'] = original_results
            config._config['output']['video_folder'] = original_videos
    
    def run_experiment(
        self,
        experiment_config: ExperimentConfig,
        verbose: bool = True
    ) -> list[RunResult]:
        """
        Run a complete experiment (multiple trials with same parameters).
        
        Args:
            experiment_config: Configuration for the experiment
            verbose: Whether to print progress
            
        Returns:
            List of RunResult objects
        """
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = self.base_output_dir / f"{experiment_config.experiment_name}_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Store max_population_limit for visualization capping
        self._current_max_pop_limit = experiment_config.max_population_limit
        
        # Save experiment configuration
        with open(experiment_dir / "experiment_config.json", 'w') as f:
            json.dump(experiment_config.to_dict(), f, indent=2)
        
        # Copy base config for reference
        shutil.copy2(self.base_config_path, experiment_dir / "base_config.yaml")
        
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# EXPERIMENT: {experiment_config.experiment_name}")
            print(f"# Runs: {experiment_config.num_runs}")
            print(f"# Output: {experiment_dir}")
            print(f"{'#'*60}\n")
        
        # Run all trials
        results = []
        for run_id in range(experiment_config.num_runs):
            try:
                result = self.run_single_trial(
                    experiment_config=experiment_config,
                    run_id=run_id,
                    output_dir=experiment_dir,
                    verbose=verbose
                )
                results.append(result)
            except Exception as e:
                print(f"ERROR in run {run_id + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Store results
        self.experiments[experiment_config.experiment_name] = results
        
        # Aggregate and visualize
        if results:
            aggregated = self.aggregate_results(results)
            self.save_aggregated_results(aggregated, experiment_dir)
            self.visualize_aggregated_results(aggregated, experiment_dir)
        
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# EXPERIMENT COMPLETE: {experiment_config.experiment_name}")
            print(f"# Successful runs: {len(results)}/{experiment_config.num_runs}")
            print(f"{'#'*60}\n")
        
        return results
    
    def run_experiment_parallel(
        self,
        experiment_config: ExperimentConfig,
        num_workers: int | None = None,
        verbose: bool = True
    ) -> list[RunResult]:
        """
        Run a complete experiment with PARALLEL execution of trials.
        
        Uses multiprocessing to run multiple independent trials simultaneously.
        Each trial runs in its own process with isolated memory space, avoiding
        MuJoCo state corruption issues.
        
        Args:
            experiment_config: Configuration for the experiment
            num_workers: Number of parallel workers (default: cpu_count - 1)
            verbose: Whether to print progress
            
        Returns:
            List of RunResult objects
        """
        from multiprocessing import Pool, cpu_count
        
        # Determine number of workers
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)  # Leave one core free
        
        num_workers = min(num_workers, experiment_config.num_runs)  # Don't spawn more than needed
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = self.base_output_dir / f"{experiment_config.experiment_name}_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Store max_population_limit for visualization capping
        self._current_max_pop_limit = experiment_config.max_population_limit
        
        # Save experiment configuration
        with open(experiment_dir / "experiment_config.json", 'w') as f:
            json.dump(experiment_config.to_dict(), f, indent=2)
        
        # Copy base config for reference
        shutil.copy2(self.base_config_path, experiment_dir / "base_config.yaml")
        
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# PARALLEL EXPERIMENT: {experiment_config.experiment_name}")
            print(f"# Runs: {experiment_config.num_runs}")
            print(f"# Workers: {num_workers}")
            print(f"# Output: {experiment_dir}")
            print(f"{'#'*60}\n")
        
        # Prepare arguments for each trial
        trial_args = []
        for run_id in range(experiment_config.num_runs):
            args = (
                experiment_config.to_dict(),
                run_id,
                str(experiment_dir),
                self.base_config_path,
                verbose
            )
            trial_args.append(args)
        
        # Run trials in parallel
        results = []
        pool = None
        try:
            # Set maxtasksperchild to recycle workers and prevent memory accumulation
            # Each worker will be recycled after processing 1 trial to avoid memory leaks
            pool = Pool(processes=num_workers, maxtasksperchild=1)
            if verbose:
                print(f"Starting {experiment_config.num_runs} trials across {num_workers} workers...")
                print(f"(Workers will be recycled after each trial to manage memory)")
                print("(Progress messages will appear as workers complete)\n")
            
            # Map trials to workers
            results_with_none = pool.map(_run_trial_subprocess, trial_args)
            
            # Filter out None results (failures)
            results = [r for r in results_with_none if r is not None]
            
            # IMPORTANT: Close pool before join to prevent workers from hanging
            pool.close()
            pool.join()
            
            # Force cleanup of any lingering processes
            if verbose:
                print("\nAll workers completed. Cleaning up...")
                
        except KeyboardInterrupt:
            print("\n\nParallel execution interrupted by user!")
            if pool:
                pool.terminate()
                pool.join()
            raise
        except Exception as e:
            print(f"\n\nError in parallel execution: {e}")
            if pool:
                pool.terminate()
                pool.join()
            import traceback
            traceback.print_exc()
        finally:
            # Ensure pool is cleaned up
            if pool is not None:
                try:
                    pool.terminate()
                    pool.join()
                except:
                    pass
        
        # Store results
        self.experiments[experiment_config.experiment_name] = results
        
        # Aggregate and visualize
        if results:
            aggregated = self.aggregate_results(results)
            self.save_aggregated_results(aggregated, experiment_dir)
            self.visualize_aggregated_results(aggregated, experiment_dir)
        
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# PARALLEL EXPERIMENT COMPLETE: {experiment_config.experiment_name}")
            print(f"# Successful runs: {len(results)}/{experiment_config.num_runs}")
            print(f"# Workers used: {num_workers}")
            if len(results) < experiment_config.num_runs:
                print(f"# WARNING: {experiment_config.num_runs - len(results)} runs failed")
            print(f"{'#'*60}\n")
        
        return results

    def run_grid_search(
        self,
        base_experiment: ExperimentConfig,
        param_grid: dict[str, list],
        num_workers: int | None = None,
        parallel_runs: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run a grid search over parameters by creating experiment variations
        from a base ExperimentConfig. The keys in `param_grid` should match
        attribute names on `ExperimentConfig` (for example: 'mutation_rate',
        'population_size', 'selection_method'). Each value should be a list
        of values to try.

        Returns a pandas DataFrame summarizing each grid point and resulting
        aggregated outcome metrics (num_completed, num_extinctions, num_explosions,
        max_generations_reached).
        """
        # Validate grid
        if not param_grid:
            raise ValueError("param_grid must be a non-empty dict")
        
        # Store max_population_limit from base experiment for visualization capping
        self._current_max_pop_limit = base_experiment.max_population_limit

        # Prepare output summary
        summary_rows = []

        # Build cartesian product of grid
        keys = list(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]

        combinations = list(itertools.product(*value_lists))

        if verbose:
            print(f"Running grid search with {len(combinations)} combinations...")

        for comb in combinations:
            # Create a copy of the base experiment config
            cfg_dict = copy.deepcopy(base_experiment.to_dict())
            # Apply combination values
            suffix_parts = []
            for k, v in zip(keys, comb):
                # set attribute on cfg_dict if present
                if k not in cfg_dict:
                    # allow setting nested attributes using dot-notation
                    if '.' in k:
                        # leave processing to apply_to_config when running
                        # instead we'll set via ExperimentConfig if possible
                        pass
                cfg_dict[k] = v
                suffix_parts.append(f"{k}={str(v)}")

            # Build experiment config object
            new_name = f"{base_experiment.experiment_name}_grid_{'_'.join(suffix_parts)}"
            cfg_dict['experiment_name'] = new_name

            new_exp = ExperimentConfig(**cfg_dict)

            if verbose:
                print(f"\n=== Grid point: {new_exp.experiment_name} ===")
                for k, v in zip(keys, comb):
                    print(f"  {k}: {v}")

            # Run experiment (parallel runs inside each experiment still honored)
            try:
                if parallel_runs:
                    results = self.run_experiment_parallel(new_exp, num_workers=num_workers, verbose=verbose)
                else:
                    results = self.run_experiment(new_exp, verbose=verbose)

                # Aggregate
                if results:
                    aggregated = self.aggregate_results(results)
                    row = {
                        **{k: v for k, v in zip(keys, comb)},
                        'experiment_name': new_exp.experiment_name,
                        'num_runs': aggregated.num_runs,
                        'num_completed': int(aggregated.num_completed),
                        'num_extinctions': int(aggregated.num_extinctions),
                        'num_explosions': int(aggregated.num_explosions),
                        'max_generations_reached': int(np.max(aggregated.generations[aggregated.runs_active > 0])) if np.any(aggregated.runs_active > 0) else 0,
                    }
                else:
                    row = {
                        **{k: v for k, v in zip(keys, comb)},
                        'experiment_name': new_exp.experiment_name,
                        'num_runs': new_exp.num_runs,
                        'num_completed': 0,
                        'num_extinctions': 0,
                        'num_explosions': 0,
                        'max_generations_reached': 0,
                    }

                summary_rows.append(row)

            except Exception as e:
                print(f"Error running grid point {new_exp.experiment_name}: {e}")
                import traceback
                traceback.print_exc()
                # record failure
                row = {
                    **{k: v for k, v in zip(keys, comb)},
                    'experiment_name': new_exp.experiment_name,
                    'num_runs': new_exp.num_runs,
                    'num_completed': 0,
                    'num_extinctions': 0,
                    'num_explosions': 0,
                    'max_generations_reached': 0,
                }
                summary_rows.append(row)
            
            # Force garbage collection between grid points to prevent memory accumulation
            gc.collect()
            if verbose:
                print(f"  Cleaned up memory after grid point")

        # Save summary to CSV
        df = pd.DataFrame(summary_rows)
        summary_path = self.base_output_dir / f"grid_search_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(summary_path, index=False)
        if verbose:
            print(f"\nGrid search complete. Summary saved to: {summary_path}")

        # Visualize grid search results
        try:
            visuals_dir = self.base_output_dir / f"grid_visuals_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            visuals_dir.mkdir(parents=True, exist_ok=True)

            # Choose primary metric for visualization
            metric = 'num_completed'

            # Simple visualizations depending on number of swept keys
            if len(keys) == 1:
                k = keys[0]
                plt.figure(figsize=(6, 4))
                if pd.api.types.is_numeric_dtype(df[k]):
                    df_sorted = df.sort_values(k)
                    plt.plot(df_sorted[k], df_sorted[metric], marker='o')
                    plt.xlabel(k)
                    plt.ylabel(metric)
                    plt.title(f"Grid: {k} vs {metric}")
                else:
                    # categorical
                    plt.bar(df[k].astype(str), df[metric])
                    plt.xlabel(k)
                    plt.ylabel(metric)
                    plt.title(f"Grid: {k} vs {metric}")
                    plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(visuals_dir / f"grid_{k}_vs_{metric}.png", dpi=200)
                plt.close()

            elif len(keys) == 2:
                kx, ky = keys[0], keys[1]
                # create pivot table for heatmap
                try:
                    pivot = df.pivot_table(index=kx, columns=ky, values=metric, aggfunc='mean')
                    # sort axes if numeric
                    ix = pivot.index
                    cx = pivot.columns
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(pivot.values, aspect='auto', origin='lower', cmap='viridis')
                    ax.set_xticks(range(len(cx)))
                    ax.set_yticks(range(len(ix)))
                    ax.set_xticklabels([str(c) for c in cx], rotation=45, ha='right')
                    ax.set_yticklabels([str(i) for i in ix])
                    ax.set_xlabel(ky)
                    ax.set_ylabel(kx)
                    fig.colorbar(im, ax=ax, label=metric)
                    ax.set_title(f"Grid heatmap: {kx} x {ky} ({metric})")
                    plt.tight_layout()
                    plt.savefig(visuals_dir / f"grid_heatmap_{kx}_x_{ky}_{metric}.png", dpi=200)
                    plt.close()
                except Exception:
                    # fallback scatter colored by metric
                    plt.figure(figsize=(6, 5))
                    plt.scatter(df[kx].astype(str), df[ky].astype(str), c=df[metric], cmap='viridis')
                    plt.colorbar(label=metric)
                    plt.xlabel(kx)
                    plt.ylabel(ky)
                    plt.title(f"Grid scatter: {kx} vs {ky} ({metric})")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(visuals_dir / f"grid_scatter_{kx}_vs_{ky}_{metric}.png", dpi=200)
                    plt.close()
                
                # --- NEW: Heatmaps for final population and final fitness ---
                try:
                    if verbose:
                        print(f"  Creating final population and fitness heatmaps for 2D grid...")
                    
                    # Build aggregated data for each parameter combination
                    heatmap_data = {}
                    for _, row in df.iterrows():
                        exp_name = row.get('experiment_name')
                        val_x = row.get(kx)
                        val_y = row.get(ky)
                        if exp_name is None:
                            continue
                        
                        key = (val_x, val_y)
                        if key not in heatmap_data:
                            heatmap_data[key] = {'pops': [], 'fits': []}
                        
                        runs = self.experiments.get(exp_name, [])
                        for r in runs:
                            if r.population_size:
                                # Cap population to max_population_limit
                                pop_val = self._cap_population_value(r.population_size[-1], self._current_max_pop_limit)
                                heatmap_data[key]['pops'].append(pop_val)
                            if r.fitness_best:
                                heatmap_data[key]['fits'].append(r.fitness_best[-1])
                    
                    if heatmap_data:
                        # Get unique sorted values for each parameter
                        try:
                            unique_x = sorted(set(k[0] for k in heatmap_data.keys()), key=float)
                        except:
                            unique_x = sorted(set(k[0] for k in heatmap_data.keys()))
                        try:
                            unique_y = sorted(set(k[1] for k in heatmap_data.keys()), key=float)
                        except:
                            unique_y = sorted(set(k[1] for k in heatmap_data.keys()))
                        
                        # Create matrices for population and fitness
                        pop_matrix = np.full((len(unique_x), len(unique_y)), np.nan)
                        fit_matrix = np.full((len(unique_x), len(unique_y)), np.nan)
                        
                        for i, vx in enumerate(unique_x):
                            for j, vy in enumerate(unique_y):
                                key = (vx, vy)
                                if key in heatmap_data:
                                    if heatmap_data[key]['pops']:
                                        pop_matrix[i, j] = np.mean(heatmap_data[key]['pops'])
                                    if heatmap_data[key]['fits']:
                                        fit_matrix[i, j] = np.mean(heatmap_data[key]['fits'])
                        
                        # 1. Final population heatmap
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(pop_matrix, aspect='auto', origin='lower', cmap='YlOrRd')
                        ax.set_xticks(range(len(unique_y)))
                        ax.set_yticks(range(len(unique_x)))
                        ax.set_xticklabels([str(v) for v in unique_y], rotation=45, ha='right')
                        ax.set_yticklabels([str(v) for v in unique_x])
                        ax.set_xlabel(ky)
                        ax.set_ylabel(kx)
                        cbar = fig.colorbar(im, ax=ax, label='Mean final population')
                        ax.set_title(f"Final Population Heatmap: {kx} x {ky}")
                        
                        # Add text annotations with values
                        for i in range(len(unique_x)):
                            for j in range(len(unique_y)):
                                if not np.isnan(pop_matrix[i, j]):
                                    text = ax.text(j, i, f'{pop_matrix[i, j]:.0f}',
                                                 ha="center", va="center", color="black", fontsize=8)
                        
                        plt.tight_layout()
                        plt.savefig(visuals_dir / f"grid_heatmap_final_population_{kx}_x_{ky}.png", dpi=200)
                        plt.close()
                        
                        # 2. Final fitness heatmap
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(fit_matrix, aspect='auto', origin='lower', cmap='RdYlGn')
                        ax.set_xticks(range(len(unique_y)))
                        ax.set_yticks(range(len(unique_x)))
                        ax.set_xticklabels([str(v) for v in unique_y], rotation=45, ha='right')
                        ax.set_yticklabels([str(v) for v in unique_x])
                        ax.set_xlabel(ky)
                        ax.set_ylabel(kx)
                        cbar = fig.colorbar(im, ax=ax, label='Mean final best fitness')
                        ax.set_title(f"Final Best Fitness Heatmap: {kx} x {ky}")
                        
                        # Add text annotations with values
                        for i in range(len(unique_x)):
                            for j in range(len(unique_y)):
                                if not np.isnan(fit_matrix[i, j]):
                                    text = ax.text(j, i, f'{fit_matrix[i, j]:.2f}',
                                                 ha="center", va="center", color="black", fontsize=8)
                        
                        plt.tight_layout()
                        plt.savefig(visuals_dir / f"grid_heatmap_final_fitness_{kx}_x_{ky}.png", dpi=200)
                        plt.close()
                        
                        if verbose:
                            print(f"  Final population and fitness heatmaps created!")
                    
                except Exception as e:
                    print(f"Warning: failed to create final population/fitness heatmaps: {e}")
                    import traceback
                    traceback.print_exc()

            else:
                # For >2 keys produce pairwise scatter plots for numeric keys colored by metric
                numeric_keys = [k for k in keys if pd.api.types.is_numeric_dtype(df[k])]
                if len(numeric_keys) >= 2:
                    # limit to first 4 numeric keys to avoid explosion
                    sel = numeric_keys[:4]
                    n = len(sel)
                    fig, axes = plt.subplots(n, n, figsize=(3*n, 3*n))
                    for i, xi in enumerate(sel):
                        for j, yj in enumerate(sel):
                            ax = axes[i, j]
                            if i == j:
                                ax.hist(df[xi].dropna(), bins=10, color='gray')
                                ax.set_xlabel(xi)
                            else:
                                sc = ax.scatter(df[xi], df[yj], c=df[metric], cmap='viridis', s=40)
                                if j == n-1:
                                    fig.colorbar(sc, ax=ax)
                            if i == n-1:
                                ax.set_xlabel(xi)
                            if j == 0:
                                ax.set_ylabel(yj)
                    plt.suptitle(f"Pairwise numeric grid scatter (colored by {metric})")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                    plt.savefig(visuals_dir / f"grid_pairwise_numeric_{metric}.png", dpi=200)
                    plt.close()

            if verbose:
                print(f"Grid visuals saved to: {visuals_dir}")
            # --- Additional distribution plots per-parameter to identify phase transitions ---
            try:
                for k in keys:
                    # Build groups of final populations AND final fitness per value for this parameter
                    groups_pop: dict[Any, list[float]] = {}
                    groups_fit: dict[Any, list[float]] = {}
                    for _, row in df.iterrows():
                        exp_name = row.get('experiment_name')
                        val = row.get(k)
                        if exp_name is None:
                            continue
                        runs = self.experiments.get(exp_name, [])
                        # Cap final populations to max_population_limit
                        final_pops = [self._cap_population_value(r.population_size[-1], self._current_max_pop_limit) 
                                      if r.population_size else 0 for r in runs]
                        final_fits = [r.fitness_best[-1] if r.fitness_best else 0.0 for r in runs]
                        groups_pop.setdefault(val, []).extend(final_pops)
                        groups_fit.setdefault(val, []).extend(final_fits)

                    # Skip if no data
                    if not groups_pop:
                        continue

                    # Prepare data for plotting (preserve sort order for numeric keys)
                    try:
                        sorted_items_pop = sorted(groups_pop.items(), key=lambda x: float(x[0]))
                        sorted_items_fit = sorted(groups_fit.items(), key=lambda x: float(x[0]))
                    except Exception:
                        # fallback to original insertion order
                        sorted_items_pop = list(groups_pop.items())
                        sorted_items_fit = list(groups_fit.items())

                    labels = [str(item[0]) for item in sorted_items_pop]
                    data_pop = [item[1] for item in sorted_items_pop]
                    data_fit = [item[1] for item in sorted_items_fit]

                    # Skip empty groups
                    if not any(len(d) for d in data_pop):
                        continue

                    safe_k = str(k).replace('/', '_').replace(' ', '_')
                    
                    # 1. Final population distribution
                    plt.figure(figsize=(8, 4))
                    plt.boxplot(data_pop, showmeans=True, patch_artist=True,
                                boxprops=dict(facecolor='lightblue', color='black'))
                    plt.xlabel(k)
                    plt.ylabel('Final population')
                    plt.title(f'Final population distribution across values of {k}')
                    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(visuals_dir / f"grid_param_distribution_pop_{safe_k}.png", dpi=200)
                    plt.close()
                    
                    # 2. Final fitness distribution
                    if any(len(d) for d in data_fit):
                        plt.figure(figsize=(8, 4))
                        plt.boxplot(data_fit, showmeans=True, patch_artist=True,
                                    boxprops=dict(facecolor='lightgreen', color='black'))
                        plt.xlabel(k)
                        plt.ylabel('Final best fitness')
                        plt.title(f'Final best fitness distribution across values of {k}')
                        plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
                        plt.tight_layout()
                        plt.savefig(visuals_dir / f"grid_param_distribution_fitness_{safe_k}.png", dpi=200)
                        plt.close()
                
                # --- NEW: Conditional distribution plots for multi-parameter grids ---
                if len(keys) >= 2:
                    if verbose:
                        print(f"  Generating conditional distribution plots for {len(keys)} parameters...")
                    
                    # For each parameter, create distributions conditioned on values of other parameters
                    for focal_param in keys:
                        other_params = [k for k in keys if k != focal_param]
                        
                        # Get unique values for other parameters
                        other_param_values = {}
                        for other_param in other_params:
                            vals = sorted(df[other_param].unique())
                            # Try numeric sort
                            try:
                                vals = sorted(vals, key=lambda x: float(x))
                            except:
                                pass
                            other_param_values[other_param] = vals
                        
                        # Generate plots for each combination of other parameter values
                        # For 2 params total: condition on the one other parameter
                        # For 3+ params: condition on each other parameter separately (marginalizing over rest)
                        if len(keys) == 2:
                            # Simple case: one conditioning variable
                            other_param = other_params[0]
                            for other_val in other_param_values[other_param]:
                                # Filter dataframe to this conditioning value
                                df_subset = df[df[other_param] == other_val]
                                
                                if len(df_subset) == 0:
                                    continue
                                
                                # Build groups for focal parameter (both pop and fitness)
                                groups_pop: dict[Any, list[float]] = {}
                                groups_fit: dict[Any, list[float]] = {}
                                for _, row in df_subset.iterrows():
                                    exp_name = row.get('experiment_name')
                                    focal_val = row.get(focal_param)
                                    if exp_name is None:
                                        continue
                                    runs = self.experiments.get(exp_name, [])
                                    # Cap final populations to max_population_limit
                                    final_pops = [self._cap_population_value(r.population_size[-1], self._current_max_pop_limit) 
                                                  if r.population_size else 0 for r in runs]
                                    final_fits = [r.fitness_best[-1] if r.fitness_best else 0.0 for r in runs]
                                    groups_pop.setdefault(focal_val, []).extend(final_pops)
                                    groups_fit.setdefault(focal_val, []).extend(final_fits)
                                
                                if not groups_pop:
                                    continue
                                
                                # Sort and plot
                                try:
                                    sorted_items_pop = sorted(groups_pop.items(), key=lambda x: float(x[0]))
                                    sorted_items_fit = sorted(groups_fit.items(), key=lambda x: float(x[0]))
                                except:
                                    sorted_items_pop = list(groups_pop.items())
                                    sorted_items_fit = list(groups_fit.items())
                                
                                labels = [str(item[0]) for item in sorted_items_pop]
                                data_pop = [item[1] for item in sorted_items_pop]
                                data_fit = [item[1] for item in sorted_items_fit]
                                
                                if not any(len(d) for d in data_pop):
                                    continue
                                
                                safe_focal = str(focal_param).replace('/', '_').replace(' ', '_')
                                safe_other = str(other_param).replace('/', '_').replace(' ', '_')
                                safe_val = str(other_val).replace('/', '_').replace(' ', '_')
                                
                                # Population plot
                                plt.figure(figsize=(8, 4))
                                plt.boxplot(data_pop, showmeans=True, patch_artist=True,
                                           boxprops=dict(facecolor='lightcoral', color='black'))
                                plt.xlabel(focal_param)
                                plt.ylabel('Final population')
                                plt.title(f'Final population across {focal_param} | {other_param}={other_val}')
                                plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
                                plt.tight_layout()
                                plt.savefig(visuals_dir / f"grid_conditional_pop_{safe_focal}_given_{safe_other}={safe_val}.png", dpi=200)
                                plt.close()
                                
                                # Fitness plot
                                if any(len(d) for d in data_fit):
                                    plt.figure(figsize=(8, 4))
                                    plt.boxplot(data_fit, showmeans=True, patch_artist=True,
                                               boxprops=dict(facecolor='lightgreen', color='black'))
                                    plt.xlabel(focal_param)
                                    plt.ylabel('Final best fitness')
                                    plt.title(f'Final best fitness across {focal_param} | {other_param}={other_val}')
                                    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
                                    plt.tight_layout()
                                    plt.savefig(visuals_dir / f"grid_conditional_fitness_{safe_focal}_given_{safe_other}={safe_val}.png", dpi=200)
                                    plt.close()
                        
                        else:
                            # For 3+ parameters: condition on each other parameter separately
                            for other_param in other_params:
                                for other_val in other_param_values[other_param]:
                                    # Filter to this conditioning value (marginalizing over remaining params)
                                    df_subset = df[df[other_param] == other_val]
                                    
                                    if len(df_subset) == 0:
                                        continue
                                    
                                    # Build groups for focal parameter (aggregating over remaining params) - both pop and fitness
                                    groups_pop: dict[Any, list[float]] = {}
                                    groups_fit: dict[Any, list[float]] = {}
                                    for _, row in df_subset.iterrows():
                                        exp_name = row.get('experiment_name')
                                        focal_val = row.get(focal_param)
                                        if exp_name is None:
                                            continue
                                        runs = self.experiments.get(exp_name, [])
                                        # Cap final populations to max_population_limit
                                        final_pops = [self._cap_population_value(r.population_size[-1], self._current_max_pop_limit) 
                                                      if r.population_size else 0 for r in runs]
                                        final_fits = [r.fitness_best[-1] if r.fitness_best else 0.0 for r in runs]
                                        groups_pop.setdefault(focal_val, []).extend(final_pops)
                                        groups_fit.setdefault(focal_val, []).extend(final_fits)
                                    
                                    if not groups_pop:
                                        continue
                                    
                                    # Sort and plot
                                    try:
                                        sorted_items_pop = sorted(groups_pop.items(), key=lambda x: float(x[0]))
                                        sorted_items_fit = sorted(groups_fit.items(), key=lambda x: float(x[0]))
                                    except:
                                        sorted_items_pop = list(groups_pop.items())
                                        sorted_items_fit = list(groups_fit.items())
                                    
                                    labels = [str(item[0]) for item in sorted_items_pop]
                                    data_pop = [item[1] for item in sorted_items_pop]
                                    data_fit = [item[1] for item in sorted_items_fit]
                                    
                                    if not any(len(d) for d in data_pop):
                                        continue
                                    
                                    # Get list of other param values that were marginalized
                                    remaining_params = [p for p in other_params if p != other_param]
                                    marg_info = ""
                                    if remaining_params:
                                        marg_info = f" (marginalizing {', '.join(remaining_params)})"
                                    
                                    safe_focal = str(focal_param).replace('/', '_').replace(' ', '_')
                                    safe_other = str(other_param).replace('/', '_').replace(' ', '_')
                                    safe_val = str(other_val).replace('/', '_').replace(' ', '_')
                                    
                                    # Population plot
                                    plt.figure(figsize=(8, 4))
                                    plt.boxplot(data_pop, showmeans=True, patch_artist=True,
                                               boxprops=dict(facecolor='lightcoral', color='black'))
                                    plt.xlabel(focal_param)
                                    plt.ylabel('Final population')
                                    plt.title(f'{focal_param} | {other_param}={other_val}{marg_info}')
                                    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
                                    plt.tight_layout()
                                    plt.savefig(visuals_dir / f"grid_conditional_pop_{safe_focal}_given_{safe_other}={safe_val}.png", dpi=200)
                                    plt.close()
                                    
                                    # Fitness plot
                                    if any(len(d) for d in data_fit):
                                        plt.figure(figsize=(8, 4))
                                        plt.boxplot(data_fit, showmeans=True, patch_artist=True,
                                                   boxprops=dict(facecolor='lightgreen', color='black'))
                                        plt.xlabel(focal_param)
                                        plt.ylabel('Final best fitness')
                                        plt.title(f'{focal_param} | {other_param}={other_val}{marg_info}')
                                        plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
                                        plt.tight_layout()
                                        plt.savefig(visuals_dir / f"grid_conditional_fitness_{safe_focal}_given_{safe_other}={safe_val}.png", dpi=200)
                                        plt.close()
                    
                    if verbose:
                        print(f"  Conditional distribution plots complete!")
                
            except Exception as e:
                print(f"Warning: failed to produce per-parameter distributions: {e}")
                # Also produce a stacked bar showing stop-reason fractions per parameter value
                try:
                    # For each swept key produce a stacked fraction bar of stop reasons
                    for k in keys:
                        # Get unique values from the dataframe for this key
                        raw_vals = list(df[k].dropna().unique())
                        # Try numeric sort, otherwise use string order
                        try:
                            raw_vals = sorted(raw_vals, key=lambda x: float(x))
                        except Exception:
                            raw_vals = [v for v in raw_vals]

                        values = []
                        completed_frac = []
                        extinct_frac = []
                        explode_frac = []
                        other_frac = []

                        for val in raw_vals:
                            rows = df[df[k] == val]
                            total = 0
                            c_cnt = e_cnt = x_cnt = o_cnt = 0
                            for _, row in rows.iterrows():
                                exp_name = row.get('experiment_name')
                                if exp_name is None:
                                    continue
                                runs = self.experiments.get(exp_name, [])
                                if not runs:
                                    continue
                                for r in runs:
                                    total += 1
                                    if not r.stopped_early:
                                        c_cnt += 1
                                    else:
                                        reason = (r.stop_reason or '').lower()
                                        if 'extinct' in reason or 'extinction' in reason:
                                            e_cnt += 1
                                        elif 'explod' in reason or 'maximum' in reason:
                                            x_cnt += 1
                                        else:
                                            o_cnt += 1

                            if total == 0:
                                continue

                            values.append(str(val))
                            completed_frac.append(c_cnt / total)
                            extinct_frac.append(e_cnt / total)
                            explode_frac.append(x_cnt / total)
                            other_frac.append(o_cnt / total)

                        if values:
                            ind = np.arange(len(values))
                            width = 0.6
                            fig, ax = plt.subplots(figsize=(max(6, len(values) * 0.8), 4))
                            ax.bar(ind, completed_frac, width, label='completed', color='C0')
                            ax.bar(ind, extinct_frac, width, bottom=completed_frac, label='extinction', color='C1')
                            bottom2 = np.array(completed_frac) + np.array(extinct_frac)
                            ax.bar(ind, explode_frac, width, bottom=bottom2, label='explosion', color='C3')
                            bottom3 = bottom2 + np.array(explode_frac)
                            if any(other_frac):
                                ax.bar(ind, other_frac, width, bottom=bottom3, label='other', color='C4')

                            ax.set_ylabel('Fraction of runs')
                            ax.set_xlabel(k)
                            ax.set_title(f'Stop-reason fractions across values of {k}')
                            ax.set_xticks(ind)
                            ax.set_xticklabels(values, rotation=45, ha='right')
                            ax.set_ylim([0, 1.05])
                            ax.legend()
                            safe_k = str(k).replace('/', '_').replace(' ', '_')
                            plt.tight_layout()
                            plt.savefig(visuals_dir / f"grid_param_stopfractions_{safe_k}.png", dpi=200)
                            plt.close()
                except Exception as e:
                    print(f"Warning: failed to produce stop-fraction stacked bars: {e}")
            except Exception as e:
                print(f"Warning: failed to produce per-parameter distributions: {e}")
        except Exception as e:
            print(f"Warning: failed to produce grid visuals: {e}")

        return df
    
    def aggregate_results(
        self,
        results: list[RunResult],
        padding_strategy: str = "forward_fill"
    ) -> AggregatedResults:
        """
        Aggregate statistics across multiple runs.
        
        Handles variable-length runs due to early stopping with configurable padding:
        - 'nan': Use NaN for missing values (only average active runs)
        - 'forward_fill': Carry last observed value forward (default)
        - 'terminal_state': Use 0 for extinction, max_limit for explosion
        
        Args:
            results: List of RunResult objects
            padding_strategy: How to handle early-stopped runs
            
        Returns:
            AggregatedResults object
        """
        if not results:
            raise ValueError("No results to aggregate")
        
        # Find maximum generation reached
        max_gens = max(len(r.generations) for r in results)
        generations = np.arange(max_gens)
        
        # Initialize arrays for each statistic
        population_data = np.full((len(results), max_gens), np.nan)
        fitness_best_data = np.full((len(results), max_gens), np.nan)
        fitness_avg_data = np.full((len(results), max_gens), np.nan)
        
        # Track which runs are actually active (not padded) at each generation
        # This is separate from the data arrays which may be padded for visualization
        truly_active = np.zeros((len(results), max_gens), dtype=bool)
        
        # Track early stopping
        num_extinctions = 0
        num_explosions = 0
        num_completed = 0
        
        # Fill data arrays
        for i, result in enumerate(results):
            n_gens = len(result.generations)
            population_data[i, :n_gens] = result.population_size
            fitness_best_data[i, :n_gens] = result.fitness_best
            fitness_avg_data[i, :n_gens] = result.fitness_avg
            
            # Mark generations where this run was truly active (actually running)
            truly_active[i, :n_gens] = True
            
            # Apply padding strategy for early-stopped runs
            if n_gens < max_gens and padding_strategy != "nan":
                if padding_strategy == "forward_fill":
                    # Carry last observed value forward
                    population_data[i, n_gens:] = result.population_size[-1] if result.population_size else np.nan
                    fitness_best_data[i, n_gens:] = result.fitness_best[-1] if result.fitness_best else np.nan
                    fitness_avg_data[i, n_gens:] = result.fitness_avg[-1] if result.fitness_avg else np.nan
                    
                elif padding_strategy == "terminal_state":
                    # Use terminal state based on stop reason
                    if "extinction" in result.stop_reason.lower():
                        population_data[i, n_gens:] = 0
                        fitness_best_data[i, n_gens:] = 0
                        fitness_avg_data[i, n_gens:] = 0
                    elif "maximum" in result.stop_reason.lower():
                        # For explosion, use the max limit if available
                        # Otherwise forward-fill
                        population_data[i, n_gens:] = result.population_size[-1] if result.population_size else np.nan
                        fitness_best_data[i, n_gens:] = result.fitness_best[-1] if result.fitness_best else np.nan
                        fitness_avg_data[i, n_gens:] = result.fitness_avg[-1] if result.fitness_avg else np.nan
            
            # Categorize stopping reason
            if result.stopped_early:
                if "extinction" in result.stop_reason.lower():
                    num_extinctions += 1
                elif "maximum" in result.stop_reason.lower():
                    num_explosions += 1
            else:
                num_completed += 1
        
        # Compute statistics (ignoring NaN)
        with np.errstate(invalid='ignore'):  # Suppress warnings for all-NaN slices
            population_mean = np.nanmean(population_data, axis=0)
            population_std = np.nanstd(population_data, axis=0)
            population_min = np.nanmin(population_data, axis=0)
            population_max = np.nanmax(population_data, axis=0)
            
            fitness_best_mean = np.nanmean(fitness_best_data, axis=0)
            fitness_best_std = np.nanstd(fitness_best_data, axis=0)
            fitness_avg_mean = np.nanmean(fitness_avg_data, axis=0)
            fitness_avg_std = np.nanstd(fitness_avg_data, axis=0)
            
            # Count truly active runs at each generation (not padded values)
            runs_active = np.sum(truly_active, axis=0)
            completion_rate = runs_active / len(results)
        
        return AggregatedResults(
            experiment_name=results[0].experiment_name,
            num_runs=len(results),
            generations=generations,
            population_mean=population_mean,
            population_std=population_std,
            population_min=population_min,
            population_max=population_max,
            fitness_best_mean=fitness_best_mean,
            fitness_best_std=fitness_best_std,
            fitness_avg_mean=fitness_avg_mean,
            fitness_avg_std=fitness_avg_std,
            runs_active=runs_active,
            completion_rate=completion_rate,
            num_extinctions=num_extinctions,
            num_explosions=num_explosions,
            num_completed=num_completed
        )
    
    def save_aggregated_results(
        self,
        aggregated: AggregatedResults,
        output_dir: Path
    ) -> None:
        """Save aggregated results to files."""
        # Save as CSV
        df = aggregated.to_dataframe()
        df.to_csv(output_dir / "aggregated_statistics.csv", index=False)
        
        # Save summary statistics
        summary = {
            'experiment_name': aggregated.experiment_name,
            'num_runs': aggregated.num_runs,
            'num_completed': aggregated.num_completed,
            'num_extinctions': aggregated.num_extinctions,
            'num_explosions': aggregated.num_explosions,
            'max_generations_reached': int(np.max(aggregated.generations[aggregated.runs_active > 0])),
            'final_population_mean': float(aggregated.population_mean[-1]) if not np.isnan(aggregated.population_mean[-1]) else None,
            'best_fitness_achieved': float(np.nanmax(aggregated.fitness_best_mean)),
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def visualize_aggregated_results(
        self,
        aggregated: AggregatedResults,
        output_dir: Path
    ) -> None:
        """Create comprehensive visualization of aggregated results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Experiment: {aggregated.experiment_name}\n"
                    f"({aggregated.num_runs} runs, "
                    f"{aggregated.num_completed} completed, "
                    f"{aggregated.num_extinctions} extinct, "
                    f"{aggregated.num_explosions} exploded)",
                    fontsize=14, fontweight='bold')
        
        gens = aggregated.generations
        
        # Plot 1: Population size over time
        ax = axes[0, 0]
        ax.plot(gens, aggregated.population_mean, 'b-', linewidth=2, label='Mean')
        ax.fill_between(
            gens,
            aggregated.population_mean - aggregated.population_std,
            aggregated.population_mean + aggregated.population_std,
            alpha=0.3, color='b', label='±1 SD'
        )
        ax.plot(gens, aggregated.population_min, 'r--', alpha=0.5, label='Min')
        ax.plot(gens, aggregated.population_max, 'g--', alpha=0.5, label='Max')
        
        # Add vertical lines at points where runs start dropping out
        if aggregated.num_extinctions + aggregated.num_explosions > 0:
            # Find where completion rate drops below certain thresholds
            for threshold in [0.75, 0.5, 0.25]:
                drop_idx = np.where(aggregated.completion_rate <= threshold)[0]
                if len(drop_idx) > 0 and drop_idx[0] > 0:
                    ax.axvline(gens[drop_idx[0]], color='gray', linestyle=':', alpha=0.4, linewidth=1)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Population Size')
        ax.set_title('Population Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Fitness over time
        ax = axes[0, 1]
        ax.plot(gens, aggregated.fitness_best_mean, 'g-', linewidth=2, label='Best (mean)')
        ax.fill_between(
            gens,
            aggregated.fitness_best_mean - aggregated.fitness_best_std,
            aggregated.fitness_best_mean + aggregated.fitness_best_std,
            alpha=0.3, color='g', label='Best ±1 SD'
        )
        ax.plot(gens, aggregated.fitness_avg_mean, 'b-', linewidth=2, label='Avg (mean)')
        ax.fill_between(
            gens,
            aggregated.fitness_avg_mean - aggregated.fitness_avg_std,
            aggregated.fitness_avg_mean + aggregated.fitness_avg_std,
            alpha=0.3, color='b', label='Avg ±1 SD'
        )
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Run completion rate
        ax = axes[1, 0]
        ax.plot(gens, aggregated.completion_rate * 100, 'r-', linewidth=2)
        ax.fill_between(gens, 0, aggregated.completion_rate * 100, alpha=0.3, color='r')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Runs Active (%)')
        ax.set_title('Run Survival Rate')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        
        # Add annotation for key milestones
        for pct in [75, 50, 25]:
            idx = np.where(aggregated.completion_rate <= pct/100)[0]
            if len(idx) > 0:
                gen = gens[idx[0]]
                ax.axvline(gen, color='gray', linestyle='--', alpha=0.5)
                ax.text(gen, pct, f' {pct}%', fontsize=9, va='center')
        
        # Plot 4: Active runs count
        ax = axes[1, 1]
        ax.bar(gens, aggregated.runs_active, color='steelblue', alpha=0.7)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Number of Active Runs')
        ax.set_title('Active Runs per Generation')
        ax.set_ylim([0, aggregated.num_runs * 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / "aggregated_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        # ------------------------------------------------------------------
        # Additional distributions across individual runs (final population
        # and generations reached). These use the stored per-run results in
        # self.experiments (if available) for the given experiment name.
        # ------------------------------------------------------------------
        try:
            runs = self.experiments.get(aggregated.experiment_name, [])
            if runs:
                # Final population distribution - cap values to max_population_limit
                final_pops = [self._cap_population_value(r.population_size[-1], self._current_max_pop_limit) 
                              if r.population_size else 0 for r in runs]
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(final_pops, bins=min(20, max(1, int(np.nanmax(final_pops) - np.nanmin(final_pops) + 1))),
                        color='C2', alpha=0.85)
                ax.axvline(np.mean(final_pops), color='k', linestyle='--', label=f"mean={np.mean(final_pops):.1f}")
                ax.axvline(np.median(final_pops), color='r', linestyle=':', label=f"median={np.median(final_pops):.1f}")
                ax.set_xlabel('Final population size')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of final populations across runs')
                ax.legend()
                plt.tight_layout()
                plt.savefig(output_dir / "final_population_distribution.png", dpi=200)
                plt.close()

                # Generations reached distribution
                gens_completed = [r.completed_generations for r in runs]
                # Use bins that reflect generation integers
                max_gen = int(max(gens_completed)) if gens_completed else 0
                bins = list(range(0, max_gen + 2)) if max_gen > 0 else 1
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(gens_completed, bins=bins, color='C3', alpha=0.85, align='left')
                ax.set_xlabel('Generations reached')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of generations reached across runs')
                plt.tight_layout()
                plt.savefig(output_dir / "generations_distribution.png", dpi=200)
                plt.close()
                # ------------------------------------------------------------------
                # Additional recommended visuals
                # 1) Stop-reason breakdown (pie chart)
                # 2) Scatter: final population vs best fitness (per run)
                # ------------------------------------------------------------------
                try:
                    # Stop reasons
                    stop_counts = {'completed': 0, 'extinction': 0, 'explosion': 0, 'other': 0}
                    for r in runs:
                        if not r.stopped_early:
                            stop_counts['completed'] += 1
                        else:
                            reason = (r.stop_reason or '').lower()
                            if 'extinct' in reason or 'extinction' in reason:
                                stop_counts['extinction'] += 1
                            elif 'explod' in reason or 'maximum' in reason:
                                stop_counts['explosion'] += 1
                            else:
                                stop_counts['other'] += 1

                    labels = []
                    sizes = []
                    for k, v in stop_counts.items():
                        if v > 0:
                            labels.append(k)
                            sizes.append(v)

                    if sizes:
                        fig, ax = plt.subplots(figsize=(5, 5))
                        ax.pie(sizes, labels=labels, autopct='%1.0f', startangle=90, colors=plt.cm.Set2.colors)
                        ax.set_title('Run stop reasons')
                        plt.tight_layout()
                        plt.savefig(output_dir / "stop_reasons_pie.png", dpi=200)
                        plt.close()

                    # Scatter: final population vs best fitness
                    final_pops = []
                    best_fitness = []
                    for r in runs:
                        if r.population_size:
                            final_pops.append(r.population_size[-1])
                        else:
                            final_pops.append(0)
                        if r.fitness_best:
                            try:
                                best_fitness.append(max(r.fitness_best))
                            except Exception:
                                best_fitness.append(np.nan)
                        else:
                            best_fitness.append(np.nan)

                    if final_pops and any(np.isfinite(best_fitness)):
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.scatter(final_pops, best_fitness, c='C4', alpha=0.8)
                        ax.set_xlabel('Final population')
                        ax.set_ylabel('Best fitness (per run)')
                        ax.set_title('Final population vs best fitness')
                        # annotate mean points
                        if final_pops:
                            ax.axvline(np.mean(final_pops), color='k', linestyle='--', alpha=0.6)
                        plt.tight_layout()
                        plt.savefig(output_dir / "finalpop_vs_bestfitness.png", dpi=200)
                        plt.close()
                except Exception as e:
                    print(f"Warning: failed to produce additional recommended visuals: {e}")
        except Exception as e:
            print(f"Warning: failed to produce per-run distributions: {e}")
    
    def analyze_genotype_clustering(
        self,
        experiment_dir: Path | str,
        generation: int | None = None,
        **kwargs
    ) -> None:
        """
        Analyze genotype clustering for a completed experiment.
        
        This automatically performs clustering analysis using DBSCAN and t-SNE
        for ALL 4 distance types (structural, weight, combined, behavioral).
        
        Args:
            experiment_dir: Path to experiment output directory
            generation: Which generation to analyze (None = final generation)
            **kwargs: Additional arguments for clustering/reduction methods
            
        Saves (for each distance type):
            - Spatial cluster plots
            - Cluster embedding visualizations (t-SNE)
            - Distance heatmaps
            - Clustering metrics report
        """
        from genotype_distance import compute_pairwise_distance_matrix, compute_genotype_diversity
        from genotype_clustering import (
            cluster_dbscan,
            reduce_dimensions_tsne,
            analyze_spatial_clustering
        )
        from clustering_visualization import (
            plot_spatial_clusters,
            plot_cluster_embedding,
            plot_distance_heatmap
        )
        
        experiment_dir = Path(experiment_dir)
        
        # Find all run directories
        run_dirs = sorted(experiment_dir.glob("run_*"))
        if not run_dirs:
            print(f"No run directories found in {experiment_dir}")
            return
        
        print(f"\n{'='*70}")
        print(f"GENOTYPE CLUSTERING ANALYSIS")
        print(f"{'='*70}")
        print(f"Experiment: {experiment_dir.name}")
        print(f"Found {len(run_dirs)} runs")
        print(f"Method: DBSCAN clustering + t-SNE dimensionality reduction")
        print(f"Distance types: structural, weight, combined, behavioral")
        print(f"{'='*70}\n")
        
        # Aggregate genotypes and positions from all runs
        all_genotypes = []
        all_positions = []
        all_fitness = []
        all_ids = []
        all_run_labels = []
        
        for run_idx, run_dir in enumerate(run_dirs):
            # Load final generation data
            results_file = run_dir / "results.json"
            if not results_file.exists():
                continue
            
            with open(results_file, 'r') as f:
                run_data = json.load(f)
            
            # Load from NPZ if available (has genotypes)
            npz_files = list((run_dir / "results").glob("final_genotypes_*.npz"))
            if not npz_files:
                print(f"  Skipping {run_dir.name}: no genotype data found")
                continue
            
            from main_spatial_ea import SpatialEA
            npz_data = SpatialEA.load_genotypes_from_npz(str(npz_files[0]))
            
            # Extract data
            genotypes = npz_data['genotypes']
            positions = npz_data['positions']
            fitness = npz_data.get('fitness', np.ones(len(genotypes)))
            
            # Convert genotypes from numpy format to dictionaries
            for i, (genotype_data, pos, fit) in enumerate(zip(genotypes, positions, fitness)):
                # Parse the genotype data structure
                genotype_dict = {
                    'nodes': genotype_data.get('nodes', []),
                    'connections': genotype_data.get('connections', []),
                    'innovation_numbers': genotype_data.get('innovation_numbers', [])
                }
                
                all_genotypes.append(genotype_dict)
                all_positions.append(pos)
                all_fitness.append(fit)
                all_ids.append(f"run{run_idx:03d}_ind{i:03d}")
                all_run_labels.append(run_idx)
            
            print(f"  {run_dir.name}: {len(genotypes)} individuals")
        
        if not all_genotypes:
            print("\nNo genotype data found across all runs. Skipping clustering analysis.")
            return
        
        print(f"\nTotal individuals collected: {len(all_genotypes)}")
        
        # Check if we have enough individuals for meaningful clustering
        n_individuals = len(all_genotypes)
        if n_individuals < 3:
            print(f"\nWARNING: Only {n_individuals} individuals found.")
            print("Clustering analysis requires at least 3 individuals.")
            print("This likely means the population went extinct early or runs failed.")
            
            # Still create a summary report explaining the situation
            clustering_dir = experiment_dir / "clustering_analysis"
            clustering_dir.mkdir(exist_ok=True)
            
            summary_path = clustering_dir / "README.txt"
            with open(summary_path, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("CLUSTERING ANALYSIS - INSUFFICIENT DATA\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Experiment: {experiment_dir.name}\n")
                f.write(f"Individuals found: {n_individuals}\n")
                f.write(f"Runs analyzed: {len(run_dirs)}\n\n")
                f.write("ISSUE\n")
                f.write("-" * 70 + "\n")
                f.write("Clustering analysis requires at least 3 individuals.\n\n")
                f.write("Possible causes:\n")
                f.write("  - Population went extinct early in all runs\n")
                f.write("  - Experiment stopped before completion\n")
                f.write("  - Genotype saving was disabled (save_individual_runs=False)\n")
                f.write("  - Configuration error preventing genotype saves\n\n")
                f.write("To enable clustering analysis:\n")
                f.write("  1. Ensure experiments run to completion\n")
                f.write("  2. Set save_individual_runs=True in experiment config\n")
                f.write("  3. Check that populations don't go extinct\n")
                f.write("  4. Adjust selection/mutation parameters if needed\n")
            
            print(f"Summary saved to: {summary_path}")
            return
        
        if n_individuals < 10:
            print(f"\nNOTE: Only {n_individuals} individuals found.")
            print("Clustering analysis may be limited with small populations.")
            print("Results should be interpreted cautiously.")
        
        # Convert to numpy arrays
        all_positions = np.array(all_positions)
        all_fitness = np.array(all_fitness)
        all_run_labels = np.array(all_run_labels)
        
        # Create base output directory for clustering analysis
        clustering_dir = experiment_dir / "clustering_analysis"
        clustering_dir.mkdir(exist_ok=True)
        
        # Analyze for each distance type
        distance_types = ['structural', 'weight', 'combined', 'behavioral']
        all_results = {}
        
        for dist_type in distance_types:
            print(f"\n{'-'*70}")
            print(f"Analyzing with {dist_type.upper()} distance...")
            print(f"{'-'*70}")
            
            try:
                # Create subdirectory for this distance type
                dist_dir = clustering_dir / dist_type
                dist_dir.mkdir(exist_ok=True)
                
                # Compute distance matrix
                print(f"  Computing pairwise distances...")
                distance_matrix = compute_pairwise_distance_matrix(
                    all_genotypes,
                    distance_type=dist_type
                )
                
                # Compute diversity metrics
                diversity = compute_genotype_diversity(distance_matrix)
                
                # Perform DBSCAN clustering
                print(f"  Performing DBSCAN clustering...")
                clustering_result = cluster_dbscan(
                    distance_matrix,
                    eps=kwargs.get('eps', 0.3),
                    min_samples=max(2, min(kwargs.get('min_samples', 2), n_individuals // 3))
                )
                
                # Check clustering quality
                if clustering_result.n_clusters == 0:
                    print(f"  WARNING: No clusters found (all noise). Population may be too diverse or eps too small.")
                elif clustering_result.n_clusters == 1:
                    print(f"  WARNING: Only 1 cluster found. Population may be homogeneous or eps too large.")
                
                # Reduce dimensions with t-SNE
                print(f"  Reducing dimensions with t-SNE...")
                # t-SNE requires perplexity < n_samples / 3
                safe_perplexity = max(2, min(30, (n_individuals - 1) // 3))
                
                if n_individuals < 5:
                    print(f"  WARNING: t-SNE may not work well with < 5 individuals")
                
                coords_2d = reduce_dimensions_tsne(
                    distance_matrix=distance_matrix,
                    perplexity=safe_perplexity
                )
                
                # Analyze spatial clustering
                print(f"  Analyzing spatial distribution...")
                spatial_stats = analyze_spatial_clustering(
                    positions=all_positions,
                    cluster_labels=clustering_result.cluster_labels
                )
                
                # Create visualizations
                print(f"  Creating visualizations...")
                
                # 1. Spatial clusters
                plot_spatial_clusters(
                    positions=all_positions,
                    cluster_labels=clustering_result.cluster_labels,
                    fitness_values=all_fitness,
                    save_path=str(dist_dir / "spatial_clusters.png"),
                    title=f"Spatial Clusters ({dist_type} distance)"
                )
                
                # 2. Cluster embedding (t-SNE)
                plot_cluster_embedding(
                    reduced_coords=coords_2d,
                    cluster_labels=clustering_result.cluster_labels,
                    fitness_values=all_fitness,
                    reduction_method="t-SNE",
                    save_path=str(dist_dir / "cluster_embedding_tsne.png"),
                    title=f"Cluster Embedding - t-SNE ({dist_type} distance)"
                )
                
                # 3. Distance heatmap
                plot_distance_heatmap(
                    distance_matrix=distance_matrix,
                    cluster_labels=clustering_result.cluster_labels,
                    save_path=str(dist_dir / "distance_heatmap.png"),
                    title=f"Genotype Distance Heatmap ({dist_type})"
                )
                
                # Save results summary
                summary_path = dist_dir / "clustering_summary.txt"
                with open(summary_path, 'w') as f:
                    f.write("=" * 70 + "\n")
                    f.write(f"CLUSTERING ANALYSIS - {dist_type.upper()} DISTANCE\n")
                    f.write("=" * 70 + "\n\n")
                    
                    f.write(f"Experiment: {experiment_dir.name}\n")
                    f.write(f"Total individuals: {len(all_genotypes)}\n")
                    f.write(f"From {len(run_dirs)} runs\n\n")
                    
                    f.write("METHOD\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Distance metric: {dist_type}\n")
                    f.write(f"Clustering: DBSCAN\n")
                    f.write(f"Dimensionality reduction: t-SNE\n\n")
                    
                    f.write("CLUSTERING RESULTS\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Number of clusters: {clustering_result.n_clusters}\n")
                    f.write(f"Number of noise points: {clustering_result.n_noise}\n")
                    
                    # Handle None values gracefully
                    sil = clustering_result.silhouette
                    ch = clustering_result.calinski_harabasz
                    db = clustering_result.davies_bouldin
                    
                    f.write(f"Silhouette score: {sil:.4f if sil is not None else 'N/A (requires 2+ clusters)'}\n")
                    f.write(f"Calinski-Harabasz index: {ch:.4f if ch is not None else 'N/A'}\n")
                    f.write(f"Davies-Bouldin index: {db:.4f if db is not None else 'N/A'}\n\n")
                    
                    # Add interpretation note if metrics are None
                    if sil is None or clustering_result.n_clusters < 2:
                        f.write("NOTE: Quality metrics unavailable - need at least 2 clusters\n")
                        if clustering_result.n_clusters == 0:
                            f.write("All individuals classified as noise. Consider:\n")
                            f.write("  - Increasing DBSCAN eps parameter\n")
                            f.write("  - Decreasing min_samples parameter\n")
                            f.write("  - Population may be truly diverse\n")
                        elif clustering_result.n_clusters == 1:
                            f.write("Only 1 cluster found. Population appears homogeneous.\n")
                        f.write("\n")
                    
                    f.write("CLUSTER SIZES\n")
                    f.write("-" * 70 + "\n")
                    for cluster_id, size in clustering_result.cluster_sizes.items():
                        if cluster_id == -1:
                            f.write(f"  Noise points: {size}\n")
                        else:
                            f.write(f"  Cluster {cluster_id}: {size} individuals\n")
                    f.write("\n")
                    
                    f.write("DIVERSITY METRICS\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Mean pairwise distance: {diversity['mean_distance']:.4f}\n")
                    f.write(f"Std pairwise distance: {diversity['std_distance']:.4f}\n")
                    f.write(f"Min pairwise distance: {diversity['min_distance']:.4f}\n")
                    f.write(f"Max pairwise distance: {diversity['max_distance']:.4f}\n\n")
                    
                    f.write("SPATIAL STATISTICS\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Spatial segregation: {spatial_stats['segregation']:.4f}\n")
                    f.write(f"  (0 = mixed, 1 = segregated)\n\n")
                    
                    for cluster_id, stats in spatial_stats['cluster_stats'].items():
                        if cluster_id == -1:
                            f.write(f"Noise points:\n")
                        else:
                            f.write(f"Cluster {cluster_id}:\n")
                        f.write(f"  Centroid: ({stats['centroid'][0]:.2f}, {stats['centroid'][1]:.2f})\n")
                        f.write(f"  Spatial spread: {stats['spread']:.2f}\n")
                        f.write(f"  Count: {stats['count']}\n\n")
                
                # Save metrics as JSON
                metrics_path = dist_dir / "clustering_metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump({
                        'distance_type': dist_type,
                        'n_clusters': clustering_result.n_clusters,
                        'n_noise': clustering_result.n_noise,
                        'silhouette': float(clustering_result.silhouette) if clustering_result.silhouette is not None else None,
                        'calinski_harabasz': float(clustering_result.calinski_harabasz) if clustering_result.calinski_harabasz is not None else None,
                        'davies_bouldin': float(clustering_result.davies_bouldin) if clustering_result.davies_bouldin is not None else None,
                        'cluster_sizes': {int(k): int(v) for k, v in clustering_result.cluster_sizes.items()},
                        'diversity': {k: float(v) for k, v in diversity.items()},
                        'spatial_segregation': float(spatial_stats['segregation'])
                    }, f, indent=2)
                
                all_results[dist_type] = {
                    'clustering': clustering_result,
                    'diversity': diversity,
                    'spatial_stats': spatial_stats
                }
                
                # Format silhouette score safely
                sil_str = f"{clustering_result.silhouette:.3f}" if clustering_result.silhouette is not None else "N/A"
                print(f"  ✓ {dist_type}: {clustering_result.n_clusters} clusters, "
                      f"silhouette={sil_str}")
                
            except Exception as e:
                print(f"  ✗ Error analyzing {dist_type} distance: {e}")
                import traceback
                traceback.print_exc()
        
        # Create comprehensive summary
        main_summary_path = clustering_dir / "README.txt"
        with open(main_summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("GENOTYPE CLUSTERING ANALYSIS - ALL DISTANCE TYPES\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Experiment: {experiment_dir.name}\n")
            f.write(f"Total individuals: {len(all_genotypes)}\n")
            f.write(f"From {len(run_dirs)} runs\n\n")
            
            f.write("ANALYSIS OVERVIEW\n")
            f.write("-" * 70 + "\n")
            f.write("This directory contains clustering analysis results for all 4 distance types:\n\n")
            f.write("  structural/  - Network topology similarity\n")
            f.write("  weight/      - Connection weight similarity\n")
            f.write("  combined/    - Weighted mix of structural + weight\n")
            f.write("  behavioral/  - Fitness and trajectory similarity\n\n")
            
            f.write("QUICK COMPARISON\n")
            f.write("-" * 70 + "\n")
            
            if all_results:
                f.write(f"{'Distance Type':<15} {'Clusters':<10} {'Silhouette':<12} {'Segregation':<12}\n")
                f.write("-" * 70 + "\n")
                
                for dist_type in distance_types:
                    if dist_type in all_results:
                        res = all_results[dist_type]
                        sil = res['clustering'].silhouette
                        sil_str = f"{sil:.4f}" if sil is not None else "N/A"
                        f.write(f"{dist_type:<15} "
                               f"{res['clustering'].n_clusters:<10} "
                               f"{sil_str:<12} "
                               f"{res['spatial_stats']['segregation']:<12.4f}\n")
                    else:
                        f.write(f"{dist_type:<15} {'FAILED':<10} {'N/A':<12} {'N/A':<12}\n")
            else:
                f.write("No distance types were successfully analyzed.\n")
                f.write("Check error messages above for details.\n")
            
            f.write("\n")
            
            # Add warnings if population was small
            if n_individuals < 10:
                f.write("WARNINGS\n")
                f.write("-" * 70 + "\n")
                f.write(f"Small population size ({n_individuals} individuals)\n")
                f.write("Results may not be statistically reliable.\n\n")
                if n_individuals < 5:
                    f.write("CRITICAL: Population size < 5\n")
                    f.write("Clustering analysis is not meaningful with so few individuals.\n")
                    f.write("Consider:\n")
                    f.write("  - Increasing population size\n")
                    f.write("  - Adjusting selection pressure\n")
                    f.write("  - Preventing early extinction\n\n")
            
            # Check for clustering issues
            all_noise = all([res['clustering'].n_clusters == 0 
                            for res in all_results.values()])
            all_single = all([res['clustering'].n_clusters == 1 
                             for res in all_results.values()])
            
            if all_noise:
                f.write("OBSERVATION: All distance types found 0 clusters (all noise)\n")
                f.write("-" * 70 + "\n")
                f.write("This suggests the population is highly diverse with no clear groupings.\n")
                f.write("Possible causes:\n")
                f.write("  - High mutation rate preventing convergence\n")
                f.write("  - Weak selection pressure\n")
                f.write("  - Random mating preventing local adaptation\n")
                f.write("  - DBSCAN eps parameter too small (try larger eps)\n\n")
            elif all_single:
                f.write("OBSERVATION: All distance types found only 1 cluster\n")
                f.write("-" * 70 + "\n")
                f.write("This suggests the population is very homogeneous.\n")
                f.write("Possible causes:\n")
                f.write("  - Strong selection causing rapid convergence\n")
                f.write("  - Low mutation rate\n")
                f.write("  - Population bottleneck or founder effect\n")
                f.write("  - DBSCAN eps parameter too large (try smaller eps)\n\n")
            
            f.write("INTERPRETATION GUIDE\n")
            f.write("-" * 70 + "\n")
            f.write("Silhouette Score: -1 to 1 (higher = better separated clusters)\n")
            f.write("  > 0.7  : Strong clustering\n")
            f.write("  0.5-0.7: Reasonable clustering\n")
            f.write("  0.25-0.5: Weak clustering\n")
            f.write("  < 0.25: No clear structure\n")
            f.write("  N/A   : Could not compute (too few clusters or identical individuals)\n\n")
            f.write("Spatial Segregation: 0 to 1 (higher = more spatially separated)\n")
            f.write("  > 0.7  : Clusters occupy distinct regions\n")
            f.write("  0.3-0.7: Moderate spatial structure\n")
            f.write("  < 0.3  : Clusters are spatially mixed\n\n")
        
        print(f"\n{'='*70}")
        print(f"Clustering analysis complete!")
        print(f"Results saved to: {clustering_dir}")
        print(f"See README.txt for overview and interpretation guide")
        print(f"{'='*70}\n")
    
    def compare_experiments(
        self,
        experiment_names: list[str],
        output_path: Path | str | None = None
    ) -> None:
        """
        Create comparative visualizations across multiple experiments.
        
        Args:
            experiment_names: List of experiment names to compare
            output_path: Path to save comparison plot
        """
        if output_path is None:
            output_path = self.base_output_dir / "experiment_comparison.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Experiment Comparison", fontsize=14, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_names)))
        
        for exp_name, color in zip(experiment_names, colors):
            if exp_name not in self.experiments:
                print(f"Warning: Experiment '{exp_name}' not found")
                continue
            
            results = self.experiments[exp_name]
            aggregated = self.aggregate_results(results)
            
            gens = aggregated.generations
            
            # Population
            axes[0, 0].plot(gens, aggregated.population_mean, 
                           color=color, linewidth=2, label=exp_name)
            axes[0, 0].fill_between(
                gens,
                aggregated.population_mean - aggregated.population_std,
                aggregated.population_mean + aggregated.population_std,
                alpha=0.2, color=color
            )
            
            # Fitness
            axes[0, 1].plot(gens, aggregated.fitness_best_mean,
                           color=color, linewidth=2, label=exp_name)
            
            # Completion rate
            axes[1, 0].plot(gens, aggregated.completion_rate * 100,
                           color=color, linewidth=2, label=exp_name)
            
            # Final statistics bar chart
            axes[1, 1].bar(exp_name, aggregated.num_completed,
                          color=color, alpha=0.7, label='Completed')
        
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Population Size')
        axes[0, 0].set_title('Population Dynamics')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Best Fitness')
        axes[0, 1].set_title('Fitness Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Runs Active (%)')
        axes[1, 0].set_title('Run Survival Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_ylabel('Number of Runs')
        axes[1, 1].set_title('Completion Statistics')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to: {output_path}")


def create_example_experiments() -> list[ExperimentConfig]:
    """Create example experiment configurations for common scenarios."""
    
    experiments = []
    
    # Experiment 1: Baseline (random pairing, no movement bias)
    experiments.append(ExperimentConfig(
        experiment_name="baseline_random",
        num_runs=5,
        pairing_method="random",
        movement_bias="none",
        population_size=10,
        num_generations=50
    ))
    
    # Experiment 2: Proximity pairing with nearest neighbor
    experiments.append(ExperimentConfig(
        experiment_name="proximity_nn",
        num_runs=5,
        pairing_method="proximity_pairing",
        movement_bias="nearest_neighbor",
        population_size=10,
        num_generations=50
    ))
    
    # Experiment 3: Mating zones with zone targeting
    experiments.append(ExperimentConfig(
        experiment_name="mating_zones",
        num_runs=5,
        pairing_method="mating_zone",
        movement_bias="nearest_zone",
        population_size=10,
        num_generations=50
    ))
    
    # Experiment 4: Energy-based selection
    experiments.append(ExperimentConfig(
        experiment_name="energy_based",
        num_runs=5,
        selection_method="energy_based",
        enable_energy=True,
        initial_energy=100.0,
        energy_depletion_rate=10.0,
        mating_energy_effect="restore",
        population_size=10,
        num_generations=50
    ))
    
    return experiments


if __name__ == "__main__":
    """
    Example usage of the experiment runner.
    
    PADDING STRATEGIES FOR EARLY-STOPPED RUNS:
    -------------------------------------------
    When runs stop early (extinction/explosion), you can control how they're
    handled in aggregated statistics:
    
    1. 'forward_fill' (DEFAULT): Carry last value forward
       - Best for: Comparing final evolutionary outcomes
       - Pros: Smooth curves, maintains sample size, reflects end state
       - Cons: Doesn't show extinction = 0 population
    
    2. 'terminal_state': Use 0 for extinction, last value for explosion  
       - Best for: Population stability analysis
       - Pros: Biologically accurate (extinct = 0)
       - Cons: Can create misleading means if many runs fail
    
    3. 'nan': Leave as NaN (only average over active runs)
       - Best for: Statistical rigor, analyzing only successful runs
       - Pros: Honest about sample size changes
       - Cons: Curves can spike/disappear suddenly
    
    To use different strategies:
        results = runner.run_experiment(exp_config)
        
        # Default (forward_fill)
        agg_default = runner.aggregate_results(results)
        
        # Terminal state (extinction = 0)
        agg_terminal = runner.aggregate_results(results, padding_strategy="terminal_state")
        
        # NaN only (no padding)
        agg_nan = runner.aggregate_results(results, padding_strategy="nan")
    """
    
    # Create runner
    runner = ExperimentRunner(base_output_dir="__experiments__")
    
    # Get example experiments
    experiments = create_example_experiments()
    
    # Run first experiment as a test
    print("Running example experiment...")
    results = runner.run_experiment(experiments[0], verbose=True)
    
    print(f"\nExample complete! Check __experiments__/ for results.")
    print(f"To run all experiments, uncomment the loop below.")
    
    # Uncomment to run all experiments:
    # for exp_config in experiments:
    #     runner.run_experiment(exp_config, verbose=True)
    # 
    # runner.compare_experiments(
    #     [exp.experiment_name for exp in experiments],
    #     output_path=runner.base_output_dir / "all_experiments_comparison.png"
    # )
