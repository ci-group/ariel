# Spatial EA Usage Guide

## Overview

The Spatial EA (Evolutionary Algorithm) simulates robot populations that evolve## Programmatic Usage

### Basic Setup

```python
from main_spatial_ea import SpatialEA
from ea_config import config

# Use default configuration
spatial_ea = SpatialEA(num_joints=8)  # Number of robot actuators

# Run evolution
best_individual = spatial_ea.run_evolution()

if best_individual:
    print(f"Best fitness: {best_individual.fitness:.4f}")
else:
    print("Evolution stopped early (extinction or explosion)")
```onstrained mating. Robots move in a physical environment, and their proximity determines mating opportunities, creating spatial selection pressures.

This system has been streamlined for better readability while maintaining full functionality. All core modules use concise docstrings and minimal logging for cleaner output.

## Quick Start

### Basic Run

```bash
cd himym/spatial_ea
uv run main_spatial_ea.py
```

This runs with default settings from `ea_config.yaml`:
- Population size: 30 robots
- Generations: 50
- Selection: fitness-based
- Movement: random pairing

### Configuration

Edit `himym/spatial_ea/ea_config.yaml` to customize:

```yaml
# Population parameters
population_size: 30
num_generations: 50
max_population_limit: 100
min_population_limit: 1

# Selection method
selection_method: "fitness_based"  # or "age_based", "energy_based"
target_population_size: 30

# Pairing method
pairing_method: "random"  # or "proximity_pairing", "mating_zone"
movement_bias: "none"     # or "nearest_neighbor", "assigned_zone"

# World settings
world_size: [10.0, 10.0, 2.0]
simulation_time: 30.0
use_periodic_boundaries: true

# Output
save_generation_snapshots: true
record_generation_videos: false
```

## Key Concepts

### 1. Pairing Methods

**Random Pairing**
- Pairs randomly selected from population
- No spatial constraints
- Baseline comparison

**Proximity Pairing**
- Only robots within `pairing_radius` can mate
- Creates local mating neighborhoods
- Promotes spatial clustering

**Mating Zone**
- Defines specific mating areas
- Robots must be inside zones to mate
- Supports static or dynamic zones

### 2. Movement Bias

**None**
- Robots move randomly (Brownian motion)
- Default baseline behavior

**Nearest Neighbor**
- Robots move toward nearest neighbor
- Creates aggregation dynamics
- Promotes clustering behavior

**Nearest Zone**
- Robots move toward nearest mating zone
- Works with multiple zones

**Assigned Zone**
- Each robot targets a specific assigned mating zone
- Combines with `mating_zone` pairing

### 3. Selection Methods

**Fitness-Based**
- Maintains target population size
- Removes least fit individuals
- Standard evolutionary pressure

**Age-Based**
- Younger individuals preferred
- Maintains target population size
- Can be combined with `max_age` limit

**Probabilistic Age**
- Death probability increases with age
- No target population enforcement
- Natural population dynamics (p_death = age/max_age)

**Energy-Based**
- Individuals have energy that depletes over time
- Death when energy reaches zero
- No target population enforcement
- Mating can cost or restore energy

**Parents Die**
- Parents automatically removed after reproduction
- Offspring and non-maters survive
- Enforces target population size

## Programmatic Usage

### Basic Setup

```python
from main_spatial_ea import SpatialEA
from ea_config import config

# Use default configuration
spatial_ea = SpatialEA(
    population_size=config.population_size,
    num_generations=config.num_generations,
    num_joints=8  # Number of robot actuators
)

# Run evolution
best_individual = spatial_ea.run_evolution()

print(f"Best fitness: {best_individual.fitness:.4f}")
```

### Custom Configuration

```python
from main_spatial_ea import SpatialEA
from ea_config import config

# Override specific parameters
config._config['population_size'] = 50
config._config['num_generations'] = 100
config._config['selection']['selection_method'] = 'energy_based'
config._config['mating']['pairing_method'] = 'proximity_pairing'
config._config['mating']['movement_bias'] = 'nearest_neighbor'

spatial_ea = SpatialEA(num_joints=8)
spatial_ea.run_evolution()
```

### Access Evolution Data

```python
from main_spatial_ea import SpatialEA

spatial_ea = SpatialEA(num_joints=8)
spatial_ea.run_evolution()

# Access data collector
dc = spatial_ea.data_collector

# Get time series data
generations = dc.generations
fitness_best = dc.fitness_best
fitness_avg = dc.fitness_avg
population_sizes = dc.population_size

# Get summary statistics
summary = dc.get_summary_stats()
print(f"Total births: {summary['total_births']}")
print(f"Total deaths: {summary['total_deaths']}")
print(f"Best fitness ever: {summary['fitness']['best_ever']:.4f}")
print(f"Mating success rate: {summary['avg_mating_success_rate']:.1f}%")
```

## Advanced Features

### 1. Incubation Period

Pre-evolve population without spatial constraints:

```yaml
# In ea_config.yaml
incubation:
  enabled: true
  num_generations: 20  # Non-spatial generations first
```

Benefits:
- Bootstraps population with viable controllers
- Speeds up spatial evolution
- Reduces early extinctions

### 2. Energy System

Add energy-based survival mechanics:

```yaml
energy:
  enable_energy: true
  initial_energy: 100.0
  energy_depletion_rate: 10.0
  mating_energy_effect: "cost"  # or "restore"
  mating_energy_amount: 20.0
```

### 3. Mating Zones

Create spatial mating constraints with multiple zones:

```yaml
mating:
  pairing_method: "mating_zone"
  num_mating_zones: 4  # Multiple zones
  mating_zone_radius: 2.5
  zone_relocation_strategy: "event_driven"  # or "generation_interval", "static"
  zone_change_interval: 3  # For generation_interval strategy
  min_zone_distance: 2.0  # Minimum spacing between zones
```

**Zone Relocation Strategies:**
- `static`: Zones never move
- `generation_interval`: Zones relocate every N generations
- `event_driven`: Zones relocate after successful matings occur within them

### 4. Population Control

Prevent extinction or explosion:

```yaml
population:
  max_population_limit: 100
  min_population_limit: 1
  stop_on_limits: true  # End run if limits reached
```

## Output Files

### Automatic Outputs

After running, files are saved to configured directories:

**Evolution Data** (`__results__/`)
- `evolution_data_TIMESTAMP.csv` - All generation statistics
- `evolution_data_TIMESTAMP.npz` - NumPy format (faster loading)
- `evolution_statistics_TIMESTAMP.png` - Basic 5-panel plot

**Final Population** (`__results__/`)
- `final_controllers_TIMESTAMP.json` - All individuals with metadata
- `final_genotypes_TIMESTAMP.npz` - Genotype arrays
- `best_controller_TIMESTAMP.txt` - Best individual (readable)

**Visualizations** (`__figures__/`)
- `mating_trajectories_gen_XXX.png` - Robot movement paths (if enabled)
- `generation_XXX_snapshot.png` - Simulation snapshots (if enabled)

**Videos** (`__videos__/`)
- `generation_XXX.mp4` - Simulation recordings (if enabled)

### Data Format

**CSV columns:**
- `generation`, `population_size`, `births`, `deaths`
- `fitness_best`, `fitness_avg`, `fitness_worst`, `fitness_std`
- `age_min`, `age_max`, `age_avg`, `age_std`
- `mating_pairs`, `unpaired`, `mating_success_rate`
- `genotype_diversity`
- `energy_min`, `energy_max`, `energy_avg`, `energy_std` (if enabled)

## Common Workflows

### 1. Basic Experiment

```bash
# Edit configuration
nano himym/spatial_ea/ea_config.yaml

# Run evolution
cd himym/spatial_ea
uv run main_spatial_ea.py

# Results automatically saved to __results__/
ls __results__/evolution_data_*.csv
```

### 2. Parameter Exploration

```python
from main_spatial_ea import SpatialEA
from ea_config import config

for mutation_rate in [0.1, 0.5, 0.9]:
    config._config['mutation']['mutation_rate'] = mutation_rate
    
    spatial_ea = SpatialEA(num_joints=8)
    spatial_ea.run_evolution()
    
    # Results saved with timestamp
    print(f"Completed mutation_rate={mutation_rate}")
```

### 3. Reproducible Runs

Set random seed for reproducibility:

```python
import numpy as np
from main_spatial_ea import SpatialEA

np.random.seed(42)  # Fixed seed

spatial_ea = SpatialEA(num_joints=8)
best = spatial_ea.run_evolution()
```

## Troubleshooting

### Population Goes Extinct

**Symptoms:** Run stops early with "Population went extinct"

**Solutions:**
- Increase initial population size
- Enable incubation period
- Reduce `energy_depletion_rate` (if using energy)
- Use less aggressive selection method
- Check mutation rates aren't too high

### Population Explodes

**Symptoms:** Run stops with "Population reached maximum limit"

**Solutions:**
- Enable fitness-based selection
- Set appropriate `target_population_size`
- Reduce reproduction rate
- Use age-based or energy-based selection

### Poor Fitness Progress

**Symptoms:** Fitness plateaus or doesn't improve

**Solutions:**
- Increase population size
- Adjust mutation rates (higher = more exploration)
- Increase `num_generations`
- Check pairing method allows sufficient mixing
- Review selection pressure (may be too weak or strong)

### Slow Simulation

**Symptoms:** Each generation takes very long

**Solutions:**
- Disable video recording: `record_generation_videos: false`
- Disable snapshots: `save_generation_snapshots: false`
- Reduce `simulation_time`
- Reduce population size
- Use simpler robot morphology

## Tips and Best Practices

1. **Start Simple**: Begin with random pairing and no movement bias
2. **Enable Logging**: Keep `save_generation_snapshots: true` for debugging
3. **Monitor Progress**: Check `__results__/` directory during runs
4. **Save Configurations**: Copy `ea_config.yaml` for each experiment
5. **Use Incubation**: Pre-evolve population for complex environments
6. **Check Convergence**: Look for fitness plateaus in results
7. **Validate Results**: Run multiple times with different seeds

## Further Reading

- **Experiment Runner**: See `EXPERIMENT_RUNNER_USAGE.md` for batch experiments
- **Visualization**: See `VISUALIZATION_USAGE.md` for analysis tools
- **Code Reference**: Check docstrings in `main_spatial_ea.py`
- **Configuration**: Full options in `ea_config.yaml` with comments

## Support

For issues or questions:
1. Check error messages in console output
2. Review configuration file syntax
3. Verify output directories exist and are writable
4. Check MuJoCo installation if simulation fails
