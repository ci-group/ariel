# Experiment Runner Usage Guide

## Overview

The Experiment Runner automates running multiple evolutionary trials with consistent parameters, handling early stopping, aggregating statistics across runs, and generating comprehensive comparisons.

The codebase has been streamlined with clean, concise code and minimal verbosity while maintaining all functionality.

## Quick Start

### 1. List Available Experiments

```bash
cd himym/spatial_ea
python run_experiments.py --list
```

Shows all predefined experiments with descriptions.

### 2. Run Single Experiment

```bash
# Run baseline experiment (10 runs default)
python run_experiments.py --experiment baseline_random_fitnessBased

# Override number of runs
python run_experiments.py --experiment baseline_random_fitnessBased --runs 5
```

### 3. Run with Parallel Execution

```bash
# Use all available CPU cores - 1
python run_experiments.py --experiment baseline_random_fitnessBased --parallel

# Specify number of workers
python run_experiments.py --experiment baseline_random_fitnessBased --parallel --num-workers 4
```

### 4. Compare Experiments

```bash
# After running multiple experiments
python run_experiments.py --compare baseline_random_fitnessBased baseline_proximity_fitnessBased
```

## Predefined Experiments

### Baseline Experiments

**baseline_random_fitnessBased**
- Random pairing, fitness-based selection
- 10 runs, 50 generations
- No movement bias
- Good baseline comparison

**baseline_proximity_fitnessBased**
- Proximity pairing, fitness-based selection
- 10 runs, 50 generations
- Spatial constraints on mating
- Compare with random pairing

**baseline_random_parents_die**
- Random pairing, parents die after mating
- 5 runs, 50 generations
- No population control
- Tests exponential growth

**baseline_random_age_based**
- Random pairing, age-based selection
- 5 runs, 50 generations
- Age pressure on survival

### Movement Bias Experiments

**nearest_neighbor_fitBased**
- Proximity pairing, nearest neighbor movement
- Includes 20-generation incubation
- Tests aggregation behavior

**nearest_neighbor_energyCost**
- Same as above but with energy system
- Mating costs energy
- Tests resource-based selection

**nearest_neighbor_probAge**
- Nearest neighbor with probabilistic age-based selection
- Max age limit of 5 generations

### Mating Zone Experiments

**dynamic_matingZone_assignedMating_fitBased**
- Multiple mating zones that move every 3 generations
- Robots assigned to specific zones
- 4 zones with 2.5m radius

**static_matingZone_assignedMating_fitBased**
- Same as above but zones don't move
- Tests fixed spatial structure

### Other Experiments

**fitness_selection** - Pure fitness-based comparison
**energy_selection** - Energy-based survival
**no_control** - No population control
**fitness_control** - Target population of 30
**low_mutation** - 0.1 mutation rate
**high_mutation** - 0.9 mutation rate

## Output Structure

Each experiment creates a timestamped directory:

```
__experiments__/
└── baseline_random_fitnessBased_20251117_123456/
    ├── experiment_config.json          # Configuration used
    ├── base_config.yaml               # Copy of base config
    ├── aggregated_statistics.csv      # Mean/std across runs
    ├── aggregated_results.png         # Visualization
    ├── summary.json                   # High-level summary
    │
    ├── run_000/                       # Individual run data
    │   ├── results.json               # Run statistics
    │   ├── statistics.csv             # Time series data
    │   ├── figures/                   # Trajectory plots
    │   ├── results/                   # Genotype saves
    │   └── videos/                    # Simulation videos
    │
    ├── run_001/
    ├── run_002/
    └── ...
```

## Key Features

### 1. Parallel Execution

Run multiple trials simultaneously:

```bash
# Automatic worker count (CPU cores - 1)
python run_experiments.py --experiment baseline_random_fitnessBased --parallel

# Specify workers
python run_experiments.py --experiment baseline_random_fitnessBased --parallel --num-workers 4
```

**Benefits:**
- Dramatically faster for multiple runs
- Independent process per run (no MuJoCo conflicts)
- Each run gets unique random seed
- Progress updates from all workers

**Note:** Fix applied to ensure different runs have different random seeds!

### 2. Statistical Aggregation

For each metric (population, fitness, etc.):
- **Mean** - Average across all active runs
- **Std** - Standard deviation (now properly computed!)
- **Min/Max** - Range across runs
- **Completion Rate** - Fraction of runs still active

### 3. Early Stopping Handling

Automatically manages runs that stop early:
- Tracks extinction (population too small)
- Tracks explosion (population too large)
- Pads data with NaN for aggregation
- Reports completion statistics

### 4. Comprehensive Visualizations

**Per Experiment:**
- Population dynamics with confidence bands
- Fitness evolution (best and average)
- Run survival rate over time
- Active runs per generation

**Cross-Experiment:**
- Population comparison
- Fitness comparison
- Completion rate comparison
- Summary statistics

## Programmatic Usage

### Basic Experiment

```python
from experiment_runner import ExperimentRunner, ExperimentConfig

runner = ExperimentRunner(base_output_dir="__experiments__")

# Define experiment
config = ExperimentConfig(
    experiment_name="my_experiment",
    num_runs=10,
    population_size=30,
    num_generations=50,
    pairing_method="random",
    selection_method="fitness_based"
)

# Run experiment
results = runner.run_experiment(config, verbose=True)

# Results are list of RunResult objects
for result in results:
    print(f"Run {result.run_id}: Best fitness = {max(result.fitness_best):.4f}")
```

### Parallel Execution

```python
# Run with parallel workers
results = runner.run_experiment_parallel(
    config, 
    num_workers=4,
    verbose=True
)
```

### Custom Experiment Configuration

```python
config = ExperimentConfig(
    experiment_name="custom_energy_experiment",
    num_runs=5,
    
    # Population
    population_size=40,
    num_generations=100,
    max_population_limit=150,
    min_population_limit=5,
    stop_on_limits=True,
    
    # Selection
    pairing_method="proximity_pairing",
    movement_bias="nearest_neighbor",
    selection_method="energy_based",
    target_population_size=40,
    pairing_radius=3.0,
    
    # Energy system
    enable_energy=True,
    initial_energy=100.0,
    energy_depletion_rate=5.0,
    mating_energy_effect="cost",
    mating_energy_amount=15.0,
    
    # Mutation
    mutation_rate=0.7,
    crossover_rate=0.9,
    
    # Output
    save_snapshots=False,
    save_trajectories=True,
    save_individual_runs=False,  # Save space
    
    # Reproducibility
    random_seed=42
)

results = runner.run_experiment_parallel(config, num_workers=5)
```

### Parameter Sweep

```python
from experiment_runner import ExperimentRunner, ExperimentConfig

runner = ExperimentRunner()

# Sweep mutation rates
for mutation_rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
    config = ExperimentConfig(
        experiment_name=f"mutation_sweep_{mutation_rate:.1f}",
        num_runs=10,
        mutation_rate=mutation_rate,
        population_size=30,
        num_generations=50
    )
    
    print(f"\nRunning mutation_rate={mutation_rate}")
    runner.run_experiment_parallel(config, num_workers=5)

# Compare all
experiment_names = [f"mutation_sweep_{r:.1f}" for r in [0.1, 0.3, 0.5, 0.7, 0.9]]
runner.compare_experiments(
    experiment_names,
    output_path="__experiments__/mutation_sweep_comparison.png"
)
```

### Access Results

```python
# After running experiment
results = runner.experiments["my_experiment"]

# Aggregate statistics
aggregated = runner.aggregate_results(results)

print(f"Experiment: {aggregated.experiment_name}")
print(f"Runs: {aggregated.num_runs}")
print(f"Completed: {aggregated.num_completed}")
print(f"Extinct: {aggregated.num_extinctions}")
print(f"Exploded: {aggregated.num_explosions}")

# Access data arrays
generations = aggregated.generations
pop_mean = aggregated.population_mean
pop_std = aggregated.population_std
fitness_best_mean = aggregated.fitness_best_mean
fitness_best_std = aggregated.fitness_best_std

# Final statistics
final_gen = -1
print(f"\nFinal generation statistics:")
print(f"  Population: {pop_mean[final_gen]:.1f} ± {pop_std[final_gen]:.1f}")
print(f"  Best fitness: {fitness_best_mean[final_gen]:.4f} ± {fitness_best_std[final_gen]:.4f}")
print(f"  Active runs: {int(aggregated.runs_active[final_gen])}/{aggregated.num_runs}")
```

## Data Analysis

### Load Aggregated Statistics

```python
import pandas as pd
import json

# Load aggregated CSV
df = pd.read_csv("__experiments__/experiment_name_timestamp/aggregated_statistics.csv")

# Load summary JSON
with open("__experiments__/experiment_name_timestamp/summary.json") as f:
    summary = json.load(f)

print(f"Experiment: {summary['experiment_name']}")
print(f"Completed runs: {summary['num_completed']}/{summary['num_runs']}")
print(f"Max generations: {summary['max_generations_reached']}")
print(f"Best fitness: {summary['best_fitness_achieved']:.4f}")
```

### Plot Custom Analysis

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("path/to/aggregated_statistics.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Population dynamics
axes[0, 0].plot(df['generation'], df['population_mean'], 'b-', linewidth=2)
axes[0, 0].fill_between(
    df['generation'],
    df['population_mean'] - df['population_std'],
    df['population_mean'] + df['population_std'],
    alpha=0.3
)
axes[0, 0].set_title('Population Dynamics')
axes[0, 0].set_xlabel('Generation')
axes[0, 0].set_ylabel('Population Size')

# Fitness evolution
axes[0, 1].plot(df['generation'], df['fitness_best_mean'], 'g-', linewidth=2)
axes[0, 1].fill_between(
    df['generation'],
    df['fitness_best_mean'] - df['fitness_best_std'],
    df['fitness_best_mean'] + df['fitness_best_std'],
    alpha=0.3, color='g'
)
axes[0, 1].set_title('Fitness Evolution')
axes[0, 1].set_xlabel('Generation')
axes[0, 1].set_ylabel('Best Fitness')

# Completion rate
axes[1, 0].plot(df['generation'], df['completion_rate'] * 100, 'r-', linewidth=2)
axes[1, 0].set_title('Run Survival Rate')
axes[1, 0].set_xlabel('Generation')
axes[1, 0].set_ylabel('Active Runs (%)')

# Active runs
axes[1, 1].bar(df['generation'], df['runs_active'], color='steelblue')
axes[1, 1].set_title('Active Runs per Generation')
axes[1, 1].set_xlabel('Generation')
axes[1, 1].set_ylabel('Number of Runs')

plt.tight_layout()
plt.savefig('custom_analysis.png', dpi=300)
```

### Compare Experiments

```python
import pandas as pd
import matplotlib.pyplot as plt

experiments = {
    'Random Pairing': 'baseline_random_20251117_120000/aggregated_statistics.csv',
    'Proximity Pairing': 'baseline_proximity_20251117_130000/aggregated_statistics.csv',
    'Mating Zones': 'mating_zones_20251117_140000/aggregated_statistics.csv'
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for name, path in experiments.items():
    df = pd.read_csv(f"__experiments__/{path}")
    
    # Fitness comparison
    axes[0].plot(df['generation'], df['fitness_best_mean'], linewidth=2, label=name)
    axes[0].fill_between(
        df['generation'],
        df['fitness_best_mean'] - df['fitness_best_std'],
        df['fitness_best_mean'] + df['fitness_best_std'],
        alpha=0.2
    )
    
    # Population comparison
    axes[1].plot(df['generation'], df['population_mean'], linewidth=2, label=name)

axes[0].set_xlabel('Generation')
axes[0].set_ylabel('Best Fitness')
axes[0].set_title('Fitness Evolution Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Generation')
axes[1].set_ylabel('Population Size')
axes[1].set_title('Population Dynamics Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiment_comparison.png', dpi=300)
```

## Advanced Features

### Reproducible Experiments

```python
config = ExperimentConfig(
    experiment_name="reproducible_experiment",
    num_runs=5,
    random_seed=42,  # Fixed seed
    # ... other params
)

# Each run will use seed: 42+run_id
# Run 0: seed=42, Run 1: seed=43, etc.
```

### Save Space with Smart Output

```python
config = ExperimentConfig(
    experiment_name="space_efficient",
    num_runs=10,
    
    # Disable expensive outputs
    save_snapshots=False,        # No trajectory images
    save_trajectories=False,     # No trajectory data
    record_videos=False,         # No simulation videos
    save_individual_runs=False,  # Keep only aggregated stats
    
    # ... other params
)
```

Individual runs still save `results.json` and `statistics.csv`, but figures/videos are cleaned up.

### Monitor Long-Running Experiments

```python
import time
from experiment_runner import ExperimentRunner, ExperimentConfig

runner = ExperimentRunner()
config = ExperimentConfig(
    experiment_name="long_experiment",
    num_runs=20,
    num_generations=200
)

start_time = time.time()

# Run with verbose output
results = runner.run_experiment_parallel(
    config, 
    num_workers=8,
    verbose=True  # Shows progress from each worker
)

duration = time.time() - start_time
print(f"\nTotal experiment time: {duration/60:.1f} minutes")
print(f"Average per run: {duration/len(results):.1f} seconds")
```

## Troubleshooting

### Runs Stop Early

**Symptoms:** Many runs marked as "extinct" or "exploded"

**Solutions:**
- Adjust `max_population_limit` and `min_population_limit`
- Enable population control: `selection_method="fitness_based"`
- Reduce energy depletion if using energy system
- Increase initial population size
- Enable incubation period

### Identical Results Across Runs

**Symptoms:** Standard deviation is zero or very small

**Solutions:**
- **Fixed in recent update!** Parallel runs now get unique seeds
- Don't set `random_seed` unless you want reproducibility
- Verify fix: Check that `fitness_best_std > 0` in aggregated stats

### Out of Memory

**Symptoms:** Process killed during parallel execution

**Solutions:**
- Reduce `num_workers`
- Disable output: `save_individual_runs=False`
- Run sequentially: don't use `--parallel`
- Reduce population size or generations

### Slow Execution

**Symptoms:** Each run takes very long

**Solutions:**
- Use parallel execution: `--parallel`
- Disable videos: `record_videos=False`
- Disable snapshots: `save_snapshots=False`
- Reduce `simulation_time`
- Reduce population size

## Tips and Best Practices

1. **Start with few runs**: Test with `--runs 3` first
2. **Use parallel execution**: Dramatically faster for multiple runs
3. **Disable expensive features**: Save snapshots only for important experiments
4. **Monitor disk space**: Videos and images can accumulate quickly
5. **Use reproducible seeds**: For debugging and validation
6. **Compare systematically**: Run baseline first, then variations
7. **Check aggregated plots**: Before diving into individual runs
8. **Save configs**: Copy experiment_config.json for your records

## Integration with Visualization

Visualize experiment results:

```python
from experiment_runner import ExperimentRunner, ExperimentConfig
from visualize_experiment import ExperimentVisualizer

# Run experiment
runner = ExperimentRunner()
config = ExperimentConfig(...)
results = runner.run_experiment_parallel(config)

# Find aggregated statistics file
import glob
exp_dir = f"__experiments__/{config.experiment_name}_*"
agg_csv = glob.glob(f"{exp_dir}/aggregated_statistics.csv")[0]

# Visualize
viz = ExperimentVisualizer(agg_csv)
viz.generate_report()
```

## Further Reading

- **Spatial EA Guide**: See `SPATIAL_EA_USAGE.md` for core EA usage
- **Visualization Guide**: See `VISUALIZATION_USAGE.md` for plot generation
- **Code Reference**: Check `experiment_runner.py` docstrings
- **Example Script**: Review `run_experiments.py` for experiment definitions

## Support

For experiment runner issues:
1. Check console output for error messages
2. Verify configuration parameters
3. Test with small runs first (`--runs 3`)
4. Review aggregated statistics and summary.json
5. Check individual run logs in run_XXX directories
