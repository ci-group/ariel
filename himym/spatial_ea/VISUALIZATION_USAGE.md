# Visualization Usage Guide

## Overview

The visualization system provides comprehensive analysis and plotting tools for Spatial EA experiments. It automatically discovers data files, generates multi-panel plots, and provides programmatic access to all metrics.

The codebase has been streamlined for better readability with concise docstrings and minimal comments while maintaining full functionality.

## Quick Start

### 1. Basic Visualization

After running an experiment, visualize results:

```bash
cd himym/spatial_ea
python visualize_experiment.py
```

**What it does:**
- Finds the latest `evolution_data_*.csv` file
- Generates comprehensive 12-panel overview
- Creates parameter analysis plots
- Saves summary statistics
- Outputs to `__results__/analysis/`

### 2. Interactive Examples

Explore visualization capabilities:

```bash
cd himym/spatial_ea
python visualize_examples.py
```

**Interactive menu:**
1. Basic Usage - Complete report generation
2. Custom Plots - Individual visualizations
3. Data Exploration - Programmatic data access
4. Compare Experiments - Multi-run comparison
5. Energy Analysis - Energy system dynamics

### 3. Specify Data File

Visualize a specific experiment:

```bash
python visualize_experiment.py --file evolution_data_20251115_123456.csv
```

## Output Files

### Default Output Directory: `__results__/analysis/`

**Main Visualizations:**
- `{experiment}_comprehensive.png` - 12-panel overview (391 KB)
- `{experiment}_parameters.png` - Controller distributions (155 KB)
- `{experiment}_heatmap.png` - Parameter correlations (96 KB)
- `{experiment}_summary.txt` - Statistics summary (1.5 KB)

**Optional Outputs:**
- `individual/` - Separate file per metric (11 plots)
- `minimal/` - Publication-ready plots (6 plots)

## Comprehensive Plot (12 Panels)

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  Population Size  │  Births/Deaths    │  Mating Success │
├─────────────────────────────────────────────────────────┤
│  Fitness Best/Avg │  Fitness Diversity│  Genotype Div   │
├─────────────────────────────────────────────────────────┤
│  Energy Levels    │  Age Distribution │  Energy Depletion│
├─────────────────────────────────────────────────────────┤
│  Fitness vs Age   │  Fitness vs Energy│  Summary Stats  │
└─────────────────────────────────────────────────────────┘
```

### What Each Panel Shows

**Row 1: Population Dynamics**
- **Population Size**: Total individuals over time
- **Births/Deaths**: New individuals and removals per generation
- **Mating Success**: Number of pairs formed and success rate

**Row 2: Fitness Metrics**
- **Fitness Evolution**: Best and average fitness with confidence bands
- **Fitness Diversity**: Coefficient of variation (CV) showing diversity
- **Genotype Diversity**: Standard deviation of all genotype parameters

**Row 3: Energy and Age** (if applicable)
- **Energy Levels**: Min/max/avg energy across population
- **Age Distribution**: Youngest/oldest/avg age
- **Energy Depletion**: Count of energy-depleted individuals

**Row 4: Correlations and Summary**
- **Fitness vs Age**: Scatter plot showing relationship
- **Fitness vs Energy**: Scatter plot showing relationship
- **Summary Statistics**: Text summary of key metrics

## Command Line Options

### Basic Options

```bash
# Visualize latest experiment
python visualize_experiment.py

# Specify data file
python visualize_experiment.py --file path/to/data.csv

# Custom output directory
python visualize_experiment.py --output /path/to/output/
```

### Plot Type Options

```bash
# Comprehensive only (default)
python visualize_experiment.py

# Add individual plots (11 separate files)
python visualize_experiment.py --individual

# Add publication plots (6 minimal plots)
python visualize_experiment.py --minimal

# Generate all types
python visualize_experiment.py --all
```

### Example Commands

```bash
# Full analysis with all plot types
python visualize_experiment.py --all --output my_analysis/

# Publication-ready plots for specific experiment
python visualize_experiment.py --file results.csv --minimal

# Individual metric plots for detailed analysis
python visualize_experiment.py --individual
```

## Programmatic Usage

### Basic Report Generation

```python
from visualize_experiment import ExperimentVisualizer

# Auto-discover latest data
viz = ExperimentVisualizer()

# Or specify file
viz = ExperimentVisualizer('path/to/evolution_data.csv')

# Generate comprehensive report
viz.generate_report()
```

### Custom Output Options

```python
# Generate with all plot types
viz.generate_report(
    individual_plots=True,
    minimal_plots=True,
    output_dir='custom_output/'
)
```

### Access Raw Data

```python
viz = ExperimentVisualizer('data.csv')

# Pandas DataFrame with all time series
df = viz.df
generations = df['generation']
fitness_best = df['fitness_best']
population_size = df['population_size']

# Final population controllers
controllers = viz.controllers['controllers']
best_controller = controllers[0]  # Sorted by fitness

# NumPy arrays (if NPZ file exists)
npz_data = viz.npz_data
fitness_array = npz_data['fitness_best']
```

### Create Individual Plots

```python
viz = ExperimentVisualizer('data.csv')

# Individual plot methods
viz.plot_all(save_path='overview.png')
viz.plot_controller_parameters(save_path='params.png')
viz.plot_parameter_evolution_heatmap(save_path='heatmap.png')

# Show interactively (don't save)
viz.plot_all(save_path=None)
```

## Advanced Usage

### 1. Compare Multiple Experiments

```python
from visualize_experiment import ExperimentVisualizer
import matplotlib.pyplot as plt
import pandas as pd

experiments = [
    'evolution_data_20251115_120000.csv',
    'evolution_data_20251115_130000.csv',
    'evolution_data_20251115_140000.csv'
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for exp_file in experiments:
    viz = ExperimentVisualizer(exp_file)
    df = viz.df
    
    # Plot fitness evolution
    axes[0, 0].plot(df['generation'], df['fitness_best'], label=exp_file)
    
    # Plot population dynamics
    axes[0, 1].plot(df['generation'], df['population_size'], label=exp_file)
    
    # Plot mating success
    axes[1, 0].plot(df['generation'], df['mating_success_rate'], label=exp_file)
    
    # Plot diversity
    axes[1, 1].plot(df['generation'], df['genotype_diversity'], label=exp_file)

for ax in axes.flat:
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiment_comparison.png', dpi=300)
```

### 2. Extract Best Controllers

```python
viz = ExperimentVisualizer('data.csv')

# Get top 5 individuals
top_5 = viz.controllers['controllers'][:5]

for rank, controller in enumerate(top_5, 1):
    print(f"\nRank {rank}:")
    print(f"  Fitness: {controller['fitness']:.6f}")
    print(f"  Age: {controller['age']} generations")
    print(f"  Energy: {controller['energy']:.2f}")
    print(f"  ID: {controller['unique_id']}")
    
    # Access genotype
    genotype = controller['genotype']
    print(f"  Genotype nodes: {len(genotype['nodes'])}")
    print(f"  Genotype connections: {len(genotype['connections'])}")
```

### 3. Statistical Analysis

```python
import numpy as np
from visualize_experiment import ExperimentVisualizer

viz = ExperimentVisualizer('data.csv')
df = viz.df

# Fitness improvement
initial_fitness = df['fitness_best'].iloc[0]
final_fitness = df['fitness_best'].iloc[-1]
improvement = final_fitness - initial_fitness
improvement_pct = (improvement / initial_fitness) * 100

print(f"Fitness improvement: {improvement:.4f} ({improvement_pct:.1f}%)")

# Convergence detection
window = len(df) // 5
recent_variance = df['fitness_best'].iloc[-window:].var()
print(f"Recent fitness variance: {recent_variance:.6f}")

if recent_variance < 0.001:
    print("Evolution has converged!")

# Population stability
pop_mean = df['population_size'].mean()
pop_std = df['population_size'].std()
pop_cv = pop_std / pop_mean
print(f"Population CV: {pop_cv:.3f}")

# Mating efficiency
avg_success = df['mating_success_rate'].mean()
print(f"Average mating success: {avg_success:.1f}%")
```

### 4. Custom Visualizations

```python
import matplotlib.pyplot as plt
from visualize_experiment import ExperimentVisualizer

viz = ExperimentVisualizer('data.csv')
df = viz.df

# Create custom plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot with custom styling
ax.plot(df['generation'], df['fitness_best'], 
        'g-', linewidth=2, label='Best Fitness')
ax.fill_between(df['generation'],
                df['fitness_best'],
                df['fitness_avg'],
                alpha=0.3, color='green',
                label='Fitness Range')

# Add annotations
max_fitness_gen = df['fitness_best'].idxmax()
max_fitness = df['fitness_best'].max()
ax.annotate(f'Peak: {max_fitness:.4f}',
           xy=(max_fitness_gen, max_fitness),
           xytext=(max_fitness_gen + 5, max_fitness - 0.1),
           arrowprops=dict(arrowstyle='->', color='red'),
           fontsize=12, fontweight='bold')

ax.set_xlabel('Generation', fontsize=14)
ax.set_ylabel('Fitness', fontsize=14)
ax.set_title('Fitness Evolution - Custom View', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('custom_fitness_plot.png', dpi=300)
```

### 5. Batch Processing

```python
from pathlib import Path
from visualize_experiment import ExperimentVisualizer

results_dir = Path('__results__')

# Process all experiments
for csv_file in results_dir.glob('evolution_data_*.csv'):
    print(f"Processing: {csv_file.name}")
    
    viz = ExperimentVisualizer(csv_file)
    
    # Custom output directory per experiment
    output_dir = results_dir / f"analysis_{csv_file.stem}"
    
    # Generate all visualizations
    viz.generate_report(
        output_dir=output_dir,
        individual_plots=True,
        minimal_plots=True
    )
    
    # Extract summary
    summary = viz.get_text_summary()
    print(f"  Best fitness: {summary['best_fitness']:.4f}")
    print(f"  Final population: {summary['final_population']}")
```

## Available Metrics

### Time Series Data (from CSV)

**Population:**
- `generation` - Generation number
- `population_size` - Number of individuals
- `births` - New individuals this generation
- `deaths` - Removed individuals this generation

**Fitness:**
- `fitness_best` - Best fitness value
- `fitness_avg` - Average fitness
- `fitness_worst` - Worst fitness
- `fitness_std` - Standard deviation

**Mating:**
- `mating_pairs` - Number of pairs formed
- `unpaired` - Individuals that didn't mate
- `mating_success_rate` - Percentage paired

**Diversity:**
- `genotype_diversity` - Std of all genotype parameters

**Age:**
- `age_min` - Youngest individual
- `age_max` - Oldest individual
- `age_avg` - Average age
- `age_std` - Age standard deviation

**Energy** (if enabled):
- `energy_min` - Minimum energy
- `energy_max` - Maximum energy
- `energy_avg` - Average energy
- `energy_std` - Energy standard deviation
- `energy_depleted_count` - Individuals at zero energy

### Controller Data (from JSON)

**Per individual:**
- `unique_id` - Individual identifier
- `fitness` - Fitness score
- `age` - Age in generations
- `energy` - Current energy level
- `generation_born` - Birth generation
- `genotype` - Full HyperNEAT genotype
  - `nodes` - Neural network nodes
  - `connections` - Neural network connections

## Publication-Ready Plots

Generate minimal plots for papers:

```bash
python visualize_experiment.py --minimal
```

**Output:** 6 plots optimized for double-column format

1. `{exp}_fitness_evolution_minimal.png` - Best/avg fitness
2. `{exp}_population_dynamics_minimal.png` - Population size
3. `{exp}_diversity_minimal.png` - Genotype diversity
4. `{exp}_mating_success_minimal.png` - Mating statistics
5. `{exp}_energy_age_minimal.png` - Energy and age (if enabled)
6. `{exp}_correlations_minimal.png` - Fitness correlations

**Features:**
- Larger fonts (11pt+)
- Clear labels and legends
- Optimized sizing (6×4 inches)
- High resolution (300 DPI)
- Minimal decorations

## Troubleshooting

### No Data Files Found

**Error:** "No evolution data files found"

**Solutions:**
- Run `main_spatial_ea.py` first to generate data
- Check files in `__results__/` directory
- Specify file explicitly: `--file path/to/data.csv`

### Missing Controller Plots

**Symptom:** Parameter plots not generated

**Solutions:**
- Ensure JSON controller files exist
- Check matching timestamps between CSV and JSON
- Controllers saved automatically by `main_spatial_ea.py`

### NaN Values in Plots

**Symptom:** Missing data or gaps in plots

**Solutions:**
- Energy plots require `enable_energy: true`
- Some metrics only available with specific features
- Check console for warnings about missing data

### Memory Issues

**Symptom:** Out of memory when visualizing

**Solutions:**
- Process one experiment at a time
- Disable individual plots: don't use `--individual`
- Reduce plot resolution in code
- Use smaller datasets

## Tips and Best Practices

1. **Always visualize after experiments** - Automate with:
   ```python
   spatial_ea.run_evolution()
   viz = ExperimentVisualizer('latest_results.csv')
   viz.generate_report()
   ```

2. **Check summary first** - Read `{exp}_summary.txt` before plots

3. **Use comprehensive for overview** - Single file, all metrics

4. **Use individual for details** - Separate analysis per metric

5. **Use minimal for papers** - Publication-ready formatting

6. **Compare experiments** - Load multiple CSV files and overlay plots

7. **Save configurations** - Document experiment settings alongside plots

## Integration with Experiment Runner

Visualize aggregated experiment results:

```python
from experiment_runner import ExperimentRunner
from visualize_experiment import ExperimentVisualizer

# Run experiment
runner = ExperimentRunner()
results = runner.run_experiment(config)

# Visualize aggregated statistics
agg_csv = f"__experiments__/{config.experiment_name}_*/aggregated_statistics.csv"
viz = ExperimentVisualizer(agg_csv)
viz.generate_report()
```

## Further Reading

- **Spatial EA Guide**: See `SPATIAL_EA_USAGE.md` for running experiments
- **Experiment Runner**: See `EXPERIMENT_RUNNER_USAGE.md` for batch processing
- **Code Reference**: Check `visualize_experiment.py` docstrings
- **Examples**: Run `visualize_examples.py` for interactive tutorials

## Support

For visualization issues:
1. Check data file exists and is readable
2. Verify CSV format matches expected columns
3. Review console output for warnings
4. Test with `visualize_examples.py` first
