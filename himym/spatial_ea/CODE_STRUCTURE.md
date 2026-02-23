# Code Structure and Organization

## Overview

This spatial EA codebase has been cleaned up for improved readability and maintainability. All core modules now feature:

- **Concise docstrings**: Single-line descriptions instead of verbose multi-paragraph documentation
- **Minimal inline comments**: Code is self-documenting with clear variable and function names
- **Streamlined logging**: Essential information only, reduced verbosity
- **Modern type hints**: Using Python 3.9+ syntax (`tuple` instead of `Tuple`, `| None` instead of `Optional`)
- **No example code**: All test/example blocks removed from production modules

## Core Modules

### Evolution Components

**`main_spatial_ea.py`** (1180 lines)
- Main evolutionary algorithm loop
- Population management and generation advancement
- Mating zone management with multiple relocation strategies
- Simulation orchestration
- Clean, streamlined implementation with minimal debug logging

**`evaluation.py`** (158 lines)
- Fitness evaluation using HyperNEAT controllers
- Target-based fitness calculation
- Simple, focused implementation

**`selection.py`** (207 lines)
- 5 selection strategies: fitness_based, age_based, probabilistic_age, energy_based, parents_die
- Each method with single-line docstring
- Clean conditional logic

**`parent_selection.py`** (306 lines)
- 3 pairing methods: random, proximity_pairing, mating_zone
- Multiple mating zone support
- Zone relocation strategies (static, generation_interval, event_driven)
- Offspring position calculation

**`crossover.py`** (70 lines)
- 6 crossover operators for HyperNEAT genomes
- Single-line docstrings for each function
- No example code, production-ready

**`mutation.py`** (101 lines)
- 8 mutation operators: uniform, gaussian, polynomial, creep, non_uniform, boundary, adaptive_gaussian, cauchy
- Concise implementation
- No verbose comments

### Support Modules

**`genetic_operators.py`**
- HyperNEAT genome creation and manipulation
- Crossover and mutation wrapper functions
- Individual cloning

**`spatial_individual.py`**
- Individual representation with spatial position
- Generation tracking, energy system, zone assignments
- Unique ID management

**`simulation_utils.py`**
- MuJoCo simulation helpers
- Population spawning and tracking
- Trajectory management
- Controller creation

**`periodic_boundary_utils.py`**
- Periodic boundary calculations
- Distance and wrapping functions
- Toroidal world support

**`visualization.py`**
- Trajectory plotting
- Mating zone visualization
- Clean plotting functions with minimal verbosity

**`incubation.py`**
- Non-spatial evolution bootstrap
- Population seeding for spatial EA
- Tournament selection

### Configuration and Data

**`ea_config.py`**
- YAML configuration loader
- Property-based access to all parameters
- Type-safe parameter retrieval

**`ea_config.yaml`**
- All EA parameters in one file
- Organized by category (population, selection, mating, energy, etc.)
- Well-commented with examples

**`evolution_data_collector.py`**
- Generation statistics tracking
- CSV and NPZ export
- Summary statistics calculation

### Experiment Infrastructure

**`experiment_runner.py`**
- Multi-run experiment orchestration
- Parallel execution support
- Statistical aggregation across runs
- Grid search functionality

**`run_experiments.py`**
- Predefined experiment configurations
- Command-line interface for running experiments
- Experiment comparison tools

### Visualization and Analysis

**`visualize_experiment.py`**
- Comprehensive visualization suite
- 12-panel overview plots
- Parameter analysis
- Publication-ready minimal plots

**`visualize_examples.py`**
- Interactive examples
- Usage demonstrations
- Data exploration patterns

## Code Quality Standards

### Docstring Style

All functions use concise single-line docstrings:

```python
def find_pairs(...) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Find mating pairs in the population."""
```

Detailed documentation is in the usage guides, not inline.

### Type Hints

Modern Python 3.9+ syntax throughout:

```python
def apply_selection(
    population: list[SpatialIndividual],
    current_positions: list[np.ndarray],
    method: str,
    paired_indices: set[int] | None = None
) -> tuple[list[SpatialIndividual], list[np.ndarray], int, list[float]]:
```

### Logging

Minimal, informative output:

```python
print(f"  Applying {method} selection: {initial_size} → {target_size}")
```

No debug-level verbosity in production code.

### Error Handling

Clear, actionable error messages:

```python
if len(zone_centers) < num_zones:
    print(f"  Warning: Could only place {len(zone_centers)} of {num_zones} zones")
```

## File Size Summary

Production code (excluding experiments and results):

```
main_spatial_ea.py          1180 lines
experiment_runner.py         931 lines
parent_selection.py          306 lines
evolution_data_collector.py  272 lines
selection.py                 207 lines
genetic_operators.py         193 lines
evaluation.py                158 lines
spatial_individual.py        145 lines
simulation_utils.py          142 lines
mutation.py                  101 lines
incubation.py                 95 lines
periodic_boundary_utils.py    90 lines
crossover.py                  70 lines
ea_config.py                 430 lines
```

Total: ~4,320 lines of clean, maintainable code

## Development Principles

1. **Readability First**: Code should be self-documenting
2. **Minimal Verbosity**: Comments only when necessary
3. **Type Safety**: Comprehensive type hints
4. **Modularity**: Clear separation of concerns
5. **No Redundancy**: DRY (Don't Repeat Yourself)
6. **Production Ready**: No test code in production modules

## Testing and Validation

All code has been tested to ensure:
- No functionality lost during cleanup
- All features working as before
- Pre-existing type annotations preserved
- Experiments run successfully
- Visualizations generate correctly

## Future Enhancements

The clean codebase makes it easier to:
- Add new selection strategies
- Implement new pairing methods
- Extend mutation operators
- Add visualization features
- Integrate new robot morphologies
- Support additional fitness functions
