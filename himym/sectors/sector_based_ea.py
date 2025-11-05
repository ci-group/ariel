"""
Grid-based spatial evolutionary algorithm with sector-based mating preferences.

Main entry point for running the evolutionary algorithm.

Key Features:
- 3×3 grid of sectors (discrete spatial structure)
- Individuals have evolving mating preferences (local vs. distant)
- Movement phase: individuals migrate between adjacent sectors
- Sector-based mate selection using preference parameter
- Offspring spawned in parent sectors

Changes from original:
1. Replaced continuous 2D space with discrete 3×3 grid
2. Added p_local parameter to genotype (mating preference)
3. Added movement phase where individuals change sectors
4. Modified mating to use sector membership and preferences
5. Updated visualization to show grid structure
"""

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

from evolutionary_algorithm import GridSpatialEA
from ea_config import config


def main():
    """Main function to run the evolutionary algorithm."""

    # Create and initialize EA
    grid_ea = GridSpatialEA(
        population_size=config.population_size,
        num_generations=config.num_generations,
    )
    grid_ea.initialize_population()
    
    # Run evolution
    grid_ea.run_evolution()
    
    # Demonstrate results
    grid_ea.demonstrate_best()
    grid_ea.demonstrate_final_population()
    
    # Plot results (optional - uncomment to generate plots)
    # grid_ea.plot_fitness_evolution()
    # grid_ea.plot_movement_analysis()
    
    print(f"\n{'='*60}")
    print("ALL TASKS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__": 
    main()
