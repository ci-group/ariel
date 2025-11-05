#!/usr/bin/env python3
import sys
import numpy as np
import mujoco
import time
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from simple_ea_class import SimpleEA
from evolutionary_algo_continuous import ContinuousContactEA
from ea_config import config


def run_generational_starter(
    num_generations: int = 20,
    population_size: int = 25
):
    print("\n" + "="*70)
    print("PHASE 1: GENERATIONAL EA STARTER POPULATION")
    print("="*70)
    print(f"Generating starter population with:")
    print(f"  Population size: {population_size}")
    print(f"  Generations: {num_generations}")
    print(f"  Selection: Tournament (standard generational EA)")
    print("="*70 + "\n")
    
    ea = SimpleEA(
        population_size=population_size,
        num_generations=num_generations,
        num_joints=8
    )
    
    ea.run_evolution()
    
    genotypes = [ind.genotype.copy() for ind in ea.population]
    
    fitness_values = [ind.fitness for ind in ea.population]
    
    print("\n" + "="*70)
    print("GENERATIONAL EA COMPLETE")
    print("="*70)
    print(f"Final population fitness:")
    print(f"  Mean: {np.mean(fitness_values):.3f}")
    print(f"  Max: {np.max(fitness_values):.3f}")
    print(f"  Min: {np.min(fitness_values):.3f}")
    print(f"  Std: {np.std(fitness_values):.3f}")
    print(f"\nExtracted {len(genotypes)} genotypes for continuous evolution")
    print("="*70 + "\n")
    
    return genotypes, fitness_values


def run_continuous_from_starter(
    starter_genotypes: list[np.ndarray],
    starter_fitness: list[float],
    duration: float = 1800.0,
    mating_cooldown: float = 30.0,
    fitness_update_interval: float = 30.0,
    checkpoint_interval: float = 300.0
):
    """
    Run continuous evolution starting from generational EA population.
    
    Args:
        starter_genotypes: Genotypes from generational EA
        starter_fitness: Fitness values from generational EA
        duration: Simulation duration in seconds
        mating_cooldown: Mating cooldown duration
        fitness_update_interval: Fitness update frequency
        checkpoint_interval: Checkpoint frequency
    """
    print("\n" + "="*70)
    print("PHASE 2: CONTINUOUS EVOLUTION FROM STARTER POPULATION")
    print("="*70)
    print(f"Configuration:")
    print(f"  Population size: {len(starter_genotypes)}")
    print(f"  Duration: {duration}s ({duration/60:.1f} minutes)")
    print(f"  Mating cooldown: {mating_cooldown}s")
    print(f"  Fitness updates: Every {fitness_update_interval}s")
    print(f"  Checkpoints: Every {checkpoint_interval}s")
    print(f"\nStarting fitness baseline:")
    print(f"  Mean: {np.mean(starter_fitness):.3f}")
    print(f"  Max: {np.max(starter_fitness):.3f}")
    print("="*70 + "\n")
    
    ea = ContinuousContactEA(
        population_size=len(starter_genotypes),
        num_joints=8,
        mating_cooldown=mating_cooldown,
        fitness_update_interval=fitness_update_interval,
        checkpoint_interval=checkpoint_interval,
        elite_archive_size=10
    )
    
    # Initialize with starter genotypes
    ea.population = ea.initialize_from_genotypes(starter_genotypes)
    
    ea.spawn_population()
    
    # Set initial fitness to zero (will accumulate from this point)
    ea.update_all_fitness()
    ea.update_elite_archive()
    
    mujoco.set_mjcb_control(lambda m, d: ea.controller(m, d))
    
    # Initial checkpoint
    ea.create_checkpoint()
    
    # Main evolution loop
    print("Starting continuous evolution...\n")
    start_time = time.time()
    
    sim_steps = int(duration / ea.model.opt.timestep)
    contact_check_interval = int(0.1 / ea.model.opt.timestep)
    trajectory_sample_steps = int(ea.trajectory_sample_interval / ea.model.opt.timestep)
    
    for step in range(sim_steps):
        mujoco.mj_step(ea.model, ea.data)
        
        # Check for contacts periodically
        if step % contact_check_interval == 0:
            ea.check_contacts()
        
        # Sample trajectories periodically
        if step % trajectory_sample_steps == 0:
            ea.sample_trajectories()
        
        # Periodic fitness updates
        if ea.data.time - ea.last_fitness_update >= ea.fitness_update_interval:
            ea.update_all_fitness()
            ea.update_elite_archive()
            ea.last_fitness_update = ea.data.time
        
        # Periodic checkpoints
        if ea.data.time - ea.last_checkpoint >= ea.checkpoint_interval:
            ea.create_checkpoint()
            ea.last_checkpoint = ea.data.time
    
    # Final checkpoint
    ea.update_all_fitness()
    ea.update_elite_archive()
    ea.create_checkpoint()
    
    elapsed_time = time.time() - start_time
    
    # Final statistics
    print(f"\n{'='*70}")
    print(f"CONTINUOUS EVOLUTION COMPLETE")
    print(f"{'='*70}")
    print(f"Simulated time: {duration}s ({duration/60:.1f} minutes)")
    print(f"Real time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    print(f"Speedup: {duration/elapsed_time:.2f}x")
    print(f"Total matings: {len(ea.mating_events)}")
    print(f"Matings per minute: {len(ea.mating_events)/(duration/60):.1f}")
    print(f"Best fitness: {ea.elite_archive[0][1]:.3f}" if ea.elite_archive else "N/A")
    print(f"{'='*70}\n")
    
    # Save results 
    ea.save_results()
    
    return ea


def save_comparison_results(
    starter_genotypes: list[np.ndarray],
    starter_fitness: list[float],
    continuous_ea: ContinuousContactEA,
    num_generations: int
):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    comparison_data = {
        'timestamp': timestamp,
        'generational_phase': {
            'num_generations': num_generations,
            'population_size': len(starter_genotypes),
            'final_fitness_mean': float(np.mean(starter_fitness)),
            'final_fitness_max': float(np.max(starter_fitness)),
            'final_fitness_min': float(np.min(starter_fitness)),
            'final_fitness_std': float(np.std(starter_fitness))
        },
        'continuous_phase': {
            'duration': continuous_ea.data.time if continuous_ea.data else 0,
            'total_matings': len(continuous_ea.mating_events),
            'num_checkpoints': len(continuous_ea.checkpoints),
            'elite_archive_best': float(continuous_ea.elite_archive[0][1]) if continuous_ea.elite_archive else 0.0,
            'final_fitness_mean': continuous_ea.checkpoints[-1]['fitness_mean'] if continuous_ea.checkpoints else 0.0,
            'final_fitness_max': continuous_ea.checkpoints[-1]['fitness_max'] if continuous_ea.checkpoints else 0.0
        }
    }
    
    comparison_file = f"{config.results_folder}/generational_to_continuous_{timestamp}.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nSaved comparison data to {comparison_file}")
    
    # Also save starter genotypes for reproducibility
    starter_file = f"{config.results_folder}/starter_genotypes_{timestamp}.npz"
    np.savez(
        starter_file,
        genotypes=np.array(starter_genotypes),
        fitness=np.array(starter_fitness),
        num_generations=num_generations
    )
    print(f"Saved starter genotypes to {starter_file}")


def main():
    """Main entry point for generational→continuous evolution pipeline."""
    parser = argparse.ArgumentParser(
        description="Run continuous evolution with generational EA starter population"
    )
    
    # Generational phase parameters
    parser.add_argument(
        '-g', '--starter-generations',
        type=int,
        default=20,
        help='Number of generations for starter population (default: 20)'
    )
    parser.add_argument(
        '-p', '--population',
        type=int,
        default=25,
        help='Population size (default: 25)'
    )
    parser.add_argument(
        '-m', '--movement',
        action='store_true',
        help='(Deprecated - SpatialEA always uses movement-based selection)'
    )
    

    parser.add_argument(
        '-d', '--duration',
        type=float,
        default=1800.0,
        help='Continuous evolution duration in seconds (default: 1800 = 30 min)'
    )
    parser.add_argument(
        '-c', '--cooldown',
        type=float,
        default=30.0,
        help='Mating cooldown in seconds (default: 30)'
    )
    parser.add_argument(
        '-f', '--fitness-interval',
        type=float,
        default=30.0,
        help='Fitness update interval in seconds (default: 30)'
    )
    parser.add_argument(
        '-k', '--checkpoint-interval',
        type=float,
        default=300.0,
        help='Checkpoint interval in seconds (default: 300 = 5 min)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("GENERATIONAL → CONTINUOUS EVOLUTION PIPELINE")
    print("="*70 + "\n")
    
    # Phase 1: Run generational EA
    starter_genotypes, starter_fitness = run_generational_starter(
        num_generations=args.starter_generations,
        population_size=args.population
    )
    
    # Phase 2: Run continuous evolution from starter
    continuous_ea = run_continuous_from_starter(
        starter_genotypes=starter_genotypes,
        starter_fitness=starter_fitness,
        duration=args.duration,
        mating_cooldown=args.cooldown,
        fitness_update_interval=args.fitness_interval,
        checkpoint_interval=args.checkpoint_interval
    )
    
    save_comparison_results(
        starter_genotypes=starter_genotypes,
        starter_fitness=starter_fitness,
        continuous_ea=continuous_ea,
        num_generations=args.starter_generations
    )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nResults saved to:")
    print(f"  - {config.results_folder}/generational_to_continuous_*.json")
    print(f"  - {config.results_folder}/starter_genotypes_*.npz")
    print(f"  - {config.results_folder}/continuous_checkpoints_*.json")
    print(f"  - {config.results_folder}/continuous_matings_*.json")
    print(f"  - {config.results_folder}/continuous_elite_*.npz")
    print(f"  - {config.results_folder}/continuous_population_*.npz")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
