"""Genetic operators for the evolutionary algorithm."""

import numpy as np
from individual import SpatialIndividual


def create_individual(next_unique_id: int, num_joints: int, config):
    """Create a new individual with random genotype.
    
    Returns:
        tuple: (individual, new_next_unique_id)
    """
    individual = SpatialIndividual(unique_id=next_unique_id)
    next_unique_id += 1
    
    # Standard joint control parameters
    genotype = []
    for _ in range(num_joints):
        amplitude = np.random.uniform(config.amplitude_init_min, config.amplitude_init_max)
        frequency = np.random.uniform(config.frequency_init_min, config.frequency_init_max)
        phase = np.random.uniform(config.phase_min, config.phase_max)
        genotype.extend([amplitude, frequency, phase])
    
    # Add mating preference parameter (p_local)
    p_local = np.random.uniform(0.0, 1.0)
    
    # genotype.append(p_local) # QUESTION: Should we add location preference to genotype?
    
    individual.genotype = genotype
    individual.p_local = p_local
    individual.fitness = 0.0
    
    return individual, next_unique_id



def crossover(parent1: SpatialIndividual, parent2: SpatialIndividual, next_unique_id: int):
    """One-point crossover including p_local parameter.
    
    Returns:
        tuple: (child1, child2, new_next_unique_id)
    """
    child1 = SpatialIndividual(unique_id=next_unique_id)
    next_unique_id += 1
    child1.parent_ids = [parent1.unique_id, parent2.unique_id]
    
    child2 = SpatialIndividual(unique_id=next_unique_id)
    next_unique_id += 1
    child2.parent_ids = [parent1.unique_id, parent2.unique_id]
    
    # Crossover point (exclude last element which is p_local)
    crossover_point = np.random.randint(1, len(parent1.genotype) - 1)
    
    child1.genotype = (parent1.genotype[:crossover_point] + 
                      parent2.genotype[crossover_point:])
    child2.genotype = (parent2.genotype[:crossover_point] + 
                      parent1.genotype[crossover_point:])
    
    # Extract p_local
    child1.p_local = child1.genotype[-1]
    child2.p_local = child2.genotype[-1]
    
    return child1, child2, next_unique_id


def mutate(individual: SpatialIndividual, next_unique_id: int, config):
    """Mutation including p_local parameter.
    
    Returns:
        tuple: (mutated_individual, new_next_unique_id)
    """
    mutated = SpatialIndividual(unique_id=next_unique_id)
    next_unique_id += 1
    mutated.parent_ids = [individual.unique_id]
    mutated.genotype = individual.genotype.copy()
    
    for i in range(len(mutated.genotype)):
        if np.random.random() < config.mutation_rate:
            # Last element is p_local
            if i == len(mutated.genotype) - 1:
                # Mutate p_local
                mutated.genotype[i] += np.random.normal(0, 0.1)
                mutated.genotype[i] = np.clip(mutated.genotype[i], 0.0, 1.0)
            else:
                # Mutate joint control parameters
                mutated.genotype[i] += np.random.normal(0, config.mutation_strength)
                
                param_type = i % 3
                if param_type == 0:  # amplitude
                    mutated.genotype[i] = np.clip(
                        mutated.genotype[i], 
                        config.amplitude_min, 
                        config.amplitude_max
                    )
                elif param_type == 1:  # frequency
                    mutated.genotype[i] = np.clip(
                        mutated.genotype[i], 
                        config.frequency_min, 
                        config.frequency_max
                    )
                else:  # phase
                    mutated.genotype[i] = mutated.genotype[i] % config.phase_max
    
    mutated.p_local = mutated.genotype[-1]
    return mutated, next_unique_id