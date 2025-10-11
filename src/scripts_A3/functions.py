"""
    This file contains some functions that will be used when running the evolutionary algorithm
"""

# import packages
import numpy as np


def crossover_robot_body(parent1, parent2, alpha):
    """
    Computes whole arithmatic crossover on two robot genotypes
    to produce two offspring.

    Parameters
    ----------
    parent1 : list of lists of floats (between 0 and 1)
        The first parents body genotype. Composed of 3 lists each of lenght 64
    parent2 : list of lists of floats (between 0 and 1)
        The second parents body genotype. Composed of 3 lists each of lenght 64
    alpha : float
        Mixing coefficient between [0,1]. Values > 0.5 bias parent1 and values < 0.5 bias parent 2
    
    Returns
    ----------
    offspring1 : list of list of float
        The first offspring's body genotype, created by blending `parent1` and `parent2`
    offspring2 : list of list of float
        The second offspring's body genotype, complementary to `offspring1`
    
    """
    
    offspring1 = []
    offspring2 = []
    for i in range(len(parent1)):

        # transform parent1 and parent2 into arrays to do element wise mutiplication
        # transform into a list before appending to keep original format
        offspring1.append(((np.array(parent1[i]) * alpha) + np.array(parent2[i]) * (1-alpha)))#.tolist())
        offspring2.append((np.array(parent1[i]) * (1-alpha) + (np.array(parent2[i]) * alpha)))#.tolist())
    return offspring1, offspring2


def fitness_evalutation(current, target = [5,0,0.5]):
    """
    Calclulates the Euclidian distance between the robots current position and the target position.
    This fitness value should be minimized.

    Parameters
    ----------
    current : list 
        Robots position after the simulation, coordinates [x,y,z]
    target : list
        Robots target position, default [5,0,0.5]
    """
    fitness = np.sqrt((current[0] - target[0])**2 + (current[1] - target[1])**2)
    return fitness



# testing functions
P1 = (0, 0, 0)  
P2 = (1, 0, 0)  
P3 = (0, 2, 0.1)  
P4 = (-1, -1, 0.2)  
print(fitness_evalutation(P1), fitness_evalutation(P2), fitness_evalutation(P3), fitness_evalutation(P4))


p1 = [[1,2,3], [4,5,6], [7,8,9]]
p2 = [[9,8,7], [6,5,4], [3,2,1]]
off1,off2 = crossover_robot_body(p1,p2,0.5)
# print(off1)
# print(off2)