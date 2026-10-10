from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List
import numpy as np
import numpy.typing as npt


class GenomeHandler(ABC):
    """Abstract base class for all genotype implementations."""
    
    def __init__(self, genome: npt.NDArray[Any] | None = None) -> None:
        """
        Initialize the genotype.

        Args:
            genome: The genome array. If None, creates empty genome.
        """
        self.fitness: float | None = None
        if genome is None:
            self.genome = self._generate_random_genome()
        else:
            self.genome = genome.copy()
    
    @abstractmethod
    def _generate_random_genome(self) -> npt.NDArray[Any]:
        """
        Generate a random genome.
        
        Returns:
            Randomly initialized genome array
        """
        pass

    @abstractmethod
    def generate_random_population(self, population_size: int) -> List[GenomeHandler]:
        """
        Generate a population of random genotypes.
        
        Args:
            population_size: Number of individuals to generate
            
        Returns:
            List of random genotype instances
        """
        pass
    
    @abstractmethod
    def crossover(self, other: GenomeHandler) -> GenomeHandler:
        """
        Perform crossover with another genotype to produce offspring.
        
        Args:
            other: The other parent genotype
            
        Returns:
            Child genotype from crossover
        """
        pass
    
    def crossover_population(
        self, 
        population1: List[GenomeHandler], 
        population2: List[GenomeHandler]
    ) -> List[GenomeHandler]:
        """
        Perform crossover on two populations.
        
        Args:
            population1: First parent population
            population2: Second parent population
            
        Returns:
            List of offspring genotypes
        """
        assert len(population1) == len(population2), "Populations must have same size"
        
        children = []
        for parent1, parent2 in zip(population1, population2):
            child = parent1.crossover(parent2)
            children.append(child)
        return children
    
    @abstractmethod
    def mutate(self) -> None:
        """Mutate this genotype in place."""
        pass
    
    def mutate_population(self, population: List[GenomeHandler]) -> None:
        """
        Mutate all individuals in a population.
        
        Args:
            population: List of genotypes to mutate
        """
        for individual in population:
            individual.mutate()
    
    @abstractmethod
    def copy(self) -> GenomeHandler:
        """
        Create a deep copy of this genotype.
        
        Returns:
            Copy of this genotype
        """
        pass
    
    @abstractmethod
    def is_valid(self) -> bool:
        """
        Check if the genotype represents a valid individual.
        
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def repair(self) -> None:
        """
        Repair the genotype to make it valid.
        Default implementation does nothing.
        """
        pass

    def compatibility_distance(self, other: GenomeHandler) -> float:
        """Compute compatibility distance to another genome for speciation."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement compatibility_distance"
        )