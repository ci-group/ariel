from abc import ABC
from enum import Enum
from ariel.ec.a000 import Mutation
from ariel.ec.a005 import Crossover
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

class GenotypeEnum(Enum):
    TREE = TreeGenome
    #LSYSTEM = LSystemGenome  # Future implementation

class Genotype(ABC):
    """Interface for different genotype types."""

    @staticmethod
    def get_crossover_object() -> "Crossover":
        """Return the crossover operator for this genotype type."""
        raise NotImplementedError("Crossover operator not implemented for this genotype type.")
    
    @staticmethod
    def get_mutator_object() -> "Mutation":
        """Return the mutator operator for this genotype type."""
        raise NotImplementedError("Mutator operator not implemented for this genotype type.")
    
    @staticmethod
    def create_individual() -> "Genotype":
        """Generate a new individual of this genotype type."""
        raise NotImplementedError("Individual generation not implemented for this genotype type.")

    