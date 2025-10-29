from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ariel.ec.a000 import Mutation
    from ariel.ec.a005 import Crossover
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
import networkx as nx


class GenotypeEnum(Enum):
    TREE = TreeGenome
    #LSYSTEM = LSystemGenome  # Future implementation

class Genotype(ABC):
    """Interface for different genotype types."""

    @staticmethod
    @abstractmethod
    def get_crossover_object() -> "Crossover":
        """Return the crossover operator for this genotype type."""
        raise NotImplementedError("Crossover operator not implemented for this genotype type.")
    
    @staticmethod
    @abstractmethod
    def get_mutator_object() -> "Mutation":
        """Return the mutator operator for this genotype type."""
        raise NotImplementedError("Mutator operator not implemented for this genotype type.")
    
    @staticmethod
    @abstractmethod
    def create_individual() -> "Genotype":
        """Generate a new individual of this genotype type."""
        raise NotImplementedError("Individual generation not implemented for this genotype type.")

    @staticmethod
    @abstractmethod
    def to_digraph(robot_genotype: "Genotype", **kwargs: dict) -> nx.DiGraph:
        """Convert the genotype to a directed graph representation."""
        raise NotImplementedError("Conversion to directed graph not implemented for this genotype type.")
    