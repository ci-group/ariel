from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ariel.ec.mutations import Mutation
    from ariel.ec.crossovers import Crossover
import networkx as nx

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
    def create_individual(**kwargs: dict) -> "Genotype":
        """Generate a new individual of this genotype type."""
        raise NotImplementedError("Individual generation not implemented for this genotype type.")

    @staticmethod
    @abstractmethod
    def to_digraph(robot_genotype: "Genotype", **kwargs: dict) -> nx.DiGraph:
        """Convert the genotype to a directed graph representation."""
        raise NotImplementedError("Conversion to directed graph not implemented for this genotype type.")
    
    @staticmethod
    @abstractmethod
    def to_json(robot_genotype: "Genotype", **kwargs: dict) -> str:
        """Convert the genotype to a JSON-serializable dictionary."""
        raise NotImplementedError("Conversion to JSON not implemented for this genotype type.")
    
    @staticmethod
    @abstractmethod
    def from_json(json_data: str, **kwargs: dict) -> "Genotype":
        """Create a genotype instance from a JSON-serializable dictionary."""
        raise NotImplementedError("Creation from JSON not implemented for this genotype type.")
    