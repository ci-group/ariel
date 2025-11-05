"""
Generic EA Settings configuration using TypeVars for genome types and operators.
"""

from typing import TypeVar, Generic, Callable, Any
from pathlib import Path
from pydantic import BaseSettings

# Define type variables
TGenome = TypeVar('TGenome')  # Generic genome type
TMutation = TypeVar('TMutation', bound=Callable[[TGenome], TGenome])  # Mutation callable
TCrossover = TypeVar('TCrossover', bound=Callable[[TGenome, TGenome], tuple[TGenome, TGenome]])  # Crossover callable

# Database handling modes
DB_HANDLING_MODES = ["delete", "append", "error"]


class EASettings(BaseSettings, Generic[TGenome]):
    """
    Generic EA Settings class that can work with any genome type.
    
    TGenome: The genome type (e.g., TreeGenome, ListGenome, etc.)
    """
    quiet: bool = False

    # EC mechanisms
    is_maximisation: bool = True
    first_generation_id: int = 0
    num_of_generations: int = 100
    target_population_size: int = 100
    
    # Generic operators - these should be callables that work with TGenome
    mutation_operator: Callable[[TGenome], TGenome]
    crossover_operator: Callable[[TGenome, TGenome], tuple[TGenome, TGenome]]
    
    # Individual creation function
    create_individual: Callable[[], TGenome]

    # Task configuration
    task: str = "evolve_to_copy"
    target_robot_file_path: Path | None = Path("examples/target_robots/small_robot_8.json")

    # Data config
    output_folder: Path = Path.cwd() / "__data__"
    db_file_name: str = "database.db"
    db_file_path: Path = output_folder / db_file_name
    db_handling: str = "delete"  # One of DB_HANDLING_MODES

    class Config:
        # Allow arbitrary types for the callable fields
        arbitrary_types_allowed = True


# Example usage with TreeGenome
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

class TreeEASettings(EASettings[TreeGenome]):
    """Concrete EA settings for TreeGenome."""
    
    def __init__(self, **kwargs):
        # Set default operators for TreeGenome
        if 'mutation_operator' not in kwargs:
            kwargs['mutation_operator'] = self._default_tree_mutation
        if 'crossover_operator' not in kwargs:
            kwargs['crossover_operator'] = self._default_tree_crossover
        if 'create_individual' not in kwargs:
            kwargs['create_individual'] = TreeGenome.create_individual
        
        super().__init__(**kwargs)
    
    @staticmethod
    def _default_tree_mutation(genome: TreeGenome) -> TreeGenome:
        """Default mutation for TreeGenome."""
        mutator = TreeGenome.get_mutator_object()
        return mutator.mutate(genome)
    
    @staticmethod
    def _default_tree_crossover(parent1: TreeGenome, parent2: TreeGenome) -> tuple[TreeGenome, TreeGenome]:
        """Default crossover for TreeGenome."""
        crossover = TreeGenome.get_crossover_object()
        return crossover.crossover(parent1, parent2)


# Alternative approach: Factory function for creating EA settings
def create_ea_settings(
    genome_type: type[TGenome],
    mutation_func: Callable[[TGenome], TGenome],
    crossover_func: Callable[[TGenome, TGenome], tuple[TGenome, TGenome]],
    create_func: Callable[[], TGenome],
    **kwargs
) -> EASettings[TGenome]:
    """
    Factory function to create EA settings for any genome type.
    
    Args:
        genome_type: The genome class (e.g., TreeGenome)
        mutation_func: Function that takes a genome and returns a mutated genome
        crossover_func: Function that takes two genomes and returns two offspring
        create_func: Function that creates a new random genome
        **kwargs: Additional settings to override defaults
    """
    settings_data = {
        'mutation_operator': mutation_func,
        'crossover_operator': crossover_func,
        'create_individual': create_func,
        **kwargs
    }
    
    return EASettings[genome_type](**settings_data)


# Example usage:
if __name__ == "__main__":
    # Using the concrete TreeEASettings class
    tree_settings = TreeEASettings(
        num_of_generations=50,
        target_population_size=50,
        target_robot_file_path=Path("examples/target_robots/medium_robot_15.json")
    )
    
    print(f"Settings for {tree_settings.num_of_generations} generations")
    print(f"Population size: {tree_settings.target_population_size}")
    print(f"Mutation operator: {tree_settings.mutation_operator}")
    print(f"Crossover operator: {tree_settings.crossover_operator}")
    
    # Using the factory function
    tree_settings2 = create_ea_settings(
        genome_type=TreeGenome,
        mutation_func=TreeGenome.get_mutator_object().mutate,
        crossover_func=TreeGenome.get_crossover_object().crossover,
        create_func=TreeGenome.create_individual,
        num_of_generations=100,
        target_population_size=100
    )