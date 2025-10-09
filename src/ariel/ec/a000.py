"""TODO(jmdm): description of script."""

# Standard library
from collections.abc import Sequence
from pathlib import Path
from typing import cast

# Third-party libraries
import numpy as np
from pydantic_settings import BaseSettings
from rich.console import Console
from rich.traceback import install
import copy
from ariel.ec.genotypes.tree.tree_genome import TreeGenome, TreeNode
import ariel.body_phenotypes.robogen_lite.config as pheno_config

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)
DB_NAME = "database.db"
DB_PATH: Path = DATA / DB_NAME
SEED = 42

# Global functions
install(width=180)
console = Console()
RNG = np.random.default_rng(SEED)

# Type Aliases
type Integers = Sequence[int]
type Floats = Sequence[float]


class IntegersGeneratorSettings(BaseSettings):
    integers_endpoint: bool = True
    choice_replace: bool = True
    choice_shuffle: bool = False


config = IntegersGeneratorSettings()


class IntegersGenerator:
    @staticmethod
    def integers(
        low: int,
        high: int,
        size: int | Sequence[int] | None = 1,
        *,
        endpoint: bool | None = None,
    ) -> Integers:
        endpoint = endpoint or config.integers_endpoint
        generated_values = RNG.integers(
            low=low,
            high=high,
            size=size,
            endpoint=endpoint,
        )
        return cast("Integers", generated_values.astype(int).tolist())

    @staticmethod
    def choice(
        value_set: int | Integers,
        size: int | Sequence[int] | None = 1,
        probabilities: Sequence[float] | None = None,
        axis: int = 0,
        *,
        replace: bool | None = None,
        shuffle: bool | None = None,
    ) -> Integers:
        replace = replace or config.choice_replace
        shuffle = shuffle or config.choice_shuffle
        generated_values = np.array(
            RNG.choice(
                a=value_set,
                size=size,
                replace=replace,
                p=probabilities,
                axis=axis,
                shuffle=shuffle,
            ),
        )
        return cast("Integers", generated_values.astype(int).tolist())


class IntegerMutator:
    @staticmethod
    def random_swap(
        individual: Integers,
        low: int,
        high: int,
        mutation_probability: float,
    ) -> Integers:
        shape = np.asarray(individual).shape
        mutator = RNG.integers(
            low=low,
            high=high,
            size=shape,
            endpoint=True,
        )
        mask = RNG.choice(
            [True, False],
            size=shape,
            p=[mutation_probability, 1 - mutation_probability],
        )
        new_genotype = np.where(mask, mutator, individual).astype(int).tolist()
        return cast("Integers", new_genotype.astype(int).tolist())

    @staticmethod
    def integer_creep(
        individual: Integers,
        span: int,
        mutation_probability: float,
    ) -> Integers:
        # Prep
        ind_arr = np.array(individual)
        shape = ind_arr.shape

        # Generate mutation values
        mutator = RNG.integers(
            low=1,
            high=span,
            size=shape,
            endpoint=True,
        )

        # Include negative mutations
        sub_mask = RNG.choice(
            [-1, 1],
            size=shape,
        )

        # Determine which positions to mutate
        do_mask = RNG.choice(
            [1, 0],
            size=shape,
            p=[mutation_probability, 1 - mutation_probability],
        )
        mutation_mask = mutator * sub_mask * do_mask
        new_genotype = ind_arr + mutation_mask
        return cast("Integers", new_genotype.astype(int).tolist())


class TreeGenerator:
    @staticmethod
    def __call__(*args, **kwargs) -> TreeGenome:
        return TreeGenome.default_init()

    @staticmethod
    def default():
        return TreeGenome.default_init()

    @staticmethod
    def linear_chain(length: int = 3) -> TreeGenome:
        """Generate a linear chain of modules (snake-like)."""
        genome = TreeGenome.default_init()  # Start with CORE
        current_node = genome.root

        for i in range(length):
            module_type = RNG.choice([pheno_config.ModuleType.BRICK, pheno_config.ModuleType.HINGE])
            rotation = RNG.choice(list(pheno_config.ModuleRotationsIdx))
            module = pheno_config.ModuleInstance(type=module_type, rotation=rotation, links={})

            # Always attach to FRONT face for linear chain
            if pheno_config.ModuleFaces.FRONT in current_node.available_faces():
                child = TreeNode(module, depth=current_node._depth + 1)
                current_node._set_face(pheno_config.ModuleFaces.FRONT, child)
                current_node = child

        return genome

    @staticmethod
    def star_shape(num_arms: int = 3) -> TreeGenome:
        """Generate a star-shaped tree with arms radiating from center."""
        genome = TreeGenome.default_init()  # Start with CORE
        available_faces = genome.root.available_faces()

        # Limit arms to available faces
        actual_arms = min(num_arms, len(available_faces))
        selected_faces = RNG.choice(available_faces, size=actual_arms, replace=False)

        for face in selected_faces:
            module_type = RNG.choice([pheno_config.ModuleType.BRICK, pheno_config.ModuleType.HINGE])
            rotation = RNG.choice(list(pheno_config.ModuleRotationsIdx))
            module = pheno_config.ModuleInstance(type=module_type, rotation=rotation, links={})

            child = TreeNode(module, depth=1)
            genome.root._set_face(face, child)

        return genome

    @staticmethod
    def binary_tree(depth: int = 2) -> TreeGenome:
        """Generate a binary-like tree structure."""
        def build_subtree(current_depth: int, max_depth: int) -> TreeNode | None:
            if current_depth >= max_depth:
                return None

            module_type = RNG.choice([pheno_config.ModuleType.BRICK, pheno_config.ModuleType.HINGE])
            rotation = RNG.choice(list(pheno_config.ModuleRotationsIdx))
            module = pheno_config.ModuleInstance(type=module_type, rotation=rotation, links={})

            node = TreeNode(module, depth=current_depth)
            available_faces = node.available_faces()

            # Add 1-2 children randomly
            if available_faces and current_depth < max_depth - 1:
                num_children = RNG.integers(1, min(3, len(available_faces) + 1))
                selected_faces = RNG.choice(available_faces, size=num_children, replace=False)

                for face in selected_faces:
                    child = build_subtree(current_depth + 1, max_depth)
                    if child:
                        node._set_face(face, child)

            return node

        genome = TreeGenome.default_init()

        # Add children to root
        available_faces = genome.root.available_faces()
        if available_faces:
            num_children = RNG.integers(1, min(3, len(available_faces) + 1))
            selected_faces = RNG.choice(available_faces, size=num_children, replace=False)

            for face in selected_faces:
                child = build_subtree(1, depth)
                if child:
                    genome.root._set_face(face, child)

        return genome

    @staticmethod
    def random_tree(max_depth: int = 4, branching_prob: float = 0.7) -> TreeGenome:
        """Generate a random tree with pheno_configurable branching probability."""
        def build_random_subtree(current_depth: int) -> TreeNode | None:
            if current_depth >= max_depth:
                return None

            module_type = RNG.choice([pheno_config.ModuleType.BRICK, pheno_config.ModuleType.HINGE])
            rotation = RNG.choice(list(pheno_config.ModuleRotationsIdx))
            module = pheno_config.ModuleInstance(type=module_type, rotation=rotation, links={})

            node = TreeNode(module, depth=current_depth)
            available_faces = node.available_faces()

            # Randomly decide to add children
            for face in available_faces:
                if RNG.random() < branching_prob:
                    child = build_random_subtree(current_depth + 1)
                    if child:
                        node._set_face(face, child)

            return node

        genome = TreeGenome.default_init()

        # Add children to root
        available_faces = genome.root.available_faces()
        for face in available_faces:
            if RNG.random() < branching_prob:
                child = build_random_subtree(1)
                if child:
                    genome.root._set_face(face, child)

        return genome

class TreeMutator:
    @staticmethod
    def random_subtree_replacement(
        individual: TreeGenome,
        max_subtree_depth: int = 1,
    ) -> TreeGenome:
        """Replace a random subtree with a new random subtree."""
        if individual.root is None:
            return individual

        new_individual = copy.copy(individual)

        # Collect all nodes in the tree
        all_nodes = new_individual.root.get_all_nodes(exclude_root=True)

        # Select a random node to replace (excluding root)
        if len(all_nodes) <= 1:
            return new_individual

        node_to_replace = RNG.choice(all_nodes[1:])  # Avoid replacing root

        # Generate a new random subtree
        new_subtree = TreeNode.random_tree_node(max_depth=max_subtree_depth)

        with new_individual.root.enable_replacement():
            new_individual.root.replace_node(node_to_replace, new_subtree)

        return new_individual

def test() -> None:
    """Entry point."""
    console.log(IntegersGenerator.integers(-5, 5, 5))
    example = IntegersGenerator.choice([1, 3, 4], (2, 5))
    console.log(example)
    example2 = IntegerMutator.integer_creep(
        example,
        span=1,
        mutation_probability=1,
    )
    console.log(example2)

    console.rule("[bold blue]Tree Generator Examples")

    genome = TreeGenome()
    genome.root = TreeNode(pheno_config.ModuleInstance(type=pheno_config.ModuleType.BRICK, rotation=pheno_config.ModuleRotationsIdx.DEG_90, links={}))
    genome.root.front = TreeNode(pheno_config.ModuleInstance(type=pheno_config.ModuleType.BRICK, rotation=pheno_config.ModuleRotationsIdx.DEG_45, links={}))
    genome.root.left = TreeNode(pheno_config.ModuleInstance(type=pheno_config.ModuleType.BRICK, rotation=pheno_config.ModuleRotationsIdx.DEG_45, links={}))
    tree_mutator = TreeMutator()
    mutated_genome = tree_mutator.random_subtree_replacement(genome, max_subtree_depth=1)
    console.log("Original Genome:", genome)
    console.log("Mutated Genome:", mutated_genome)




if __name__ == "__main__":
    test()
