"""TODO(jmdm): description of script."""

# Standard library
from abc import ABC, abstractmethod
from pathlib import Path

# Third-party libraries
import numpy as np
from rich.console import Console
from rich.traceback import install

# Local libraries
from ariel.ec.genotypes.genotype import Genotype
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.a000 import IntegersGenerator
from ariel.ec.a001 import JSONIterable

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

class Crossover(ABC):
    which_crossover: str = ""

    @abstractmethod
    @classmethod
    def __call__(
        cls,
        parent_i: Genotype,
        parent_j: Genotype,
        **kwargs: dict,
    ) -> tuple[Genotype, Genotype]:
        """Perform crossover on two genotypes.

        Parameters
        ----------
        parent_i : Genotype
            The first parent genotype (list or nested list of integers).
        parent_j : Genotype
            The second parent genotype (list or nested list of integers).

        Returns
        -------
        tuple[Genotype, Genotype]
            Two child genotypes resulting from the crossover.
        """
        pass

class IntegerCrossover(Crossover):
    @classmethod
    def __call__(
        cls,
        parent_i: Genotype,
        parent_j: Genotype,
        **kwargs: dict,
    ) -> tuple[Genotype, Genotype]:
        if cls.which_crossover == "one_point":
            return cls.one_point(parent_i, parent_j, **kwargs)
        else:
            msg = f"Crossover type '{cls.which_crossover}' not recognized."
            raise ValueError(msg)

    @staticmethod
    def one_point(
        parent_i: Genotype,
        parent_j: Genotype,
    ) -> tuple[Genotype, Genotype]:
        # Prep
        parent_i_arr_shape = np.array(parent_i).shape
        parent_j_arr_shape = np.array(parent_j).shape
        parent_i_arr = np.array(parent_i).flatten().copy()
        parent_j_arr = np.array(parent_j).flatten().copy()

        # Ensure parents have the same length
        if parent_i_arr_shape != parent_j_arr_shape:
            msg = "Parents must have the same length"
            raise ValueError(msg)

        # Select crossover point
        crossover_point = RNG.integers(0, len(parent_i_arr))

        # Copy over parents
        child1 = parent_i_arr.copy()
        child2 = parent_j_arr.copy()

        # Perform crossover
        child1[crossover_point:] = parent_j_arr[crossover_point:]
        child2[crossover_point:] = parent_i_arr[crossover_point:]

        # Correct final shape
        child1 = child1.reshape(parent_i_arr_shape).astype(int).tolist()
        child2 = child2.reshape(parent_j_arr_shape).astype(int).tolist()
        return child1, child2

class TreeCrossover(Crossover):
    @classmethod
    def __call__(
        cls,
        parent_i: Genotype,
        parent_j: Genotype,
        **kwargs: dict,
    ) -> tuple[Genotype, Genotype]:
        if cls.which_crossover == "koza_default":
            return cls.koza_default(parent_i, parent_j, **kwargs)
        elif cls.which_crossover == "normal":
            return cls.normal(parent_i, parent_j, **kwargs)
        else:
            msg = f"Crossover type '{cls.which_crossover}' not recognized."
            raise ValueError(msg)

    @staticmethod
    def koza_default(
        parent_i: Genotype,
        parent_j: Genotype,
        koza_internal_node_prob: float = 0.9,
    ) -> tuple[Genotype, Genotype]:
        """
        Koza default:
            -   In Parent A: choose an internal node with high probability (e.g., 90%) excluding root.
                Falls back to any node if A has no internal nodes.
            -   In Parent B: choose any node uniformly (internal or terminal).

        Forcing at least one internal node increases the chance you actually change structure
        (not just swapping a leaf for a leaf), while letting the other parent be unrestricted adds variety.
        """
        parent_i_root, parent_j_root = parent_i.root, parent_j.root
        parent_i_internal_nodes = parent_i_root.get_internal_nodes(mode="dfs", exclude_root=True)

        if RNG.random() > koza_internal_node_prob and parent_i_internal_nodes:
            node_a = RNG.choice(parent_i_internal_nodes)
        else:
            node_a = RNG.choice(parent_i_root.get_all_nodes(mode="dfs", exclude_root=True))

        parent_j_all_nodes = parent_j_root.get_all_nodes()
        node_b = RNG.choice(parent_j_all_nodes)
        if node_a is None or node_b is None:
            # If either tree is just a root, return copies of parents
            return parent_i.copy(), parent_j.copy()

        parent_i_old = parent_i.copy()
        parent_j_old = parent_j.copy()
        child1 = parent_i
        child2 = parent_j

        with child1.root.enable_replacement():
            child1.root.replace_node(node_a, node_b)
        with child2.root.enable_replacement():
            child2.root.replace_node(node_b, node_a)

        parent_i = parent_i_old
        parent_j = parent_j_old
        return child1, child2
    
    @staticmethod
    def normal(
        parent_i: Genotype,
        parent_j: Genotype,
    ) -> tuple[Genotype, Genotype]:
        """
        Normal tree crossover:
            - Pick a random node from Parent A (uniform over all nodes).
            - Pick a random node from Parent B (uniform over all nodes).
            - Swap the selected subtrees.

        Returns two children produced by swapping the chosen subtrees.
        """
        parent_i_root, parent_j_root = parent_i.root, parent_j.root

        # Uniformly choose any node (root, internal, or leaf)
        node_a = RNG.choice(parent_i_root.get_all_nodes(exclude_root=True))
        node_b = RNG.choice(parent_j_root.get_all_nodes(exclude_root=True))

        if not node_a or not node_b:
            # If either tree is just a root, return copies of parents
            return parent_i.copy(), parent_j.copy()

        # Preserve originals (same pattern as in koza_default)
        parent_i_old = parent_i.copy()
        parent_j_old = parent_j.copy()
        child1 = parent_i
        child2 = parent_j

        # Perform the swap
        with child1.root.enable_replacement():
            child1.root.replace_node(node_a, node_b)
        with child2.root.enable_replacement():
            child2.root.replace_node(node_b, node_a)

        # Restore parent handles for caller (as in your koza_default)
        parent_i = parent_i_old
        parent_j = parent_j_old
        return child1, child2


def tree_main():
    import ariel.body_phenotypes.robogen_lite.config as config
    from ariel.ec.genotypes.tree.tree_genome import TreeNode, TreeGenome

    # Create first tree
    genome1 = TreeGenome()
    genome1.root = TreeNode(
        config.ModuleInstance(type=config.ModuleType.CORE, rotation=config.ModuleRotationsIdx.DEG_0, links={}))
    genome1.root.front = TreeNode(
        config.ModuleInstance(type=config.ModuleType.BRICK, rotation=config.ModuleRotationsIdx.DEG_90, links={}))
    genome1.root.back = TreeNode(
        config.ModuleInstance(type=config.ModuleType.HINGE, rotation=config.ModuleRotationsIdx.DEG_45, links={}))

    # Create second tree
    genome2 = TreeGenome()
    genome2.root = TreeNode(
        config.ModuleInstance(type=config.ModuleType.CORE, rotation=config.ModuleRotationsIdx.DEG_0, links={}))
    genome2.root.right = TreeNode(
        config.ModuleInstance(type=config.ModuleType.BRICK, rotation=config.ModuleRotationsIdx.DEG_180, links={}))
    genome2.root.back = TreeNode(
        config.ModuleInstance(type=config.ModuleType.HINGE, rotation=config.ModuleRotationsIdx.DEG_270, links={}))

    console.log("Parent 1:", genome1)
    console.log("Parent 2:", genome2)

    genome2.root.replace_node(genome1, genome2)

    # Perform crossover
    child1, child2 = TreeCrossover.koza_default(genome1, genome2)

    console.log("Child 1:", child1)
    console.log("Child 2:", child2)


def main() -> None:
    """Entry point."""
    p1 = IntegersGenerator.integers(-5, 5, (2, 5))
    p2 = IntegersGenerator.choice([1, 3, 4], (2, 5))
    console.log(p1, p2)

    c1, c2 = Crossover.one_point(p1, p2)
    console.log(c1, c2)


if __name__ == "__main__":
    tree_main()
