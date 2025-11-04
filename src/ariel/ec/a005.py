"""TODO(jmdm): description of script."""
from __future__ import annotations

# Standard library
from abc import ABC, abstractmethod
from pathlib import Path
import random

# Third-party libraries
import numpy as np
from rich.console import Console
from rich.traceback import install

from typing import TYPE_CHECKING

# Local libraries
from ariel.ec.genotypes.lsystem.l_system_genotype import LSystemDecoder
if TYPE_CHECKING:
    from ariel.ec.genotypes.genotype import Genotype
    from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.a000 import IntegersGenerator

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
    crossovers_mapping: dict[str, function] = NotImplemented
    which_crossover: str = ""

    @classmethod
    def set_which_crossover(cls, crossover_type: str) -> None:
        cls.which_crossover = crossover_type

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.crossovers_mapping = {
            name: getattr(cls, name)
            for name, val in cls.__dict__.items()
            if isinstance(val, staticmethod)
        }

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
        if cls.which_crossover in cls.crossovers_mapping:
            return cls.crossovers_mapping[cls.which_crossover](
                parent_i,
                parent_j,
                **kwargs
            )
        else:
            msg = f"Crossover type '{cls.which_crossover}' not recognized."
            raise ValueError(msg)

class IntegerCrossover(Crossover):
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
    @staticmethod
    def koza_default(
        parent_i: TreeGenome,
        parent_j: TreeGenome,
        koza_internal_node_prob: float = 0.9,
    ) -> tuple[TreeGenome, TreeGenome]:
        """
        Koza default:
            -   In Parent A: choose an internal node with high probability (e.g., 90%) excluding root.
                Falls back to any node if A has no internal nodes.
            -   In Parent B: choose any node uniformly (internal or terminal).

        Forcing at least one internal node increases the chance you actually change structure
        (not just swapping a leaf for a leaf), while letting the other parent be unrestricted adds variety.
        """
        parent_i_root, parent_j_root = parent_i.root, parent_j.root

        nodes_a = parent_i_root.get_all_nodes(exclude_root=True)
        nodes_b = parent_j_root.get_all_nodes(exclude_root=True)

        # If either tree is just a root, return copies of parents
        if not nodes_a or not nodes_b:
            return parent_i.copy(), parent_j.copy()

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
        parent_i: TreeGenome,
        parent_j: TreeGenome,
    ) -> tuple[TreeGenome, TreeGenome]:
        """
        Normal tree crossover:
            - Pick a random node from Parent A (uniform over all nodes).
            - Pick a random node from Parent B (uniform over all nodes).
            - Swap the selected subtrees.

        Returns two children produced by swapping the chosen subtrees.
        """
        parent_i_root, parent_j_root = parent_i.root, parent_j.root

        nodes_a = parent_i_root.get_all_nodes(exclude_root=True)
        nodes_b = parent_j_root.get_all_nodes(exclude_root=True)

        # If either tree is just a root, return copies of parents
        if not nodes_a or not nodes_b:
            return parent_i.copy(), parent_j.copy()

        # Uniformly choose any node (root, internal, or leaf)
        node_a = RNG.choice(nodes_a)
        node_b = RNG.choice(nodes_b)

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
    
class LSystemCrossover(Crossover):
    @staticmethod
    def crossover_uniform_rules_lsystem(lsystem_parent1,lsystem_parent2,mutation_rate):
        axiom_offspring1="C"
        axiom_offspring2="C"
        rules_offspring1={}
        rules_offspring2={}
        iter_offspring1=0
        iter_offspring2=0
        if random.random()>mutation_rate:
            rules_offspring1['C']=lsystem_parent2.rules['C']
            rules_offspring2['C']=lsystem_parent1.rules['C']
            iter_offspring1+=lsystem_parent2.iterations
            iter_offspring2+=lsystem_parent1.iterations
        else:
            rules_offspring1['C']=lsystem_parent1.rules['C']
            rules_offspring2['C']=lsystem_parent2.rules['C']
            iter_offspring1+=lsystem_parent1.iterations
            iter_offspring2+=lsystem_parent2.iterations
        if random.random()>mutation_rate:
            rules_offspring1['B']=lsystem_parent2.rules['B']
            rules_offspring2['B']=lsystem_parent1.rules['B']
            iter_offspring1+=lsystem_parent2.iterations
            iter_offspring2+=lsystem_parent1.iterations
        else:
            rules_offspring1['B']=lsystem_parent1.rules['B']
            rules_offspring2['B']=lsystem_parent2.rules['B']
            iter_offspring1+=lsystem_parent1.iterations
            iter_offspring2+=lsystem_parent2.iterations
        if random.random()>mutation_rate:
            rules_offspring1['H']=lsystem_parent2.rules['H']
            rules_offspring2['H']=lsystem_parent1.rules['H']
            iter_offspring1+=lsystem_parent2.iterations
            iter_offspring2+=lsystem_parent1.iterations
        else:
            rules_offspring1['H']=lsystem_parent1.rules['H']
            rules_offspring2['H']=lsystem_parent2.rules['H']
            iter_offspring1+=lsystem_parent1.iterations
            iter_offspring2+=lsystem_parent2.iterations
        if random.random()>mutation_rate:
            rules_offspring1['N']=lsystem_parent2.rules['N']
            rules_offspring2['N']=lsystem_parent1.rules['N']
            iter_offspring1+=lsystem_parent2.iterations
            iter_offspring2+=lsystem_parent1.iterations
        else:
            rules_offspring1['N']=lsystem_parent1.rules['N']
            rules_offspring2['N']=lsystem_parent2.rules['N']
            iter_offspring1+=lsystem_parent1.iterations
            iter_offspring2+=lsystem_parent2.iterations
        iteration_offspring1=int(iter_offspring1/4)
        iteration_offspring2=int(iter_offspring2/4)
        offspring1=LSystemDecoder(axiom_offspring1,rules_offspring1,iteration_offspring1,lsystem_parent1.max_elements,lsystem_parent1.max_depth,lsystem_parent1.verbose)
        offspring2=LSystemDecoder(axiom_offspring2,rules_offspring2,iteration_offspring2,lsystem_parent2.max_elements,lsystem_parent2.max_depth,lsystem_parent2.verbose)
        return offspring1,offspring2

    @staticmethod
    def crossover_uniform_genes_lsystem(lsystem_parent1,lsystem_parent2,mutation_rate):
        axiom_offspring1="C"
        axiom_offspring2="C"
        rules_offspring1={}
        rules_offspring2={}
        iter_offspring1=0
        iter_offspring2=0

        rules_parent1 = lsystem_parent1.rules["C"].split()
        rules_parent2 = lsystem_parent2.rules["C"].split()
        enh_parent1 = []
        enh_parent2 = []
        i = 0
        while i < len(rules_parent1):
            if rules_parent1[i][:4] in ['addf','addk','addl','addr','addb','addt']:
                new_token= rules_parent1[i] + " " + rules_parent1[i+1]
                enh_parent1.append(new_token)
                i+=1
            if rules_parent1[i][:4] in ['movf','movk','movl','movr','movb','movt']:
                enh_parent1.append(rules_parent1[i])
            if rules_parent1[i]=='C':
                enh_parent1.append(rules_parent1[i])
            i+=1 
        i = 0
        while i < len(rules_parent2):
            if rules_parent2[i][:4] in ['addf','addk','addl','addr','addb','addt']:
                new_token= rules_parent2[i] + " " + rules_parent2[i+1]
                enh_parent2.append(new_token)
                i+=1
            if rules_parent2[i][:4] in ['movf','movk','movl','movr','movb','movt']:
                enh_parent2.append(rules_parent2[i])
            if rules_parent2[i]=='C':
                enh_parent2.append(rules_parent2[i])
            i+=1 
        r_offspring1=""
        r_offspring2=""
        le_common = min(len(enh_parent1),len(enh_parent2))
        for i in range(0,le_common):
            if random.random()>mutation_rate:
                r_offspring1+=enh_parent2[i]+" "
                r_offspring2+=enh_parent1[i]+" "
            else:
                r_offspring1+=enh_parent1[i]+" "
                r_offspring2+=enh_parent2[i]+" "
        if len(enh_parent1)>le_common:
            for j in range(le_common,len(enh_parent1)):
                if random.random()>mutation_rate:
                    r_offspring1+=enh_parent1[j]+" "
                else:   
                    r_offspring2+=enh_parent1[j]+" "
        if len(enh_parent2)>le_common:
            for j in range(le_common,len(enh_parent2)):
                if random.random()>mutation_rate:
                    r_offspring1+=enh_parent2[j]+" "
                else:
                    r_offspring2+=enh_parent2[j]+" "
        rules_offspring1['C']=r_offspring1
        rules_offspring2['C']=r_offspring2

        rules_parent1 = lsystem_parent1.rules["B"].split()
        rules_parent2 = lsystem_parent2.rules["B"].split()
        enh_parent1 = []
        enh_parent2 = []
        i = 0
        while i < len(rules_parent1):
            if rules_parent1[i][:4] in ['addf','addk','addl','addr','addb','addt']:
                new_token= rules_parent1[i] + " " + rules_parent1[i+1]
                enh_parent1.append(new_token)
                i+=1
            if rules_parent1[i][:4] in ['movf','movk','movl','movr','movb','movt']:
                enh_parent1.append(rules_parent1[i])
            if rules_parent1[i]=='B':
                enh_parent1.append(rules_parent1[i])
            i+=1 
        i = 0
        while i < len(rules_parent2):
            if rules_parent2[i][:4] in ['addf','addk','addl','addr','addb','addt']:
                new_token= rules_parent2[i] + " " + rules_parent2[i+1]
                enh_parent2.append(new_token)
                i+=1
            if rules_parent2[i][:4] in ['movf','movk','movl','movr','movb','movt']:
                enh_parent2.append(rules_parent2[i])
            if rules_parent2[i]=='B':
                enh_parent2.append(rules_parent2[i])
            i+=1 
        r_offspring1=""
        r_offspring2=""
        le_common = min(len(enh_parent1),len(enh_parent2))
        for i in range(0,le_common):
            if random.random()>mutation_rate:
                r_offspring1+=enh_parent2[i]+" "
                r_offspring2+=enh_parent1[i]+" "
            else:
                r_offspring1+=enh_parent1[i]+" "
                r_offspring2+=enh_parent2[i]+" "
        if len(enh_parent1)>le_common:
            for j in range(le_common,len(enh_parent1)):
                if random.random()>mutation_rate:
                    r_offspring1+=enh_parent1[j]+" "
                else:   
                    r_offspring2+=enh_parent1[j]+" "
        if len(enh_parent2)>le_common:
            for j in range(le_common,len(enh_parent2)):
                if random.random()>mutation_rate:
                    r_offspring1+=enh_parent2[j]+" "
                else:
                    r_offspring2+=enh_parent2[j]+" "
        rules_offspring1['B']=r_offspring1
        rules_offspring2['B']=r_offspring2

        rules_parent1 = lsystem_parent1.rules["H"].split()
        rules_parent2 = lsystem_parent2.rules["H"].split()
        enh_parent1 = []
        enh_parent2 = []
        i = 0
        while i < len(rules_parent1):
            if rules_parent1[i][:4] in ['addf','addk','addl','addr','addb','addt']:
                new_token= rules_parent1[i] + " " + rules_parent1[i+1]
                enh_parent1.append(new_token)
                i+=1
            if rules_parent1[i][:4] in ['movf','movk','movl','movr','movb','movt']:
                enh_parent1.append(rules_parent1[i])
            if rules_parent1[i]=='H':
                enh_parent1.append(rules_parent1[i])
            i+=1 
        i = 0
        while i < len(rules_parent2):
            if rules_parent2[i][:4] in ['addf','addk','addl','addr','addb','addt']:
                new_token= rules_parent2[i] + " " + rules_parent2[i+1]
                enh_parent2.append(new_token)
                i+=1
            if rules_parent2[i][:4] in ['movf','movk','movl','movr','movb','movt']:
                enh_parent2.append(rules_parent2[i])
            if rules_parent2[i]=='H':
                enh_parent2.append(rules_parent2[i])
            i+=1 
        r_offspring1=""
        r_offspring2=""
        le_common = min(len(enh_parent1),len(enh_parent2))
        for i in range(0,le_common):
            if random.random()>mutation_rate:
                r_offspring1+=enh_parent2[i]+" "
                r_offspring2+=enh_parent1[i]+" "
            else:
                r_offspring1+=enh_parent1[i]+" "
                r_offspring2+=enh_parent2[i]+" "
        if len(enh_parent1)>le_common:
            for j in range(le_common,len(enh_parent1)):
                if random.random()>mutation_rate:
                    r_offspring1+=enh_parent1[j]+" "
                else:   
                    r_offspring2+=enh_parent1[j]+" "
        if len(enh_parent2)>le_common:
            for j in range(le_common,len(enh_parent2)):
                if random.random()>mutation_rate:
                    r_offspring1+=enh_parent2[j]+" "
                else:
                    r_offspring2+=enh_parent2[j]+" "
        rules_offspring1['H']=r_offspring1
        rules_offspring2['H']=r_offspring2

        rules_parent1 = lsystem_parent1.rules["N"].split()
        rules_parent2 = lsystem_parent2.rules["N"].split()
        enh_parent1 = []
        enh_parent2 = []
        i = 0
        while i < len(rules_parent1):
            if rules_parent1[i][:4] in ['addf','addk','addl','addr','addb','addt']:
                new_token= rules_parent1[i] + " " + rules_parent1[i+1]
                enh_parent1.append(new_token)
                i+=1
            if rules_parent1[i][:4] in ['movf','movk','movl','movr','movb','movt']:
                enh_parent1.append(rules_parent1[i])
            if rules_parent1[i]=='N':
                enh_parent1.append(rules_parent1[i])
            i+=1 
        i = 0
        while i < len(rules_parent2):
            if rules_parent2[i][:4] in ['addf','addk','addl','addr','addb','addt']:
                new_token= rules_parent2[i] + " " + rules_parent2[i+1]
                enh_parent2.append(new_token)
                i+=1
            if rules_parent2[i][:4] in ['movf','movk','movl','movr','movb','movt']:
                enh_parent2.append(rules_parent2[i])
            if rules_parent2[i]=='N':
                enh_parent2.append(rules_parent2[i])
            i+=1 
        r_offspring1=""
        r_offspring2=""
        le_common = min(len(enh_parent1),len(enh_parent2))
        for i in range(0,le_common):
            if random.random()>mutation_rate:
                r_offspring1+=enh_parent2[i]+" "
                r_offspring2+=enh_parent1[i]+" "
            else:
                r_offspring1+=enh_parent1[i]+" "
                r_offspring2+=enh_parent2[i]+" "
        if len(enh_parent1)>le_common:
            for j in range(le_common,len(enh_parent1)):
                if random.random()>mutation_rate:
                    r_offspring1+=enh_parent1[j]+" "
                else:   
                    r_offspring2+=enh_parent1[j]+" "
        if len(enh_parent2)>le_common:
            for j in range(le_common,len(enh_parent2)):
                if random.random()>mutation_rate:
                    r_offspring1+=enh_parent2[j]+" "
                else:
                    r_offspring2+=enh_parent2[j]+" "
        rules_offspring1['N']=r_offspring1
        rules_offspring2['N']=r_offspring2

        iter_offspring1+=lsystem_parent2.iterations
        iter_offspring2+=lsystem_parent1.iterations
        offspring1=LSystemDecoder(axiom_offspring1,rules_offspring1,iter_offspring1,lsystem_parent1.max_elements,lsystem_parent1.max_depth,lsystem_parent1.verbose)
        offspring2=LSystemDecoder(axiom_offspring2,rules_offspring2,iter_offspring2,lsystem_parent2.max_elements,lsystem_parent2.max_depth,lsystem_parent2.verbose)
        return offspring1,offspring2


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
    main()
