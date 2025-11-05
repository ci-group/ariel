"""TODO(jmdm): description of script."""
from __future__ import annotations
# Standard library
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import cast, TYPE_CHECKING, List
import random

# Third-party libraries
import numpy as np
from pydantic_settings import BaseSettings
from rich.console import Console
from rich.traceback import install
import copy
if TYPE_CHECKING:
    from ariel.ec.genotypes.genotype import Genotype
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

class Mutation(ABC):
    mutations_mapping: dict[str, function] = NotImplemented
    which_mutation: str = ""

    @classmethod
    def set_which_mutation(cls, mutation_type: str) -> None:
        cls.which_mutation = mutation_type

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.mutations_mapping = {
            name: getattr(cls, name)
            for name, val in cls.__dict__.items()
            if isinstance(val, staticmethod)
        }

    @classmethod
    def __call__(
        cls,
        individual: Genotype,
        **kwargs: dict,
    ) -> Genotype:
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
        if cls.which_mutation in cls.mutations_mapping:
            return cls.mutations_mapping[cls.which_mutation](
                individual,
                **kwargs
            )
        else:
            msg = f"Mutation type '{cls.which_mutation}' not recognized."
            raise ValueError(msg)

class IntegerMutator(Mutation):

    @staticmethod
    def random_swap(
        individual: Genotype,
        low: int,
        high: int,
        mutation_probability: float,
    ) -> Genotype:
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
        individual: Genotype,
        span: int,
        mutation_probability: float,
    ) -> Genotype:
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


# class TreeGenerator:
#     @staticmethod
#     def __call__(*args, **kwargs) -> TreeGenome:
#         return TreeGenome.default_init()

#     @staticmethod
#     def default():
#         return TreeGenome.default_init()

#     @staticmethod
#     def linear_chain(length: int = 3) -> TreeGenome:
#         """Generate a linear chain of modules (snake-like)."""
#         genome = TreeGenome.default_init()  # Start with CORE
#         current_node = genome.root

#         for i in range(length):
#             module_type = RNG.choice([pheno_config.ModuleType.BRICK, pheno_config.ModuleType.HINGE])
#             rotation = RNG.choice(list(pheno_config.ModuleRotationsIdx))
#             module = pheno_config.ModuleInstance(type=module_type, rotation=rotation, links={})

#             # Always attach to FRONT face for linear chain
#             if pheno_config.ModuleFaces.FRONT in current_node.available_faces():
#                 child = TreeNode(module, depth=current_node._depth + 1)
#                 current_node._set_face(pheno_config.ModuleFaces.FRONT, child)
#                 current_node = child

#         return genome

#     @staticmethod
#     def star_shape(num_arms: int = 3) -> TreeGenome:
#         """Generate a star-shaped tree with arms radiating from center."""
#         genome = TreeGenome.default_init()  # Start with CORE
#         available_faces = genome.root.available_faces()

#         # Limit arms to available faces
#         actual_arms = min(num_arms, len(available_faces))
#         selected_faces = RNG.choice(available_faces, size=actual_arms, replace=False)

#         for face in selected_faces:
#             module_type = RNG.choice([pheno_config.ModuleType.BRICK, pheno_config.ModuleType.HINGE])
#             rotation = RNG.choice(list(pheno_config.ModuleRotationsIdx))
#             module = pheno_config.ModuleInstance(type=module_type, rotation=rotation, links={})

#             child = TreeNode(module, depth=1)
#             genome.root._set_face(face, child)

#         return genome

#     @staticmethod
#     def binary_tree(depth: int = 2) -> TreeGenome:
#         """Generate a binary-like tree structure."""
#         def build_subtree(current_depth: int, max_depth: int) -> TreeNode | None:
#             if current_depth >= max_depth:
#                 return None

#             module_type = RNG.choice([pheno_config.ModuleType.BRICK, pheno_config.ModuleType.HINGE])
#             rotation = RNG.choice(list(pheno_config.ModuleRotationsIdx))
#             module = pheno_config.ModuleInstance(type=module_type, rotation=rotation, links={})

#             node = TreeNode(module, depth=current_depth)
#             available_faces = node.available_faces()

#             # Add 1-2 children randomly
#             if available_faces and current_depth < max_depth - 1:
#                 num_children = RNG.integers(1, min(3, len(available_faces) + 1))
#                 selected_faces = RNG.choice(available_faces, size=num_children, replace=False)

#                 for face in selected_faces:
#                     child = build_subtree(current_depth + 1, max_depth)
#                     if child:
#                         node._set_face(face, child)

#             return node

#         genome = TreeGenome.default_init()

#         # Add children to root
#         available_faces = genome.root.available_faces()
#         if available_faces:
#             num_children = RNG.integers(1, min(3, len(available_faces) + 1))
#             selected_faces = RNG.choice(available_faces, size=num_children, replace=False)

#             for face in selected_faces:
#                 child = build_subtree(1, depth)
#                 if child:
#                     genome.root._set_face(face, child)

#         return genome

#     @staticmethod
#     def random_tree(max_depth: int = 4, branching_prob: float = 0.7) -> TreeGenome:
#         """Generate a random tree with pheno_configurable branching probability."""
#         genome = TreeGenome.default_init()  # Start with CORE
#         face = RNG.choice(genome.root.available_faces())
#         subtree = TreeNode.random_tree_node(max_depth=max_depth - 1, branch_prob=branching_prob)
#         if subtree:
#             genome.root._set_face(face, subtree)
#         return genome

class TreeMutator(Mutation):
        
    @staticmethod
    def _random_tree(max_depth: int = 2, branching_prob: float = 0.5) -> Genotype:
        """Generate a random tree with pheno_configurable branching probability."""
        from ariel.ec.genotypes.tree.tree_genome import TreeGenome, TreeNode
        genome = TreeGenome.default_init()  # Start with CORE
        face = RNG.choice(genome.root.available_faces())
        subtree = TreeNode.random_tree_node(max_depth=max_depth - 1, branch_prob=branching_prob)
        if subtree:
            genome.root._set_face(face, subtree)
        return genome

    @staticmethod
    def random_subtree_replacement(
        individual: Genotype,
        max_subtree_depth: int = 2,
        branching_prob: float = 0.5,
    ) -> Genotype:
        """Replace a random subtree with a new random subtree."""
        from ariel.ec.genotypes.tree.tree_genome import TreeNode
        new_individual = copy.copy(individual)

        # Collect all nodes in the tree
        all_nodes = new_individual.root.get_all_nodes(exclude_root=True)

        if not all_nodes:
            # print("Tree has no nodes to replace; generating a new random tree.")
            return TreeMutator._random_tree(max_depth=max_subtree_depth, branching_prob=branching_prob)

        # Generate a new random subtree
        new_subtree = TreeNode.random_tree_node(max_depth=max_subtree_depth, branch_prob=branching_prob)

        node_to_replace = RNG.choice(all_nodes)

        with new_individual.root.enable_replacement():
            new_individual.root.replace_node(node_to_replace, new_subtree)

        return new_individual
    
class LSystemMutator(Mutation):

    @staticmethod
    def mutate_one_point_lsystem(lsystem,mut_rate,add_temperature=0.5):
        op_completed = ""
        if random.random()<mut_rate:
            action=random.choices(['add_rule','rm_rule'],weights=[add_temperature,1-add_temperature])[0]
            rules = lsystem.rules
            rule_to_change=random.choice(range(0,len(rules)))
            rl_tmp = list(rules.values())[rule_to_change]
            splitted_rules=rl_tmp.split()
            gene_to_change=random.choice(range(0,len(splitted_rules)))
            match action:
                case 'add_rule':
                    operator=random.choice(['addf','addk','addl','addr','addb','addt','movf','movk','movl','movr','movt','movb'])
                    if operator in ['addf','addk','addl','addr','addb','addt']:
                        if splitted_rules[gene_to_change][:4] in ['addf','addk','addl','addr','addb','addt']:
                            rotation = random.choice([0,45,90,135,180,225,270])
                            op_to_add=operator+"("+str(rotation)+")"
                            item_to_add=random.choice(['B','H','N'])
                            splitted_rules.insert(gene_to_change+2,item_to_add)
                            splitted_rules.insert(gene_to_change+2,op_to_add)
                            op_completed="ADDED : "+op_to_add+" "+item_to_add
                        elif splitted_rules[gene_to_change][:4] in ['movf','movk','movl','movr','movb','movt']:
                            rotation = random.choice([0,45,90,135,180,225,270])
                            op_to_add=operator+"("+str(rotation)+")"
                            item_to_add=random.choice(['B','H','N'])
                            splitted_rules.insert(gene_to_change,item_to_add)
                            splitted_rules.insert(gene_to_change,op_to_add)
                            op_completed="ADDED : "+op_to_add+" "+item_to_add
                        elif splitted_rules[gene_to_change] in ['C','B','H','N']:
                            rotation = random.choice([0,45,90,135,180,225,270])
                            op_to_add=operator+"("+str(rotation)+")"
                            item_to_add=random.choice(['B','H','N'])
                            splitted_rules.insert(gene_to_change+1,item_to_add)
                            splitted_rules.insert(gene_to_change+1,op_to_add)
                            op_completed="ADDED : "+op_to_add+" "+item_to_add
                    if operator in ['movf','movk','movl','movr','movb','movt']:
                        if splitted_rules[gene_to_change][:4] in ['addf','addk','addl','addr','addb','addt']:
                            splitted_rules.insert(gene_to_change+2,operator)
                            op_completed="ADDED : "+operator
                        elif splitted_rules[gene_to_change][:4] in ['movf','movk','movl','movr','movb','movt']:
                            splitted_rules.insert(gene_to_change,operator)
                            op_completed="ADDED : "+operator
                        elif splitted_rules[gene_to_change] in ['C','B','H','N']:
                            splitted_rules.insert(gene_to_change+1,operator)
                            op_completed="ADDED : "+operator
                case 'rm_rule':
                    if splitted_rules[gene_to_change][:4] in ['addf','addk','addl','addr','addb','addt']:
                        op_completed="REMOVED : "+splitted_rules[gene_to_change]+" "+splitted_rules[gene_to_change+1]
                        splitted_rules.pop(gene_to_change)
                        splitted_rules.pop(gene_to_change)
                    elif splitted_rules[gene_to_change] in ['H','B','N']:
                        op_completed="REMOVED : "+splitted_rules[gene_to_change-1]+" "+splitted_rules[gene_to_change]
                        if gene_to_change-1>=0:
                            splitted_rules.pop(gene_to_change-1)
                            splitted_rules.pop(gene_to_change-1)
                    elif splitted_rules[gene_to_change][:4] in ['movf','movk','movl','movr','movt','movb']:
                        op_completed="REMOVED : "+splitted_rules[gene_to_change]
                        splitted_rules.pop(gene_to_change)
            new_rule = ""
            for j in range(0,len(splitted_rules)):
                new_rule+=splitted_rules[j]+" "
            if new_rule!="":
                lsystem.rules[list(rules.keys())[rule_to_change]]=new_rule
            else:
                lsystem.rules[list(rules.keys())[rule_to_change]]=lsystem.rules[list(rules.keys())[rule_to_change]]
        return op_completed


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

    treeGenerator = TreeGenerator()
    random_tree = treeGenerator.random_tree(max_depth=3, branching_prob=0.7)
    console.log("Random Tree:", random_tree)

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
