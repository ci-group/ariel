from torch.fx.experimental.graph_gradual_typechecker import Refine as Refine
from torch.fx.experimental.unification import Var as Var, unify as unify
from torch.fx.tensor_type import TensorType as TensorType

def infer_symbolic_types_single_pass(traced) -> None:
    """
    Calls our symbolic inferencer once.
    """
def infer_symbolic_types(traced) -> None:
    """
    Calls our symbolic inferencer twice.
    This is useful when one pass is not enough
    to infer all the information such as the case
    for braodcasting.
    """
def convert_eq(list_of_eq):
    """
    Convert equality constraints in the right format
    to be used by unification library.
    """
def unify_eq(list_of_eq):
    """
    Apply unification to a set of
    equality constraints
    """
def substitute_solution_one_type(mapping, t):
    """
    Apply the most general unifier to a type
    """
def substitute_all_types(graph, mapping) -> None:
    """
    Apply the most general unifier to all types in a graph
    till reaching a fixed point. If the input and output graph
    are the same, we converge.
    """
def check_for_type_equality(g1, g2):
    """
    A check equality to be used in fixed points.
    We do not use graph equality but instead type
    equality.
    """
