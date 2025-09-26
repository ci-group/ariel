from torch.fx.experimental.migrate_gradual_types.constraint import BVar as BVar, BinConstraintD as BinConstraintD, BinConstraintT as BinConstraintT, Conj as Conj, DVar as DVar, Disj as Disj, F as F, Prod as Prod, T as T, TVar as TVar, is_algebraic_expression as is_algebraic_expression, is_bool_expr as is_bool_expr, is_dim as is_dim
from torch.fx.experimental.migrate_gradual_types.constraint_generator import ConstraintGenerator as ConstraintGenerator
from torch.fx.experimental.migrate_gradual_types.constraint_transformation import transform_constraint as transform_constraint
from torch.fx.experimental.migrate_gradual_types.operation import op_add as op_add, op_div as op_div, op_eq as op_eq, op_gt as op_gt, op_leq as op_leq, op_lt as op_lt, op_mod as op_mod, op_mul as op_mul, op_neq as op_neq, op_sub as op_sub
from torch.fx.experimental.migrate_gradual_types.z3_types import D as D, tensor_type as tensor_type, z3_dyn as z3_dyn
from torch.fx.tensor_type import Dyn as Dyn, TensorType as TensorType

HAS_Z3: bool

def transform_to_z3(constraint, counter, dimension_dict): ...
def transform_var(tensor, counter, dimension_dict):
    """
        Transforms tensor variables to a format understood by z3
        Args:
            tensor: Tensor variable or a tensor type potentially with variable dimensions
        Returns: Transformed variable to a z3 format

        """
def transform_dimension(dimension, counter, dimension_dict):
    """
        Takes a dimension variable or a number and transforms it to a tuple
        according to our scheme
        Args:
            dimension: The dimension to be transformed
            counter: variable tracking

        Returns:  tuple and the current counter

        """
def transform_algebraic_expression(expr, counter, dimension_dict):
    """
        Transforms an algebraic expression to z3 format
        Args:
            expr: An expression is either a dimension variable or an algebraic-expression


        Returns: the transformed expression

        """
def transform_all_constraints(traced, counter: int = 0):
    """
        Given a trace, generates constraints and transforms them to z3 format

        """
def iterate_till_fixed_point(constraints, counter):
    """
        Transform constraints till reaching a fixed point
        """
def transform_all_constraints_trace_time(tracer_root, graph, node, counter: int = 0):
    """
        Takes a node and a graph and generates two sets of constraints.
        One set constraints the node's constraints and another set
        constraints the negation of the node's constraints
        Args:
            tracer_root: the root for getting the module instances
            graph: the graph so far in the tracing process
            node: node that represents a conditional
            counter: variable tracking

        Returns: Two sets of constraints. One with a conjunction with the
        the conditional constraint and the other with a conjunction with
        its negation.

        """
def evaluate_conditional_with_constraints(tracer_root, graph, node, counter: int = 0, user_constraints=None):
    """
        Given an IR and a node representing a conditional, evaluate the conditional
        and its negation
        Args:
            tracer_root: Tracer root for module instances
            node: The node to be evaluated

        Returns: the results of evaluating the condition and the negation with
        the rest of the constraints

        """
