from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting as ApplyBroadcasting, BinConstraintD as BinConstraintD, CalcConv as CalcConv, CalcMaxPool as CalcMaxPool, CalcProduct as CalcProduct, CanReshape as CanReshape, Conj as Conj, Constraint as Constraint, DGreatestUpperBound as DGreatestUpperBound, DVar as DVar, Disj as Disj, F as F, GetItem as GetItem, GetItemTensor as GetItemTensor, IndexSelect as IndexSelect, Prod as Prod, T as T, TGreatestUpperBound as TGreatestUpperBound, TVar as TVar, Transpose as Transpose
from torch.fx.experimental.migrate_gradual_types.constraint_generator import BinConstraintT as BinConstraintT, MAX_TENSOR_RANK as MAX_TENSOR_RANK
from torch.fx.experimental.migrate_gradual_types.operation import op_add as op_add, op_consistency as op_consistency, op_div as op_div, op_eq as op_eq, op_leq as op_leq, op_matching as op_matching, op_mod as op_mod, op_mul as op_mul, op_neq as op_neq, op_precision as op_precision, op_sub as op_sub
from torch.fx.experimental.migrate_gradual_types.util import gen_dvar as gen_dvar, gen_nat_constraints as gen_nat_constraints, gen_tensor_dims as gen_tensor_dims
from torch.fx.tensor_type import Dyn as Dyn, TensorType as TensorType
from typing import Callable

_TRANSFORMATION_RULES: dict[Constraint, Callable]

def register_transformation_rule(call_target): ...
def valid_index(index, dims):
    """
    Given a list of dimensions, checks if an index is valid in the list
    """
def transform_transpose(constraint, counter):
    """
    Similar to a sequence of two index-selects
    """
def transform_index_select(constraint, counter):
    """
    The constraints consider the given tensor size, checks if the index is valid
    and if so, generates a constraint for replacing the input dimension
    with the required dimension
    """
def transform_get_item(constraint, counter):
    """
    generate an equality of the form:
    t = [a1, ..., an]
    then generate constraints that check if the given index is valid
    given this particular tensor size.
    If the index is valid, generate a constraint to get the item
    Note that we already handled the Dyn input case in the previous
    step.
    Args:
        constraint: GetItem which assumes we are getting an item from a tensor (not Dyn)
        counter: variable tracking
    Returns: simplified constraints for GetItem

    """
def valid_index_tensor(index, dims):
    """
    if the slice instances exceed the length of the dimensions
    then this is a type error so we return False
    """
def transform_get_item_tensor(constraint, counter):
    """
    When the index is a tuple, then the output will be a tensor
    TODO: we have to check if this is the case for all HF models

    The cases we are covering here are a tuple with one of:
     - slice with default argument
     - None

     None appends 1 to the input tensor dimensions
     so each occurrence of 'None' increases the rank by 1

     slice with default arguments does not change the rank
    """
def generate_binconstraint_t(constraint, counter):
    """
    Transform binary constraints for tensors
    """
def generate_binconstraint_d(constraint, counter):
    """
    Transform binary constraints for dimensions
    """
def generate_conj(constraint, counter):
    """
    Transform conjunctions
    """
def generate_disj(constraint, counter):
    """
    Transform disjunctions
    """
def generate_gub(constraint, counter):
    """
    Transform greatest upper bound for tensors. Results in equality and Greatest Upper Bound
    on dimensions
    """
def generate_d_gub(constraint, counter):
    """
    Transform greatest upper bound for dimensions into equality constraints
    """
def generate_calc_conv(constraint, counter): ...
def generate_calc_maxpool(constraint, counter):
    """
    Transform maxpool constraints
    """
def generate_calc_product(constraint, counter):
    """
    Transform flatten constraints
    """
def generate_reshape(constraint, counter):
    """
    Transform reshape constraints
    """
def generate_broadcasting(constraint, counter):
    """
    Transform broadcasting constraints
    """
def transform_constraint(constraint: Constraint, counter: int):
    """
    Transforms a constraint into a simpler constraint.
    Ex: precision and consistency are transformed to equality
    Args:
        constraint: constraint to be transformed
        counter: for variable tracking

    Returns: Constraint

    """
def calc_last_two_dims(constraint, d: list[DVar]):
    """
    Generates constraints for the last two dimensions of a convolution or a maxpool output
    Args:
        constraint: CalcConv or CalcMaxPool
        d: The list of output dimensions

    Returns: Constraints for calculating the last two dimensions of the output

    """
def generate_all_int_dyn_dim_possibilities(my_list: list[DVar]):
    """
    Generate all possibilities of being equal or not equal to dyn for my_list
    Args:
        my_list: List of tensor dimensions

    Returns: A list of a list of constraints. Each list of constraints corresponds to
    one possibility about the values of the dimension variables
    """
def is_target_div_by_dim(target: list[int], dim: list[DVar]):
    """
    Generate constraints to check if the target dimensions are divisible by the input dimensions
    Args:
        target: Target dimensions
        dim: Input dimensions

    Returns: Constraints to check divisibility

    """
def is_dim_div_by_target(target: list[int], dim: list[DVar]):
    """
    Generate constraints to check if the input dimensions is divisible by the target dimensions
    Args:
        target: Target dimensions
        dim:  Input dimensions

    Returns: Constraints to check divisibility

    """
def gen_all_reshape_possibilities(list_of_dims, target):
    """
    Consider all possibilities what the input dimensions could be (number or dynamic)
    Then generate the appropriate constraints using multiplication or mod depending on the possibility
    The possibilities we consider here are the cross product of being equal to dyn or not equal to dyn
    for the input. Target is fixed because at most one dimension could be dyn.
    We have different cases for this.

    Args:
        list_of_dims: The input list of dimensions
        target: The tensor we want to reshape to

    Returns: A disjunction of transformed reshape constraints

    """
def broadcast_dim(tensor_input1, tensor_input2, res1, res2, index, padding: bool = False):
    """
    Apply broadcasting to the 'index' dimension of tensor_input1.
    Args:
        tensor_input1: should represent [d1, ..., d_index, ...] where d_index = 1
        tensor_input2: represents the second input
        res1: broadcasted result 1
        res2: broadcasted result 2
        index: the index to broadcast
        padding: If padding was used, then tensor_input1[index] does not exist

    Returns:

    """
def apply_padding(e1_var: TVar, e11: BinConstraintT, e2: BinConstraintT, e12: BinConstraintT, d2: list[DVar], d11: list[DVar], d12: list[DVar], counter: int):
    """
    We are considering the possibility where one input has less dimensions than
    another input, so we apply padding to the broadcasted results

    Args:
        e1_var: Variable representing the first input where padding will be
        e11: constraint of the form e11 = Tensortype[d1, ..., dn]
        e2:  constraint of the form e2 = Tensortype[d1, ..., dn]
        e12: constraint of the form e11 = Tensortype[d1, ..., dn]
        d2: Tensor variables for the second input
        d11: Tensor variables for the broadcasted first input
        d12: Tensor variables for the broadcasted second input
        counter: variable tracking

    Returns: A new constraint whose goal is to apply padding to the broadcasted result

    """
def no_broadcast_dim_with_index(d1: list[DVar], d2: list[DVar], d3: list[DVar], d4: list[DVar], i: int):
    """
    Args:
        d1: input 1
        d2: input 2
        d3: simulated broadcasting for input 1
        d4: simulated broadcasting for input 2
        i: the rank of the resulting tensor addition

    Returns: Constraints for when no broadcasting occurs
    """
def gen_lists_of_dims(num_tensors: int, dim_size: int, counter: int):
    """
    Generate lists of DVar to represent tensor dimensions
    Args:
        num_tensors: the required number of tensors
        dim_size: the number of dimensions for each tensor
        counter: variable tracking

    Returns: A list of a list of tensor dimensions

    """
def create_equality_constraints_for_broadcasting(e1: TVar, e2: TVar, e11: TVar, e12: TVar, d1: list[DVar], d2: list[DVar], d11: list[DVar], d12: list[DVar]):
    """
    Create equality constraints for when no broadcasting occurs
    Args:
        e1: Input 1
        e2: Input 2
        e11: Broadcasted input 1
        e12: Broadcasted input 2
        d1: Variables that store dimensions for e1
        d2: Variables that store dimensions for e2
        d11: Variables that store dimensions for e11
        d12: Variables that store dimensions for e22

    Returns: Four equality constraints

    """
def gen_consistency_constraints(constraint: Constraint, counter: int):
    """
    Args:
        constraint: Consistency constraint on tensors
        counter: for variable tracking

    Returns: Equality and consistency constraints on dimensions

    """
def gen_greatest_upper_bound(constraint: TGreatestUpperBound, counter: int):
    """
    Args:
        constraint: Greatest upper bound on tensors
        counter: variable tracking

    Returns: A set of equality constraints and DGreatestUpperBound constraints

    """
def generate_all_broadcasting_possibilities_no_padding(d1: list[DVar], d2: list[DVar], d11: list[DVar], d12: list[DVar]):
    """
    Generate broadcasting constraints assuming no padding. Broadcasting can happen at any dimension.
    We look at all combinations for all dimensions in d1 and d2
    Args:
        d1: input1 dimensions
        d2: input2 dimensions
        d11: broadcasted input1 dimensions
        d12: broadcasted input2 dimensions

    Returns: broadcasting constraints relating the input dimensions to the broadcasted dimensions

    """
def gen_broadcasting_constraints(e1: TVar, e2: TVar, e11: TVar, e12: TVar, i: int, counter: int):
    """
    Simulates broadcasting on e1 and e2 and returns the results
    respectively in e11 and e12. Because of gradual types,
    e1 and e2 may not be equal. Similarly, e11 and e12 may not
    be equal. e11 and e12 should be guaranteed to be consistent
    as they represent the shapes of the tensors to be added after
    broadcasting.
    Args:
        e1: TVar representing the type of input 1
        e2: TVar representing the type of input 2
        e11: TVar representing the representing broadcasted input 1
        e12: TVar representing the representing broadcasted input 2
        i: The rank of the resulting type of addition
        counter: for variable tracking

    Returns: Simplified broadcasting constraints

    """
