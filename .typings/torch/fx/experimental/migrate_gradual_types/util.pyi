from torch.fx.experimental.migrate_gradual_types.constraint import BVar as BVar, BinConstraintD as BinConstraintD, DVar as DVar, TVar as TVar
from torch.fx.experimental.migrate_gradual_types.operation import op_leq as op_leq

def gen_tvar(curr):
    """
    Generate a tensor variable
    :param curr: The current counter
    :return: a tensor variable and the updated counter
    """
def gen_dvar(curr):
    """
    Generate a dimension variable
    :param curr: the current counter
    :return: a dimension variable and an updated counter
    """
def gen_bvar(curr):
    """
    Generate a boolean variable
    :param curr: the current counter
    :return: a boolean variable and an updated counter
    """
def gen_tensor_dims(n, curr):
    """
    Generate a list of tensor dimensions
    :param n:  the number of dimensions
    :param curr: the current counter
    :return: a list of dimension variables and an updated counter
    """
def gen_nat_constraints(list_of_dims):
    """
    Generate natural number constraints for dimensions
    """
