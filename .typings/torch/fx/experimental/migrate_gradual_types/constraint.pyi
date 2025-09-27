from _typeshed import Incomplete
from torch.fx.experimental.migrate_gradual_types.operation import op_add as op_add, op_div as op_div, op_eq as op_eq, op_gt as op_gt, op_lt as op_lt, op_mod as op_mod, op_mul as op_mul, op_neq as op_neq, op_sub as op_sub
from torch.fx.tensor_type import Dyn as Dyn, TensorType as TensorType

class Constraint: ...

class Conj(Constraint):
    conjucts: Incomplete
    def __init__(self, conjuncts) -> None:
        """
        :param conjuncts: Conjunction of constraints
        """
    def __eq__(self, other): ...
    def __repr__(self) -> str: ...

class Disj(Constraint):
    disjuncts: Incomplete
    def __init__(self, disjuncts) -> None:
        """
        :param disjuncts: Disjunction of constraints
        """
    def __eq__(self, other): ...
    def __repr__(self) -> str: ...

class Prod(Constraint):
    products: Incomplete
    def __init__(self, products) -> None:
        """
        :param products: lists of dimensions to multiply
        """
    def __eq__(self, other): ...
    def __repr__(self) -> str: ...

class T(Constraint):
    """
    True
    """
    def __init__(self) -> None: ...
    def __eq__(self, other): ...
    def __repr__(self) -> str: ...

class F(Constraint):
    """
    False
    """
    def __init__(self) -> None: ...
    def __eq__(self, other): ...
    def __repr__(self) -> str: ...

class BinaryConstraint(Constraint):
    """
    Represents all binary operations
    """
    lhs: Incomplete
    rhs: Incomplete
    op: Incomplete
    def __init__(self, lhs, rhs, op) -> None:
        """
        :param lhs: lhs of the constraint
        :param rhs: rhs of the constraint
        :param op: string representing the operation
        """
    def __eq__(self, other): ...
    def __repr__(self) -> str: ...

class BinConstraintT(BinaryConstraint):
    """
    Binary constraints about tensors
    """
    def __init__(self, lhs, rhs, op) -> None: ...
    def __eq__(self, other): ...

class BinConstraintD(BinaryConstraint):
    """
    Binary constraints about dimensions
    """
    def __init__(self, lhs, rhs, op) -> None: ...
    def __eq__(self, other): ...

class TGreatestUpperBound(Constraint):
    """
    Greatest Upper bound for tensors with dynamic type
    """
    res: Incomplete
    rhs1: Incomplete
    rhs2: Incomplete
    def __init__(self, res, rhs1, rhs2) -> None:
        """
        :param res: tensor variable that stores the result of the outout
        :param rhs1: tensor or tensor variable
        :param rhs2: tensor or tensor variabke
        """
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...

class DGreatestUpperBound(Constraint):
    """
    Greatest Upper bound for dimensions
    """
    res: Incomplete
    rhs1: Incomplete
    rhs2: Incomplete
    def __init__(self, res, rhs1, rhs2) -> None:
        """
        :param res: Dimension variable to store the result
        :param rhs1: dimension variable 1
        :param rhs2: dimension variable 2
        """
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...

class CanReshape(Constraint):
    """
    can_reshape constraint
    """
    src: Incomplete
    target: Incomplete
    def __init__(self, src, target) -> None:
        """
        :param src: tensor variable
        :param target: tensor
        """
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...

class IndexSelect(Constraint):
    input_var: Incomplete
    tensor_size: Incomplete
    dim_replace: Incomplete
    index: Incomplete
    output: Incomplete
    def __init__(self, tensor_size, input_var, dim_replace, index, output) -> None:
        '''
        Args:
            input_var: input to index_select
            tensor_size: tensor size we are considering
            dim_replace: the dimension of the output at "index"
            index: location of the dimensions to replace in the input
            output: variable to store the result
        '''
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...

class Transpose(Constraint):
    input_var: Incomplete
    tensor_size: Incomplete
    index1: Incomplete
    index2: Incomplete
    output: Incomplete
    def __init__(self, tensor_size, input_var, index1, index2, output) -> None:
        """
        Args:
            tensor_size: current tensor size
            input_var: variable to hold input
            index1: dimension 1
            index2: dimension 2
            output: output that stores result
        """
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...

class GetItem(Constraint):
    res: Incomplete
    tensor_size: Incomplete
    index: Incomplete
    input_var: Incomplete
    def __init__(self, tensor_size, index, res, input_var) -> None:
        """
        Constraint for getting item given a tensor size
        :param tensor_size: actual number
        :param index: actual number representing the index
        :param res: dimension variable to carry the item we get
        :param input_var: a tensor variable from which we will get item
        """
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...

class GetItemTensor(Constraint):
    res: Incomplete
    tensor_size: Incomplete
    index_tuple: Incomplete
    input_var: Incomplete
    def __init__(self, tensor_size, index_tuple, res, input_var) -> None:
        """
        Constraint for getting item given a tensor size
        However, when the argument is a tuple, we will
        expect a tensor
        :param tensor_size: actual number representing the rank
        :param index_tuple: tuple for indexing
        :param res: tensor variable to carry the item we get
        :param input_var: a tensor variable from which we will get item
        """
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...

class CalcConv(Constraint):
    conv_result: Incomplete
    input_var: Incomplete
    c_out: Incomplete
    kernel: Incomplete
    padding: Incomplete
    stride: Incomplete
    dilation: Incomplete
    matching_constraint: Incomplete
    def __init__(self, conv_result, input_var, c_out, kernel, padding, stride, dilation, matching_constraint_vars) -> None:
        """
        :param conv_result: the convolution result
        :param input_var: input to convolution
        :param c_out: output chanel type
        :param kernel: kernel tuple
        """
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...

class CalcMaxPool(Constraint):
    maxpool_result: Incomplete
    input_var: Incomplete
    kernel: Incomplete
    padding: Incomplete
    stride: Incomplete
    dilation: Incomplete
    matching_constraint: Incomplete
    def __init__(self, maxpool_result, input_var, kernel, padding, stride, dilation, matching_constraint_vars) -> None:
        """
        :param maxpool_result: the result of maxpool
        :param input_var: input to convolution
        :param kernel: kernel tuple
        """
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...

class ApplyBroadcasting(Constraint):
    res1: Incomplete
    res2: Incomplete
    input1: Incomplete
    input2: Incomplete
    def __init__(self, res1, res2, input1, input2) -> None:
        """
        :param res1: resulting tensor 1
        :param res2: resulting tensor 2
        :param input1: tensor variable 1
        :param input2: tensor variable 2
        """
    def __eq__(self, other): ...
    def __repr__(self) -> str: ...

class CalcProduct(Constraint):
    """
    Given correct dimensions, calculate the product for flatten accounting for Dyn
    """
    start: Incomplete
    end: Incomplete
    dims_to_flatten: Incomplete
    flattened: Incomplete
    def __init__(self, start, end, flattened, dims_to_flatten) -> None:
        """
        :param start: start index
        :param end: end index
        :param flattened: variable to store the product
        :param dims_to_flatten: the type which we will flatten
        """
    def __eq__(self, other): ...
    def __repr__(self) -> str: ...

class TVar:
    """
    Tensor variable with no tensor constructor
    """
    tvar: Incomplete
    def __init__(self, tvar) -> None:
        """
        :param tvar: tensor variable
        """
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...

class DVar:
    """
    Dimension variable
    """
    c: Incomplete
    def __init__(self, c) -> None:
        """
        :param c: character or number
        """
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...

class BVar:
    """
    Boolean variable
    """
    c: Incomplete
    def __init__(self, c) -> None:
        """
        :param c: character or number
        """
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...

def is_algebraic_expression(constraint): ...
def is_bool_expr(constraint): ...
def is_dim(d): ...
