from _typeshed import Incomplete
from torch.fx.node import Node, Target
from typing import Callable, TypeVar
from typing_extensions import ParamSpec

__all__ = ['GraphTypeChecker', 'Refine', 'adaptiveavgpool2d_check', 'adaptiveavgpool2d_inference_rule', 'add_inference_rule', 'all_eq', 'bn2d_inference_rule', 'broadcast_types', 'calculate_out_dimension', 'conv2d_inference_rule', 'conv_refinement_rule', 'conv_rule', 'element_wise_eq', 'expand_to_tensor_dim', 'first_two_eq', 'flatten_check', 'flatten_inference_rule', 'flatten_refinement_rule', 'get_attr_inference_rule', 'get_greatest_upper_bound', 'get_parameter', 'linear_check', 'linear_inference_rule', 'linear_refinement_rule', 'maxpool2d_check', 'maxpool2d_inference_rule', 'register_algebraic_expressions_inference_rule', 'register_inference_rule', 'register_refinement_rule', 'relu_inference_rule', 'reshape_inference_rule', 'transpose_inference_rule']

_T = TypeVar('_T')
_P = ParamSpec('_P')

def expand_to_tensor_dim(t, n):
    """
    Expand a type to the desired tensor dimension if possible
    Raise an error otherwise.
    - t is the given type
    - n is a number of dimensions to expand to
    """
def broadcast_types(t1, t2):
    """
    Applies broadcasting to both given types such that they
    become consistent with eachother and returns two new
    resulting types
    """
def register_inference_rule(call_target: Target) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
def register_refinement_rule(call_target: Target) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
def register_algebraic_expressions_inference_rule(call_target: Target) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
def add_inference_rule(n: Node):
    """
    Apply the addition inference rule. This includes:
    - scalar addition
    - broadcasting semantics

    Note that we always return the least precise type between
    the operands (after applying broadcasting) to be the final type of the operation

    Note that we do not modify the operand types themselves after applying broadcasting
    to them. We only use them to calculate the final type
    """
def get_attr_inference_rule(n: Node, traced):
    '''
    The current getattr rule only handles the shape attribute
    Can be extended to other attributes
    The most representitive type we have is "Dyn" but the system
    can be extended with more types, such as a type to represent shapes
    '''
def transpose_inference_rule(n: Node):
    """
    We check that dimensions for the transpose operations
    are within range of the tensor type of the node
    """
def reshape_inference_rule(n: Node):
    """
    Without dynamism, the rule checks that the
    product of the elements of the argument tensor
    type is equal to the product of the elements
    of the required shape. We gradualize this rule
    by adding a case to handle fully dynamic input
    as well as input where some of the tensor dimensions
    are unknown. In this case we check for divisibility
    """
def bn2d_inference_rule(n: Node, module_instance):
    """
    Given a BatchNorm2D instance and a node check the following conditions:
    - the input type can be expanded to a size 4 tensor: t =  (x_1, x_2, x_3, x_4)
    - the current node type can be expanded to a size 4 tensor: t' =  (x_1', x_2', x_3', x_4')
    - t is consistent with t'
    - x_2 is consistent with the module's num_features
    - x_2' is consistent with the module's num_features
    output type: the more precise type of t and t'
    """
def calculate_out_dimension(d_in, module_instance, index):
    """
    For calculating h_in and w_out according to the conv2D documentation
    """
def get_greatest_upper_bound(type1, type2):
    """
    Get the most precise type that's consistent with the given types
    """
def conv2d_inference_rule(n: Node, module_instance):
    """
    Given a Conv2D instance and a node check the following conditions:
    - the input type can be expanded to a size 4 tensor: t =  (x_1, x_2, H, W)
    - the current node type can be expanded to a size 4 tensor: t' =  (x_1', x_2', x_3', x_4')
    - x_2 is consistent with the module's in_channels
    - let o = (x_1, out_channels, H_out, W_out)
    then the output is the greatest upper bound of o and the existing node type t'.
    """
def relu_inference_rule(n: Node, module_instance):
    """
    Input and output shapes should be equal.
    """
def maxpool2d_check(typ, module_instance):
    """
    Applies the maxpool2d shape information to the input
    this affects the last two dimensions
    """
def maxpool2d_inference_rule(n: Node, module_instance):
    """
    Given a MaxPool2D instance and a node check the following conditions:
    - Input size matches size 3 or 4
    - Current node type is consistent with the output type we will calculate
    - Input size matches output size and the last two dimensions of the output
      are w_out and h_out. The remaining dimensions are the same as the input
    - Our final result is the greatest upper bound of the output we calculate
      and the current node type.
    """
def linear_check(tensor_type, module_instance):
    """
    Checks that an input tensor type satisfies the conditions for linear operation
    and returns the output type based on in and out features given by module_instance
    """
def linear_inference_rule(n: Node, module_instance):
    """
    Applies the shape information to the input then gets the greatest upper bound
    of the resulting type and the existing type
    """
def adaptiveavgpool2d_check(tensor_type, module_instance): ...
def adaptiveavgpool2d_inference_rule(n: Node, module_instance):
    """
    The input and output sizes should be the same except for the last
    two dimensions taken from the input, which represent width and height
    """
def flatten_check(tensor_type, start_dim, end_dim): ...
def flatten_inference_rule(n: Node):
    """
    Applies the flatten shape information to the input then gets the
    greatest upper bound of the resulting type and the existing type
    """

class GraphTypeChecker:
    env: Incomplete
    traced: Incomplete
    def __init__(self, env, traced) -> None: ...
    def type_check(self):
        """
        A gradual type checker for graphs
        Effect: every node's field type will be
        populated with a type after type-checking is done
        """
    def type_check_node(self, n: Node):
        """
        Type check a given fx node.
        Current operations:
        - Reshape
        - Transpose
        - Add
        - Relu
        - conv2d
        - batchnorm2d
        - flatten
        - maxpool2d
        - adaptiveavgpool2d
        - linear
        """

def conv_refinement_rule(n: Node):
    """
    The equality constraints are between the first dimension of
    the input and output
    """
def linear_refinement_rule(n: Node):
    """
    The equality constraints are between the first dimension of
    the input and output
    """
def all_eq(n: Node):
    """
    For operations where the input shape is equal to the output shape
    """
def first_two_eq(n: Node):
    """
    For operations where the first two dimensions of the input and output shape
    are equal
    """
def element_wise_eq(n: Node):
    """
    For element-wise operations and handles broadcasting.
    Note that after applying broadcasting to the arguments
    we are able to determine if certain dimensions have not been broadcast
    if they are symbolicallu equal.

    in this case, we can establish equality between those dimensions and the
    corresponding output dimensions.

    Note that it takes two iterations for this result. One iteration to establish
    equality between certain dimensions of the operands (requiring the whole solver
    including unification) and another iteration to establish equality between the operands
    and the resulting type, requiring another round of constraint generation and unificaiton.
    """
def flatten_refinement_rule(n: Node):
    """
    Generates equality constraints between the dimensions of the input and output
    that will not be involved in the flatten operation
    """
def conv_rule(n: Node, module_instance):
    """
    Represents the outout in terms of an algrbraic expression w.r.t
    the input when possible
    """

class Refine:
    """
    Symbolic shape inference.
    Generates constraints over type variables.
    Currently all constraints are equality constraints.
    """
    constraints: Incomplete
    traced: Incomplete
    symbol_iter: Incomplete
    def __init__(self, traced) -> None: ...
    def refine(self):
        """
        Generates constraints for
        every node in the graph based on
        the operation.
        """
    def symbolic_relations(self):
        """
        Infers algebraic relations
        """
    def replace_dyn_with_fresh_var(self, typ):
        """
        Replace all unknown types with fresh type variables.
        """
    def convert_to_sympy_symbols(self, typ):
        """
        Replace all unknown types with fresh type variables.
        """
    def refine_node(self, n: Node):
        """
        Returns a list of equality constraints for
        call_module and call_function nodes.
        Models the relation between input and output dimensions
        using constraints in case they are both tensors.
        All operations used in resnet50 are defined.
        """
    def infer_symbolic_relations(self, n: Node): ...

def get_parameter(traced, target: str):
    """
    Returns the parameter given by ``target`` if it exists,
    otherwise throws an error.

    See the docstring for ``get_submodule`` for a more detailed
    explanation of this method's functionality as well as how to
    correctly specify ``target``.

    Args:
        target: The fully-qualified string name of the Parameter
            to look for. (See ``get_submodule`` for how to specify a
            fully-qualified string.)

    Returns:
        torch.nn.Parameter: The Parameter referenced by ``target``

    Raises:
        AttributeError: If the target string references an invalid
            path or resolves to something that is not an
            ``nn.Parameter``
    """
