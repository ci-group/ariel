import dataclasses
from _typeshed import Incomplete
from enum import Enum
from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
from typing import Any, Callable

__all__ = ['Constraint', 'Dim', 'dims', 'refine_dynamic_shapes_from_suggested_fixes', 'AdditionalInputs']

class _DimHintType(Enum):
    """
    Enum for dynamic shape hints.
    - AUTO means automatic inference of shape (static or dynamic).
    - STATIC means static shape (always specialized).
    - DYNAMIC means dynamic, will error out if specialized.
    """
    AUTO = ...
    STATIC = ...
    DYNAMIC = ...

@dataclasses.dataclass
class _DimHint:
    type: _DimHintType
    min: int | None = ...
    max: int | None = ...
    _factory: bool | None = ...
    @staticmethod
    def AUTO(): ...
    @staticmethod
    def DYNAMIC(): ...
    @staticmethod
    def STATIC(): ...
    def __call__(self, min=None, max=None) -> _DimHint: ...

class Dim:
    '''
    The `Dim` class allows users to specify dynamism in their exported programs. By marking a dimension with a `Dim`,
    the compiler associates the dimension with a symbolic integer containing a dynamic range.

    The API can be used in 2 ways: Dim hints (i.e. automatic dynamic shapes: `Dim.AUTO`, `Dim.DYNAMIC`, `Dim.STATIC`),
    or named Dims (i.e. `Dim("name", min=1, max=2)`).

    Dim hints provide the lowest barrier to exportability, with the user only needing to specify if a dimension
    if dynamic, static, or left for the compiler to decide (`Dim.AUTO`). The export process will automatically
    infer the remaining constraints on min/max ranges and relationships between dimensions.

    Example::

        class Foo(nn.Module):
            def forward(self, x, y):
                assert x.shape[0] == 4
                assert y.shape[0] >= 16
                return x @ y


        x = torch.randn(4, 8)
        y = torch.randn(8, 16)
        dynamic_shapes = {
            "x": {0: Dim.AUTO, 1: Dim.AUTO},
            "y": {0: Dim.AUTO, 1: Dim.AUTO},
        }
        ep = torch.export(Foo(), (x, y), dynamic_shapes=dynamic_shapes)

    Here, export would raise an exception if we replaced all uses of `Dim.AUTO` with `Dim.DYNAMIC`,
    as x.shape[0] is constrained to be static by the model.

    More complex relations between dimensions may also be codegened as runtime assertion nodes by the compiler,
    e.g. (x.shape[0] + y.shape[1]) % 4 == 0, to be raised if runtime inputs do not satisfy such constraints.

    You may also specify min-max bounds for Dim hints, e.g. `Dim.AUTO(min=16, max=32)`, `Dim.DYNAMIC(max=64)`,
    with the compiler inferring the remaining constraints within the ranges. An exception will be raised if
    the valid range is entirely outside the user-specified range.

    Named Dims provide a stricter way of specifying dynamism, where exceptions are raised if the compiler
    infers constraints that do not match the user specification. For example, exporting the previous
    model, the user would need the following `dynamic_shapes` argument::

        s0 = Dim("s0")
        s1 = Dim("s1", min=16)
        dynamic_shapes = {
            "x": {0: 4, 1: s0},
            "y": {0: s0, 1: s1},
        }
        ep = torch.export(Foo(), (x, y), dynamic_shapes=dynamic_shapes)

    Named Dims also allow specification of relationships between dimensions, up to univariate linear relations.
    For example, the following indicates one dimension is a multiple of another plus 4::

        s0 = Dim("s0")
        s1 = 3 * s0 + 4

    '''
    AUTO: Incomplete
    DYNAMIC: Incomplete
    STATIC: Incomplete
    __name__: Incomplete
    min: Incomplete
    max: Incomplete
    def __init__(self, name: str, *, min: int | None = None, max: int | None = None) -> None: ...
    def __add__(self, other) -> Dim: ...
    def __radd__(self, other) -> Dim: ...
    def __sub__(self, other) -> Dim: ...
    def __rsub__(self, other) -> Dim: ...
    def __mul__(self, other) -> Dim: ...
    def __rmul__(self, other) -> Dim: ...
    def _derived_name(self, fn) -> str: ...
    def _derive(self, fn) -> Dim: ...
    @staticmethod
    def _readable(name: str, min_: int, max_: int) -> str: ...
    def __repr__(self) -> str: ...
_Dim = Dim

class _StaticDim(Dim):
    """
    Class for static :func:`Dim` types.

    This class is only for setting and checking static dim constraints,
    and the user should never interact with it.
    """
    __name__: Incomplete
    value: Incomplete
    def __init__(self, value: int) -> None: ...
    @property
    def min(self): ...
    @property
    def max(self): ...

class _DerivedDim(Dim):
    """
    Class for derived :func:`Dim` types.

    Currently we only support increasing linear expressions with integer coefficients.
    In other words, a derived Dim can always be written in the form Ax + B, where
    x is a regular Dim (i.e., non-derived Dim), A and B are integers, and A is positive.
    (In particular, the latter ensures that x < y => Ax + B < Ay + B.)
    These restrictions on the form of derived Dims makes the metatheory simpler: e.g.,
    it simplifies computing ranges for derived Dims, solving for underlying regular Dims,
    deciding equalities between derived Dims, and so on.

    The function lambda x: Ax + B is expressed by `fn`, where x is a normal Dim, `root`.
    The range of a derived Dim is computed by mapping `fn` over the range of its `root`.
    """
    __name__: Incomplete
    root: Incomplete
    fn: Incomplete
    def __init__(self, name: str, root: Dim, fn: Callable) -> None: ...
    @property
    def min(self): ...
    @property
    def max(self): ...
    def _derive(self, fn): ...
    def __repr__(self) -> str: ...

def dims(*names: str, min: int | None = None, max: int | None = None) -> tuple[Dim, ...]:
    """
    Util to create multiple :func:`Dim` types.

    Returns:
        A tuple of :func:`Dim` types.
    """

@dataclasses.dataclass
class _ConstraintTarget:
    """
    This represents input tensor dimensions.
    """
    t_id: int
    dim: int

@dataclasses.dataclass
class _Constraint(_ConstraintTarget):
    """
    This represents a Dim describing a constraint target.

    `name` is the name of the Dim.
    `constraint_range` contains the min/max bounds of the Dim.
    """
    name: str
    constraint_range: StrictMinMaxConstraint
    def _clone_with_range(self, lower: int = 0, upper=None): ...
    def __ge__(self, lower): ...
    def __gt__(self, lower): ...
    def __le__(self, upper): ...
    def __lt__(self, upper): ...
    def __bool__(self) -> bool: ...
    @property
    def serializable_spec(self): ...

@dataclasses.dataclass
class _PhantomRoot:
    '''
    This represents the root of a derived Dim where the root does not directly
    specify the shape of any input dimension, but the derived Dim does.

    e.g., the input shapes 2*dim and dim + 1 are related via a "phantom" dim.

    The fields `name`, `constraint_range`, and `val` carried by a phantom root
    help create a symbol for it. Any derived dims with this phantom root are
    backed by expressions over this symbol.
    '''
    name: str
    constraint_range: StrictMinMaxConstraint
    val: int

@dataclasses.dataclass
class _DerivedConstraint(_ConstraintTarget):
    """
    This represents a derived Dim, whose root is either a regular constraint target
    (which directly specifies the shape of some input dimension) or a phantom root
    (which does so indirectly).

    It can be thought of as a subclass of `_Constraint`, except that it does not
    support <, <=, >, >= operations.
    """
    name: str
    constraint_range: StrictMinMaxConstraint
    root: _ConstraintTarget | _PhantomRoot
    fn: Callable
    @property
    def serializable_spec(self): ...

@dataclasses.dataclass
class _RelaxedConstraint(_ConstraintTarget):
    """
    This represents a dim marked with Dim.AUTO/DYNAMIC (i.e. mark_dynamic() or maybe_mark_dynamic()),
    which leaves relations & min/max ranges for inference, instead of requiring explicit specification.
    The intention is for constraint violations to not be raised if produce_guards() finds equalities or
    relations between a _RelaxedConstraint and another type of _Constraint.
    """
    @property
    def serializable_spec(self): ...
Constraint = _Constraint | _DerivedConstraint | _RelaxedConstraint

@dataclasses.dataclass
class _IntWrapper:
    """
    Dummy wrapper class to wrap around integer inputs so that when we parse the
    dynamic_shapes structure, we can mark if any of the integers were marked as
    dynamic.
    """
    val: int
    dynamism: _DimHint | int | None = dataclasses.field(init=False, default=None)

class ShapesCollection:
    '''
    Builder for dynamic_shapes.
    Used to assign dynamic shape specifications to tensors that appear in inputs.

    This is useful particularly when :func:`args` is a nested input structure, and it\'s
    easier to index the input tensors, than to replicate the structure of :func:`args` in
    the :func:`dynamic_shapes` specification.

    Example::

        args = {"x": tensor_x, "others": [tensor_y, tensor_z]}

        dim = torch.export.Dim(...)
        dynamic_shapes = torch.export.ShapesCollection()
        dynamic_shapes[tensor_x] = (dim, dim + 1, 8)
        dynamic_shapes[tensor_y] = {0: dim * 2}
        # This is equivalent to the following (now auto-generated):
        # dynamic_shapes = {"x": (dim, dim + 1, 8), "others": [{0: dim * 2}, None]}

        torch.export(..., args, dynamic_shapes=dynamic_shapes)

    To specify dynamism for integers, we need to first wrap the integers using
    _IntWrapper so that we have a "unique identification tag" for each integer.

    Example::

        args = {"x": tensor_x, "others": [int_x, int_y]}
        # Wrap all ints with _IntWrapper
        mapped_args = pytree.tree_map_only(int, lambda a: _IntWrapper(a), args)

        dynamic_shapes = torch.export.ShapesCollection()
        dynamic_shapes[tensor_x] = (dim, dim + 1, 8)
        dynamic_shapes[mapped_args["others"][0]] = Dim.DYNAMIC

        # This is equivalent to the following (now auto-generated):
        # dynamic_shapes = {"x": (dim, dim + 1, 8), "others": [Dim.DYNAMIC, None]}

        torch.export(..., args, dynamic_shapes=dynamic_shapes)
    '''
    _shapes: Incomplete
    def __init__(self) -> None: ...
    def __setitem__(self, t, shape) -> None: ...
    def __getitem__(self, t): ...
    def __len__(self) -> int: ...
    def dynamic_shapes(self, m, args, kwargs=None):
        """
        Generates the :func:`dynamic_shapes` pytree structure according to :func:`args` and :func:`kwargs`.
        """

class AdditionalInputs:
    """
    Infers dynamic_shapes based on additional inputs.

    This is useful particularly for deployment engineers who, on the one hand, may
    have access to ample testing or profiling data that can provide a fair sense of
    representative inputs for a model, but on the other hand, may not know enough
    about the model to guess which input shapes should be dynamic.

    Input shapes that are different than the original are considered dynamic; conversely,
    those that are the same as the original are considered static. Moreover, we verify
    that the additional inputs are valid for the exported program. This guarantees that
    tracing with them instead of the original would have generated the same graph.

    Example::

        args0, kwargs0 = ...  # example inputs for export

        # other representative inputs that the exported program will run on
        dynamic_shapes = torch.export.AdditionalInputs()
        dynamic_shapes.add(args1, kwargs1)
        ...
        dynamic_shapes.add(argsN, kwargsN)

        torch.export(..., args0, kwargs0, dynamic_shapes=dynamic_shapes)
    """
    _examples: Incomplete
    def __init__(self) -> None: ...
    def add(self, args, kwargs=None) -> None:
        """
        Additional input :func:`args` and :func:`kwargs`.
        """
    def dynamic_shapes(self, m, args, kwargs=None):
        """
        Infers a :func:`dynamic_shapes` pytree structure by merging shapes of the
        original input :func:`args` and :func:`kwargs` and of each additional input
        args and kwargs.
        """
    def verify(self, ep) -> None:
        """
        Verifies that an exported program is valid for each additional input.
        """

def refine_dynamic_shapes_from_suggested_fixes(msg: str, dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any]) -> dict[str, Any] | tuple[Any] | list[Any]:
    """
    When exporting with :func:`dynamic_shapes`, export may fail with a ConstraintViolation error if the specification
    doesn't match the constraints inferred from tracing the model. The error message may provide suggested fixes -
    changes that can be made to :func:`dynamic_shapes` to export successfully.

    Example ConstraintViolation error message::

        Suggested fixes:

            dim = Dim('dim', min=3, max=6)  # this just refines the dim's range
            dim = 4  # this specializes to a constant
            dy = dx + 1  # dy was specified as an independent dim, but is actually tied to dx with this relation

    This is a helper function that takes the ConstraintViolation error message and the original :func:`dynamic_shapes` spec,
    and returns a new :func:`dynamic_shapes` spec that incorporates the suggested fixes.

    Example usage::

        try:
            ep = export(mod, args, dynamic_shapes=dynamic_shapes)
        except torch._dynamo.exc.UserError as exc:
            new_shapes = refine_dynamic_shapes_from_suggested_fixes(
                exc.msg, dynamic_shapes
            )
            ep = export(mod, args, dynamic_shapes=new_shapes)

    """
