from _typeshed import Incomplete
from torch._dynamo.exc import TensorifyScalarRestartAnalysis as TensorifyScalarRestartAnalysis
from torch._dynamo.symbolic_convert import TensorifyState as TensorifyState
from torch._dynamo.utils import get_metrics_context as get_metrics_context
from torch._prims_common import get_computation_dtype as get_computation_dtype
from torch._subclasses import fake_tensor as fake_tensor
from torch._subclasses.fake_tensor import FakeTensor as FakeTensor
from torch._utils_internal import justknobs_check as justknobs_check
from torch.fx._utils import lazy_format_graph_code as lazy_format_graph_code
from torch.fx.experimental.symbolic_shapes import ShapeEnv as ShapeEnv, guard_scalar as guard_scalar, has_free_symbols as has_free_symbols
from torch.fx.graph_module import GraphModule as GraphModule
from torch.fx.passes.runtime_assert import _get_sym_val as _get_sym_val
from torch.fx.proxy import MetaProxy as MetaProxy
from torch.utils._sympy.interp import _run_sympy_handler as _run_sympy_handler, sympy_interp as sympy_interp
from torch.utils._sympy.reference import TensorReferenceAnalysis as TensorReferenceAnalysis
from torch.utils._sympy.symbol import SymT as SymT, symbol_is_type as symbol_is_type

__all__: list[str]
log: Incomplete
graph_code_log: Incomplete
SUPPORTED_OPS: Incomplete

def tensorify_python_scalars(gm: GraphModule, shape_env: ShapeEnv, fake_mode: fake_tensor.FakeTensorMode) -> None:
    """
    Converts Python scalar operations into Tensor operations within the graph. This pass looks for
    Tensor operations that involve SymFloat arguments and transforms them into equivalent operations
    that use only Tensor inputs.

    Args:
        gm: The FX graph module representing the computation graph.
        shape_env: The shape environment responsible for symbolic shape tracking and propagation
        during graph transformations.

    Returns:
        None
    """
