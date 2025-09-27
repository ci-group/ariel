import dataclasses
import sympy
import torch
import torch.utils._pytree as pytree
from .graph_signature import ArgumentSpec, ExportGraphSignature
from _typeshed import Incomplete
from collections.abc import Iterator
from contextlib import contextmanager
from torch._export.verifier import Verifier
from torch.export.decomp_utils import CustomDecompTable
from torch.fx._symbolic_trace import _ConstantAttributeType
from torch.fx.passes.infra.pass_base import PassResult
from torch.utils._sympy.value_ranges import ValueRanges
from typing import Any, Callable, final

__all__ = ['ExportedProgram', 'ModuleCallEntry', 'ModuleCallSignature', 'default_decompositions']

PassType = Callable[[torch.fx.GraphModule], PassResult | None]

@dataclasses.dataclass
class ModuleCallSignature:
    inputs: list[ArgumentSpec]
    outputs: list[ArgumentSpec]
    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec
    forward_arg_names: list[str] | None = ...
    def replace_all_uses_with(self, original_node, new_node) -> None: ...

@dataclasses.dataclass
class ModuleCallEntry:
    fqn: str
    signature: ModuleCallSignature | None = ...

def default_decompositions() -> CustomDecompTable:
    """
    This is the default decomposition table which contains decomposition of
    all ATEN operators to core aten opset. Use this API together with
    :func:`run_decompositions()`
    """

class ExportedProgram:
    """
    Package of a program from :func:`export`. It contains
    an :class:`torch.fx.Graph` that represents Tensor computation, a state_dict containing
    tensor values of all lifted parameters and buffers, and various metadata.

    You can call an ExportedProgram like the original callable traced by
    :func:`export` with the same calling convention.

    To perform transformations on the graph, use ``.module`` property to access
    an :class:`torch.fx.GraphModule`. You can then use
    `FX transformation <https://pytorch.org/docs/stable/fx.html#writing-transformations>`_
    to rewrite the graph. Afterwards, you can simply use :func:`export`
    again to construct a correct ExportedProgram.
    """
    _graph_module: Incomplete
    _graph_signature: ExportGraphSignature
    _state_dict: dict[str, Any]
    _range_constraints: dict[sympy.Symbol, ValueRanges]
    _module_call_graph: list[ModuleCallEntry]
    _example_inputs: Incomplete
    _constants: Incomplete
    _verifiers: Incomplete
    def __init__(self, root: torch.nn.Module | dict[str, Any], graph: torch.fx.Graph, graph_signature: ExportGraphSignature, state_dict: dict[str, torch.Tensor | torch.nn.Parameter], range_constraints: dict[sympy.Symbol, Any], module_call_graph: list[ModuleCallEntry], example_inputs: tuple[tuple[Any, ...], dict[str, Any]] | None = None, constants: dict[str, _ConstantAttributeType] | None = None, *, verifiers: list[type[Verifier]] | None = None) -> None: ...
    @property
    def graph_module(self): ...
    @graph_module.setter
    def graph_module(self, value) -> None: ...
    @property
    def graph(self): ...
    @graph.setter
    def graph(self, value) -> None: ...
    @property
    def graph_signature(self): ...
    @graph_signature.setter
    def graph_signature(self, value) -> None: ...
    @property
    def state_dict(self): ...
    @state_dict.setter
    def state_dict(self, value) -> None: ...
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """
        Returns an iterator over original module's parameters.
        """
    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]:
        """
        Returns an iterator over original module parameters, yielding
        both the name of the parameter as well as the parameter itself.
        """
    def buffers(self) -> Iterator[torch.Tensor]:
        """
        Returns an iterator over original module buffers.
        """
    def named_buffers(self) -> Iterator[tuple[str, torch.Tensor]]:
        """
        Returns an iterator over original module buffers, yielding
        both the name of the buffer as well as the buffer itself.
        """
    @property
    def range_constraints(self): ...
    @range_constraints.setter
    def range_constraints(self, value) -> None: ...
    @property
    def module_call_graph(self): ...
    @module_call_graph.setter
    def module_call_graph(self, value) -> None: ...
    @property
    def example_inputs(self): ...
    @example_inputs.setter
    def example_inputs(self, value) -> None: ...
    @property
    def call_spec(self): ...
    @call_spec.setter
    def call_spec(self, value) -> None: ...
    @property
    def verifier(self) -> Any: ...
    @verifier.setter
    def verifier(self, value) -> None: ...
    @property
    def dialect(self) -> str: ...
    @dialect.setter
    def dialect(self, value) -> None: ...
    @property
    def verifiers(self): ...
    @verifiers.setter
    def verifiers(self, value) -> None: ...
    @property
    def tensor_constants(self): ...
    @tensor_constants.setter
    def tensor_constants(self, value) -> None: ...
    @property
    def constants(self): ...
    @constants.setter
    def constants(self, value) -> None: ...
    def _get_flat_args_with_check(self, args, kwargs):
        """Flatten args, kwargs using pytree, then, check specs.

        Args:
            args: List[Any] original args passed to __call__
            kwargs: Dict[str, Any] original kwargs passed to __call

        Returns:
            A tuple of (flat_args, received_spec)
            flat_args is flattend args / kwargs
            received_spec is the pytree spec produced while flattening the
            tuple (args, kwargs)
        """
    def _graph_module_flat_inputs(self, args: Any, kwargs: Any) -> Any:
        """Transform args, kwargs of __call__ to args for graph_module.

        self.graph_module takes stuff from state dict as inputs.
        The invariant is for ep: ExportedProgram is
        ep(args, kwargs) ==
          ep.postprocess(ep.graph_module(ep.graph_module_flat_inputs(args, kwargs)))
        """
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def _postprocess_graph_module_outputs(self, res, orig_args, orig_kwargs):
        """Process potential mutations to the input.

        Because self.graph_module is functional, so mutations has to be written
        back after execution of graph_module.
        """
    def __str__(self) -> str: ...
    def module(self) -> torch.nn.Module:
        """
        Returns a self contained GraphModule with all the parameters/buffers inlined.
        """
    def _num_lifted_params_buffers(self): ...
    @_disable_prexisiting_fake_mode
    def run_decompositions(self, decomp_table: dict[torch._ops.OperatorBase, Callable] | None = None, decompose_custom_triton_ops: bool = False) -> ExportedProgram:
        """
        Run a set of decompositions on the exported program and returns a new
        exported program. By default we will run the Core ATen decompositions to
        get operators in the
        `Core ATen Operator Set <https://pytorch.org/docs/stable/torch.compiler_ir.html>`_.

        For now, we do not decompose joint graphs.

        Args:
            decomp_table:
             An optional argument that specifies decomp behaviour for Aten ops
             (1) If None, we decompose to core aten decompositions
             (2) If empty, we don't decompose any operator


        Some examples:

        If you don't want to decompose anything

        .. code-block:: python

            ep = torch.export.export(model, ...)
            ep = ep.run_decompositions(decomp_table={})

        If you want to get a core aten operator set except for certain operator, you can do following:

        .. code-block:: python

            ep = torch.export.export(model, ...)
            decomp_table = torch.export.default_decompositions()
            decomp_table[your_op] = your_custom_decomp
            ep = ep.run_decompositions(decomp_table=decomp_table)
        """
    def _transform_do_not_use(self, *passes: PassType) -> ExportedProgram: ...
    def _check_input_constraints(self, flat_args_with_path) -> None: ...
    def validate(self) -> None: ...
    @final
    def _validate(self) -> None: ...
    def _update(self, graph_module, graph_signature, *, state_dict=None, constants=None, verifiers=None) -> ExportedProgram: ...
