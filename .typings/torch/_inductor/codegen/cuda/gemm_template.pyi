import abc
import torch
from . import cutlass_utils as cutlass_utils
from ... import ir as ir
from ...ir import Buffer as Buffer, CUDATemplateBuffer as CUDATemplateBuffer, ChoiceCaller as ChoiceCaller, FixedLayout as FixedLayout, IRNode as IRNode, Layout as Layout, ReinterpretView as ReinterpretView
from ...utils import Placeholder as Placeholder, is_dynamic as is_dynamic
from ...virtualized import V as V
from ..common import IndentedBuffer as IndentedBuffer
from .cuda_kernel import CUDATemplateKernel as CUDATemplateKernel
from .cuda_template import CUTLASSTemplate as CUTLASSTemplate
from .cutlass_presets import gen_cutlass_presets as gen_cutlass_presets
from .cutlass_python_evt import CutlassEVTCodegen as CutlassEVTCodegen, scaled_mm_evt as scaled_mm_evt
from .cutlass_utils import torch_dtype_to_cutlass_type as torch_dtype_to_cutlass_type
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from torch._inductor.codegen.cuda.cutlass_cache import maybe_fetch_ops as maybe_fetch_ops
from torch._inductor.scheduler import BaseSchedulerNode as BaseSchedulerNode
from torch._inductor.select_algorithm import create_inputs_key as create_inputs_key
from torch._inductor.utils import clear_on_fresh_cache as clear_on_fresh_cache
from typing import Any

GemmOperation = Any
log: Incomplete
GEMM_TEMPLATE_CUTLASS_3X: str
GEMM_ARGS_CUTLASS_3X: str
GEMM_ARGS_CUTLASS_3X_EPILOGUE: str
GEMM_TEMPLATE_CUTLASS_2X: str
GEMM_ARGS_CUTLASS_2X: str
GEMM_ARGS_SPARSE_CUTLASS_2X: str
GEMM_STANDALONE_RUNNER_ADDITIONAL_INCLUDES: str
GEMM_STANDALONE_RUNNER_TEMPLATE: str

class CUTLASSGemmTemplate(CUTLASSTemplate, ABC, metaclass=abc.ABCMeta):
    """
    CUTLASS GEMM Template, which is used to generate CUTLASS GEMM kernels
    including those which allow flexible fusions with epilogues.
    """
    filtered_ops_cache: dict[str, list[Any]]
    cache_clear: Incomplete
    alpha: Incomplete
    beta: Incomplete
    use_fast_accum: Incomplete
    cache_key: str
    def __init__(self, input_nodes: list[Buffer], layout: Layout, alpha: float, beta: float, input_reorder: list[int] | None = None, use_fast_accum: bool | None = None) -> None:
        """
        Args:
            input_nodes (List[Buffer]): List of input nodes of the GEMM kernel.
            layout (Layout): Layout type of the resulting output node.
            alpha (float): The scaling factor for the product of the inputs in the GEMM operation.
            beta (float): The scaling factor applied to the output matrix.
            input_reorder (Optional[List[int]]): Specifies the reordering of the input nodes. If not provided,
                            no reordering is performed. Defaults to None.
        """
    @staticmethod
    @abstractmethod
    def add_cutlass_gemm_choices(choices: list[ChoiceCaller], layout: ir.Layout, input_nodes: list[Buffer], alpha: float | int = 1, beta: float | int = 0, input_reorder: list[int] | None = None, use_fast_accum: bool | None = None, **extra_kwargs) -> None: ...
    @staticmethod
    @abstractmethod
    def _get_supported_ops() -> list[cutlass_library.gemm_operation.GemmOperation]: ...
    @staticmethod
    @abstractmethod
    def _has_tma_epilogue(self) -> bool: ...
    @abstractmethod
    def _get_template(self) -> str: ...
    @abstractmethod
    def _get_template_args(self, op: cutlass_library.gemm_op.GemmOperation) -> tuple[str, str | None]: ...
    @abstractmethod
    def _are_inputs_layout_compatible(self, layouts: list[Layout]) -> bool: ...
    @abstractmethod
    def _shape_match(self, op: cutlass_library.gemm_op.GemmOperation) -> bool: ...
    @abstractmethod
    def _alignment_match(self, op: cutlass_library.gemm_op.GemmOperation) -> bool: ...
    @abstractmethod
    def _set_bias_layout_and_alignment(self, op: cutlass_library.gemm_op.GemmOperation) -> bool: ...
    @abstractmethod
    def _define_gemm_instance(self, op: GemmOperation, evt_name: str | None = None) -> tuple[str, str]: ...
    @abstractmethod
    def _get_extra_inputs_and_names(self, op: cutlass_gemm_op.GemmOperation = None) -> tuple[Buffer | None, list[Buffer | None], list[str]]: ...
    @abstractmethod
    def _update_arg_names_for_test_call_statement(self, arg_names: list[str], input_nodes: list[Buffer]) -> list[str]: ...
    def _add_cutlass_gemm_choices(self, choices: list[ChoiceCaller], layout: ir.Layout, input_nodes: list[Buffer], alpha: float | int = 1, beta: float | int = 0, input_reorder: list[int] | None = None, **extra_kwargs) -> None:
        """
        Adds Cutlass GEMM configurations choices to the auto-tuning list.

        This function mutates the passed list of choices by appending the choices for Cutlass GEMM configs to it.

        Args:
            choices (list): The list to which choices are appended.
            layout (ir.Layout): The layout configuration.
            input_nodes (list): The list of input nodes.
            alpha (float,int): Scaling factor, defaults to 1.
            beta (float,int): Offset, defaults to 0.
            input_reorder (list, optional): Order of the inputs, defaults to None.
            **extra_kwargs: Additional keyword arguments.

        """
    def header(self) -> IndentedBuffer:
        """
        Returns a buffer containing CUDA C++ code for the header section of the CUTLASS GEMM template.
        This section primarily includes the necessary header files.

        Returns:
            IndentedBuffer: An instance of IndentedBuffer that contains the generated CUDA C++ header code.
        """
    @staticmethod
    def cutlass_layout(torch_layout: ir.Layout) -> cutlass_lib.LayoutType | None:
        """
        Converts an ir.Layout instance into the corresponding cutlass_library.LayoutType enum value
        (RowMajor, ColumnMajor, or None if no matching value is found ).

        Args:
            torch_layout (ir.Layout): The layout that needs to be looked up.

        Returns:
            cutlass_lib.LayoutType: The converted layout corresponding to the `torch_layout` or None if no matching
            value is found.
        """
    @staticmethod
    def flip_cutlass_layout(cutlass_layout: cutlass_lib.LayoutType) -> cutlass_lib.LayoutType:
        """Helper method: Flips a given cutlass layout (cutlass_lib.LayoutType) from RowMajor
        to ColumnMajor or vice versa"""
    @staticmethod
    def layout_match(torch_layout: ir.Layout, cutlass_layout: cutlass_lib.LayoutType) -> bool:
        """Helper Method: Determines whether a given torch layout matches a given Cutlass layout"""
    @staticmethod
    def set_alignment(torch_layout, op_element) -> bool:
        """
        Helper method to update the alignment of a given CUTLASS GEMM op operand's element.

        This method modifies the alignment of the given Cutlass GEMM op operand's element to match the
        layout of the corresponding ir.Buffer node.

        Args:
            torch_layout: The layout of the corresponding ir.Buffer node.
            op_element: The Cutlass GEMM op operand's element whose alignment is to be updated.

        Returns:
            bool: True if the alignment was successfully updated, False otherwise.
        """
    @staticmethod
    def should_swap_XW(bias: IRNode) -> bool:
        """
        Helper method to determine whether we should do an explicit transpose by switching the order of the
        matmul operands. This might be necessary when we can't otherwise arrive at the right memory
        layout for the given Bias operand.

        Note: This method is a workaround for CUDA Errors that seemingly non-deterministically
        occurred in practice in some CUTLASS GEMM Kernels with Linear epilogues that have a bias term.
        it might make sense to check on newer Cutlass releases whether it makes sense to keep
        returning True in certain cases or whether it becomes unnecessary.
        """
    @staticmethod
    def swap_XW(op: cutlass_library.gemm_op.GemmOperation) -> cutlass_library.gemm_op.GemmOperation:
        """
        Swap operands X and W (aka operans A and B) of the GEMM operation. This
        requires transposing the operands, which is done by swapping the strides.
        Note that we don't change the apparent external layout, just the operand layout.
        this is intentional.
        """
    def fix_op_layout(self, op: cutlass_library.gemm_op.GemmOperation, X: Buffer, W: Buffer, Bias: Buffer | None, Y: Buffer | ReinterpretView) -> cutlass_library.gemm_op.GemmOperation: ...
    def _dtype_match(self, op: cutlass_library.gemm_op.GemmOperation) -> bool:
        """
        Checking dtypes of A, B, acc, D here.

        Empirically speaking, CUTLASS2x ops have same dtype for C and D.
        """
    def filter_op(self, op: cutlass_library.gemm_op.GemmOperation) -> cutlass_library.gemm_op.GemmOperation:
        """
        Helper method:

        Determines whether a given Cutlass GEMM op definition is suitable for the current
        input / output of the operation that this template is supposed to implement.

        Takes memory layout, dtype and support for EVT operations into account,
        and filters potentially problematic ops.

        Returns None if the op is not suitable, otherwise returns the op to be used, which might
        have been mutated.
        """
    def gen_ops(self) -> list[tuple[str, cutlass_gemm_op.GemmOperation]]:
        """
        Creates a list of Cutlass GemmOperation instances that match the operation this template is designed to represent.
        The matching is carried out with respect to the input and output specifications of the operation.

        No function arguments.

        Returns:
            List[Tuple[str, cutlass_gemm_op.GemmOperation]]: A list of (cutlass_name, GemmOperation)
            tuples that are compatible with the operation requirements of this template.
        """
    def gemm_mode(self) -> str:
        '''
        Returns a Cutlass GEMM mode string for the current operation, dependent on whether this op implements
        a batched GEMM or a simple GEMM without batch dimension.

        Returns:
        str: A string indicating the Cutlass GEMM mode. If the output node has more than two dimensions,
            "cutlass::gemm::GemmUniversalMode::kBatched" is returned, otherwise
            "cutlass::gemm::GemmUniversalMode::kGemm" is returned.
        '''
    def render(self, kernel: CUDATemplateKernel, op: cutlass_gemm_op.GemmOperation = None, template_buffer_node: CUDATemplateBuffer | None = None, epilogue_nodes: list[BaseSchedulerNode] | None = None, **kwargs) -> str:
        """
        The primary entry point for the code rendering process used in this template.
        Renders the Cutlass based CUDA C++ code for the GEMM Kernel that this template is designed to implement,
        including potentially fused epilogues.

        Args:
            kernel (CUDATemplateKernel): The kernel to be rendered.
            op (cutlass_gemm_op.GemmOperation, optional): A GEMM operation that is required to be compatible with the
                input and output definitions as well as a possible epilogue. Defaults to None.
            **kwargs: Additional keyword arguments. Currently unused.

        Returns:
            str: Cutlass based CUDA C++ code fragment as a string, to be used by the current
            CUDATemplateKernel or autotuning code.

        Note:
            All inputs and their corresponding buffer addresses and names take precedence over previously
            passed inputs to the template at construction time. However, they should be layout compatible.
        """
    def test_call_statement(self, kernel, input_nodes, names_str: str = '') -> str:
        """
        Helper method to render the Cutlass CUDA C++ code required for calling the GEMM operation in the standalone
        test runner that might also be generated along with the rest of the code, if the corresponding config is
        enabled.

        Returns a C++ statement that calls the GEMM operation with the correct arguments.
        """
    def _render_evt(self, op: GemmOperation, evt_py_code: str, buffer_renames: dict[str, str], output_dtype: torch.dtype, accumulator_dtype: torch.dtype) -> tuple[str, str, str]: ...

class CUTLASS3xGemmTemplate(CUTLASSGemmTemplate):
    """
    CUTLASS 3x GEMM Template, which is used to generate CUTLASS GEMM kernels
    including those which allow flexible fusions with epilogues.
    """
    def __init__(self, input_nodes: list[Buffer], layout: Layout, alpha: float, beta: float, input_reorder: list[int] | None = None, use_fast_accum: bool | None = None) -> None: ...
    @staticmethod
    def add_cutlass_gemm_choices(choices: list[ChoiceCaller], layout: ir.Layout, input_nodes: list[Buffer], alpha: float | int = 1, beta: float | int = 0, input_reorder: list[int] | None = None, use_fast_accum: bool | None = None, **extra_kwargs) -> None: ...
    @staticmethod
    def _get_supported_ops() -> list[cutlass_library.gemm_operation.GemmOperation]: ...
    def _get_template(self) -> str: ...
    def _get_template_args(self, op: cutlass_library.gemm_op.GemmOperation) -> tuple[str, str | None]: ...
    @staticmethod
    def _has_tma_epilogue(op: cutlass_library.gemm_op.GemmOperation) -> bool:
        """Helper method: Determine whether a given Cutlass GEMM op has a TMA Epilogue"""
    @staticmethod
    def supports_epilogue_fusion(op: GemmOperation) -> bool: ...
    def _are_inputs_layout_compatible(self, layouts: list[Layout]) -> bool:
        """
        Evaluates whether input layouts are compatible for General Matrix Multiply (GEMM).

        This function checks compatibility of A, B, and possibly C operand layouts for
        a General Matrix Multiply (GEMM) operation, expressed as 'alpha * matmul(A, B) + beta * C'.
        It verifies requirements such as matching data types, minimum rank, and suitability
        for broadcasting, as defined by PyTorch operations like `torch.matmul`, `torch.aten.mm`,
        `addmm`, `bmm`, `baddbmm`, etc.

        Args:
            layouts (List[Layout]): List containing 2 or 3 Layout objects representing
                                    the input matrices A, B, and possibly C.

        Returns:
            bool: True if layouts are GEMM compatible, otherwise False.
        """
    def _render_evt(self, op: GemmOperation, evt_py_code: str, var_name_to_buffer_name: dict[str, str], output_dtype: torch.dtype, accumulator_dtype: torch.dtype) -> tuple[str, str, str]: ...
    def _shape_match(self, op: cutlass_library.gemm_op.GemmOperation) -> bool: ...
    def _alignment_match(self, op: cutlass_library.gemm_op.GemmOperation) -> bool: ...
    def _set_bias_layout_and_alignment(self, op: cutlass_library.gemm_op.GemmOperation) -> bool: ...
    def _define_gemm_instance(self, op: GemmOperation, evt_name: str | None = None) -> tuple[str, str]:
        """Defines and renders the Cutlass / CUDA C++ code for a given GEMM operation instance.

        This function uses the Cutlass library to generate key parts of the codegen process. General Matrix Multiply
        forms a core part of a number of scientific applications, so this efficient and adaptable implementation is
        crucial.

        Args:
            op (cutlass_library.gemm_op.GemmOperation): This is the core GEMM operation that we are defining and rendering.

        Returns:
            Tuple[str, str]: A tuple where the first part is a string that constitutes the defined GEMM operation in C++
                             code (render) and the second part is the string that specifies the operation type.
        """
    def _get_extra_inputs_and_names(self, op: cutlass_gemm_op.GemmOperation = None) -> tuple[Buffer | None, list[Buffer | None], list[str]]: ...
    def _update_arg_names_for_test_call_statement(self, arg_names: list[str], input_nodes: list[Buffer]) -> list[str]: ...
    def render_gemm_arguments(self, argument_template: str, epilogue_template: str, should_swap_xw: bool, X: IRNode, W: IRNode, Bias: IRNode, Y: IRNode, alpha: float, beta: float, kernel: CUDATemplateKernel, epilogue_args) -> str:
        """
        Render the Cutlass CUDA C++ code required for passing arguments to the GEMM operation.

        Args:
            argument_template (str): Template for the GEMM operation arguments.
            epilogue_template (str): Template for the epilogue arguments.
            should_swap_xw (bool): Determines whether X, W operands should be swapped. If True, applies an explicit
            transpose operation to X and W.
            X (IRNode): The X input tensor.
            W (IRNode): The W input tensor.
            Bias (IRNode): The bias tensor.
            Y (IRNode): The output tensor.
            alpha (float): Scaling factor for the product of the inputs.
            beta (float): Scaling factor for the output tensor.
            kernel (CUDATemplateKernel): CUDA Template kernel for the operation.
            epilogue_args (any): Additional arguments for the epilogue state.

        Returns:
            str: A block of CUDA C++ code as a string, ready to be used as arguments for the GEMM operation.

        Note: If `should_swap_xw` is True, a transpose operation will be applied to the X, W, Bias, and Y
        tensors. This operation also implies the M and N dimensions of Bias and GEMM output to be swapped
        before the function call.
        """

class CUTLASS2xGemmTemplate(CUTLASSGemmTemplate):
    def __init__(self, input_nodes: list[Buffer], layout: Layout, alpha: float, beta: float, input_reorder: list[int] | None = None) -> None: ...
    @staticmethod
    def add_cutlass_gemm_choices(choices: list[ChoiceCaller], layout: ir.Layout, input_nodes: list[Buffer], alpha: float | int = 1, beta: float | int = 0, input_reorder: list[int] | None = None, use_fast_accum: bool | None = False, **extra_kwargs) -> None: ...
    @staticmethod
    def _get_supported_ops() -> list[cutlass_library.gemm_operation.GemmOperation]: ...
    @staticmethod
    def _has_tma_epilogue(self) -> bool: ...
    def _get_template(self) -> str: ...
    def _get_template_args(self, op: cutlass_library.gemm_op.GemmOperation) -> tuple[str, str | None]: ...
    def _are_inputs_layout_compatible(self, layouts: list[Layout]) -> bool:
        """
        Evaluates whether input layouts are compatible for set of operations supported by this class.

        Args:
            layouts (List[Layout]): List containing Layout objects representing
                                    the input matrices.

        Returns:
            bool: True if layouts are GEMM compatible, otherwise False.
        """
    def _shape_match(self, op: cutlass_library.gemm_op.GemmOperation) -> bool: ...
    def _alignment_match(self, op: cutlass_library.gemm_op.GemmOperation) -> bool: ...
    def _set_bias_layout_and_alignment(self, op: cutlass_library.gemm_op.GemmOperation) -> bool: ...
    def _define_gemm_instance(self, op: GemmOperation, evt_name: str | None = None) -> tuple[str, str]:
        """Defines and renders the Cutlass / CUDA C++ code for a given GEMM operation instance.

        This function uses the Cutlass library to generate key parts of the codegen process. General Matrix Multiply
        forms a core part of a number of scientific applications, so this efficient and adaptable implementation is
        crucial.

        Args:
            op (cutlass_library.gemm_op.GemmOperation): This is the core GEMM operation that we are defining and rendering.

        Returns:
            Tuple[str, str]: A tuple where the first part is a string that constitutes the defined GEMM operation in C++
                             code (render) and the second part is the string that specifies the operation type.
        """
    def _get_extra_inputs_and_names(self, op: cutlass_gemm_op.GemmOperation = None) -> tuple[Buffer | None, list[Buffer | None], list[str]]: ...
    def _update_arg_names_for_test_call_statement(self, arg_names: list[str], input_nodes: list[Buffer]) -> list[str]: ...
    def render_gemm_arguments(self, instance_type: str, argument_template: str, epilogue_template: str, should_swap_xw: bool, X: IRNode, W: IRNode, Bias: IRNode, Meta: IRNode, Y: IRNode, alpha: float, beta: float, kernel: CUDATemplateKernel, epilogue_args) -> str:
        """
        Render the Cutlass CUDA C++ code required for passing arguments to the GEMM operation.

        Args:
            instance_type (str): GEMM instance type.
            argument_template (str): Template for the GEMM operation arguments.
            epilogue_template (str): Template for the epilogue arguments.
            should_swap_xw (bool): Determines whether X, W operands should be swapped. If True, applies an explicit
            transpose operation to X and W.
            X (IRNode): The X input tensor.
            W (IRNode): The W input tensor.
            Bias (IRNode): The bias tensor.
            Meta (IRNode): The meta tensor.
            Y (IRNode): The output tensor.
            alpha (float): Scaling factor for the product of the inputs.
            beta (float): Scaling factor for the output tensor.
            kernel (CUDATemplateKernel): CUDA Template kernel for the operation.
            epilogue_args (any): Additional arguments for the epilogue state.

        Returns:
            str: A block of CUDA C++ code as a string, ready to be used as arguments for the GEMM operation.

        Note: If `should_swap_xw` is True, a transpose operation will be applied to the X, W, Bias, and Y
        tensors. This operation also implies the M and N dimensions of Bias and GEMM output to be swapped
        before the function call.
        """
