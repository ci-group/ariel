import abc
import torch.nn as nn
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.placement_types import Placement
from typing import Any

__all__ = ['ParallelStyle', 'RowwiseParallel', 'SequenceParallel', 'ColwiseParallel', 'PrepareModuleInput', 'PrepareModuleInputOutput', 'PrepareModuleOutput']

class ParallelStyle(ABC, metaclass=abc.ABCMeta):
    """
    The parallel style contract defines how the module or submodule should be parallelized.

    It only defines the ``apply`` method for ``parallelize_module`` to use, this allows maximum
    flexibility for different kind of style implementations.
    """
    src_data_rank: int | None
    @abstractmethod
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...

class ColwiseParallel(ParallelStyle):
    '''
    Partition a compatible nn.Module in a column-wise fashion. Currently supports nn.Linear and nn.Embedding.
    Users can compose it together with RowwiseParallel to achieve the sharding of more complicated modules.
    (i.e. MLP, Attention)

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Module, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be replicated.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Module, this is used to ensure the output of the nn.Module
            with the user desired layout. If not specified, the output tensor is sharded on the last dimension.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: True.
    Returns:
        A :class:`ParallelStyle` object that represents Colwise sharding of the nn.Module.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> m = Model(...)  # m is a nn.Module that contains a "w1" nn.Linear submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # By default, the input of the "w1" Linear will be converted to Replicated DTensor
        >>> # and the output of "w1" will return :class:`torch.Tensor` that shards on the last dim.
        >>>
        >>> sharded_mod = parallelize_module(m, tp_mesh, {"w1": ColwiseParallel()})
        >>> ...

    .. note:: By default ``ColwiseParallel`` output is sharded on the last dimension if the ``output_layouts`` not
        specified, if there\'re operators that require specific tensor shape (i.e. before the paired ``RowwiseParallel``),
        keep in mind that if the output is sharded the operator might need to be adjusted to the sharded size.
    '''
    input_layouts: Incomplete
    output_layouts: Incomplete
    desired_input_layouts: Incomplete
    use_local_output: Incomplete
    def __init__(self, *, input_layouts: Placement | None = None, output_layouts: Placement | None = None, use_local_output: bool = True) -> None: ...
    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh): ...
    def _partition_linear_fn(self, name, module, device_mesh) -> None: ...
    def _partition_embedding_fn(self, name, module, device_mesh) -> None: ...
    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh): ...
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...
    def __repr__(self) -> str: ...

class RowwiseParallel(ParallelStyle):
    '''
    Partition a compatible nn.Module in a row-wise fashion. Currently supports nn.Linear and nn.Embedding.
    Users can compose it with ColwiseParallel to achieve the sharding of more complicated modules.
    (i.e. MLP, Attention)

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Module, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be sharded on the last dimension.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Module, this is used to ensure the output of the nn.Module
            with the user desired layout. If not specified, the output tensor is replicated.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: True.
    Returns:
        A :class:`ParallelStyle` object that represents Rowwise sharding of the nn.Module.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> m = Model(...)  # m is a nn.Module that contains a "w2" nn.Linear submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # By default, the input of the "w2" Linear will be converted to DTensor that shards on the last dim
        >>> # and the output of "w2" will return a replicated :class:`torch.Tensor`.
        >>>
        >>> sharded_mod = parallelize_module(m, tp_mesh, {"w2": RowwiseParallel()}),
        >>> ...
    '''
    input_layouts: Incomplete
    output_layouts: Incomplete
    use_local_output: Incomplete
    def __init__(self, *, input_layouts: Placement | None = None, output_layouts: Placement | None = None, use_local_output: bool = True) -> None: ...
    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh): ...
    def _partition_linear_fn(self, name, module, device_mesh) -> None: ...
    def _partition_embedding_fn(self, name, module, device_mesh) -> None: ...
    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh): ...
    desired_input_layouts: tuple[Placement, ...]
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...
    def __repr__(self) -> str: ...

class SequenceParallel(ParallelStyle):
    '''
    SequenceParallel replicates a compatible ``nn.Module`` parameters and runs the sharded computation with
    input sharded on the sequence dimension. This currently supports ``nn.LayerNorm``, ``nn.Dropout``, and the
    `RMSNorm python implementation <https://github.com/facebookresearch/llama/blob/main/llama/model.py#L34>`__

    This style implements the operation that is described in the paper
    `Reducing Activation Recomputation in Large Transformer Models <https://arxiv.org/abs/2205.05198>`__

    If the input passed in to this ``nn.Module`` is a :class:`torch.Tensor`, it assumes that the input is already sharded
    on the sequence dimension and converts the input to a :class:`DTensor` sharded on the sequence dimension. If the input
    passed in to this ``nn.Module`` is already a :class:`DTensor` but is not sharded on the sequence dimension, it would
    redistribute the input to be sharded on the sequence dimension.

    The output of the ``nn.Module`` will be sharded on the sequence dimension.

    Keyword Args:
        sequence_dim (int, optional):
            The sequence dimension of the input tensor for the ``nn.Module``, this is used to annotate the input tensor to
            become a DTensor that is sharded on the sequence dimension, default: 1.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: False.
    Returns:
        A :class:`ParallelStyle` object that represents Sequence Parallel of the ``nn.Module``.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, SequenceParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> m = Model(...)  # m is a nn.Module that contains a "norm" nn.LayerNorm submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # By default, the input of the "norm" will be converted to DTensor that shards on the sequence dim
        >>> # and the output of "norm" will return a sharded on sequence dimension :class:`DTensor`.
        >>>
        >>> sharded_mod = parallelize_module(m, tp_mesh, {"norm": SequenceParallel()}),
        >>> ...

    .. note:: SequenceParallel style assumes ones initialization if there are weights in the nn.Module (i.e.
        ``nn.LayerNorm`` or ``RMSNorm``, and they by default have ones initialization). If you have custom
        inits for the weights on those modules, you need to broadcast the weights before/after parallelizing
        to ensure that they are replicated.
    '''
    sequence_sharding: Incomplete
    use_local_output: Incomplete
    def __init__(self, *, sequence_dim: int = 1, use_local_output: bool = False) -> None: ...
    def _replicate_module_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh): ...
    @staticmethod
    def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh): ...
    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh): ...
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...
    def __repr__(self) -> str: ...

class PrepareModuleInput(ParallelStyle):
    '''
    Configure the nn.Module\'s inputs to convert the input tensors of the nn.Module to DTensors at runtime according to
    ``input_layouts``, and perform layout redistribution according to the ``desired_input_layouts``.

    Keyword Args:
        input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The DTensor layouts of input tensors for the nn.Module, this is used to convert the input tensors to
            DTensors. If some inputs are not torch.Tensor or no need to convert to DTensors, ``None`` need to be specified
            as a placeholder. default: None.
        desired_input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The desired DTensor layout of input tensors for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. This argument needs to have the same length with ``input_layouts``. default: None.
        input_kwarg_layouts (Dict[str, Placement]):
            The DTensor layouts of input kwargs for the nn.Module, this is used to convert the input kwarg tensors to DTensors.
            default: None
        desired_input_kwarg_layouts: (Dict[str, Placement]):
            The desired DTensor layout of input kwargs for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. default: None.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module inputs, default: False.
    Returns:
        A :class:`ParallelStyle` object that prepares the sharding layouts of the nn.Module\'s inputs.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleInput
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Module that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # According to the style specified below, the first input of attn will be annotated to Sharded DTensor
        >>> # and then redistributed to Replicated DTensor.
        >>> parallelize_module(
        >>>     block, # this can be a submodule or module
        >>>     tp_mesh,
        >>>     parallelize_plan={
        >>>         "attn": PrepareModuleInput(
        >>>             input_layouts=(Shard(0), None, None, ...),
        >>>             desired_input_layouts=(Replicate(), None, None, ...)
        >>>         ),
        >>>     }
        >>> )
    '''
    input_layouts: Incomplete
    desired_input_layouts: Incomplete
    use_local_output: Incomplete
    with_kwargs: Incomplete
    input_kwarg_layouts: Incomplete
    desired_input_kwarg_layouts: Incomplete
    def __init__(self, *, input_layouts: Placement | tuple[Placement | None] | None = None, desired_input_layouts: Placement | tuple[Placement | None] | None = None, input_kwarg_layouts: dict[str, Placement] | None = None, desired_input_kwarg_layouts: dict[str, Placement] | None = None, use_local_output: bool = False) -> None: ...
    def _prepare_input_arg(self, input: Any, mesh: DeviceMesh, input_layout: Placement | None, desired_layout: Placement | None): ...
    def _prepare_input_fn(self, inputs, device_mesh): ...
    def _prepare_input_kwarg_fn(self, inputs, kwarg_inputs, device_mesh): ...
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...
    def __repr__(self) -> str: ...

class PrepareModuleOutput(ParallelStyle):
    '''
    Configure the nn.Module\'s outputs to convert the output tensors of the nn.Module to DTensors at runtime according to
    ``output_layouts``, and perform layout redistribution according to the ``desired_output_layouts``.

    Keyword Args:
        output_layouts (Union[Placement, Tuple[Placement]]):
            The DTensor layouts of output tensors for the nn.Module, this is used to convert the output tensors to
            DTensors if they are :class:`torch.Tensor`. If some outputs are not torch.Tensor or no need to convert to DTensors,
            ``None`` need to be specified as a placeholder.
        desired_output_layouts (Union[Placement, Tuple[Placement]]):
            The desired DTensor layouts of output tensors for the nn.Module, this is used to ensure the outputs of the nn.Module
            have the desired DTensor layouts.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module outputs, default: True.
    Returns:
        A ParallelStyle object that prepares the sharding layouts of the nn.Module\'s outputs.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleOutput
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Module that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # According to the style specified below, the output of the TransformerBlock will be converted to Replicated DTensor
        >>> # and then redistributed to Sharded DTensor.
        >>> parallelize_module(
        >>>     block, # this can be a submodule or module
        >>>     tp_mesh,
        >>>     parallelize_plan = PrepareModuleOutput(
        >>>         output_layouts=Replicate(),
        >>>         desired_output_layouts=Shard(0)
        >>>     )
        >>> )
    '''
    output_layouts: Incomplete
    desired_output_layouts: Incomplete
    use_local_output: Incomplete
    def __init__(self, *, output_layouts: Placement | tuple[Placement], desired_output_layouts: Placement | tuple[Placement], use_local_output: bool = True) -> None: ...
    def _prepare_out_fn(self, outputs, device_mesh): ...
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...
    def __repr__(self) -> str: ...

class PrepareModuleInputOutput(ParallelStyle):
    '''
    Configure the nn.Module\'s inputs (and outputs) to convert the input tensors (and output tensors, respectively) of the nn.Module
    to DTensors at runtime according to ``input_layouts`` (and output_layouts, respectively), and perform layout redistribution
    according to the ``desired_input_layouts`` (and ``desired_output_layouts``, respectively). This is a combination of
    :class:`PrepareModuleInput` and :class:`PrepareModuleOutput`.

    Keyword Args:
        input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The DTensor layouts of input tensors for the nn.Module, this is used to convert the input tensors to
            DTensors. If some inputs are not torch.Tensor or no need to convert to DTensors, ``None`` need to be specified
            as a placeholder. default: None.
        desired_input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The desired DTensor layout of input tensors for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. This argument needs to have the same length with ``input_layouts``. default: None.
        input_kwarg_layouts (Dict[str, Placement]):
            The DTensor layouts of input kwargs for the nn.Module, this is used to convert the input kwarg tensors to DTensors.
            default: None
        desired_input_kwarg_layouts: (Dict[str, Placement]):
            The desired DTensor layout of input kwargs for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. default: None.
        use_local_input (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module inputs, default: False.
        output_layouts (Union[Placement, Tuple[Placement]]):
            The DTensor layouts of output tensors for the nn.Module, this is used to convert the output tensors to
            DTensors if they are :class:`torch.Tensor`. If some outputs are not torch.Tensor or no need to convert to DTensors,
            ``None`` need to be specified as a placeholder.
        desired_output_layouts (Union[Placement, Tuple[Placement]]):
            The desired DTensor layouts of output tensors for the nn.Module, this is used to ensure the outputs of the nn.Module
            have the desired DTensor layouts.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module outputs, default: True.
    Returns:
        A :class:`ParallelStyle` object that prepares the sharding layouts of the nn.Module\'s inputs and outputs.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleInputOutput
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Module that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # According to the style specified below, the first input of attn will be annotated as Sharded DTensor
        >>> # and then redistributed to Replicated DTensor, and the output of the TransformerBlock will be annotated
        >>> # as Replicated DTensor and then redistributed to Sharded DTensor.
        >>> parallelize_module(
        >>>     block, # this can be a submodule or module
        >>>     tp_mesh,
        >>>     parallelize_plan={
        >>>         "attn": PrepareModuleInputOutput(
        >>>             input_layouts=(Shard(0), None, None, ...),
        >>>             desired_input_layouts=(Replicate(), None, None, ...),
        >>>             output_layouts=Replicate(),
        >>>             desired_output_layouts=Shard(0),
        >>>         ),
        >>>     }
        >>> )
    '''
    prepare_module_input: Incomplete
    prepare_module_output: Incomplete
    def __init__(self, *, input_layouts: Placement | tuple[Placement | None] | None = None, desired_input_layouts: Placement | tuple[Placement | None] | None = None, input_kwarg_layouts: dict[str, Placement] | None = None, desired_input_kwarg_layouts: dict[str, Placement] | None = None, use_local_input: bool = False, output_layouts: Placement | tuple[Placement], desired_output_layouts: Placement | tuple[Placement], use_local_output: bool = True) -> None: ...
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...
    def __repr__(self) -> str: ...
