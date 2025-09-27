import abc
import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
from ._utils import PipeInfo
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from typing import Any, Callable

__all__ = ['PipelineStage', 'build_stage']

class _RootArgPlaceholder:
    """
    Placeholder for model-level inputs.
    """
    meta: Incomplete
    def __init__(self, tensor) -> None: ...

class _RecvInfo:
    """
    Represents a stage input.
    """
    input_name: Incomplete
    source: Incomplete
    buffer: Incomplete
    def __init__(self, input_name: str, source: int, buffer: torch.Tensor) -> None: ...
    def __repr__(self) -> str: ...
InputInfo = _RecvInfo | _RootArgPlaceholder

class _PipelineStageBase(ABC, metaclass=abc.ABCMeta):
    """
    Base class for pipeline stages.
    Defines or implements common methods used by the `_PipelineStage` used by
    the tracing frontend and `PipelineStage` used by manual frontend.
    """
    submod: Incomplete
    stage_index: Incomplete
    num_stages: Incomplete
    device: Incomplete
    group: Incomplete
    dw_builder: Incomplete
    backward_state: dict[int, tuple[Any, ...]]
    dw_runner: dict[int, Callable[..., None]]
    group_rank: Incomplete
    group_size: Incomplete
    _outputs_meta: tuple[torch.Tensor, ...] | None
    fwd_cache: dict[int, tuple[Any, list[torch.Tensor]]]
    bwd_cache: dict[int, tuple[torch.Tensor | None, ...]]
    output_chunks: list[Any]
    log_prefix: Incomplete
    args_recv_info: dict[int, tuple[InputInfo, ...]]
    act_send_info: dict[int, list]
    grad_recv_info: dict
    grad_send_info: list | None
    chunks: int | None
    stage_index_to_group_rank: dict[int, int]
    def __init__(self, submodule: torch.nn.Module, stage_index: int, num_stages: int, device: torch.device, group: dist.ProcessGroup | None = None, dw_builder: Callable[[], Callable[..., None]] | None = None) -> None:
        """
        Args:
            submodule (torch.nn.Module): The module to be executed in this stage.
            stage_index (int): The index of this stage.
            num_stages (int): The total number of stages in this pipeline.
            device (torch.device): The device to run this stage on.
            group (Optional[dist.ProcessGroup]): The process group to use for communication.
                If `None`, the default process group will be used.
                Default: `None`.
            dw_builder (Optional[Callable[[], Callable[..., None]]): If provided, dw_builder is a builder function
                that will build a new dw_runner function that will run parts of module backward that were intentionally
                skipped during the module's actual backward pass. The builder must be invoked by stage after stage runs
                model backwards, and stage should save the latest dw_runner to run during weight pas (W).
                If not provided, a dw_runner will be generated automatically by traversing the autograd graph.
                When used with schedules that only have F and B steps, the fresh dw_runner function will be called as
                part of I (input backwards). When used with F,I,W schedules, the dw_runner function implements 'W'.
        """
    @property
    def has_backward(self) -> bool:
        """
        Returns true if this stage has a backward pass.
        """
    _has_backward: Incomplete
    @has_backward.setter
    def has_backward(self, has_backward: bool): ...
    @property
    def is_first(self):
        """
        Returns true if this stage is the first stage in the pipeline.
        """
    @property
    def is_last(self):
        """
        Returns true if this stage is the last stage in the pipeline.
        """
    def _check_chunk_id(self, chunk_id: int): ...
    def _configure_outputs_meta(self, outputs_meta: tuple[torch.Tensor, ...]):
        """
        Track the output shapes/dtype of this stage since they determine the send operation(s) which must match
        recv operations of the next stage.  The next stage _will_ be freezing its recv buffers based on its initial
        configuration, so it's important to also freeze/validate the output side to avoid any send/recv mismatches
        which could show up as hangs, silent corruption, or other errors.
        """
    def get_outputs_meta(self) -> tuple[torch.Tensor, ...]:
        """Get the output metadata (meta tensors) reprensenting the outputs of this stage"""
    def _create_grad_send_info(self, args_recv_info: tuple) -> list[int | None]:
        """
        Create a list of stage indices to send gradients to.
        """
    @abstractmethod
    def _prepare_forward_infra(self, num_microbatches: int, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None) -> tuple[Any, ...]: ...
    def _prepare_backward_infra(self, num_microbatches: int): ...
    @abstractmethod
    def _create_grad_recv_info(self, act_send_info: dict) -> tuple[_RecvInfo, ...]: ...
    def _get_recv_ops(self, recv_infos: tuple[InputInfo, ...]) -> list[dist.P2POp]:
        """
        Helper function shared by `get_fwd_recv_ops` and `get_bwd_recv_ops`.
        Returns a list of ops that correspond to the recv infos.
        """
    def set_local_fwd_input(self, prev_stage_outputs: Any, mb_index: int) -> None:
        """
        Moves 'prev_stage_outputs' from another stage on the same rank into place as inputs for this stage. Avoids
        copying tensor data or using send/recv op.  Detaches original tensor and sets requires_grad so the
        tensor can serve as a leaf for autograd and gradients can be collected from it during backward.
        """
    def get_local_bwd_output(self, mb_index):
        """
        Returns the input grad tensors for this stage, which correspond to the stage inputs during forward.
        """
    def set_local_bwd_input(self, next_stage_bwd_outputs: tuple[torch.Tensor | None, ...], mb_index: int) -> None:
        """
        Moves 'grad input' tensors from the next stage to 'grad_output' on this stage, avoiding a copy or send/recv.
        Does not detach or set '_requires_grad'.
        """
    def get_fwd_recv_ops(self, fwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the input arguments
        for this stage.
        """
    def get_bwd_recv_ops(self, bwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the gradients
        for this stage.
        """
    def get_fwd_send_ops(self, fwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Get the activation send ops for current stage's forward.
        """
    def get_bwd_send_ops(self, bwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Get the gradient send ops for current stage's backward.
        """
    def clear_runtime_states(self) -> None:
        """
        Clear runtime states of the stage.
        """
    def _map_tensor_from_recv_info(self, recv_infos: tuple[InputInfo, ...]):
        """
        Map tensors from recv infos to a list.
        """
    def _retrieve_recv_activations(self, fwd_chunk_id: int):
        """
        Retrieve the activations received for the current stage during forward.
        """
    def _retrieve_recv_grads(self, bwd_chunk_id: int):
        """
        Retrieve the gradients received for the current stage during backward.
        """
    def forward_maybe_with_nosync(self, *args, **kwargs): ...
    def scale_grads(self, grad_scale_factor: int) -> None:
        """Scale gradients model gradients by `grad_scale_factor`, which should be specified in coordination with the
        loss function used with pipelining.  For loss functions which perform 'mean' loss reduction, `grad_scale_factor`
        should be set to num_microbatches.  For loss functions that use `sum` reduction, `grad_scale_factor` should
        be set to 1.

        Should only be called once per pipeline schedule step, after all backwards passes have completed.
        """
    def backward_maybe_with_nosync(self, backward_type, bwd_kwargs: dict, last_backward: bool = False) -> tuple[tuple[torch.Tensor | None, ...], list[dict[str, Any]] | None]:
        """
        Whether using PP with FSDP or DDP, there are some runtime differences between the last backward step and the
        other steps.  Namely, we need to accumulate gradients on previous steps and reduce them on the last step, but
        there are additional state-variables and performance considerations depending on the data parallelism used.
        This helper should adapt any pipeline parallel schedule to work with common/supported data parallel libraries.
        """
    def forward_one_chunk(self, fwd_chunk_id: int, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None):
        """
        Perform forward pass on the stage with one microbatch.
        `args` and `kwargs` are the inputs from *external* to this stage.
        As of Sept 2024:
        - `args` applies to the first stage only, other stages receives args
          through activation transmission.
        - `kwargs` can be passed to all stages via respective `step` calls.
        """
    def backward_one_chunk(self, bwd_chunk_id: int, loss=None, full_backward: bool = True, last_backward: bool = False):
        """
        Perform backward pass on the module.
        This should only be called once per microbatch.

        If full_backward is True (the default), the full backward pass including weight and input gradients will be run,
        and it is an error to call `backward_weight_one_chunk` for this bwd_chunk_id.

        If full_backward is False, it is optional that `dw_runner` was provided to the PipelineStage at __init__ time,
        and a subsequent call to `backward_weight_one_chunk` is required to invoke dw_runner and complete the backward.

        last_backward is controlled by the schedule and signals synchronization of gradients across DP groups
        after the last backward.
        """
    def backward_weight_one_chunk(self, bwd_chunk_id: int, last_backward: bool = False): ...
    def _validate_fwd_input(self, args, kwargs) -> None:
        """Raises a RuntimeError if shapes of input args/kwargs do not match the shapes configured for this stage."""
    def _validate_fwd_outputs(self, outputs: tuple[torch.Tensor, ...]):
        """Raises a RuntimeError if this stage produces an output of unexpected shape/dtype.
        Most likely, this could be cause either by incorrect user specification of output shapes, or because
        shape inference was done on the original model but then at runtime the model is wrapped with something like
        mixed precision which changes output dtype.
        """

class _PipelineStage(_PipelineStageBase):
    pipe_info: Incomplete
    node: Incomplete
    name: Incomplete
    submod_to_stage_index: dict[str, int]
    def __init__(self, stage_module: torch.nn.Module, stage_index: int, pipe_info: PipeInfo, device: torch.device, group: dist.ProcessGroup | None = None) -> None:
        """
        Create a pipeline stage given a stage_module to be wrapped by this stage
        and a `pipe_info` describing the stage relationship of the pipeline.

        Args:
            stage_module (torch.nn.Module): the module to be wrapped by this stage
            stage_index (int): the index of this stage in the pipeline
            pipe_info (PipeInfo): information about the pipeline, can be retrieved by `pipe.info()`
            device (torch.device): the device to be used by this stage
            group (Optional[dist.ProcessGroup]): the process group to be used by this stage
        """
    def _move_submod_to_device(self) -> None: ...
    act_send_info: Incomplete
    def _prepare_forward_infra(self, num_microbatches: int, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None) -> tuple[Any, ...]:
        """
        Create send/recv infrastructures for activations (during forward)
        """
    def get_stage_index_of_submod(self, submod_name: str):
        """
        Given a submodule name, return the stage index of the submodule.
        """
    def _create_act_recv_info(self):
        """
        Create a tuple of `_RecvInfo` for inputs to the stage.
        """
    def find_dst_rank(self, user: fx.Node) -> int | None:
        """
        Find the destination rank of a `user` node.
        If the `user` is not a submod, `None` may be returned.
        """
    def _create_act_send_info(self):
        """
        Create a dict of send info for activations.
        The dict is of the form:
        {
            output_index: [dst_rank_0, dst_rank_1, ...],
            ...
        }
        where the list of `dst_rank`s covers the case where an output value may
        be consumed by multiple stages.
        """
    def _get_output_node(self): ...
    def _create_grad_recv_info(self, act_send_info: dict) -> tuple[_RecvInfo, ...]:
        """
        Create a tuple of `_RecvInfo` for gradients.
        """

def build_stage(stage_module: torch.nn.Module, stage_index: int, pipe_info: PipeInfo, device: torch.device, group: dist.ProcessGroup | None = None) -> _PipelineStage:
    """
    Create a pipeline stage given a stage_module to be wrapped by this stage
    and pipeline information.

    Args:
        stage_module (torch.nn.Module): the module to be wrapped by this stage
        stage_index (int): the index of this stage in the pipeline
        pipe_info (PipeInfo): information about the pipeline, can be retrieved by `pipe.info()`
        device (torch.device): the device to be used by this stage
        group (Optional[dist.ProcessGroup]): the process group to be used by this stage

    Returns:
        _PipelineStage: a pipeline stage that can run with `PipelineSchedules`.
    """

class PipelineStage(_PipelineStageBase):
    """
    A class representing a pipeline stage in a pipeline parallelism setup.

    PipelineStage assumes sequential partitioning of the model, i.e. the model is split into chunks where outputs from
    one chunk feed into inputs of the next chunk, with no skip connections.

    PipelineStage performs runtime shape/dtype inference automatically by propagating the outputs from stage0 to
    stage1 and so forth, in linear order.  To bypass shape inference, pass the `input_args` and `output_args` to each
    PipelineStage instance.

    Args:
        submodule (nn.Module): The PyTorch module wrapped by this stage.
        stage_index (int): The ID of this stage.
        num_stages (int): The total number of stages.
        device (torch.device): The device where this stage is located.
        input_args (Union[torch.Tensor, Tuple[torch.tensor]], optional): The input arguments for the submodule.
        output_args (Union[torch.Tensor, Tuple[torch.tensor]], optional): The output arguments for the submodule.
        group (dist.ProcessGroup, optional): The process group for distributed training. If None, default group.
        dw_builder (Optional[Callable[[], Callable[..., None]]): If provided, dw_builder will build a new dw_runner function
            that will the W action (input weights) for F, I, W (Fwd, Input, Weight) zero bubble schedules.
    """
    inputs: list[torch.Tensor] | None
    inputs_meta: tuple[torch.Tensor, ...] | None
    outputs_grad: list[torch.Tensor]
    def __init__(self, submodule: nn.Module, stage_index: int, num_stages: int, device: torch.device, input_args: torch.Tensor | tuple[torch.Tensor, ...] | None = None, output_args: torch.Tensor | tuple[torch.Tensor, ...] | None = None, group: dist.ProcessGroup | None = None, dw_builder: Callable[[], Callable[..., None]] | None = None) -> None: ...
    def _shape_inference(self, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None): ...
    act_send_info: dict[int, list]
    def _prepare_forward_infra(self, num_microbatches: int, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None) -> tuple[Any, ...]: ...
    def _create_grad_recv_info(self, act_send_info: dict) -> tuple[_RecvInfo, ...]: ...
