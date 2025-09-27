import abc
import torch
from .microbatch import TensorChunkSpec
from .stage import _PipelineStageBase
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from enum import Enum
from torch.nn.modules.loss import _Loss
from typing import Any, Callable, NamedTuple

__all__ = ['get_schedule_class', 'PipelineScheduleSingle', 'PipelineScheduleMulti', 'Schedule1F1B', 'ScheduleGPipe', 'ScheduleInterleaved1F1B', 'ScheduleLoopedBFS', 'ScheduleInterleavedZeroBubble', 'ScheduleZBVZeroBubble']

class _ComputationType(Enum):
    FORWARD = 1
    BACKWARD_INPUT = 2
    BACKWARD_WEIGHT = 3
    UNSHARD = 4
    RESHARD = 5
    SEND_F = 6
    RECV_F = 7
    SEND_B = 8
    RECV_B = 9
    FULL_BACKWARD = 10
    def __str__(self) -> str: ...
    @staticmethod
    def from_str(action): ...
F = FORWARD
I = BACKWARD_INPUT
W = BACKWARD_WEIGHT
B = FULL_BACKWARD

class _Action(NamedTuple):
    stage_index: int
    computation_type: _ComputationType
    microbatch_index: int | None = ...
    def __repr__(self) -> str: ...
    @staticmethod
    def from_str(action_string: str):
        """
        Reverse of __repr__

        String should be formatted as [stage][action type][(microbatch)]
            e.g. `2F0`, `1UNSHARD`, `3SEND_F1`
        """

class _PipelineSchedule(ABC, metaclass=abc.ABCMeta):
    _n_microbatches: Incomplete
    _loss_fn: Incomplete
    scale_grads: Incomplete
    _args_chunk_spec: Incomplete
    _kwargs_chunk_spec: Incomplete
    _output_merge_spec: Incomplete
    _has_backward: Incomplete
    _internal_losses: list[torch.Tensor]
    def __init__(self, n_microbatches: int, loss_fn: Callable[..., torch.Tensor] | None = None, args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None, kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None, output_merge_spec: dict[str, Any] | tuple[Any] | None = None, scale_grads: bool = True) -> None: ...
    def _maybe_compute_loss(self, stage, output, target_mbs, mb_index) -> None: ...
    def _maybe_get_loss(self, stage, mb_index): ...
    def _update_losses(self, stages, losses) -> None:
        """
        Update the losses to those in the internal state
        """
    @abstractmethod
    def _step_microbatches(self, arg_mbs: list | None = None, kwarg_mbs: list | None = None, target_mbs: list | None = None, losses: list | None = None):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the schedule
        implementation.

        Args:
            microbatches: list of microbatch args.
        """
    @abstractmethod
    def step(self, *args, target=None, losses: list | None = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """
    def _check_inputs(self, arg_mbs: list | None = None, kwarg_mbs: list | None = None, target_mbs: list | None = None, losses: list | None = None):
        """
        Pre-process/check inputs
        """
    def _compute_loss(self, output, target): ...
    def _split_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None):
        """
        Splits a full-batch input into chunks (i.e. microbatches) and returns
        the chunks
        """
    def _merge_outputs(self, output_chunks: list[Any]) -> Any:
        """
        Merge output chunks back to a batch state.
        If output_merge_spec is None, the utility will merge output chunks by dimension 0 (batch dim).
        """

class PipelineScheduleSingle(_PipelineSchedule, metaclass=abc.ABCMeta):
    """
    Base class for single-stage schedules.
    Implements the `step` method.
    Derived classes should implement `_step_microbatches`.

    Gradients are scaled by num_microbatches depending on the `scale_grads` argument, defaulting to True.  This setting
    should match the configuration of your loss_fn, which may either average losses (scale_grads=True)
    or sum losses (scale_grads=False).
    """
    _stage: Incomplete
    _num_stages: Incomplete
    _stage_initialized: bool
    pipeline_order: dict[int, list[_Action | None]] | None
    def __init__(self, stage: _PipelineStageBase, n_microbatches: int, loss_fn: Callable | None = None, args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None, kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None, output_merge_spec: dict[str, Any] | tuple[Any] | None = None, scale_grads: bool = True) -> None: ...
    def _initialize_stage(self, args, kwargs) -> None: ...
    def step(self, *args, target=None, losses: list | None = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """
    def _get_pipeline_order(self) -> dict[int, list[_Action | None]] | None:
        """
        Returns the pipeline execution order as a schedule IR.

        The returned IR is a dictionary mapping rank IDs to lists of actions.
        Each action is either an _Action object representing computation to perform,
        or None representing a deliberate idle step.

        The None values are used to represent pipeline bubbles where a rank
        must wait for dependencies from other ranks before proceeding. However
        during execution, with  the _PipelineScheduleRuntime, these Nones are
        skipped since the relevant communication (send/recv) will be scheduled and waited on.

        Returns:
            A dictionary mapping rank -> list of actions
        """

class _ScheduleForwardOnly(PipelineScheduleSingle):
    """
    The forward-only schedule.
    Will go through all the microbatches and perform only the forward pass
    """
    def _step_microbatches(self, arg_mbs: list | None = None, kwarg_mbs: list | None = None, target_mbs: list | None = None, losses: list | None = None):
        """
        Run one iteration of the pipeline schedule
        """

class ScheduleGPipe(PipelineScheduleSingle):
    """
    The GPipe schedule.
    Will go through all the microbatches in a fill-drain manner.
    """
    def _step_microbatches(self, arg_mbs: list | None = None, kwarg_mbs: list | None = None, target_mbs: list | None = None, losses: list | None = None):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the GPipe schedule.

        Args:
            microbatches: list of microbatch args.
        """
    def _get_pipeline_order(self) -> dict[int, list[_Action | None]] | None:
        """
        Returns the pipeline order for GPipe schedule.

        See base method in PipelineScheduleSingle for details on the schedule IR format.
        """

class Schedule1F1B(PipelineScheduleSingle):
    """
    The 1F1B schedule.
    Will perform one forward and one backward on the microbatches in steady state.
    """
    def _step_microbatches(self, arg_mbs: list | None = None, kwarg_mbs: list | None = None, target_mbs: list | None = None, losses: list | None = None):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the 1F1B schedule.

        Args:
            microbatches: list of microbatch args.
        """
    def _get_pipeline_order(self) -> dict[int, list[_Action | None]] | None:
        """
        Returns the pipeline order for 1F1B schedule.

        See base method in PipelineScheduleSingle for details on the schedule IR format.
        """

class PipelineScheduleMulti(_PipelineSchedule):
    """
    Base class for multi-stage schedules.
    Implements the `step` method.

    Gradients are scaled by num_microbatches depending on the `scale_grads` argument, defaulting to True.  This setting
    should match the configuration of your loss_fn, which may either average losses (scale_grads=True)
    or sum losses (scale_grads=False).
    """
    _stages: Incomplete
    _num_stages: Incomplete
    pp_group_size: Incomplete
    rank: Incomplete
    stage_index_to_group_rank: Incomplete
    _stages_initialized: bool
    _should_compute_loss: Incomplete
    pipeline_order: dict[int, list[_Action | None]]
    def __init__(self, stages: list[_PipelineStageBase], n_microbatches: int, loss_fn: Callable | None = None, args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None, kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None, output_merge_spec: dict[str, Any] | tuple[Any] | None = None, use_full_backward: bool | None = None, scale_grads: bool = True) -> None: ...
    def _initialize_stages(self, args: tuple[Any, ...], kwargs): ...
    def _validate_and_set_stage_mapping(self, actions: dict[int, list[_Action | None]]) -> None:
        """
        Allocates the stage index to rank mapping which is needed for communication
        """
    def _dump_csv(self, filename) -> None:
        """Dump a CSV representation of the schedule into a file with the provided filename."""
    def _load_csv(self, filename, format: str = 'compute_only') -> None:
        '''Load a CSV representation of the schedule from a file with the provided filename.
        This API will most likely get renamed/refactored so is marked as internal for now.

        format must be "compute_only" for PipelineScheduleMulti.
        '''
    def step(self, *args, target=None, losses: list | None = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """
    def _step_microbatches(self, arg_mbs: list | None = None, kwarg_mbs: list | None = None, target_mbs: list | None = None, losses: list | None = None):
        """
        Operate on the microbatches for looped schedules (multiple stages on each rank).

        TODO: Does not use sorted_batch_isend_irecv(). As a result, this schedule does
        not support models with skip connections.
        """

class _PipelineScheduleRuntime(PipelineScheduleMulti):
    """
    Provides a simple runtime that requires a 'schedule IR' including specified communication operations.

    Can be instantiated directly by creating _PipelineScheduleRuntime and calling load_csv, or can be
    subclassed and the subclass can be responsible for creating a schedule IR.
    """
    pipeline_order_with_comms: dict[int, list[_Action]]
    def _load_actions(self, actions: dict[int, list[_Action | None]], format: str = 'compute_only'):
        """
        Given an in-memory representation for a simple compute-only schedule, lower it to a complex schedule including
        communication actions.  Stores the schedule in self, and must be called before running step_mo()
        """
    def _load_csv(self, filename: str, format: str = 'compute_only'):
        '''Loads a csv in simple format and then lowers it to include communication actions

        format must be either "compute_only" or "compute_comms".  If compute_only, the lowering passes
        will automatically be run to generate a compute_comms schedule.
        '''
    def _dump_csv(self, filename: str):
        """Dump a CSV representation of the compute + comms schedule into a file with the provided filename."""
    def _simulate(self): ...
    def _step_microbatches(self, arg_mbs: list | None = None, kwarg_mbs: list | None = None, target_mbs: list | None = None, losses: list | None = None):
        """
        Operate on the microbatches for looped schedules (multiple stages on each rank).

        TODO: Does not use sorted_batch_isend_irecv(). As a result, this schedule does
        not support models with skip connections.
        """

class ScheduleLoopedBFS(PipelineScheduleMulti):
    """
    Breadth-First Pipeline Parallelism.
    See https://arxiv.org/abs/2211.05953 for details.
    Similar to Interleaved 1F1B, Looped BFS supports multiple stages per rank.
    What is different is that when microbatches are ready for multiple local
    stages, Loops BFS will prioritizes the earlier stage, running all available
    microbatches at once.
    """
    pipeline_order: dict[int, list[_Action | None]]
    def __init__(self, stages: list[_PipelineStageBase], n_microbatches: int, loss_fn: Callable | _Loss | None = None, output_merge_spec: dict[str, Any] | tuple[Any] | None = None, scale_grads: bool = True) -> None: ...
    def _calculate_single_rank_operations(self, rank): ...

class ScheduleInterleaved1F1B(PipelineScheduleMulti):
    '''
    The Interleaved 1F1B schedule.
    See https://arxiv.org/pdf/2104.04473 for details.
    Will perform one forward and one backward on the microbatches in steady
    state and supports multiple stages per rank. When microbatches are ready for
    multiple local stages, Interleaved 1F1B prioritizes the earlier microbatch
    (also called "depth first").

    This schedule is mostly similar to the original paper.
    It differs by being relaxing the requirement of num_microbatch % pp_size == 0.
    Using the flex_pp schedule, we will have num_rounds = max(1, n_microbatches // pp_group_size) and
    it works as long as n_microbatches % num_rounds is 0. As a few examples, support

    1. pp_group_size = 4, n_microbatches = 10. We will have num_rounds = 2 and n_microbatches % 2 is 0.
    2. pp_group_size = 4, n_microbatches = 3. We will have num_rounds = 1 and n_microbatches % 1 is 0.
    '''
    pp_group_size: Incomplete
    n_local_stages: Incomplete
    rank: Incomplete
    number_of_rounds: Incomplete
    microbatches_per_round: Incomplete
    pipeline_order: dict[int, list[_Action | None]]
    def __init__(self, stages: list[_PipelineStageBase], n_microbatches: int, loss_fn: Callable | None = None, args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None, kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None, output_merge_spec: dict[str, Any] | tuple[Any] | None = None, scale_grads: bool = True) -> None: ...
    def _calculate_single_rank_operations(self, rank) -> list[_Action | None]: ...

class ScheduleInterleavedZeroBubble(PipelineScheduleMulti):
    """
    The Interleaved Zero Bubble schedule.
    See https://arxiv.org/pdf/2401.10241 for details.
    Will perform one forward and one backward on inputs for the microbatches in steady
    state and supports multiple stages per rank. Uses the backward for weights to fill in
    the pipeline bubble.

    In particular this is implementing the ZB1P schedule in the paper.
    """
    pp_group_size: Incomplete
    n_local_stages: Incomplete
    rank: Incomplete
    number_of_rounds: Incomplete
    microbatches_per_round: Incomplete
    pipeline_order: dict[int, list[_Action | None]]
    def __init__(self, stages: list[_PipelineStageBase], n_microbatches: int, loss_fn: Callable | None = None, args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None, kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None, output_merge_spec: dict[str, Any] | tuple[Any] | None = None, scale_grads: bool = True) -> None: ...
    def _calculate_single_rank_operations(self, rank) -> list[_Action | None]: ...
    def _add_bubbles_to_actions(self, num_stages_global): ...

class ScheduleZBVZeroBubble(PipelineScheduleMulti):
    '''
    The Zero Bubble schedule (ZBV variant).
    See https://arxiv.org/pdf/2401.10241 Section 6 for details.

    This schedules requires exactly two stages per rank.

    This schedule will perform one forward and one backward on inputs for the microbatches in steady
    state and supports multiple stages per rank. Uses backward with respect to weights to fill in
    the pipeline bubble.

    This ZB-V schedule would have the "zero bubble" property only if time forward == time backward input == time backward weights.
    In practice, this is not likely true for real models so alternatively
    a greedy scheduler could be implemented for unequal/unbalanced time.
    '''
    pp_group_size: Incomplete
    stage_index_to_group_rank: Incomplete
    n_local_stages: Incomplete
    rank: Incomplete
    num_stages: Incomplete
    pipeline_order: dict[int, list[_Action | None]]
    def __init__(self, stages: list[_PipelineStageBase], n_microbatches: int, loss_fn: Callable | None = None, args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None, kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None, output_merge_spec: dict[str, Any] | tuple[Any] | None = None, scale_grads: bool = True) -> None: ...
    def _calculate_single_rank_operations(self, rank) -> list[_Action | None]: ...

def get_schedule_class(schedule_name: str):
    """
    Maps a schedule name (case insensitive) to its corresponding class object.

    Args:
        schedule_name (str): The name of the schedule.
    """
