import torch
import torch.distributed.tensor._dispatch as op_dispatch
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Sequence
from torch._export.wrappers import mark_subclass_constructor_exportable_experimental
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import Placement
from typing import Any, Callable

__all__ = ['DTensor', 'distribute_tensor', 'distribute_module', 'ones', 'empty', 'full', 'rand', 'randn', 'zeros']

class _ToTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: DTensor, grad_placements: Sequence[Placement] | None): ...
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor): ...

class _FromTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, device_mesh: DeviceMesh, placements: tuple[Placement, ...], run_check: bool, shape: torch.Size | None = None, stride: tuple[int, ...] | None = None) -> DTensor: ...
    @staticmethod
    def backward(ctx, grad_output: DTensor): ...

class DTensor(torch.Tensor):
    """
    ``DTensor`` (Distributed Tensor) is a subclass of ``torch.Tensor`` that provides single-device like
    abstraction to program with multi-device ``torch.Tensor``. It describes the distributed tensor sharding
    layout (DTensor Layout) through the :class:`DeviceMesh` and following types of :class:`Placement`:

    * :class:`Shard`: Tensor sharded on the tensor dimension ``dim`` on the devices of the ``DeviceMesh`` dimension
    * :class:`Replicate`: Tensor replicated on the devices of the ``DeviceMesh`` dimension
    * :class:`Partial`: Tensor is pending reduction on the devices of the ``DeviceMesh`` dimension

    When calling PyTorch operators, ``DTensor`` overrides the PyTorch operators to perform sharded computation and issue
    communications whenever necessary. Along with the operator computation, ``DTensor`` will transform or propagate the
    placements (DTensor Layout) properly (based on the operator semantic itself) and generate new ``DTensor`` outputs.

    To ensure numerical correctness of the ``DTensor`` sharded computation when calling PyTorch operators, ``DTensor``
    requires every Tensor argument of the operator be DTensor.

    .. note:: Directly using the Tensor subclass constructor here is not the recommended way to create a ``DTensor``
        (i.e. it does not handle autograd correctly hence is not the public API). Please refer to the `create_dtensor`_
        section to see how to create a ``DTensor``.
    """
    _local_tensor: torch.Tensor
    _spec: DTensorSpec
    __slots__: Incomplete
    _op_dispatcher: op_dispatch.OpDispatcher
    @staticmethod
    @torch._disable_dynamo
    def __new__(cls, local_tensor: torch.Tensor, spec: DTensorSpec, *, requires_grad: bool) -> DTensor:
        '''
        Construct a DTensor from a local tensor, device mesh, and placement and
        other tensor properties (i.e. shape, requires_grad, strides, etc).

        .. note:: This is not a public API and it\'s only supposed to be used by the
            operator implementations and internals. If you want to construct a
            DTensor from a local tensor, consider using ``DTensor.from_local``, if
            you want to construct a DTensor from a "global" tensor (where you
            already have tensor initialized and want to shard this tensor),
            consider using ``distribute_tensor``.
        '''
    @torch._disable_dynamo
    @mark_subclass_constructor_exportable_experimental
    def __init__(self, *args, **kwargs) -> None: ...
    def __repr__(self) -> str: ...
    def __tensor_flatten__(self):
        """
        protocol to inform how to flatten a DTensor to local tensor
        for PT2 tracing
        """
    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride): ...
    def __coerce_tangent_metadata__(self): ...
    def __coerce_same_metadata_as_tangent__(self, flatten_spec, expected_type=None): ...
    @classmethod
    @torch._disable_dynamo
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None): ...
    @staticmethod
    def from_local(local_tensor: torch.Tensor, device_mesh: DeviceMesh | None = None, placements: Sequence[Placement] | None = None, *, run_check: bool = False, shape: torch.Size | None = None, stride: tuple[int, ...] | None = None) -> DTensor:
        """
        Create a :class:`DTensor` from a local torch.Tensor on each rank
        according to the ``device_mesh`` and ``placements`` specified.

        Args:
            local_tensor (torch.Tensor): local torch.Tensor on each rank.
            device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
                tensor, if not specified, must be called under a DeviceMesh
                context manager, default: None
            placements (List[:class:`Placement`], optional): the placements that
                describes how to place the local torch.Tensor on DeviceMesh, must
                have the same number of elements as ``device_mesh.ndim``.

        Keyword args:
            run_check (bool, optional): at a cost of extra communications, perform
                sanity check across ranks to check each local tensor's meta information
                to ensure correctness. If have :class:`Replicate` in ``placements``, the
                data on first rank of the device mesh dimension will be broadcasted
                to other ranks. default: False
            shape (torch.Size, optional): A List of int which specifies the size of
                DTensor which build on top of `local_tensor`. Note this needs to be
                provided if the shape of ``local_tensor`` are different across the ranks.
                If not provided, ``shape`` will be computed assuming the given distributed
                tensor is evenly sharded across ranks. default: None
            stride (tuple, optional): A List of int which specifies the stride of DTensor.
                If not provided, ``stride`` will be computed assuming the given distributed
                tensor is evenly sharded across ranks. default: None

        Returns:
            A :class:`DTensor` object

        .. note:: When ``run_check=False``, it is the user's responsibility to ensure the
            local tensor passed in is correct across ranks (i.e. the tensor is sharded for
            the ``Shard(dim)`` placement or replicated for the ``Replicate()`` placement).
            If not, the behavior of the created DTensor is undefined.

        .. note:: ``from_local`` is differentiable, the `requires_grad` of the created
            `DTensor` object will depend on if `local_tensor` requires_grad or not.
        """
    def to_local(self, *, grad_placements: Sequence[Placement] | None = None) -> torch.Tensor:
        """
        Get the local tensor of this DTensor on its current rank. For sharding it returns
        a local shard of the logical tensor view, for replication it returns the replica on
        its current rank.

        Keyword args:
            grad_placements (List[:class:`Placement`], optional): the placements describes
                the future layout of any gradient layout of the Tensor returned from this
                function.
                `to_local` converts DTensor to local tensor and the returned local tensor
                might not be used as the original DTensor layout later in the code. This
                argument is the hint that user can give to autograd in case the gradient
                layout of the returned tensor does not match the original DTensor layout.
                If not specified, we will assume the gradient layout remains the same
                as the original DTensor and use that for gradient computation.

        Returns:
            A :class:`torch.Tensor` or ``AsyncCollectiveTensor`` object. it represents the
            local tensor on its current rank. When an ``AsyncCollectiveTensor`` object is returned,
            it means the local tensor is not ready yet (i.e. communication is not finished). In this
            case, user needs to call ``wait`` to wait the local tensor to be ready.

        .. note:: ``to_local`` is differentiable, the ``requires_grad`` of the local tensor returned
            will depend on if the `DTensor` requires_grad or not.
        """
    def redistribute(self, device_mesh: DeviceMesh | None = None, placements: Sequence[Placement] | None = None, *, async_op: bool = False, forward_dtype: torch.dtype | None = None, backward_dtype: torch.dtype | None = None) -> DTensor:
        """
        ``redistribute`` performs necessary collective operations that redistribute the current
        DTensor from its current placements to a new placements, or from its current DeviceMesh
        to a new DeviceMesh. i.e. we can turn a Sharded DTensor to a Replicated DTensor by
        specifying a Replicate placement for each dimension of the DeviceMesh.

        When redistributing from current to the new placements on one device mesh dimension, we
        will perform the following operations including communication collective or local operation:

        1. ``Shard(dim)`` -> ``Replicate()``: ``all_gather``
        2. ``Shard(src_dim)`` -> ``Shard(dst_dim)``: ``all_to_all``
        3. ``Replicate()`` -> ``Shard(dim)``: local chunking (i.e. ``torch.chunk``)
        4. ``Partial()`` -> ``Replicate()``: ``all_reduce``
        5. ``Partial()`` -> ``Shard(dim)``: ``reduce_scatter``


        ``redistribute`` would correctly figure out the necessary redistribute steps for DTensors
        that are created either on 1-D or N-D DeviceMesh.

        Args:
            device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
                DTensor. If not specified, it would use the current DTensor's DeviceMesh.
                default: None
            placements (List[:class:`Placement`], optional): the new placements that
                describes how to place the DTensor into the DeviceMesh, must
                have the same number of elements as ``device_mesh.ndim``.
                default: replicate on all mesh dimensions

        Keyword args:
            async_op (bool, optional): whether to perform the DTensor redistribute operation
                asynchronously or not. Default: False
            forward_dtype (torch.dtype, optional): the local tensor datatype can be converted to
                ``forward_dtype`` before redistributing the local tensor in its forward.
                The result DTensor will be in ``forward_dtype`` Default: None.
            backward_dtype (torch.dtype, optional): the local tensor datatype can be converted to
                ``backward_dtype`` before redistributing the local tensor in its backward.
                The result DTensor gradient would be converted back to the current DTensor dtype. Default: None

        Returns:
            A :class:`DTensor` object

        .. note:: ``redistribute`` is differentiable, which means user do not need to worry about
            the backward formula of the redistribute operation.

        .. note:: ``redistribute`` currently only supports redistributing DTensor on the same DeviceMesh,
            Please file an issue if you need to redistribute DTensor to different DeviceMesh.
        """
    def full_tensor(self, *, grad_placements: Sequence[Placement] | None = None) -> torch.Tensor:
        """
        Return the full tensor of this DTensor. It will perform necessary collectives
        to gather the local tensors from other ranks in its DeviceMesh and concatenate
        them together. It's a syntatic sugar of the following code:

        ``dtensor.redistribute(placements=[Replicate()] * mesh.ndim).to_local()``

        Keyword args:
            grad_placements (List[:class:`Placement`], optional): the placements describes
                the future layout of any gradient layout of the full Tensor returned from this
                function.
                `full_tensor` converts DTensor to a full torch.Tensor and the returned torch.tensor
                might not be used as the original replicated DTensor layout later in the code. This
                argument is the hint that user can give to autograd in case the gradient
                layout of the returned tensor does not match the original replicated DTensor layout.
                If not specified, we will assume the gradient layout of the full tensor be replicated.

        Returns:
            A :class:`torch.Tensor` object that represents the full tensor of this DTensor.

        .. note:: ``full_tensor`` is differentiable.
        """
    @property
    def device_mesh(self) -> DeviceMesh:
        """
        The :class:`DeviceMesh` attribute that associates with this DTensor object.

        .. note:: ``device_mesh`` is a read-only property, it can not be set.
        """
    @property
    def placements(self) -> tuple[Placement, ...]:
        """
        The placements attribute of this DTensor that describes the layout of this
        DTensor on the its DeviceMesh.

        .. note:: ``placements`` is a read-only property, it can not be set.
        """
    def __create_write_items__(self, fqn: str, object: Any): ...
    def __create_chunk_list__(self):
        """
        Return a list of ChunkStorageMetadata, which is a dataclass that describes the size/offset of the local shard/replica
        on current rank. For DTensor, each rank will have a single local shard/replica, so the returned list usually only
        has one element.

        This dunder method is primariy used for distributed checkpoint purpose.

        Returns:
            A List[:class:`ChunkStorageMetadata`] object that represents the shard size/offset on the current rank.
        """
    def __get_tensor_shard__(self, index): ...

def distribute_tensor(tensor: torch.Tensor, device_mesh: DeviceMesh | None = None, placements: Sequence[Placement] | None = None, *, src_data_rank: int | None = 0) -> DTensor:
    '''
    Distribute a leaf ``torch.Tensor`` (i.e. nn.Parameter/buffers) to the ``device_mesh`` according
    to the ``placements`` specified. The rank of ``device_mesh`` and ``placements`` must be the
    same. The ``tensor`` to distribute is the logical or "global" tensor, and the API would use
    the ``tensor`` from first rank of the DeviceMesh dimension as the source of truth to preserve
    the single-device semantic. If you want to construct a DTensor in the middle of the Autograd
    computation, please use :meth:`DTensor.from_local` instead.

    Args:
        tensor (torch.Tensor): torch.Tensor to be distributed. Note that if you
            want to shard a tensor on a dimension that is not evenly divisible by
            the number of devices in that mesh dimension, we use ``torch.chunk``
            semantic to shard the tensor and scatter the shards. The uneven sharding
            behavior is experimental and subject to change.
        device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to distribute the
            tensor, if not specified, must be called under a DeviceMesh context
            manager, default: None
        placements (List[:class:`Placement`], optional): the placements that
            describes how to place the tensor on DeviceMesh, must have the same
            number of elements as ``device_mesh.ndim``. If not specified, we will
            by default replicate the tensor across the ``device_mesh`` from the
            first rank of each dimension of the `device_mesh`.

    Keyword args:
        src_data_rank (int, optional): the rank of the source data for the logical/global tensor, it is
            used by :meth:`distribute_tensor` to scatter/broadcast the shards/replicas to other ranks.
            By default, we use ``group_rank=0`` on each DeviceMesh dimension as the source data to preserve
            the single-device semantic. If passing ``None`` explicitly, :meth:`distribute_tensor` simply uses
            its local data instead of trying to preserve the single-device semantic via scatter/broadcast.
            Default: 0

    Returns:
        A :class:`DTensor` or ``XLAShardedTensor`` object.

    .. note::
        When initialize the DeviceMesh with the ``xla`` device_type, ``distribute_tensor``
        return `XLAShardedTensor` instead. see `this issue <https://github.com/pytorch/pytorch/issues/92909>`__
        for more details. The XLA integration is experimental and subject to change.
    '''
def distribute_module(module: nn.Module, device_mesh: DeviceMesh | None = None, partition_fn: Callable[[str, nn.Module, DeviceMesh], None] | None = None, input_fn: Callable[[nn.Module, Any, DeviceMesh], None] | None = None, output_fn: Callable[[nn.Module, Any, DeviceMesh], None] | None = None) -> nn.Module:
    """
    This function expose three functions to control the parameters/inputs/outputs of the module:

    1. To perform sharding on the module before runtime execution by specifying the
    ``partition_fn`` (i.e. allow user to convert Module parameters to :class:`DTensor`
    parameters according to the `partition_fn` specified).
    2. To control the inputs or outputs of the module during runtime execution by
    specifying the ``input_fn`` and ``output_fn``. (i.e. convert the input to
    :class:`DTensor`, convert the output back to ``torch.Tensor``)

    Args:
        module (:class:`nn.Module`): user module to be partitioned.
        device_mesh (:class:`DeviceMesh`): the device mesh to place the module.
        partition_fn (Callable): the function to partition parameters (i.e. shard certain
            parameters across the ``device_mesh``). If ``partition_fn`` is not specified,
            by default we replicate all module parameters of ``module`` across the mesh.
        input_fn (Callable): specify the input distribution, i.e. could control how the
            input of the module is sharded. ``input_fn`` will be installed as a module
            ``forward_pre_hook`` (pre forward hook).
        output_fn (Callable): specify the output distribution, i.e. could control how the
            output is sharded, or convert it back to torch.Tensor. ``output_fn`` will be
            installed as a module ``forward_hook`` (post forward hook).

    Returns:
        A module that contains parameters/buffers that are all ``DTensor`` s.

    .. note::
        When initialize the DeviceMesh with the ``xla`` device_type, ``distribute_module``
        return nn.Module with PyTorch/XLA SPMD annotated parameters. See
        `this issue <https://github.com/pytorch/pytorch/issues/92909>`__
        for more details. The XLA integration is experimental and subject to change.

    """
def ones(*size, dtype: torch.dtype | None = None, layout: torch.layout = ..., requires_grad: bool = False, device_mesh: DeviceMesh | None = None, placements: Sequence[Placement] | None = None) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 1, with the shape defined
    by the variable argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
def empty(*size, dtype: torch.dtype | None = None, layout: torch.layout = ..., requires_grad: bool = False, device_mesh: DeviceMesh | None = None, placements: Sequence[Placement] | None = None) -> DTensor:
    """
    Returns a :class:`DTensor` filled with uninitialized data. The shape of the :class:`DTensor`
    is defined by the variable argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: empty(1,2,3..) or empty([1,2,3..]) or empty((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).        layout (:class:`torch.layout`, optional): the desired layout of returned :class:`DTensor`.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
def full(size, fill_value, *, dtype: torch.dtype | None = None, layout: torch.layout = ..., requires_grad: bool = False, device_mesh: DeviceMesh | None = None, placements: Sequence[Placement] | None = None) -> DTensor:
    """
    Returns a :class:`DTensor` filled with ``fill_value`` according to ``device_mesh`` and
    ``placements``, with the shape defined by the argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))
        fill_value(Scalar): the value to fill the output tensor with.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
def rand(*size, requires_grad: bool = False, dtype: torch.dtype | None = None, layout: torch.layout = ..., device_mesh: DeviceMesh | None = None, placements: Sequence[Placement] | None = None) -> DTensor:
    """
    Returns a :class:`DTensor` filled with random numbers from a uniform distribution
    on the interval ``[0, 1)``. The shape of the tensor is defined by the variable
    argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
def randn(*size, requires_grad: bool = False, dtype: torch.dtype | None = None, layout: torch.layout = ..., device_mesh: DeviceMesh | None = None, placements: Sequence[Placement] | None = None) -> DTensor:
    """
    Returns a :class:`DTensor` filled with random numbers from a normal distribution
    with mean 0 and variance 1. The shape of the tensor is defined by the variable
    argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
def zeros(*size, requires_grad: bool = False, dtype: torch.dtype | None = None, layout: torch.layout = ..., device_mesh: DeviceMesh | None = None, placements: Sequence[Placement] | None = None) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 0.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: zeros(1,2,3..) or zeros([1,2,3..]) or zeros((1,2,3..))
    Keyword args:
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned :class:`DTensor`.
            Default: ``torch.strided``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
