import contextlib
import torch
from _typeshed import Incomplete
from torch.distributed.device_mesh import DeviceMesh as DeviceMesh
from torch.fx.experimental.proxy_tensor import get_proxy_mode as get_proxy_mode
from torch.utils._pytree import tree_map_only as tree_map_only
from typing import Any

def is_torchdynamo_compiling():
    """Can't import torchdynamo in torchdeploy builds currently."""

RANK_TYPES: Incomplete

def wait_tensor(tensor):
    """
    Wait on a tensor returned by the collectives ops.

    Waiting follows device semantics, which means blocking on CPU and synchronizing streams on CUDA.
    """
def broadcast(self, src: int, group: RANK_TYPES, tag: str = ''):
    """
    Broadcasts the tensor to all processes in the given process group.

    Args:
        src (int): Source rank
        group (ProcessGroup or List[int]): The process group to work on.
        tag (str, optional): A unique identifier for the collective. Default: empty string
    """
def all_reduce(self, reduceOp: str, group: RANK_TYPES, tag: str = ''):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    The input tensor is left unmodified.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
def all_gather_tensor(self, gather_dim: int, group: RANK_TYPES, tag: str = '') -> torch.Tensor:
    """
    Gather tensor data across from all machines and concatenate over ``gather_dim``.

    Note that it currently only supports gather_dim = 0.

    The input tensor is left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
def all_gather_tensor_autograd(self, gather_dim: int, group: RANK_TYPES, tag: str = ''):
    """
    Gather tensor data across from all machines and concatenate over ``gather_dim``.

    Note that it currently only supports gather_dim = 0.

    This function is the same as all_gather_tensor but will propagate the
    backwards gradient across workers.

    See all_gather_tensor for more details on usage.
    """
def reduce_scatter_tensor(self, reduceOp: str, scatter_dim: int, group: RANK_TYPES, tag: str = ''):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result, then scatter the results to corresponding ranks.


    The input tensor is left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh
    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
def reduce_scatter_tensor_autograd(self, reduceOp: str, scatter_dim: int, group: RANK_TYPES, tag: str = ''):
    '''
    Reduces the tensor data across all machines in such a way that all get
    the final result, then scatter the results to corresponding ranks.

    This function is the same as reduce_scatter_tensor but will propagate the
    backwards gradient across workers.

    Currently only the "sum" reduceOp is supported.

    See reduce_scatter_tensor for more details on usage.
    '''
def all_reduce_coalesced(self, reduceOp: str, group: RANK_TYPES, tag: str = '') -> list[torch.Tensor]:
    """
    Reduces a list of tensors across all machines in such a way that all get
    the final result.

    The all tensors in the input list are left unmodified.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
def all_gather_into_tensor_coalesced(self, group: RANK_TYPES, tag: str = '') -> list[torch.Tensor]:
    """
    Gather a list of tensors across from all machines.

    Note that it currently only supports gather_dim = 0.

    The input tensor is left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
def reduce_scatter_tensor_coalesced(inputs: list[torch.Tensor], reduceOp: str, scatter_dim: list[int], group: RANK_TYPES, tag: str = '') -> list[torch.Tensor]:
    """
    Reduces a list of tensors across all machines in such a way that all get
    the final result, then scatter the results to corresponding ranks.

    The input tensors are left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
def _is_view_op(tgt): ...
def all_to_all_single(self, output_split_sizes: list[int] | None, input_split_sizes: list[int] | None, group: RANK_TYPES, tag: str = '') -> torch.Tensor:
    """
    Each process splits input tensor and then scatters the split list
    to all processes in a group. Then concatenate the received tensors from all
    the processes in the group and return single output tensor.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
def all_to_all_single_autograd(self, output_split_sizes: list[int] | None, input_split_sizes: list[int] | None, group: RANK_TYPES, tag: str = '') -> torch.Tensor:
    """
    Same as all_to_all_single but supports autograd.
    """
def permute_tensor(self, src_dst: list[int], group: RANK_TYPES, tag: str = '') -> torch.Tensor:
    """
    Permutes the elements of the tensor according to the given source/destination pairs. `src_dst` should
    be defined such that src_dst[m] == n means m sends to n.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one
    """

class AsyncCollectiveTensor(torch.Tensor):
    """
    A Tensor wrapper subclass that is used to trigger a call to wait
    prior to first use of the underlying tensor.
    Use it inside functional collective pytorch wrappers like the following:
    def functional_collective(self, group, tag):
        tag, rankset, group_size = _expand_group(group, tag)
        tensor = torch.ops.c10d_functional.{collective}(self, tag, rankset, group_size)
        return _maybe_wrap_tensor(tensor)
    """
    elem: torch.Tensor
    completed: bool
    __slots__: Incomplete
    @staticmethod
    def __new__(cls, elem: torch.Tensor): ...
    def __tensor_flatten__(self): ...
    def tolist(self): ...
    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride): ...
    def __coerce_same_metadata_as_tangent__(self, expected_metadata: Any, expected_type: type | None = None): ...
    def __repr__(self) -> str: ...
    def trigger_wait(self): ...
    def wait(self) -> torch.Tensor: ...
    def _get_acs_underlying_tensor(self):
        """This method enables  _functional_collectives_impl to test if a tensor is an ACS"""
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None): ...
    def numpy(self): ...

def _expand_group(group: RANK_TYPES, tag: str = '') -> tuple[str, list[int], int]:
    """
    _expand_group desugars the different RANK_TYPES types into a canonical format that is traceable.

    By having this be part of the explicit eager codepath, we avoid having to specialize behavior inside
    torchdynamo and can still interoperate with processgroup objects or other untraceable forms.
    """
def _resolve_group_name(group: RANK_TYPES, tag: str = '') -> str:
    """
    Given group in RANK_TYPES, return the group name.
    """

class _FromTorchTensor(torch.autograd.Function):
    """
    _FromTorchTensor allows autograd to propagate from a normal Tensor to an
    AsyncCollectiveTensor.
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor: ...
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor: ...

def _are_we_tracing() -> bool: ...
def _maybe_wrap_tensor(self) -> torch.Tensor: ...
@contextlib.contextmanager
def allow_inflight_collective_as_graph_input_ctx(value: bool = True):
    '''
    Context manager to temporarily set whether inflight collectives are allowed as torch.compile graph inputs.
    Common use case is when the collective is issued in eager (with `async_op=True`) but waited in compiled region:
    ```
    def all_reduce_eager(x):
        y = x * x
        req = dist.all_reduce(y, op=dist.ReduceOp.SUM, async_op=True)
        return y


    @torch.compile(fullgraph=True)
    def all_reduce_wait_compiled(y):
        torch.ops.c10d_functional.wait_tensor(y)
        return y * y


    x = torch.ones(1280, 1280, device="cuda") + self.rank
    # the context manager ensures that `wait_tensor(y)` will wait on the correct work object
    with allow_inflight_collective_as_graph_input_ctx():
        y = all_reduce_eager(x)
        z = all_reduce_wait_compiled(y)
    ```
    With this context manager, when a collective is called, under the hood the work object of the collective
    will be registered in the work registry, and the wait_tensor() in compiled region called on
    the output tensor of the collective will wait on the correct work object.
    '''
def _all_gather_into_tensor_coalesced_meta(self, tag, rankset, group_size): ...
def _broadcast_meta(self, *args): ...
def _all_reduce_meta(self, *args): ...
def _wait_tensor_meta(self, *args): ...
def _all_gather_into_tensor_meta(shard, tag, rankset, group_size): ...
def _reduce_scatter_tensor_meta(input, reduce_op, tag, rankset, group_size): ...
def _all_reduce_coalesced_meta(self, *args): ...
def _all_reduce__meta(inp, *args): ...
def _broadcast__meta(inp, *args): ...
def _all_reduce_coalesced__meta(inputs, *args): ...
def _reduce_scatter_tensor_coalesced_meta(inputs, reduceOp, tag, rankset, group_size): ...
def _all_to_all_single_meta(input, output_split_sizes, input_split_sizes, *args, **kwargs): ...
def _all_gather_into_tensor_out_native_meta(input, group_size, group_name, *, out): ...
def _all_gather_into_tensor_native_meta(input, group_size, group_name): ...
def _all_gather_into_tensor_coalesced_native_meta(inputs, group_size, group_name): ...
def _reduce_scatter_tensor_native_meta(inp, reduce_op, group_size, group_name): ...
def _reduce_scatter_tensor_coalesced_native_meta(inputs, reduce_op, group_size, group_name): ...

lib_impl: Incomplete
legacy_lib: Incomplete
legacy_lib_impl: Incomplete
ops_defs: Incomplete
my_module: Incomplete
op_name: Incomplete
backend_impl: Incomplete

def all_gather_tensor_inplace(output_tensor: torch.Tensor, input_tensor: torch.Tensor, group=None, async_op: bool = False, tag: str = '', gather_dim: int = 0): ...
def reduce_scatter_tensor_inplace(output: torch.Tensor, input: torch.Tensor, op: str = 'sum', group=None, async_op: bool = False, scatter_dim: int = 0, tag: str = ''): ...

REDUCE_OP_TO_STR: Incomplete

def all_reduce_inplace(tensor: torch.Tensor, op: str = 'sum', group=None, async_op: bool = False, tag: str = ''): ...
def all_to_all_inplace(output: torch.Tensor, input: torch.Tensor, output_split_sizes=None, input_split_sizes=None, group=None, async_op: bool = False, tag: str = ''): ...
def all_gather_inplace(tensor_list: list[torch.Tensor], tensor: torch.Tensor, group=None, async_op: bool = False, tag: str = ''): ...

traceable_collective_remaps: Incomplete
