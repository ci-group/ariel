from torch.autograd import Function as Function
from torch.distributed import ReduceOp as ReduceOp, group as group

def broadcast(tensor, src, group=...):
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Arguments:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process.
        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Received tensor from the broadcast op.

    """
def gather(tensor, dst: int = 0, group=...):
    """
    Gathers a list of tensors in a single process.

    Arguments:
        tensor (Tensor): Input tensor.
        dst (int, optional): Destination rank (default is 0).
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple[Tensor]: List of appropriately-sized tensors with the gathered data.
    """
def scatter(tensors, src: int = 0, group=...):
    """
    Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Arguments:
        tensors (list[Tensor]): List of tensors to scatter on the source rank.
            Receivers must pass ``None`.
        src (int, optional): Source rank (default is 0).
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output tensor from the scatter operation.

    """
def reduce(tensor, dst, op=..., group=...):
    """
    Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Arguments:
        tensor (Tensor): Input of the collective.
        dst (int): Destination rank.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective.

    """
def reduce_scatter(output, input_list, op=..., group=...):
    """
    Reduces, then scatters a list of tensors to all processes in a group.

    Arguments:
        output (Tensor): Output tensor.
        input_list (list[Tensor]): List of tensors to reduce and scatter.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective.

    """
def all_gather(tensor, group=...):
    """
    Gathers tensors from the whole group in a list.

    Arguments:
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple([Tensor]): Output of the collective.

    """
def _all_gather_base(output_tensor, input_tensor, group=...):
    '''
    Single tensor all gather. Gathers a single tensor from all ranks, and puts them in a single output tensor.

    Args:
        output_tensor (Tensor): Output tensor. It should contain
            correctly-sized tensors to be used for output of the collective.
        input_tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Examples:
        >>> # All tensors below are of torch.int64 dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> # xdoctest: +SKIP("incorrect want text")
        >>> output_tensor = torch.zeros(2, dtype=torch.int64)
        >>> output_tensor
        [tensor([0, 0])] # Rank 0 and 1
        >>> tensor = torch.arange(1, dtype=torch.int64) + 1 + rank
        >>> tensor
        tensor([1]) # Rank 0
        tensor([2]) # Rank 1
        >>> dist.all_gather_base(output_tensor, tensor)
        >>> output_tensor
        tensor([1,2]) # Rank 0
        tensor([1,2]) # Rank 1

    .. warning::
        `_all_gather_base` is experimental and subject to change.
        It is the caller\'s responsibility to ensure the output_tensor
        is correctly sized.

    '''
def all_to_all(output_tensor_list, input_tensor_list, group=...):
    """
    Each process scatters list of input tensors to all processes in a group and return gathered list of tensors in output list.

    Arguments:
        output_tensor_list (list[Tensor]): list of tensors to gather one per rank.
        input_tensor_list (list[Tensor]): List of tensors to scatter one per rank.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple([Tensor]): Output of the collective.

    """
def all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=...):
    """
    Each process splits input tensor and then scatters the split list to all processes in a group.

    Then concatenate the received tensors from all the processes in the group and return single output tensor.

    Arguments:
        output (Tensor): Gathered concatenated output tensor.
        input (Tensor): Input tensor to scatter.
        output_split_sizes: (list[Int], optional): Output split sizes for dim 0
            if specified None or empty, dim 0 of ``output`` tensor must divide
            equally by ``world_size``.
        input_split_sizes: (list[Int], optional): Input split sizes for dim 0
            if specified None or empty, dim 0 of ``input`` tensor must divide
            equally by ``world_size``.

    Returns:
        Tensor: Output of the collective.

    """
def all_reduce(tensor, op=..., group=...):
    """
    Reduces the tensor data across all machines in such a way that all get the final result.

    After the call the returned tensor is going to be bitwise
    identical in all processes.

    Arguments:
        tensor (Tensor): Input of the collective.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective

    """

class _Broadcast(Function):
    @staticmethod
    def forward(ctx, src, group, tensor): ...
    @staticmethod
    def backward(ctx, grad_output): ...

class _Gather(Function):
    @staticmethod
    def forward(ctx, dst, group, tensor): ...
    @staticmethod
    def backward(ctx, *grad_outputs): ...

class _Scatter(Function):
    @staticmethod
    def forward(ctx, src, group, *tensors): ...
    @staticmethod
    def backward(ctx, grad_output): ...

class _Reduce(Function):
    @staticmethod
    def forward(ctx, src, op, group, tensor): ...
    @staticmethod
    def backward(ctx, grad_output): ...

class _Reduce_Scatter(Function):
    @staticmethod
    def forward(ctx, op, group, tensor, *input_tensor_list): ...
    @staticmethod
    def backward(ctx, grad_output): ...

class _AllGather(Function):
    @staticmethod
    def forward(ctx, group, tensor): ...
    @staticmethod
    def backward(ctx, *grad_outputs): ...

class _AllGatherBase(Function):
    @staticmethod
    def forward(ctx, output_tensor, input_tensor, group): ...
    @staticmethod
    def backward(ctx, grad_output): ...

class _AlltoAll(Function):
    @staticmethod
    def forward(ctx, group, out_tensor_list, *tensors): ...
    @staticmethod
    def backward(ctx, *grad_outputs): ...

class _AlltoAllSingle(Function):
    @staticmethod
    def forward(ctx, group, output, output_split_sizes, input_split_sizes, input): ...
    @staticmethod
    def backward(ctx, grad_output): ...

class _AllReduce(Function):
    @staticmethod
    def forward(ctx, op, group, tensor): ...
    @staticmethod
    def backward(ctx, grad_output): ...
