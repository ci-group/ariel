from torch.distributed._shard.common_op_utils import _basic_validation as _basic_validation
from torch.distributed._shard.sharded_tensor import Shard as Shard, ShardedTensor as ShardedTensor, _sharded_op_impl as _sharded_op_impl

def _sharded_op_common(op, early_stop_func, extra_check):
    '''
    Inject sharded tensor op registration with common logics executed before
    different behaviors are done on either local shards or a local tensor.

    Example::
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> op = torch.transpose
        >>> @_sharded_op_impl(op)
        >>> @_sharded_op_common(op, early_stop_func, extra_check)
        >>> def sharded_tensor_op(types, args, kwargs, process_group):
        >>>   ...
        >>>
        >>> st = sharded_tensor.rand(32, 16)
        >>> st.transpose(1, 2)
        >>> # This will call \'_sharded_op_common\'

    Args:
        op: The op to be registered and applied to all shards of the st.
        early_stop_func (Callable, optional): the func for early stop.
            Default: if ``None``, no early stop.
        extra_check (Callable, optional): the func for extra condition check.
            Default: if ``None``, no extra check.

    Return:
        func (Callable): Torch function for which we want to provide a sharded
            implementation (ex: torch.transpose)
    '''
def _register_sharded_op_on_local_shards(op, early_stop_func=None, extra_check=None, customized_func=None):
    """
    Handles ``__torch_function__`` dispatch for ops which are performed on
    each shard of the sharded tensor such as elementwise op like
    ``torch.nn.functional.gelu`` or ``torch.nn.functional.relu``.

    For more complicated ops, a customized func can be used to generate
    the new shards and sharded tensor size.

    This function expects that the original ShardingSpec for the ShardedTensor
    is preserved irrespective of whether or not a customized function is used.

    Args:
        op: The op to be registered and applied to all shards of the st.
        early_stop_func (Callable, optional): the func for early stop.
            Default: if ``None``, no early stop.
        extra_check (Callable, optional): the func for extra condition check.
            Default: if ``None``, no extra check.
        customized_func (Callable, optional): the func for customized logic
            to generate new shards and sharded tensor size.
            Default: if ``None``, we simply lower to the real op call with
                all local shards of the st.

    Return:
        func (Callable): registered implementation for sharded op for
        ``__torch_function__`` dispatch.
    """
