from torch._ops import OpOverload

__all__ = ['register_sharding']

def register_sharding(op: OpOverload | list[OpOverload]):
    '''
    :meth:`register_sharding` is an experimental API that allows users to register sharding
    strategies for an operator when the tensor inputs and outputs are DTensor.
    It can be useful when: (1) there doesn\'t exist a default sharding strategy for ``op``,
    e.g. when ``op`` is a custom operator that is not supported by :class:`DTensor`; (2)
    when users would like to overwrite default sharding strategies of existing operators.

    Args:
        op (Union[OpOverload, List[OpOverload]]):
            An op or a list of ops to register the customized sharding function.

    Returns:
        A function decorator which can be used to wrap a function that defines the sharding
        strategy for the operator specified in ``op``. The defined sharding strategy will be
        registered to DTensor and will override the default sharding strategy if DTensor has
        already implemented the operator. The customized sharding function takes the same inputs
        as the original op (except that if an arg is a :class:`torch.Tensor`, it will be
        replaced by a tensor-like object that DTensor uses internally). The function should
        return a sequence of 2-tuples, each specifying acceptable output placements and its
        corresponding intput placements.

    Example:
        >>> # xdoctest: +SKIP("distributed")
        >>> @register_sharding(aten._softmax.default)
        >>> def custom_softmax_sharding(x, dim, half_to_float):
        >>>     softmax_dim = dim if dim >= 0 else dim + x.ndim
        >>>     acceptable_shardings = []
        >>>
        >>>     all_replicate = ([Replicate()], [Replicate(), None, None])
        >>>     acceptable_shardings.append(all_replicate)
        >>>
        >>>     for sharding_dim in range(x.ndim):
        >>>         if sharding_dim != softmax_dim:
        >>>             all_sharded = (
        >>>                 [Shard(sharding_dim)],
        >>>                 [Shard(sharding_dim), None, None],
        >>>             )
        >>>             acceptable_shardings.append(all_sharded)
        >>>
        >>>     return acceptable_shardings

    .. note:: This API is currently experimental and subject to change
    '''
