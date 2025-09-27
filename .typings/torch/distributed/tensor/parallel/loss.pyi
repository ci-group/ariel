import contextlib
from collections.abc import Generator

__all__ = ['loss_parallel']

@contextlib.contextmanager
def loss_parallel() -> Generator[None]:
    '''
    A context manager that enables loss parallelism, where efficient parallelized loss computation
    can be performed when the input is sharded on the class dimension. Currently only the cross-entropy
    loss is supported.

    Within this context manager, one can use :func:`~torch.nn.functional.cross_entropy` or
    :class:`~torch.nn.CrossEntropyLoss` as usual, with the following assumptions on the input parameters.
    The corresponding ``backward()`` call, if any, also needs to happen under this context manager.

    Args:
        input (:class:`DTensor`):
            Input logits. Assumed to be sharded on the class dimension.
        target (Union[:class:`torch.Tensor`, :class:`DTensor`]):
            Must be ground truth class indices (class probabilities currently not supported).
            Assumed to be replicated across the ``DeviceMesh``.
        weight (Union[:class:`torch.Tensor`, :class:`DTensor`], optional):
            If given, assumed to be replicated across the ``DeviceMesh``.
        label_smoothing:
            Currently not supported.

    Returns:
        A replicated :class:`DTensor`.

    Example:
        A sharded DTensor is manually created here to showcase the usage.
        In practice, it is usually the output of a TP module.

        >>> # xdoctest: +SKIP("distributed")
        >>> from torch.distributed.tensor.parallel import loss_parallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> device_mesh = init_device_mesh("cuda", (8,))
        >>> input = torch.randn(4, 16, device="cuda", requires_grad=True)
        >>> dist_input = distribute_tensor(input, device_mesh, placements=[Shard(1)])
        >>> target = torch.randint(16, (4,), device="cuda")
        >>> with loss_parallel():
        >>>     loss = F.cross_entropy(dist_input, target, reduction="mean")
        >>>     loss.backward()
        >>> ...
    '''
