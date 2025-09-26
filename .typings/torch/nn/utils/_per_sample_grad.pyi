from torch.nn.utils._expanded_weights.expanded_weights_impl import ExpandedWeight as ExpandedWeight

def call_for_per_sample_grads(module, *, batch_size=None, loss_reduction: str = 'sum', batch_first: bool = True):
    '''
    Return a forward function for a module, populating grad_sample with per sample gradients on backward invocation.

    Args:
        module: The ``nn.Module`` to get per sample gradients with respect to. All trainable
          parameters will compute per sample gradients, located in a ``grad_sample``
          field when ``backward`` is invoked
        batch_size: The batch size of the input. If None is passed, all tensor arguments in args and kwargs must have
          the same batch size, which is the size of the first dimension. Otherwise, it must be passed manually.
          Default: None
        loss_reduction: Indicates if the loss reduction (for aggregating the gradients) is a sum or a mean operation. If
          "mean", per sample gradients will be scaled by the batch size to offset the crossbatch interaction from
          running mean across a batch. Must be "mean" or "sum". Default: "sum"
        batch_first: Indicates if the batch dimension is the first dimension. If True, the batch dimension is the first
          dimension. If False, it\'s the second dimension. Default: True.

    Examples::
        >>> # xdoctest: +SKIP
        >>> model = nn.Linear(4, 3)
        >>> batched_input = torch.randn(5, 4)  # batch size of 5
        >>> res = call_for_per_sample_grads(model)(batched_input).sum()
        >>> res.backward()
        >>> assert model.weight.shape == (3, 4)
        >>> assert model.weight.grad_sample.shape == (5, 3, 4)
        >>> assert model.weight.grad is None
        >>> assert model.bias.shape == (3,)
        >>> assert model.bias.grad_sample.shape == (5, 3)
        >>> assert model.bias.grad is None

    An example using "mean" loss reduction. The grad_sample fields will be scaled by batch_size from what they would be
    if we ran the same code with loss_reduction="sum". This is because the mean at the end will scale all
    grad_outputs by 1 / batch_size from cross batch interaction.
        >>> model = nn.Linear(4, 3)
        >>> batched_input = torch.randn(5, 4)  # batch size of 5
        >>> res = call_for_per_sample_grads(model, 5, loss_reduction="mean")(
        ...     batched_input
        ... ).mean()
        >>> res.backward()

    Note::
        Does not work with any `nn.RNN`, including `nn.GRU` or `nn.LSTM`. Please use custom
        rewrites that wrap an `nn.Linear` module. See Opacus for an example
    '''
