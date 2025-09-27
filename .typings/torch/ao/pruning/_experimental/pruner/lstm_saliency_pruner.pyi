from .base_structured_sparsifier import BaseStructuredSparsifier as BaseStructuredSparsifier, FakeStructuredSparsity as FakeStructuredSparsity

class LSTMSaliencyPruner(BaseStructuredSparsifier):
    """
    Prune packed LSTM weights based on saliency.
    For each layer {k} inside a LSTM, we have two packed weight matrices
    - weight_ih_l{k}
    - weight_hh_l{k}

    These tensors pack the weights for the 4 linear layers together for efficiency.

    [W_ii | W_if | W_ig | W_io]

    Pruning this tensor directly will lead to weights being misassigned when unpacked.
    To ensure that each packed linear layer is pruned the same amount:
        1. We split the packed weight into the 4 constituent linear parts
        2. Update the mask for each individual piece using saliency individually

    This applies to both weight_ih_l{k} and weight_hh_l{k}.
    """
    def update_mask(self, module, tensor_name, **kwargs) -> None: ...
