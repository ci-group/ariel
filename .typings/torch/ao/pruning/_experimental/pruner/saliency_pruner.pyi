from .base_structured_sparsifier import BaseStructuredSparsifier as BaseStructuredSparsifier

class SaliencyPruner(BaseStructuredSparsifier):
    """
    Prune rows based on the saliency (L1 norm) of each row.

    This pruner works on N-Dimensional weight tensors.
    For each row, we will calculate the saliency, whic is the sum the L1 norm of all weights in that row.
    We expect that the resulting saliency vector has the same shape as our mask.
    We then pick elements to remove until we reach the target sparsity_level.
    """
    def update_mask(self, module, tensor_name, **kwargs) -> None: ...
