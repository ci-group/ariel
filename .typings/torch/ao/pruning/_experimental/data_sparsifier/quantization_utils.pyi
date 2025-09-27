import torch.nn as nn
from _typeshed import Incomplete
from torch.ao.pruning.sparsifier.utils import fqn_to_module as fqn_to_module, module_to_fqn as module_to_fqn

SUPPORTED_MODULES: Incomplete

def _fetch_all_embeddings(model):
    """Fetches Embedding and EmbeddingBag modules from the model"""
def post_training_sparse_quantize(model, data_sparsifier_class, sparsify_first: bool = True, select_embeddings: list[nn.Module] | None = None, **sparse_config):
    """Takes in a model and applies sparsification and quantization to only embeddings & embeddingbags.
    The quantization step can happen before or after sparsification depending on the `sparsify_first` argument.

    Args:
        - model (nn.Module)
            model whose embeddings needs to be sparsified
        - data_sparsifier_class (type of data sparsifier)
            Type of sparsification that needs to be applied to model
        - sparsify_first (bool)
            if true, sparsifies first and then quantizes
            otherwise, quantizes first and then sparsifies.
        - select_embeddings (List of Embedding modules)
            List of embedding modules to in the model to be sparsified & quantized.
            If None, all embedding modules with be sparsified
        - sparse_config (Dict)
            config that will be passed to the constructor of data sparsifier object.

    Note:
        1. When `sparsify_first=False`, quantization occurs first followed by sparsification.
            - before sparsifying, the embedding layers are dequantized.
            - scales and zero-points are saved
            - embedding layers are sparsified and `squash_mask` is applied
            - embedding weights are requantized using the saved scales and zero-points
        2. When `sparsify_first=True`, sparsification occurs first followed by quantization.
            - embeddings are sparsified first
            - quantization is applied on the sparsified embeddings
    """
