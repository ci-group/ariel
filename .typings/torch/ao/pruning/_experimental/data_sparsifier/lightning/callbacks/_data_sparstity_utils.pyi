import logging
from torch.ao.pruning._experimental.data_sparsifier.base_data_sparsifier import SUPPORTED_TYPES as SUPPORTED_TYPES

logger: logging.Logger

def _attach_model_to_data_sparsifier(module, data_sparsifier, config=None) -> None:
    """Attaches a data sparsifier to all the layers of the module.
    Essentially, loop over all the weight parameters in the module and
    attach it to the data sparsifier.
    Note::
        The '.' in the layer names are replaced with '_' (refer to _get_valid_name() below)
        before attaching to the sparsifier. This is because, the data
        sparsifier uses a dummy model inside to store the weight parameters.
    """
def _get_valid_name(name): ...
def _log_sparsified_level(model, data_sparsifier) -> None: ...
