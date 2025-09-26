import torch.nn as nn

def _annotate_modules_for_dynamo(module: nn.Module, ignored_modules: set[nn.Module], use_orig_params: bool) -> None:
    """
    Annotates the submodules in ``module`` 's tree, except those in
    ``ignored_modules``, indicating that the submodules are FSDP-managed and
    saving the ``use_orig_params`` setting passed to the FSDP constructor.
    """
