__all__ = ['get_storage_info', 'hierarchical_pickle', 'get_model_info', 'get_inline_skeleton', 'burn_in_info', 'get_info_and_burn_skeleton']

def get_storage_info(storage): ...
def hierarchical_pickle(data): ...
def get_model_info(path_or_file, title=None, extra_file_size_limit=...):
    """Get JSON-friendly information about a model.

    The result is suitable for being saved as model_info.json,
    or passed to burn_in_info.
    """
def get_inline_skeleton():
    """Get a fully-inlined skeleton of the frontend.

    The returned HTML page has no external network dependencies for code.
    It can load model_info.json over HTTP, or be passed to burn_in_info.
    """
def burn_in_info(skeleton, info):
    """Burn model info into the HTML skeleton.

    The result will render the hard-coded model info and
    have no external network dependencies for code or data.
    """
def get_info_and_burn_skeleton(path_or_bytesio, **kwargs): ...
