def check_serializing_named_tensor(tensor) -> None: ...
def build_dim_map(tensor):
    """Returns a map of { dim: dim_name } where dim is a name if the dim is named
    and the dim index otherwise."""
def unzip_namedshape(namedshape): ...
def namer_api_name(inplace): ...
def is_ellipsis(item): ...
def single_ellipsis_index(names, fn_name): ...
def expand_single_ellipsis(numel_pre_glob, numel_post_glob, names): ...
def replace_ellipsis_by_position(ellipsis_idx, names, tensor_names): ...
def resolve_ellipsis(names, tensor_names, fn_name):
    """
    Expands ... inside `names` to be equal to a list of names from `tensor_names`.
    """
def update_names_with_list(tensor, names, inplace): ...
def update_names_with_mapping(tensor, rename_map, inplace): ...
def update_names(tensor, names, rename_map, inplace):
    """There are two usages:

    tensor.rename(*names) returns a view on tensor with named dims `names`.
    `names` must be of length `tensor.dim()`; otherwise, if '...' is in `names`,
    then it is expanded greedily to be equal to the corresponding names from
    `tensor.names`.

    For example,
    ```
    >>> # xdoctest: +SKIP
    >>> x = torch.empty(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
    >>> x.rename('...', 'height', 'width').names
    ('N', 'C', 'height', 'width')

    >>> # xdoctest: +SKIP
    >>> x.rename('batch', '...', 'width').names
    ('batch', 'C', 'H', 'width')

    ```

    tensor.rename(**rename_map) returns a view on tensor that has rename dims
        as specified in the mapping `rename_map`.

    For example,
    ```
    >>> # xdoctest: +SKIP
    >>> x = torch.empty(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
    >>> x.rename(W='width', H='height').names
    ('N', 'C', 'height', 'width')

    ```

    Finally, tensor.rename has an in-place version called tensor.rename_.
    """
