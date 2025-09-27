import contextlib
import os
import types
from _typeshed import Incomplete
from torch.serialization import MAP_LOCATION
from typing import Any

__all__ = ['download_url_to_file', 'get_dir', 'help', 'list', 'load', 'load_state_dict_from_url', 'set_dir']

class _Faketqdm:
    total: Incomplete
    disable: Incomplete
    n: int
    def __init__(self, total=None, disable: bool = False, unit=None, *args, **kwargs) -> None: ...
    def update(self, n) -> None: ...
    def set_description(self, *args, **kwargs) -> None: ...
    def write(self, s) -> None: ...
    def close(self) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...
tqdm = _Faketqdm

def get_dir() -> str:
    """
    Get the Torch Hub cache directory used for storing downloaded models & weights.

    If :func:`~torch.hub.set_dir` is not called, default path is ``$TORCH_HOME/hub`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesystem layout, with a default value ``~/.cache`` if the environment
    variable is not set.
    """
def set_dir(d: str | os.PathLike) -> None:
    """
    Optionally set the Torch Hub directory used to save downloaded models & weights.

    Args:
        d (str): path to a local folder to save downloaded models & weights.
    """
def list(github, force_reload: bool = False, skip_validation: bool = False, trust_repo=None, verbose: bool = True):
    '''
    List all callable entrypoints available in the repo specified by ``github``.

    Args:
        github (str): a string with format "repo_owner/repo_name[:ref]" with an optional
            ref (tag or branch). If ``ref`` is not specified, the default branch is assumed to be ``main`` if
            it exists, and otherwise ``master``.
            Example: \'pytorch/vision:0.10\'
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.
            Default is ``False``.
        skip_validation (bool, optional): if ``False``, torchhub will check that the branch or commit
            specified by the ``github`` argument properly belongs to the repo owner. This will make
            requests to the GitHub API; you can specify a non-default GitHub token by setting the
            ``GITHUB_TOKEN`` environment variable. Default is ``False``.
        trust_repo (bool, str or None): ``"check"``, ``True``, ``False`` or ``None``.
            This parameter was introduced in v1.12 and helps ensuring that users
            only run code from repos that they trust.

            - If ``False``, a prompt will ask the user whether the repo should
              be trusted.
            - If ``True``, the repo will be added to the trusted list and loaded
              without requiring explicit confirmation.
            - If ``"check"``, the repo will be checked against the list of
              trusted repos in the cache. If it is not present in that list, the
              behaviour will fall back onto the ``trust_repo=False`` option.
            - If ``None``: this will raise a warning, inviting the user to set
              ``trust_repo`` to either ``False``, ``True`` or ``"check"``. This
              is only present for backward compatibility and will be removed in
              v2.0.

            Default is ``None`` and will eventually change to ``"check"`` in v2.0.
        verbose (bool, optional): If ``False``, mute messages about hitting
            local caches. Note that the message about first download cannot be
            muted. Default is ``True``.

    Returns:
        list: The available callables entrypoint

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> entrypoints = torch.hub.list("pytorch/vision", force_reload=True)
    '''
def help(github, model, force_reload: bool = False, skip_validation: bool = False, trust_repo=None):
    '''
    Show the docstring of entrypoint ``model``.

    Args:
        github (str): a string with format <repo_owner/repo_name[:ref]> with an optional
            ref (a tag or a branch). If ``ref`` is not specified, the default branch is assumed
            to be ``main`` if it exists, and otherwise ``master``.
            Example: \'pytorch/vision:0.10\'
        model (str): a string of entrypoint name defined in repo\'s ``hubconf.py``
        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.
            Default is ``False``.
        skip_validation (bool, optional): if ``False``, torchhub will check that the ref
            specified by the ``github`` argument properly belongs to the repo owner. This will make
            requests to the GitHub API; you can specify a non-default GitHub token by setting the
            ``GITHUB_TOKEN`` environment variable. Default is ``False``.
        trust_repo (bool, str or None): ``"check"``, ``True``, ``False`` or ``None``.
            This parameter was introduced in v1.12 and helps ensuring that users
            only run code from repos that they trust.

            - If ``False``, a prompt will ask the user whether the repo should
              be trusted.
            - If ``True``, the repo will be added to the trusted list and loaded
              without requiring explicit confirmation.
            - If ``"check"``, the repo will be checked against the list of
              trusted repos in the cache. If it is not present in that list, the
              behaviour will fall back onto the ``trust_repo=False`` option.
            - If ``None``: this will raise a warning, inviting the user to set
              ``trust_repo`` to either ``False``, ``True`` or ``"check"``. This
              is only present for backward compatibility and will be removed in
              v2.0.

            Default is ``None`` and will eventually change to ``"check"`` in v2.0.
    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> print(torch.hub.help("pytorch/vision", "resnet18", force_reload=True))
    '''
def load(repo_or_dir, model, *args, source: str = 'github', trust_repo=None, force_reload: bool = False, verbose: bool = True, skip_validation: bool = False, **kwargs):
    '''
    Load a model from a github repo or a local directory.

    Note: Loading a model is the typical use case, but this can also be used to
    for loading other objects such as tokenizers, loss functions, etc.

    If ``source`` is \'github\', ``repo_or_dir`` is expected to be
    of the form ``repo_owner/repo_name[:ref]`` with an optional
    ref (a tag or a branch).

    If ``source`` is \'local\', ``repo_or_dir`` is expected to be a
    path to a local directory.

    Args:
        repo_or_dir (str): If ``source`` is \'github\',
            this should correspond to a github repo with format ``repo_owner/repo_name[:ref]`` with
            an optional ref (tag or branch), for example \'pytorch/vision:0.10\'. If ``ref`` is not specified,
            the default branch is assumed to be ``main`` if it exists, and otherwise ``master``.
            If ``source`` is \'local\'  then it should be a path to a local directory.
        model (str): the name of a callable (entrypoint) defined in the
            repo/dir\'s ``hubconf.py``.
        *args (optional): the corresponding args for callable ``model``.
        source (str, optional): \'github\' or \'local\'. Specifies how
            ``repo_or_dir`` is to be interpreted. Default is \'github\'.
        trust_repo (bool, str or None): ``"check"``, ``True``, ``False`` or ``None``.
            This parameter was introduced in v1.12 and helps ensuring that users
            only run code from repos that they trust.

            - If ``False``, a prompt will ask the user whether the repo should
              be trusted.
            - If ``True``, the repo will be added to the trusted list and loaded
              without requiring explicit confirmation.
            - If ``"check"``, the repo will be checked against the list of
              trusted repos in the cache. If it is not present in that list, the
              behaviour will fall back onto the ``trust_repo=False`` option.
            - If ``None``: this will raise a warning, inviting the user to set
              ``trust_repo`` to either ``False``, ``True`` or ``"check"``. This
              is only present for backward compatibility and will be removed in
              v2.0.

            Default is ``None`` and will eventually change to ``"check"`` in v2.0.
        force_reload (bool, optional): whether to force a fresh download of
            the github repo unconditionally. Does not have any effect if
            ``source = \'local\'``. Default is ``False``.
        verbose (bool, optional): If ``False``, mute messages about hitting
            local caches. Note that the message about first download cannot be
            muted. Does not have any effect if ``source = \'local\'``.
            Default is ``True``.
        skip_validation (bool, optional): if ``False``, torchhub will check that the branch or commit
            specified by the ``github`` argument properly belongs to the repo owner. This will make
            requests to the GitHub API; you can specify a non-default GitHub token by setting the
            ``GITHUB_TOKEN`` environment variable. Default is ``False``.
        **kwargs (optional): the corresponding kwargs for callable ``model``.

    Returns:
        The output of the ``model`` callable when called with the given
        ``*args`` and ``**kwargs``.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> # from a github repo
        >>> repo = "pytorch/vision"
        >>> model = torch.hub.load(
        ...     repo, "resnet50", weights="ResNet50_Weights.IMAGENET1K_V1"
        ... )
        >>> # from a local directory
        >>> path = "/some/local/path/pytorch/vision"
        >>> # xdoctest: +SKIP
        >>> model = torch.hub.load(path, "resnet50", weights="ResNet50_Weights.DEFAULT")
    '''
def download_url_to_file(url: str, dst: str, hash_prefix: str | None = None, progress: bool = True) -> None:
    '''Download object at the given URL to a local path.

    Args:
        url (str): URL of the object to download
        dst (str): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix (str, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> # xdoctest: +REQUIRES(POSIX)
        >>> torch.hub.download_url_to_file(
        ...     "https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth",
        ...     "/tmp/temporary_file",
        ... )

    '''
def load_state_dict_from_url(url: str, model_dir: str | None = None, map_location: MAP_LOCATION = None, progress: bool = True, check_hash: bool = False, file_name: str | None = None, weights_only: bool = False) -> dict[str, Any]:
    '''Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it\'s deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (str): URL of the object to download
        model_dir (str, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (str, optional): name for the downloaded file. Filename from ``url`` will be used if not set.
        weights_only(bool, optional): If True, only weights will be loaded and no complex pickled objects.
            Recommended for untrusted sources. See :func:`~torch.load` for more details.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)
        >>> state_dict = torch.hub.load_state_dict_from_url(
        ...     "https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth"
        ... )

    '''
