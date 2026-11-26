from mujoco._callbacks import *
from mujoco._constants import *
from mujoco._enums import *
from mujoco._errors import *
from mujoco._functions import *
from mujoco._specs import *
from mujoco._structs import *
from mujoco._render import *
from mujoco.rendering.classic.gl_context import *
from _typeshed import Incomplete
from mujoco import _specs
from mujoco.rendering.classic.renderer import Renderer as Renderer
from typing import Any, IO, Sequence, TypeAlias

proc_translated: Incomplete
is_rosetta: Incomplete
MJTNUM_DTYPE: Incomplete
MjStruct: TypeAlias

def to_zip(spec: _specs.MjSpec, file: str | IO[bytes]) -> None:
    """Converts an MjSpec to a zip file.

  Args:
    spec: The mjSpec to save to a file.
    file: The path to the file to save to or the file object to write to.
  """
def from_zip(file: str | IO[bytes]) -> _specs.MjSpec:
    """Reads a zip file and returns an MjSpec.

  Args:
    file: The path to the file to read from or the file object to read from.
  Returns:
    An MjSpec object.
  """

class _MjBindModel:
    """Wrapper for MjModel that allows binding multiple specs."""
    def __init__(self, elements: Sequence[Any]) -> None: ...
    def __getattr__(self, key: str): ...
    def __setattr__(self, key: str, value: Any): ...

class _MjBindData:
    """Wrapper for MjData that allows binding multiple specs."""
    def __init__(self, elements: Sequence[Any]) -> None: ...
    def __getattr__(self, key: str): ...
    def __setattr__(self, key: str, value: Any): ...

__version__: Incomplete
HEADERS_DIR: Incomplete
PLUGINS_DIR: Incomplete
PLUGIN_HANDLES: Incomplete
