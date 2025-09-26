from mujoco._callbacks import *
from mujoco._constants import *
from mujoco._enums import *
from mujoco._errors import *
from mujoco._functions import *
from mujoco._render import *
from mujoco._specs import *
from mujoco._structs import *
from mujoco.gl_context import *
from _typeshed import Incomplete
from mujoco import _specs as _specs, _structs as _structs
from mujoco.renderer import Renderer as Renderer
from typing import Any, IO, Sequence
from typing_extensions import TypeAlias

__path__: Incomplete
_SYSTEM: Incomplete
proc_translated: Incomplete
is_rosetta: Incomplete
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
    elements: Incomplete
    def __init__(self, elements: Sequence[Any]) -> None: ...
    def __getattr__(self, key: str): ...

class _MjBindData:
    elements: Incomplete
    def __init__(self, elements: Sequence[Any]) -> None: ...
    def __getattr__(self, key: str): ...

def _bind_model(model: _structs.MjModel, specs: Sequence[MjStruct] | MjStruct):
    """Bind a Mujoco spec to a mjModel.

  Args:
    model: The mjModel to bind to.
    specs: The mjSpec elements to use for binding, can be a single element or a
      sequence.
  Returns:
    A MjModelGroupedViews object or a list of the same type.
  """
def _bind_data(data: _structs.MjData, specs: Sequence[MjStruct] | MjStruct):
    """Bind a Mujoco spec to a mjData.

  Args:
    data: The mjData to bind to.
    specs: The mjSpec elements to use for binding, can be a single element or a
      sequence.
  Returns:
    A MjDataGroupedViews object or a list of the same type.
  """

HEADERS_DIR: Incomplete
PLUGINS_DIR: Incomplete
PLUGIN_HANDLES: Incomplete

def _load_all_bundled_plugins() -> None: ...

__version__: Incomplete
