from _typeshed import Incomplete
from torch.cuda._memory_viz import _block_extra as _block_extra, _frames_fmt as _frames_fmt
from typing import NamedTuple

logger: Incomplete

def observe_garbage(observer): ...
def _get_cell_type(): ...

CellType: Incomplete

def annotated_references(obj):
    """
    Return known information about references held by the given object.

    Returns a mapping from referents to lists of descriptions.  Note that there
    may be more than one edge leading to any particular referent; hence the
    need for a list.  Descriptions are currently strings.

    """

BASE_TYPES: Incomplete
FRAME_FILENAME_LIMIT: int

def object_annotation(obj):
    """
    Return a string to be used for Graphviz nodes.

    The string should be short but as informative as possible.
    """

class Node(NamedTuple):
    label: str
    context: str | None
    root: bool
    referrents: list[tuple[str, int]]

def create_graph(objects, *, context=None, filter=None): ...
def escape(n): ...
def is_cuda_tensor(obj): ...
def cuda_allocation_context(): ...
def to_dot(nodes): ...

_template: str
_listener_template: str

def to_html(nodes): ...
def observe_tensor_cycles(callback): ...
def warn_tensor_cycles():
    """
    Install a warning that reports whenever a cycle that is holding CUDA memory is observed.

    The warning produces an .html file that visualizes the cycle,
    and links it to the stack frame that allocted the CUDA tensor.

    Reference cycles are freed by the cycle collector rather than being cleaned up
    when the objects in the cycle first become unreachable. If a cycle points to a tensor,
    the CUDA memory for that tensor will not be freed until garbage collection runs.
    Accumulation of CUDA allocations can lead to out of memory errors (OOMs), as well as
    non-deterministic allocation behavior which is harder to debug.
    """
