from . import config as config
from .backends.registry import list_backends as list_backends, lookup_backend as lookup_backend, register_backend as register_backend
from .convert_frame import replay as replay
from .decorators import allow_in_graph as allow_in_graph, assume_constant_result as assume_constant_result, disable as disable, disallow_in_graph as disallow_in_graph, dont_skip_tracing as dont_skip_tracing, forbid_in_graph as forbid_in_graph, graph_break as graph_break, mark_dynamic as mark_dynamic, mark_static as mark_static, mark_static_address as mark_static_address, maybe_mark_dynamic as maybe_mark_dynamic, nonstrict_trace as nonstrict_trace, patch_dynamo_config as patch_dynamo_config, run as run, set_stance as set_stance, skip_frame as skip_frame, substitute_in_graph as substitute_in_graph
from .eval_frame import OptimizedModule as OptimizedModule, explain as explain, export as export, optimize as optimize, optimize_assert as optimize_assert
from .external_utils import is_compiling as is_compiling

__all__ = ['allow_in_graph', 'assume_constant_result', 'disallow_in_graph', 'dont_skip_tracing', 'forbid_in_graph', 'substitute_in_graph', 'graph_break', 'mark_dynamic', 'maybe_mark_dynamic', 'mark_static', 'mark_static_address', 'nonstrict_trace', 'optimize', 'optimize_assert', 'patch_dynamo_config', 'skip_frame', 'export', 'explain', 'run', 'replay', 'disable', 'set_stance', 'reset', 'OptimizedModule', 'is_compiling', 'register_backend', 'list_backends', 'lookup_backend', 'config']

def reset() -> None:
    """
    Clear all compile caches and restore initial state.  This function is intended
    to reset Dynamo's state *as if* you had started a fresh process invocation, which
    makes it good for testing scenarios where you want to behave as if you started
    a new process.  It does NOT affect any file system caches.

    NB: this does NOT reset logging state.  Don't use this to test logging
    initialization/reinitialization.
    """
