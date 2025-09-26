import contextlib
import torch
from torch.fx.graph_module import GraphModule as GraphModule

_EMPTY_NN_MODULE_STACK_KEY: str

def _node_metadata_hook(node: torch.fx.Node, stack_trace: str | None = None) -> None:
    '''
    Hook for adding the appropriate metadata to nodes that are created during a
    pass using graph.create_node. An example of how to use it:

    ```
    with _set_node_metadata_hook(gm,
        functools.partial(_node_metadata_hook, stack_trace="file")
    ):
        pass(gm)
    ```

    This hook should not work for all generic cases -- specifically it assumes
    that nodes being added are only call_function nodes, and copies over the
    first argument node\'s nn_module_stack.
    '''
@contextlib.contextmanager
def _set_node_metadata_hook(gm: torch.fx.GraphModule, f):
    """
    Takes a callable which will be called after we create a new node. The
    callable takes the newly created node as input and returns None.
    """
