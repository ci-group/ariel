def dump_tensorboard_summary(graph_executor, logdir) -> None: ...
def visualize(graph, name_prefix: str = '', pb_graph=None, executors_it=None):
    """Visualizes an independent graph, or a graph executor."""
def visualize_graph_executor(state, name_prefix, pb_graph, inline_graph):
    """Append the state of a given GraphExecutor to the graph protobuf.

    Args:
        state (GraphExecutor or GraphExecutorState): GraphExecutor to display.
        name_prefix (str): Name prefix of the containing subgraph.
        pb_graph (GraphDef): graph to append to.
        inline_graph (Callable): a function that handles setting up a value_map,
            so that some graphs in here can be inlined. This is necessary, because
            this will simply be `visualize` for the top-level GraphExecutor,
            or `inline_graph` for all nested ones.

            The signature should look like (Graph, name_prefix) -> ().
            It will be called exactly once.

    The strategy is to embed all different configurations as independent subgraphs,
    while inlining the original graph as the one that actually produces the values.
    """
def visualize_rec(graph, value_map, name_prefix, pb_graph, executors_it=None):
    """Recursive part of visualize (basically skips setting up the input and output nodes)."""
