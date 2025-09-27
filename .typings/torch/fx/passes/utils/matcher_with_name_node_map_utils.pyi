from .matcher_utils import InternalMatch, SubgraphMatcher
from _typeshed import Incomplete
from torch.fx import Graph, GraphModule

__all__ = ['SubgraphMatcherWithNameNodeMap']

class SubgraphMatcherWithNameNodeMap(SubgraphMatcher):
    '''Extends SubgraphMatcher to support querying the matched subgraph nodes through node name,
    this requires pattern to have specific format (returning and additional dictionary at the output,
    that has node name as key, and the node in the pattern graph as value, see Example for more details)

    Difference with SubgraphMatcher is that it takes a `pattern_gm` GraphModule as input during
    initialization since we need to modify the graph (which requires `recompile` the GraphModule)

    Example::
        def pattern(x, weight):
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            return relu, {"conv": conv, "relu": relu}


        def target_graph(x, weight):
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            relu *= 2
            return relu


        pattern_gm = export_for_training(pattern, example_inputs).module()
        target_gm = export_for_training(target_graph, example_inputs).module()
        matcher = SubgraphMatcherWithNameNodeMap(pattern_gm)
        matches = matcher.match(target_gm)
        for match in matches:
            match.name_node_map["conv"].meta["annotation"] = ...

    '''
    name_node_map: Incomplete
    def __init__(self, pattern_gm: GraphModule, match_output: bool = False, match_placeholder: bool = False, remove_overlapping_matches: bool = True, ignore_literals: bool = False) -> None: ...
    def match(self, graph: Graph) -> list[InternalMatch]:
        '''The returned InternalMatch will have name_node_map populated with a map
        from node name (str) to the target node, e.g.
        {"conv": target_conv_ndoe, "relu": target_relu_node}

        this requires the pattern graph returns an additional
        output of node name to node, e.g. instead of:
        ```
        def pattern(...):
            ...
            return relu
        ```
        we should do:
        ```
        def pattern(...):
            ...
            return relu, {"conv": conv, "relu": relu}
        ``` instead
        '''
