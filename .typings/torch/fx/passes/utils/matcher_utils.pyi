from _typeshed import Incomplete
from dataclasses import dataclass, field
from torch.fx import Graph, Node
from typing import Any

__all__ = ['SubgraphMatcher', 'InternalMatch']

@dataclass
class InternalMatch:
    anchors: list[Node]
    nodes_map: dict[Node, Node] = field(default_factory=dict)
    placeholder_nodes: list[Node] = field(default_factory=list)
    returning_nodes: list[Node] = field(default_factory=list)
    name_node_map: dict[str, Node] = field(default_factory=dict)
    def __copy__(self): ...

class SubgraphMatcher:
    pattern: Incomplete
    match_output: Incomplete
    match_placeholder: Incomplete
    remove_overlapping_matches: Incomplete
    ignore_literals: Incomplete
    pattern_placeholder_nodes: Incomplete
    pattern_returning_nodes: list[Node]
    pattern_anchors: list[Node]
    def __init__(self, pattern: Graph, match_output: bool = False, match_placeholder: bool = False, remove_overlapping_matches: bool = True, ignore_literals: bool = False) -> None:
        """
        Args:
            pattern: the targeted matching pattern, represented in fx.Graph.
            match_output: If True, output node in the pattern graph will be treated as a part of the targeted pattern.
                If False, output node is ignored during match.
            match_placeholder: If True, placeholder node in the pattern graph will be treated as a part of
                the targeted pattern. If False, placeholder nodes will be used a wildcard.
            remove_overlapping_matches: If True, in the case of overlapping matches, only the first match
                will be returned.
            ignore_literals: If True, will not check if literals are equal and
                will instead treat them as wildcards.
        """
    def _match_attributes(self, pn: Node, gn: Node) -> bool: ...
    def _nodes_are_equal(self, pn: Node, gn: Node) -> bool: ...
    def _is_contained(self, nodes_map: dict[Node, Node]) -> bool: ...
    def _remove_overlapping_matches(self, matches: list[InternalMatch]) -> list[InternalMatch]: ...
    def _match_literals(self, pn: Any, gn: Any, match: InternalMatch) -> bool: ...
    def _match_nodes(self, pn: Node, gn: Node, match: InternalMatch) -> bool: ...
    def match(self, graph: Graph) -> list[InternalMatch]:
        '''
        Returns:
            The matched subgraphs.
            Thre returned subgraph would be fully self-contained, meaning the nodes (except placeholder
            and nodes returned by output) can only be consumed by nodes within the matched subgraph.

        Subgraph pattern matcher is implemented with the backtracking style in the following steps:

        1. We first identify all the anchor nodes in the pattern graph. The anchor nodes
        are the "sinks" (nodes with no user other than the output node) of the pattern graph.
        One pattern graph could have multiple anchors if it has multiple return values.

        2. In the target graph, we identify the potential candidate nodes that can be matched
        with each anchor. These anchor-candidate pairs are the starting points for
        pairwise per-node matching.

        3. For each anchor-candidate pair, we simultaneously traverse backwards (DFS) in both
        pattern and target graphs. For every pattern nodes along traversal path, we compare it
        against the target nodes. In case any comparison failed, the match for this anchor-candidate
        pair fails. A match is found when DFS completes traversing the graph. See `self._match_nodes`
        for more details.

        4. In the case of multiple anchors, every anchor will need to find a match using step 3.
        In addition, the matches found between anchors need to have a common intersection node
        in order for the match to be valid. This is implemented with backtracking. See `backtracking`
        for more details.

        Notice: graph traversal must be done in the reverser order because a tensor can have multiple
        consumers, but can only have a single producer. Only with reverser order, we can we jointly
        traverse the pattern and target graph in a deterministic path.

        Warning: In theory, this backtracking algorithm have an **exponential** time complexity. However,
        in practice, it\'s unlikely to blow up.

        '''
