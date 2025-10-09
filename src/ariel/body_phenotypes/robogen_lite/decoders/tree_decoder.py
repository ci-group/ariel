from __future__ import annotations

from typing import Optional, Dict, Callable
import networkx as nx
from ariel.ec.genotypes.tree.tree_genome import TreeNode, TreeGenome
from ariel.body_phenotypes.robogen_lite import config

def to_digraph(genome: TreeGenome, use_node_ids: bool = True) -> nx.DiGraph:
    """
    Convert this genome (rooted at `genome.root`) to a NetworkX directed graph.

    The graph has a parent→child edge for each attachment in the tree. By default,
    nodes in the graph are keyed by the `TreeNode.id` values maintained in your
    structure. Optionally, you can request contiguous integer IDs (0..N-1) in DFS
    order via `use_node_ids=False`.

    Node attributes
    ----------------
    type : str
        The module type (enum `.name`).
    rotation : str
        The rotation (enum `.name`).
    depth : int
        The tree depth stored on the node (`node._depth`).
    raw_id : int
        The original `TreeNode.id` (always present, even when `use_node_ids=False`).

    Edge attributes
    ----------------
    face : str
        The face label (enum `.name`) used to attach the child to its parent.

    Parameters
    ----------
    use_node_ids : bool, optional (default: True)
        If True, graph nodes are keyed by `TreeNode.id`. If False, assigns
        contiguous integer IDs in DFS order starting at 0.

    Returns
    -------
    nx.DiGraph
        A directed graph representation of the tree. If the genome is empty
        (`self.root is None`), returns an empty graph.
    """
    g = nx.DiGraph()
    root = genome.root
    if root is None:
        return g

    # Stable mapping: either identity (node.id) or contiguous DFS ids.
    node_key: Callable[[TreeNode], int]
    if use_node_ids:
        node_key = lambda n: n.id
    else:
        # Assign 0..N-1 in first-seen (DFS) order
        seen: Dict[int, int] = {}
        next_id = 0
        def node_key(n: TreeNode) -> int:
            nonlocal next_id
            if n.id not in seen:
                seen[n.id] = next_id
                next_id += 1
            return seen[n.id]

    def dfs(parent: TreeNode | None, child: TreeNode, via_face: config.ModuleFaces | None) -> None:
        cid = node_key(child)
        # Add/update child node with attributes (use enum names for JSON-friendliness)
        g.add_node(
            cid,
            type=child.module_type.name,
            rotation=child.rotation.name,
            raw_id=child.id,
        )

        if parent is not None:
            pid = node_key(parent)
            g.add_edge(pid, cid, face=via_face.name if via_face is not None else None)

        # Recurse over children (face -> subnode)
        for face, sub in child.children.items():
            dfs(child, sub, face)

    dfs(None, root, None)
    return g

def test():
    # Create a simple tree genome for testing
    genome = TreeGenome()
    genome.root = TreeNode(config.ModuleInstance(type=config.ModuleType.BRICK, rotation=config.ModuleRotationsIdx.DEG_90, links={}))
    genome.root.front = TreeNode(config.ModuleInstance(type=config.ModuleType.BRICK, rotation=config.ModuleRotationsIdx.DEG_45, links={}))
    genome.root.left = TreeNode(config.ModuleInstance(type=config.ModuleType.BRICK, rotation=config.ModuleRotationsIdx.DEG_45, links={}))

    # Convert to directed graph
    digraph = to_digraph(genome, use_node_ids=False)

    # Print the graph nodes and edges with attributes
    print("Nodes:")
    for node, attrs in digraph.nodes(data=True):
        print(f"  {node}: {attrs}")

    print("\nEdges:")
    for u, v, attrs in digraph.edges(data=True):
        print(f"  {u} -> {v}: {attrs}")

# Test code
if __name__ == "__main__":
    test()