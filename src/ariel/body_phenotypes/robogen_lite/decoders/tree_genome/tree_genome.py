from __future__ import annotations
from ast import Dict
from typing import Any
from zipfile import Path
import matplotlib.pyplot as plt
import ariel.src.ariel.body_phenotypes.robogen_lite.config as config

import networkx as nx
from networkx import DiGraph
from networkx.readwrite import json_graph

class Tree:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.root = TreeNode(config.ModuleInstance(type=config.ModuleType.CORE, rotation=config.ModuleRotationsIdx.DEG_0, links={}), depth=0)

    def tree_to_digraph(self) -> nx.DiGraph:
        """
        Convert Tree (rooted at self.root) to a NetworkX DiGraph.
        Nodes are given integer ids (0..N-1) in DFS order.

        Node attrs:  type=<ModuleType.name>, rotation=<ModuleRotationsIdx.name>, depth=<int>
        Edge attrs:  face=<ModuleFaces.name>
        """
        # Stable ids for each TreeNode instance
        node_id: Dict[TreeNode, int] = {}
        next_id = 0

        def get_id(n: TreeNode) -> int:
            nonlocal next_id
            if n not in node_id:
                node_id[n] = next_id
                next_id += 1
            return node_id[n]

        def dfs(parent: TreeNode | None, child: TreeNode, via_face: config.ModuleFaces | None):
            id = get_id(child)
            # add/update the node with attributes (use .name to make JSON-friendly)
            self.graph.add_node(
                id,
                type=child.module_type.name,
                rotation=child.rotation.name,
                # depth=child._depth,
            )

            if parent is not None:
                parent_id = get_id(parent)
                # face stored as string (Enum.name) for readability / JSON
                self.graph.add_edge(parent_id, id, face=via_face.name if via_face else None)

            # descend
            for face, sub in child.children.items():
                # Expect sub to be a TreeNode
                dfs(child, sub, face)

        dfs(None, self.root, None)

class TreeNode:
    def __init__(self, module: config.ModuleInstance, depth: int = 0):
        self.module_type = module.type
        self.rotation = module.rotation
        # type: dict[ModuleFaces, TreeNode]
        self.children = module.links
        self._depth = depth

    def add_child(self, face: config.ModuleFaces, child_module: config.ModuleInstance):
        if face in self.children:
            raise ValueError(f"Face {face} already has a child.")
        if face not in config.ALLOWED_FACES[self.module_type]:
            raise ValueError(f"Face {face} is not allowed for module type {self.module_type}.")
        self.children[face] = TreeNode(child_module, depth=self._depth + 1)

# from matplotlib.figure import Figure
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from pathlib import Path

# def draw_graph(
#     graph: DiGraph[Any],
#     title: str = "NetworkX Directed Graph",
#     save_file: Path | str = "graph.png",
# ) -> None:
#     # --- NO pyplot here; use Figure + Agg canvas ---
#     fig = Figure()
#     canvas = FigureCanvas(fig)
#     ax = fig.add_subplot(111)

#     # Layouts (deterministic seed)
#     pos = nx.spectral_layout(graph)
#     pos = nx.spring_layout(graph, pos=pos, k=1, iterations=20, seed=42)

#     # Draw on explicit axes
#     nx.draw(
#         graph,
#         pos,
#         with_labels=True,
#         node_size=150,
#         node_color="#FFFFFF00",
#         edgecolors="blue",
#         font_size=8,
#         width=0.5,
#         ax=ax,
#     )

#     edge_labels = nx.get_edge_attributes(graph, "face")
#     nx.draw_networkx_edge_labels(
#         graph,
#         pos,
#         edge_labels=edge_labels,
#         font_color="red",
#         font_size=8,
#         ax=ax,
#     )

#     ax.set_title(title)
#     fig.tight_layout()

#     # Save via Agg canvas (no GUI backend involved)
#     fig.savefig(save_file, dpi=300, bbox_inches="tight")


# Generate a simple tree for demonstration
tree = Tree()
tree.root.add_child(config.ModuleFaces.FRONT, config.ModuleInstance(type=config.ModuleType.BRICK, rotation=config.ModuleRotationsIdx.DEG_90, links={}))
tree.root.add_child(config.ModuleFaces.TOP, config.ModuleInstance(type=config.ModuleType.HINGE, rotation=config.ModuleRotationsIdx.DEG_0, links={}))
tree.root.children[config.ModuleFaces.FRONT].add_child(config.ModuleFaces.TOP, config.ModuleInstance(type=config.ModuleType.BRICK, rotation=config.ModuleRotationsIdx.DEG_45, links={}))
tree.tree_to_digraph()
graph = tree.graph
print(graph.nodes(data=True))
# draw_graph(graph, title="Tree Structure", save_file="tree_structure.png")

