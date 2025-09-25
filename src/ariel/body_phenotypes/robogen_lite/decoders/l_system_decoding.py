"""Example of L-system-based decoding for modular robot graphs.

Author:     omn (with help from GitHub Copilot)
Date:       2025-09-25
Py Ver:     3.12
OS:         macOS Sequoia 15.3.1
Status:     Prototype

Notes
-----
    * This decoder uses an L-system string as the genotype to generate a directed graph (DiGraph) using NetworkX.
    * The L-system rules and axiom define the growth of the modular robot structure.

References
----------
    [1] https://en.wikipedia.org/wiki/L-system
    [2] https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.tree_data.html

"""

# Standard library
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Third-party libraries
import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph
from networkx.readwrite import json_graph

# Local libraries (reuse draw/save from hi_prob_decoding if available)
# from .hi_prob_decoding import draw_graph, save_graph_as_json

SEED = 42
DPI = 300

class LSystemDecoder:
    """Implements an L-system-based decoder for modular robot graphs."""

    def __init__(
        self,
        axiom: str,
        rules: Dict[str, str],
        iterations: int = 2,
        module_type_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize the L-system decoder.

        Parameters
        ----------
        axiom : str
            The initial string (genotype) of the L-system.
        rules : dict
            Production rules for the L-system.
        iterations : int
            Number of iterations to apply the rules.
        module_type_map : dict, optional
            Maps L-system symbols to module types (for node attributes).
        """
        self.axiom = axiom
        self.rules = rules
        self.iterations = iterations
        self.module_type_map = module_type_map or {}
        self.lsystem_string = self._generate_lsystem()
        self.graph = nx.DiGraph()
        self._decode_lsystem_to_graph()

    def _generate_lsystem(self) -> str:
        """Generate the L-system string after the given number of iterations."""
        current = self.axiom
        for _ in range(self.iterations):
            next_str = "".join(self.rules.get(c, c) for c in current)
            current = next_str
        return current

    def _decode_lsystem_to_graph(self) -> None:
        """
        Decode the L-system string into a directed graph.
        Supports branches using '[' (push) and ']' (pop) as in classic L-systems.
        Each symbol is a module; edges are created from the current parent node.
        """
        stack = []  # Stack for branching
        prev_node = None
        idx = 0
        for symbol in self.lsystem_string:
            if symbol == '[':
                # Push the current node onto the stack
                stack.append(prev_node)
            elif symbol == ']':
                # Pop the node from the stack
                if stack:
                    prev_node = stack.pop()
            else:
                node_label = f"{symbol}{idx}"
                self.graph.add_node(
                    node_label,
                    type=self.module_type_map.get(symbol, symbol),
                )
                if prev_node is not None:
                    self.graph.add_edge(prev_node, node_label)
                prev_node = node_label
                idx += 1

    def get_graph(self) -> DiGraph:
        """Return the generated NetworkX DiGraph."""
        return self.graph

    def save_graph_as_json(self, save_file: Path | str | None = None) -> None:
        """Save the graph as a JSON file (node-link format)."""
        if save_file is None:
            return
        data = json_graph.node_link_data(self.graph, edges="edges")
        json_string = json.dumps(data, indent=4)
        with Path(save_file).open("w", encoding="utf-8") as f:
            f.write(json_string)

    def draw_graph(
        self,
        title: str = "L-System Decoded Graph",
        save_file: Path | str | None = None,
    ) -> None:
        """Draw the decoded graph using matplotlib and networkx."""
        plt.figure()
        pos = nx.spring_layout(self.graph, seed=SEED)
        options = {
            "with_labels": True,
            "node_size": 150,
            "node_color": "#FFFFFF00",
            "edgecolors": "blue",
            "font_size": 8,
            "width": 0.5,
        }
        nx.draw(self.graph, pos, **options)
        plt.title(title)
        if save_file:
            plt.savefig(save_file, dpi=DPI)
        else:
            plt.show()

# Example usage (can be removed or moved to a test file)
if __name__ == "__main__":
    # Example: F->F+F, axiom F, 3 iterations
    axiom = "F"
    rules = {"F": "C[H][B][H][N]",
             "H" : "HB",
             "B" : "BH"}
    module_type_map = {"C": "CORE", "H": "HINGE", "B": "BRICK"}
    decoder = LSystemDecoder(axiom, rules, iterations=3, module_type_map=module_type_map)
    decoder.draw_graph()
    decoder.save_graph_as_json("lsystem_graph.json")