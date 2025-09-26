
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
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from enum import Enum

# Third-party libraries
import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph
from networkx.readwrite import json_graph


# Local libraries
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleRotationsTheta, ModuleType

SEED = 42
DPI = 300

class SymbolToModuleType(Enum):
    """Enum for module types."""

    C = 'CORE'
    B = 'BRICK'
    H = 'HINGE'
    N = 'NONE'

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
        Each gene can encode orientation and face as SYMBOL(rotation,face), e.g., B(90,FRONT).
        Orientation and face are stored as node attributes (defaults: 0, FRONT).
        """
        stack = []  # Stack for branching
        prev_node = None
        idx = 0
        # Regex to match symbol with optional rotation and face, e.g., B(90,FRONT)
        token_pattern = re.compile(r"([A-Za-z])(?:\((\d{1,3})(?:,([A-Za-z]+))?\))?")
        # Tokenize the string, preserving brackets
        tokens = []
        i = 0
        s = self.lsystem_string
        core_count = 0
        while i < len(s):
            if s[i] in '[]':
                tokens.append(s[i])
                i += 1
            elif s[i].isalpha():
                m = token_pattern.match(s, i)
                if m:
                    symbol = m.group(1)
                    # Enforce node type from ModuleType enum
                    try:
                        symbol_to_look = SymbolToModuleType[symbol]
                        node_type = ModuleType[symbol_to_look.value]
                    except KeyError:
                        raise ValueError(f"Symbol '{symbol}' is not a valid ModuleType enum name.")
                    if node_type == ModuleType.CORE:
                        core_count += 1
                        if core_count > 1:
                            raise ValueError("L-system string contains more than one CORE module.")
                    # Parse and validate orientation
                    if m.group(2) is not None:
                        try:
                            rotation_val = int(m.group(2))
                            rotation_enum = next((r for r in ModuleRotationsTheta if r.value == rotation_val), ModuleRotationsTheta.DEG_0)
                        except Exception:
                            rotation_enum = ModuleRotationsTheta.DEG_0
                    else:
                        rotation_enum = ModuleRotationsTheta.DEG_0
                    face_str = m.group(3) if m.group(3) is not None else "FRONT"
                    try:
                        face = ModuleFaces[face_str]
                    except KeyError:
                        face = ModuleFaces.FRONT
                    tokens.append((symbol, node_type, rotation_enum, face))
                    i = m.end()
                else:
                    # fallback: treat as NONE
                    tokens.append((s[i], ModuleType.NONE, ModuleRotationsTheta.DEG_0, ModuleFaces.FRONT))
                    i += 1
            else:
                i += 1  # skip any other character

        for token in tokens:
            if token == '[':
                stack.append(prev_node)
            elif token == ']':
                if stack:
                    prev_node = stack.pop()
            else:
                symbol, node_type, rotation_enum, face = token
                node_label = f"{symbol}{idx}"
                self.graph.add_node(
                    node_label,
                    type=node_type,
                    rotation=rotation_enum,
                    face=face,
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
