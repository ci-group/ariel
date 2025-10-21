"""Example of L-system-based decoding for modular robot graphs.

Author:     omn
Date:       2025-09-26
Py Ver:     3.12
OS:         macOS Tahoe 26
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

class SymbolToModuleType(Enum): # for auto-transcoding between L-system string characters and ModuleType elements
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
    ) -> None:
        """
        Initialize the L-system decoder.
        Automatically expands the L-system and builds the graph.
        """
        self.axiom = axiom
        self.rules = rules
        self.iterations = iterations
        self.graph = nx.DiGraph()
        self.lsystem_string = self.expand_lsystem() # first we expand the string applying recursively all the rules
        self.build_graph_from_string(self.lsystem_string) # we create the graph with networkx from a fully expanded L-system string

    def expand_lsystem(self, axiom: str = None, rules: Dict[str, str] = None, iterations: int = None) -> str:
        """
        Generate the L-system string after the given number of iterations, recursively expanding inside brackets as well, but stopping at the required depth.
        Each token is replaced in place by its rule expansion (not appended after).
        """
        # Match C as a single character, and other genes as X(num,FACE)
        gene_pattern = re.compile(r"(C|[A-Za-z]\(\d{1,3},[A-Za-z]+\))|\[|\]")
        axiom = axiom if axiom is not None else self.axiom # if we call it without axiom defined then we pick the one already assigned
        rules = rules if rules is not None else self.rules # same for rules
        iterations = iterations if iterations is not None else self.iterations # same for iterations

        def expand_all(s, depth):
            if depth == 0: # end of recursion ... just return the string.
                return s
            tokens = [m.group(0) for m in gene_pattern.finditer(s)]
            result = []
            i = 0
            while i < len(tokens): #go through all the token identified
                token = tokens[i]
                if token == '[':
                    # Find the matching closing bracket
                    bracket_level = 1
                    j = i + 1
                    while j < len(tokens) and bracket_level > 0:
                        if tokens[j] == '[':
                            bracket_level += 1
                        elif tokens[j] == ']':
                            bracket_level -= 1
                        j += 1
                    # Recursively expand the inside of the brackets with depth-1
                    inside = expand_all(''.join(tokens[i+1:j-1]), depth-1)
                    result.append('[' + inside + ']')
                    i = j
                elif token == ']':
                    i += 1
                elif token in rules:
                    # Replace the token in place with its expansion
                    replacement = rules[token]
                    expanded = expand_all(replacement, depth-1)
                    result.append(expanded)  # This replaces the token at this position
                    i += 1
                else:
                    result.append(token)
                    i += 1
            return ''.join(result)

        return expand_all(axiom, iterations)

    def build_graph_from_string(self, lsystem_string: str) -> None:
        """
        Build the graph from a fully expanded L-system string.
        """
        self.graph = nx.DiGraph()
        # Match C as a single character, and other genes as X(num,FACE)
        token_pattern = re.compile(r"C|([A-Za-z])\((\d{1,3}),(\w+)\)")
        s = lsystem_string
        core_count = 0
        idx_counter = [0]  # mutable counter for unique node labels

        def parse_tokens(s):
            # Parse the string into a tree of (gene, [children])
            tokens = []
            i = 0
            while i < len(s):
                if s[i] == '[':
                    # Find matching bracket
                    bracket_level = 1
                    j = i + 1
                    while j < len(s) and bracket_level > 0:
                        if s[j] == '[':
                            bracket_level += 1
                        elif s[j] == ']':
                            bracket_level -= 1
                        j += 1
                    subtree = parse_tokens(s[i+1:j-1])
                    tokens.append(subtree)
                    i = j
                elif s[i] == ']':
                    i += 1
                elif s[i].isalpha():
                    m = token_pattern.match(s, i)
                    if m:
                        tokens.append(s[i:m.end()])
                        i = m.end()
                    else:
                        tokens.append(s[i])
                        i += 1
                else:
                    i += 1
            return tokens

        def build_graph(tree, parent=None):
            nonlocal core_count
            current_parent = parent
            for node in tree:
                if isinstance(node, list):
                    # This is a branch, attach to the same parent (do not update current_parent)
                    build_graph(node, current_parent)
                else:
                    m = token_pattern.match(node)
                    if m:
                        if m.group(0) == "C":
                            symbol = "C"
                            node_type = ModuleType.CORE
                            rotation_enum = ModuleRotationsTheta.DEG_0
                            face = ModuleFaces.FRONT
                        else:
                            symbol = m.group(1)
                            try: # check if the type of elements is authorized (part of ModuleType enum)
                                symbol_to_look = SymbolToModuleType[symbol]
                                node_type = ModuleType[symbol_to_look.value]
                            except KeyError:
                                raise ValueError(f"Symbol '{symbol}' is not a valid ModuleType enum name.")
                            if node_type == ModuleType.CORE:
                                core_count += 1
                                if core_count > 1:
                                    raise ValueError("L-system string contains more than one CORE module.")
                            if m.group(2) is not None:
                                try: # check if the rotation is part of the allowed rotations (Module RotationsTheta enum)
                                    rotation_val = int(m.group(2))
                                    rotation_enum = next((r for r in ModuleRotationsTheta if r.value == rotation_val), ModuleRotationsTheta.DEG_0)
                                except Exception: # if error then default to 0
                                    rotation_enum = ModuleRotationsTheta.DEG_0
                            else: # if no rotation is provided then is is defaulted to 0
                                rotation_enum = ModuleRotationsTheta.DEG_0
                            face_str = m.group(3) if m.group(3) is not None else "FRONT"
                            try: # check if the face is in the allowed faces (Module ModuleFaces enum)
                                face = ModuleFaces[face_str]
                            except KeyError: # if error then default to FRONT
                                face = ModuleFaces.FRONT
                        node_label = f"{symbol}{idx_counter[0]}" # generate a unique ID for the node
                        self.graph.add_node(
                            node_label,
                            type=node_type,
                            rotation=rotation_enum,
                        ) #create and add the node to the graph
                        if current_parent is not None: # if there is a parent, create a link in the graph
                            self.graph.add_edge(current_parent, node_label,face=face_str)
                        idx_counter[0] += 1
                        current_parent = node_label  # Only update parent after a single node, not after a branch
            return current_parent

        tree = parse_tokens(s)
        build_graph(tree)

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
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_size=150,
            node_color="#FFFFFF00",
            edgecolors="blue",
            font_size=8,
            width=0.5,
        )

        edge_labels = nx.get_edge_attributes(self.graph, "face")

        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels=edge_labels,
            font_color="red",
            font_size=8,
        )

        plt.title(title)
        if save_file:
            plt.savefig(save_file, dpi=DPI)
        else:
            plt.show()

# in case we want to test on an example

def main():
    # Example: axiom with orientation and face, and C expands into branches
    axiom = "C[H(0,FRONT)][H(0,LEFT)][H(0,RIGHT)]"
    rules = {
        "H(0,FRONT)": "H(0,FRONT)B(0,FRONT)",
        "H(0,LEFT)": "H(0,LEFT)B(0,FRONT)",
        "H(0,RIGHT)": "H(0,RIGHT)B(0,FRONT)" # Example for N, can be expanded as needed
    }
    decoder = LSystemDecoder(axiom, rules, iterations=2)
    print("Nodes and attributes:")
    for n, d in decoder.graph.nodes(data=True):
        print(n, d)
    print(decoder.lsystem_string)
    decoder.draw_graph()

if __name__ == "__main__":
    main()
