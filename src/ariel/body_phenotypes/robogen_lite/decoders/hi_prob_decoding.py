"""Highest-probability-decoding algorithm for ARIEL-robots.

Note
-----
* Graphs are represented as directed graphs (DiGraph) using NetworkX.
* Graphs are saved as JSON [1]_.

References
----------
.. [1] `NetworkX JSON Graph <https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.tree_data.html#networkx.readwrite.json_graph.tree_data>`_
Todo
----
    - [ ] for loops to be replaced with vectorized operations
    - [ ] DiGraph positioning use cartesian coordinates instead of spring layout
    - [ ] Should probably move the graph functions to a separate script
"""

# Evaluate type annotations in a deferred manner (ruff: UP037)
from __future__ import annotations

# Standard library
import json
from pathlib import Path
from typing import Any

# Third-party libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from networkx import DiGraph
from networkx.readwrite import json_graph

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    NUM_OF_FACES,
    ModuleFaces,
    ModuleInstance,
    ModuleRotationsIdx,
    ModuleType,
)

# Global constants
SEED = 42
DPI = 300

# Global functions
RNG = np.random.default_rng(SEED)


class HighProbabilityDecoder:
    """Implements the high-probability-decoding algorithm."""

    def __init__(self, num_modules: int) -> None:
        """
        Initialize the high-probability-decoding algorithm.

        Parameters
        ----------
        num_modules : int
            Number of modules to be decoded.
        """
        self.num_modules = num_modules

        # Data structure to hold the decoded graph (not networkx graph)
        self._graph: dict[int, ModuleInstance] = {}

        # NetworkX graph
        self.graph: DiGraph[Any] = nx.DiGraph()

    def probability_matrices_to_graph(
        self,
        type_probability_space: npt.NDArray[np.float32],
        connection_probability_space: npt.NDArray[np.float32],
        rotation_probability_space: npt.NDArray[np.float32],
    ) -> DiGraph[Any]:
        """
        Convert probability matrices to a graph.

        Parameters
        ----------
        type_probability_space
            Probability space for module types.
        connection_probability_space
            Probability space for connections between modules.
        rotation_probability_space
            Probability space for module rotations.

        Returns
        -------
        DiGraph
            A graph representing the decoded modules and their connections.
        """
        self.type_p_space = type_probability_space
        self.conn_p_space = connection_probability_space
        self.rot_p_space = rotation_probability_space

        # Apply constraints
        self.apply_connection_constraints()

        # Initialize module types and rotations
        self.set_module_types_and_rotations()

        # Decode probability spaces into a graph
        self.decode_probability_to_graph()

        # Create the final graph from the simple graph
        self.generate_networkx_graph()
        return self.graph

    def generate_networkx_graph(self) -> None:
        """Generate a NetworkX graph from the decoded graph."""
        for parent, module_instance in self._graph.items():
            self.graph.add_node(
                parent,
                type=module_instance.type.name,
                rotation=module_instance.rotation.name,
            )
            for face, child in module_instance.links.items():
                self.graph.add_node(
                    child,
                    type=self._graph[child].type.name,
                    rotation=self._graph[child].rotation.name,
                )

                self.graph.add_edge(
                    parent,
                    child,
                    face=face.name,
                )

    def decode_probability_to_graph(
        self,
    ) -> None:
        """Decode the probability spaces into a graph."""
        available_faces = np.zeros_like(self.conn_p_space)
        available_faces[IDX_OF_CORE, :, :] = 1.0
        selected_faces = np.zeros_like(self.conn_p_space)

        for _ in range(self.num_modules):
            # Contrast the connection probabilities with the available faces
            current_space = available_faces * self.conn_p_space

            # Get index of max values
            max_index = np.unravel_index(
                np.argmax(current_space),
                current_space.shape,
            )
            x, y, z = max_index

            # Get parent and child types and rotations
            parent_type = ModuleType(int(np.argmax(self.type_p_space[x])))
            parent_rotation = ModuleRotationsIdx(
                int(np.argmax(self.rot_p_space[x])),
            )
            child_type = ModuleType(int(np.argmax(self.type_p_space[y])))
            child_rotation = ModuleRotationsIdx(
                int(np.argmax(self.rot_p_space[y])),
            )

            # Get max value, and check if it is zero, if so, break
            max_value = current_space[x, y, z]
            if max_value == 0.0:
                break

            # Enable newly connected block
            available_faces[y, :, :] = 1.0

            # Avoid re-selection
            self.conn_p_space[x, :, z] = 0.0  # disable taken face
            self.conn_p_space[:, y, :] = 0.0  # child has only one parent

            # Update selected faces
            selected_faces[x, y, z] = 1.0

            # Update graph with new edge
            parent: int = int(x)
            child: int = int(y)
            face: int = int(z)

            # If the child is not in the final graph, add it
            if child not in self._graph:
                self._graph[child] = ModuleInstance(
                    type=child_type,
                    rotation=child_rotation,
                    links={},
                )

            # If the parent is not in the final graph, add it
            if parent not in self._graph:
                self._graph[parent] = ModuleInstance(
                    type=parent_type,
                    rotation=parent_rotation,
                    links={
                        ModuleFaces(face): child,
                    },
                )
            else:
                # If the parent is already in the graph, update its links
                self._graph[parent].links[ModuleFaces(face)] = child

    def set_module_types_and_rotations(self) -> None:
        """Set the module types and rotations using probability spaces."""
        for i in range(self.num_modules):
            # Update the type probability space
            module_type = ModuleType(int(np.argmax(self.type_p_space[i])))
            self.type_p_space[i, :] = 0.0
            self.type_p_space[i, module_type.value] = 1.0

            # Update the rotation probability space
            rotation_type = ModuleRotationsIdx(
                int(np.argmax(self.rot_p_space[i])),
            )
            self.rot_p_space[i, :] = 0.0
            self.rot_p_space[i, rotation_type.value] = 1.0

    def apply_connection_constraints(
        self,
    ) -> None:
        """Apply connection constraints to probability spaces."""
        # Self connection not allowed
        for i in range(NUM_OF_FACES):
            np.fill_diagonal(self.conn_p_space[:, :, i], 0.0)

        # Core is unique
        self.type_p_space[:, int(ModuleType.CORE.value)] = 0.0
        self.type_p_space[IDX_OF_CORE, int(ModuleType.CORE.value)] = 1.0

        # Core is always a parent, never a child
        self.conn_p_space[:, IDX_OF_CORE, :] = 0.0

        # Set the allowed faces for the module type
        conn_p_space_mask = np.zeros_like(
            self.conn_p_space,
        )

        # Set the allowed rotations for the module type
        rot_p_space_mask = np.zeros_like(
            self.rot_p_space,
        )

        # Face and rotation constraints
        for i in range(self.num_modules):
            # Get the type of the module
            module_type = ModuleType(int(np.argmax(self.type_p_space[i])))

            for face_i in ALLOWED_FACES[module_type]:
                conn_p_space_mask[i, :, face_i.value] = 1.0

            for rotation_i in ALLOWED_ROTATIONS[module_type]:
                rot_p_space_mask[i, rotation_i.value] = 1.0

        self.conn_p_space = np.multiply(
            self.conn_p_space,
            conn_p_space_mask,
        )
        self.rot_p_space = np.multiply(
            self.rot_p_space,
            rot_p_space_mask,
        )


def save_graph_as_json(
    graph: DiGraph[Any],
    save_file: Path | str | None = None,
) -> None:
    """
    Save a directed graph as a JSON file.

    Parameters
    ----------
    graph : DiGraph
        The directed graph to save.
    save_file : Path | str | None, optional
        The file path to save the graph JSON, by default None
    """
    if save_file is None:
        return

    data = json_graph.node_link_data(graph, edges="edges")
    json_string = json.dumps(data, indent=4)

    with Path(save_file).open("w", encoding="utf-8") as f:
        f.write(json_string)


def draw_graph(
    graph: DiGraph[Any],
    title: str = "NetworkX Directed Graph",
    save_file: Path | str | None = None,
) -> None:
    """
    Draw a directed graph.

    Parameters
    ----------
    graph : DiGraph
        The directed graph to draw.
    title : str
        The title of the graph.
    save_file : Path | str | None, optional
        The file path to save the graph image, by default None
    """
    plt.figure()

    pos = nx.spectral_layout(graph)

    pos = nx.spring_layout(graph, pos=pos, k=1, iterations=20, seed=SEED)

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=150,
        node_color="#FFFFFF00",
        edgecolors="blue",
        font_size=8,
        width=0.5,
    )

    edge_labels = nx.get_edge_attributes(graph, "face")

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_color="red",
        font_size=8,
    )

    plt.title(title)

    # Save the graph visualization
    if save_file:
        plt.savefig(save_file, dpi=DPI)
    else:
        # Show the plot
        plt.show()
