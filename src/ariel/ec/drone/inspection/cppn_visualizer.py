"""Visualization of CPPN network graphs.

Follows the DroneVisualizer / VisualizationConfig pattern from drone_visualizer.py.
Renders the CPPN as a layered directed graph with node-type coloring, weight-scaled
connections, and optional activation/weight/bias labels.
"""

from __future__ import annotations

import copy
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from ariel.ec.drone.genome_handlers.cppn.network import (
    CPPNNetwork,
    NodeType,
)
from ariel.ec.drone.genome_handlers.cppn.evaluation import topological_sort


# Short display names for activation functions
_ACTIVATION_SHORT = {
    "identity": "id",
    "sigmoid": "sig",
    "tanh": "tanh",
    "sin": "sin",
    "cos": "cos",
    "gaussian": "gauss",
    "abs": "abs",
    "relu": "relu",
    "step": "step",
}


@dataclass
class CPPNVisualizationConfig:
    """Configuration for CPPN network graph visualization."""

    # Node colors by type
    input_node_color: str = "#66bb6a"   # green
    hidden_node_color: str = "#ffa726"  # orange
    output_node_color: str = "#42a5f5"  # blue

    # Connection colors by weight sign
    positive_weight_color: str = "#1565c0"  # dark blue
    negative_weight_color: str = "#c62828"  # dark red
    disabled_connection_color: str = "#9e9e9e"  # gray

    # Connection line width scaled by |weight|
    min_line_width: float = 0.5
    max_line_width: float = 4.0

    # Disabled connections
    disabled_alpha: float = 0.3
    disabled_linestyle: str = "--"

    # Layout params
    layer_spacing: float = 2.0
    node_vertical_spacing: float = 1.2
    node_radius: float = 0.35

    # Label toggles
    show_activation_labels: bool = True
    show_weight_labels: bool = False
    show_bias_values: bool = False

    # Legend
    show_legend: bool = True

    # Figure
    figsize: Tuple[float, float] = (12, 8)
    title_fontsize: int = 14
    label_fontsize: int = 9
    weight_fontsize: int = 7


class CPPNVisualizer:
    """Renders a CPPN network as a layered directed graph."""

    def __init__(self, config: Optional[CPPNVisualizationConfig] = None) -> None:
        self.config = config if config is not None else CPPNVisualizationConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_network(
        self,
        network_or_handler: Any,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        show_disabled: bool = True,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the CPPN network graph.

        Args:
            network_or_handler: A ``CPPNNetwork`` instance, or any object with
                a ``.genome`` attribute that is a ``CPPNNetwork`` (duck-typed).
            ax: Existing matplotlib axes. Creates a new figure if *None*.
            title: Optional title.
            show_disabled: Whether to draw disabled connections.
            **kwargs: Override any ``CPPNVisualizationConfig`` field.

        Returns:
            ``(fig, ax)`` tuple.
        """
        network = self._extract_network(network_or_handler)
        config = self._apply_config_overrides(**kwargs)

        if ax is None:
            fig, ax = plt.subplots(figsize=config.figsize)
        else:
            fig = ax.figure

        # Compute layout
        positions, layers = self._compute_positions(network, config)

        # Draw connections first (behind nodes)
        self._draw_connections(ax, network, positions, config, show_disabled)

        # Draw nodes on top
        self._draw_nodes(ax, network, positions, config)

        # Legend
        if config.show_legend:
            self._draw_legend(ax, config)

        # Title
        if title:
            ax.set_title(title, fontsize=config.title_fontsize)

        ax.set_aspect("equal")
        ax.axis("off")
        ax.autoscale_view()
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_network(obj: Any) -> CPPNNetwork:
        if isinstance(obj, CPPNNetwork):
            return obj
        if hasattr(obj, "genome"):
            return obj.genome
        raise TypeError(
            f"Expected CPPNNetwork or an object with a .genome attribute, got {type(obj)}"
        )

    def _apply_config_overrides(self, **kwargs) -> CPPNVisualizationConfig:
        config = copy.deepcopy(self.config)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    # ---- layer assignment ----

    @staticmethod
    def _compute_layers(network: CPPNNetwork) -> Dict[int, int]:
        """Assign each node to a layer using longest-path depth from inputs."""
        input_ids = {n.node_id for n in network.get_input_nodes()}
        output_ids = {n.node_id for n in network.get_output_nodes()}

        # Build adjacency from enabled connections
        children: Dict[int, List[int]] = defaultdict(list)
        for conn in network.get_enabled_connections():
            children[conn.source_id].append(conn.target_id)

        # BFS / longest-path depth from inputs
        depth: Dict[int, int] = {nid: 0 for nid in input_ids}
        queue = deque(input_ids)
        while queue:
            nid = queue.popleft()
            for child in children[nid]:
                new_depth = depth[nid] + 1
                if child not in depth or new_depth > depth[child]:
                    depth[child] = new_depth
                    queue.append(child)

        # Assign any unreachable nodes (isolated hidden/output) a depth
        for nid in network.nodes:
            if nid not in depth:
                depth[nid] = 1

        # Force output nodes to max(hidden_depths) + 1
        hidden_depths = [
            depth[nid] for nid in depth if nid not in input_ids and nid not in output_ids
        ]
        output_layer = (max(hidden_depths) + 1) if hidden_depths else 1
        for nid in output_ids:
            depth[nid] = output_layer

        # Force input nodes to layer 0
        for nid in input_ids:
            depth[nid] = 0

        return depth

    def _compute_positions(
        self, network: CPPNNetwork, config: CPPNVisualizationConfig
    ) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, int]]:
        """Return ``{node_id: (x, y)}`` and ``{node_id: layer}``."""
        layers = self._compute_layers(network)

        # Group nodes by layer
        layer_nodes: Dict[int, List[int]] = defaultdict(list)
        for nid, layer in layers.items():
            layer_nodes[layer].append(nid)

        # Sort within each layer
        input_nodes = {n.node_id for n in network.get_input_nodes()}
        output_nodes = {n.node_id: n.output_index for n in network.get_output_nodes()}

        for layer_idx in layer_nodes:
            nids = layer_nodes[layer_idx]
            if all(nid in output_nodes for nid in nids):
                nids.sort(key=lambda n: output_nodes.get(n, 0))
            else:
                nids.sort()

        # Compute positions
        positions: Dict[int, Tuple[float, float]] = {}
        for layer_idx, nids in layer_nodes.items():
            x = layer_idx * config.layer_spacing
            n = len(nids)
            y_start = -(n - 1) / 2.0 * config.node_vertical_spacing
            for i, nid in enumerate(nids):
                positions[nid] = (x, y_start + i * config.node_vertical_spacing)

        return positions, layers

    # ---- drawing helpers ----

    def _draw_connections(
        self,
        ax: plt.Axes,
        network: CPPNNetwork,
        positions: Dict[int, Tuple[float, float]],
        config: CPPNVisualizationConfig,
        show_disabled: bool,
    ) -> None:
        # Count parallel edges between the same pair to offset curvature
        pair_count: Dict[Tuple[int, int], int] = defaultdict(int)
        pair_index: Dict[Tuple[int, int], int] = defaultdict(int)
        for conn in network.connections.values():
            pair = (min(conn.source_id, conn.target_id), max(conn.source_id, conn.target_id))
            pair_count[pair] += 1

        for conn in network.connections.values():
            if not conn.enabled and not show_disabled:
                continue
            if conn.source_id not in positions or conn.target_id not in positions:
                continue

            src = positions[conn.source_id]
            dst = positions[conn.target_id]

            # Curvature for parallel edges
            pair = (min(conn.source_id, conn.target_id), max(conn.source_id, conn.target_id))
            idx = pair_index[pair]
            pair_index[pair] += 1
            total = pair_count[pair]
            if total > 1:
                rad = 0.15 * (idx - (total - 1) / 2.0)
            else:
                rad = 0.0

            # Color and style
            if not conn.enabled:
                color = config.disabled_connection_color
                alpha = config.disabled_alpha
                linestyle = config.disabled_linestyle
            elif conn.weight >= 0:
                color = config.positive_weight_color
                alpha = 0.8
                linestyle = "-"
            else:
                color = config.negative_weight_color
                alpha = 0.8
                linestyle = "-"

            # Line width scaled by |weight|
            weight_abs = abs(conn.weight)
            lw = config.min_line_width + (config.max_line_width - config.min_line_width) * min(
                weight_abs / 3.0, 1.0
            )

            arrow = FancyArrowPatch(
                posA=src,
                posB=dst,
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={rad}",
                color=color,
                alpha=alpha,
                linewidth=lw,
                linestyle=linestyle,
                mutation_scale=12,
                zorder=1,
            )
            ax.add_patch(arrow)

            # Weight labels
            if config.show_weight_labels and conn.enabled:
                mx = (src[0] + dst[0]) / 2.0
                my = (src[1] + dst[1]) / 2.0 + rad * 0.5
                ax.text(
                    mx, my, f"{conn.weight:.2f}",
                    fontsize=config.weight_fontsize,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7),
                    zorder=3,
                )

    def _draw_nodes(
        self,
        ax: plt.Axes,
        network: CPPNNetwork,
        positions: Dict[int, Tuple[float, float]],
        config: CPPNVisualizationConfig,
    ) -> None:
        for node in network.nodes.values():
            if node.node_id not in positions:
                continue
            x, y = positions[node.node_id]

            # Color by type
            if node.node_type == NodeType.INPUT:
                color = config.input_node_color
            elif node.node_type == NodeType.OUTPUT:
                color = config.output_node_color
            else:
                color = config.hidden_node_color

            circle = plt.Circle(
                (x, y), config.node_radius,
                facecolor=color, edgecolor="black", linewidth=1.2, zorder=4,
            )
            ax.add_patch(circle)

            # Label
            label = self._node_label(node, config)
            ax.text(
                x, y, label,
                fontsize=config.label_fontsize,
                ha="center", va="center",
                fontweight="bold", zorder=5,
            )

            # Bias annotation
            if config.show_bias_values and node.node_type != NodeType.INPUT:
                ax.text(
                    x, y - config.node_radius - 0.15,
                    f"b={node.bias:.2f}",
                    fontsize=config.weight_fontsize,
                    ha="center", va="top", color="gray",
                    zorder=5,
                )

    @staticmethod
    def _node_label(node, config: CPPNVisualizationConfig) -> str:
        if node.node_type == NodeType.INPUT:
            return node.input_label or f"in{node.node_id}"
        if node.node_type == NodeType.OUTPUT:
            # The handler stores output labels in _OUTPUT_LABELS but we don't import it;
            # fall back to output_index
            return f"out{node.output_index}" if node.output_index is not None else f"o{node.node_id}"
        # Hidden: show activation short name
        if config.show_activation_labels:
            return _ACTIVATION_SHORT.get(node.activation.value, node.activation.value)
        return str(node.node_id)

    def _draw_legend(self, ax: plt.Axes, config: CPPNVisualizationConfig) -> None:
        handles = [
            mpatches.Patch(facecolor=config.input_node_color, edgecolor="black", label="Input"),
            mpatches.Patch(facecolor=config.hidden_node_color, edgecolor="black", label="Hidden"),
            mpatches.Patch(facecolor=config.output_node_color, edgecolor="black", label="Output"),
            plt.Line2D([0], [0], color=config.positive_weight_color, lw=2, label="+weight"),
            plt.Line2D([0], [0], color=config.negative_weight_color, lw=2, label="-weight"),
            plt.Line2D(
                [0], [0], color=config.disabled_connection_color,
                lw=1, linestyle="--", label="Disabled",
            ),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=config.label_fontsize, framealpha=0.8)
