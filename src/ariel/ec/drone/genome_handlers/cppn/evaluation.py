"""Feed-forward evaluation of a CPPN network."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List

import numpy as np

from .network import CPPNNetwork, NodeType, apply_activation


def topological_sort(network: CPPNNetwork) -> List[int]:
    """Return node IDs in topological (feed-forward) evaluation order.

    Uses Kahn's algorithm on enabled connections only.

    Raises:
        ValueError: If a cycle is detected among enabled connections.
    """
    enabled = network.get_enabled_connections()

    # Build adjacency and in-degree info for nodes reachable via enabled edges
    in_degree: Dict[int, int] = {nid: 0 for nid in network.nodes}
    children: Dict[int, List[int]] = defaultdict(list)

    for conn in enabled:
        children[conn.source_id].append(conn.target_id)
        in_degree[conn.target_id] += 1

    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    order: List[int] = []

    while queue:
        nid = queue.popleft()
        order.append(nid)
        for child in children[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(network.nodes):
        raise ValueError(
            "Cycle detected in CPPN network; topological sort failed."
        )
    return order


def evaluate_cppn(
    network: CPPNNetwork,
    inputs: np.ndarray,
) -> np.ndarray:
    """Evaluate the CPPN in feed-forward mode.

    Args:
        network: The CPPN to evaluate.
        inputs: Array of shape ``(n_inputs,)`` or ``(batch, n_inputs)``.

    Returns:
        Array of shape ``(n_outputs,)`` or ``(batch, n_outputs)`` with output
        node values ordered by ``output_index``.
    """
    inputs = np.asarray(inputs, dtype=np.float64)
    single = inputs.ndim == 1
    if single:
        inputs = inputs[np.newaxis, :]  # (1, n_inputs)

    batch_size = inputs.shape[0]

    # Map input nodes to columns of ``inputs``
    input_nodes = network.get_input_nodes()  # sorted by node_id
    output_nodes = network.get_output_nodes()  # sorted by output_index

    activations: Dict[int, np.ndarray] = {}
    for idx, node in enumerate(input_nodes):
        activations[node.node_id] = inputs[:, idx]  # (batch,)

    # Build incoming connections index
    incoming: Dict[int, list] = defaultdict(list)
    for conn in network.get_enabled_connections():
        incoming[conn.target_id].append(conn)

    order = topological_sort(network)

    for nid in order:
        node = network.nodes[nid]
        if node.node_type == NodeType.INPUT:
            continue  # already set

        # Weighted sum of incoming activations + bias
        total = np.full(batch_size, node.bias, dtype=np.float64)
        for conn in incoming[nid]:
            if conn.source_id in activations:
                total += conn.weight * activations[conn.source_id]

        activations[nid] = apply_activation(node.activation, total)

    # Gather outputs in order
    result = np.column_stack(
        [activations.get(n.node_id, np.zeros(batch_size)) for n in output_nodes]
    )

    if single:
        return result[0]  # (n_outputs,)
    return result
