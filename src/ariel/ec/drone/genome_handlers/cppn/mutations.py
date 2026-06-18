"""NEAT-style mutation operators for CPPN networks."""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional

import numpy as np

from .network import (
    ActivationFunction,
    CPPNNetwork,
    ConnectionGene,
    NodeGene,
    NodeType,
)
from .innovation import InnovationCounter


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def mutate_cppn(
    network: CPPNNetwork,
    innovation_counter: InnovationCounter,
    rng: np.random.Generator,
    prob_add_node: float = 0.03,
    prob_add_connection: float = 0.05,
    prob_remove_node: float = 0.01,
    prob_remove_connection: float = 0.02,
    prob_mutate_weights: float = 0.80,
    prob_mutate_activation: float = 0.05,
    prob_toggle_connection: float = 0.02,
    weight_perturb_std: float = 0.5,
    weight_replace_prob: float = 0.1,
    weight_range: float = 3.0,
    bias_perturb_std: float = 0.3,
    bias_replace_prob: float = 0.1,
    bias_range: float = 3.0,
) -> None:
    """Select and apply one mutation to *network* in-place."""
    probs = np.array([
        prob_add_node,
        prob_add_connection,
        prob_remove_node,
        prob_remove_connection,
        prob_mutate_weights,
        prob_mutate_activation,
        prob_toggle_connection,
    ])
    # Remainder goes to "no mutation"
    remainder = max(0.0, 1.0 - probs.sum())
    probs = np.append(probs, remainder)
    probs /= probs.sum()  # normalise to handle floating-point drift

    choice = rng.choice(len(probs), p=probs)

    if choice == 0:
        _add_node(network, innovation_counter, rng)
    elif choice == 1:
        _add_connection(network, innovation_counter, rng, weight_range=weight_range)
    elif choice == 2:
        _remove_node(network, rng)
    elif choice == 3:
        _remove_connection(network, rng)
    elif choice == 4:
        _mutate_weights(
            network, rng,
            perturb_std=weight_perturb_std,
            replace_prob=weight_replace_prob,
            weight_range=weight_range,
            bias_perturb_std=bias_perturb_std,
            bias_replace_prob=bias_replace_prob,
            bias_range=bias_range,
        )
    elif choice == 5:
        _mutate_activation(network, rng)
    elif choice == 6:
        _toggle_connection(network, rng)
    # else: no mutation


# ---------------------------------------------------------------------------
# Individual mutation operators
# ---------------------------------------------------------------------------

def _add_node(
    network: CPPNNetwork,
    innovation_counter: InnovationCounter,
    rng: np.random.Generator,
) -> None:
    """Split a random enabled connection by inserting a hidden node."""
    enabled = network.get_enabled_connections()
    if not enabled:
        return

    conn = enabled[rng.integers(len(enabled))]
    conn.enabled = False

    new_id = network.next_node_id
    network.next_node_id += 1

    activation = rng.choice(list(ActivationFunction))
    new_node = NodeGene(
        node_id=new_id,
        node_type=NodeType.HIDDEN,
        activation=activation,
        bias=0.0,
    )
    network.nodes[new_id] = new_node

    inn1 = innovation_counter.get_innovation(conn.source_id, new_id)
    inn2 = innovation_counter.get_innovation(new_id, conn.target_id)

    network.connections[inn1] = ConnectionGene(
        innovation_number=inn1,
        source_id=conn.source_id,
        target_id=new_id,
        weight=1.0,
        enabled=True,
    )
    network.connections[inn2] = ConnectionGene(
        innovation_number=inn2,
        source_id=new_id,
        target_id=conn.target_id,
        weight=conn.weight,
        enabled=True,
    )


def _add_connection(
    network: CPPNNetwork,
    innovation_counter: InnovationCounter,
    rng: np.random.Generator,
    max_attempts: int = 50,
    weight_range: float = 3.0,
) -> None:
    """Add a new connection between two previously unconnected nodes."""
    node_ids = list(network.nodes.keys())
    existing_pairs = {
        (c.source_id, c.target_id) for c in network.connections.values()
    }

    for _ in range(max_attempts):
        src_id = node_ids[rng.integers(len(node_ids))]
        tgt_id = node_ids[rng.integers(len(node_ids))]

        if src_id == tgt_id:
            continue
        src_node = network.nodes[src_id]
        tgt_node = network.nodes[tgt_id]
        if src_node.node_type == NodeType.OUTPUT:
            continue
        if tgt_node.node_type == NodeType.INPUT:
            continue
        if (src_id, tgt_id) in existing_pairs:
            continue
        if _would_create_cycle(network, src_id, tgt_id):
            continue

        inn = innovation_counter.get_innovation(src_id, tgt_id)
        network.connections[inn] = ConnectionGene(
            innovation_number=inn,
            source_id=src_id,
            target_id=tgt_id,
            weight=rng.uniform(-weight_range, weight_range),
            enabled=True,
        )
        return


def _remove_node(network: CPPNNetwork, rng: np.random.Generator) -> None:
    """Remove a random hidden node and all connections involving it."""
    hidden = network.get_hidden_nodes()
    if not hidden:
        return

    node = hidden[rng.integers(len(hidden))]
    nid = node.node_id

    # Remove connections involving the node
    to_remove = [
        inn for inn, c in network.connections.items()
        if c.source_id == nid or c.target_id == nid
    ]
    for inn in to_remove:
        del network.connections[inn]

    del network.nodes[nid]


def _remove_connection(network: CPPNNetwork, rng: np.random.Generator) -> None:
    """Disable a random enabled connection."""
    enabled = network.get_enabled_connections()
    if not enabled:
        return
    conn = enabled[rng.integers(len(enabled))]
    conn.enabled = False


def _mutate_weights(
    network: CPPNNetwork,
    rng: np.random.Generator,
    perturb_std: float = 0.5,
    replace_prob: float = 0.1,
    weight_range: float = 3.0,
    bias_perturb_std: float = 0.3,
    bias_replace_prob: float = 0.1,
    bias_range: float = 3.0,
) -> None:
    """Perturb or replace all connection weights and node biases."""
    for conn in network.connections.values():
        if rng.random() < replace_prob:
            conn.weight = rng.uniform(-weight_range, weight_range)
        else:
            conn.weight += rng.normal(0.0, perturb_std)
            conn.weight = np.clip(conn.weight, -weight_range, weight_range)

    for node in network.nodes.values():
        if node.node_type == NodeType.INPUT:
            continue
        if rng.random() < bias_replace_prob:
            node.bias = rng.uniform(-bias_range, bias_range)
        else:
            node.bias += rng.normal(0.0, bias_perturb_std)
            node.bias = np.clip(node.bias, -bias_range, bias_range)


def _mutate_activation(network: CPPNNetwork, rng: np.random.Generator) -> None:
    """Change the activation function of a random hidden node."""
    hidden = network.get_hidden_nodes()
    if not hidden:
        return

    node = hidden[rng.integers(len(hidden))]
    options = [a for a in ActivationFunction if a != node.activation]
    if options:
        node.activation = options[rng.integers(len(options))]


def _toggle_connection(network: CPPNNetwork, rng: np.random.Generator) -> None:
    """Toggle enabled/disabled on a random connection."""
    conns = list(network.connections.values())
    if not conns:
        return
    conn = conns[rng.integers(len(conns))]
    if not conn.enabled and _would_create_cycle(network, conn.source_id, conn.target_id):
        return  # Don't re-enable if it would create a cycle
    conn.enabled = not conn.enabled


# ---------------------------------------------------------------------------
# Cycle detection helper
# ---------------------------------------------------------------------------

def _would_create_cycle(
    network: CPPNNetwork,
    source_id: int,
    target_id: int,
) -> bool:
    """Return True if adding source→target would create a cycle (DFS)."""
    # Check: is *source_id* reachable from *target_id* via enabled connections?
    visited = set()
    stack = [target_id]
    children = defaultdict(list)
    for conn in network.get_enabled_connections():
        children[conn.source_id].append(conn.target_id)

    while stack:
        nid = stack.pop()
        if nid == source_id:
            return True
        if nid in visited:
            continue
        visited.add(nid)
        stack.extend(children[nid])

    return False
