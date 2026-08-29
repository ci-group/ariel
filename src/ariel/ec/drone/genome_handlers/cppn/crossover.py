"""NEAT-style crossover for CPPN networks."""

from __future__ import annotations

import copy
from collections import defaultdict, deque
from typing import Optional

import numpy as np

from .network import (
    CPPNNetwork,
    ConnectionGene,
    NodeGene,
    NodeType,
)


def crossover_cppn(
    net1: CPPNNetwork,
    net2: CPPNNetwork,
    fitness1: Optional[float],
    fitness2: Optional[float],
    rng: np.random.Generator,
    disable_gene_prob: float = 0.75,
) -> CPPNNetwork:
    """Produce a child network by NEAT-style aligned crossover.

    Matching genes (same innovation number) are inherited randomly from either
    parent.  Disjoint and excess genes are inherited only from the fitter
    parent.  When fitnesses are equal or unknown, each disjoint/excess gene
    is randomly included with 50% probability (per the original NEAT paper).

    Parameters
    ----------
    net1, net2 : CPPNNetwork
        Parent networks.
    fitness1, fitness2 : float | None
        Fitness of each parent.  ``None`` is treated as equal.
    rng : np.random.Generator
        Random number generator.
    disable_gene_prob : float
        Probability that a matching gene is disabled in the child when it is
        disabled in *either* parent.
    """
    keys1 = set(net1.connections.keys())
    keys2 = set(net2.connections.keys())
    all_keys = keys1 | keys2

    # Determine fitter parent (None / equal -> treat as equal)
    if fitness1 is None or fitness2 is None or fitness1 == fitness2:
        fitter = 0  # 0 = equal
    elif fitness1 > fitness2:
        fitter = 1
    else:
        fitter = 2

    max_innov1 = max(keys1, default=-1)
    max_innov2 = max(keys2, default=-1)

    child_connections: dict[int, ConnectionGene] = {}
    # Track which parent contributed each connection (for node inheritance)
    node_source: dict[int, CPPNNetwork] = {}  # node_id -> source network

    for inn in all_keys:
        in1 = inn in keys1
        in2 = inn in keys2

        if in1 and in2:
            # Matching gene — inherit randomly
            if rng.random() < 0.5:
                conn = copy.deepcopy(net1.connections[inn])
                src_net = net1
            else:
                conn = copy.deepcopy(net2.connections[inn])
                src_net = net2
            # Disable with probability if disabled in either parent
            if not net1.connections[inn].enabled or not net2.connections[inn].enabled:
                if rng.random() < disable_gene_prob:
                    conn.enabled = False
            child_connections[inn] = conn
            node_source.setdefault(conn.source_id, src_net)
            node_source.setdefault(conn.target_id, src_net)

        elif in1 and not in2:
            # Gene only in net1 — disjoint or excess
            if fitter == 1:
                include = True
            elif fitter == 0:
                include = rng.random() < 0.5
            else:
                include = False
            if include:
                conn = copy.deepcopy(net1.connections[inn])
                child_connections[inn] = conn
                node_source.setdefault(conn.source_id, net1)
                node_source.setdefault(conn.target_id, net1)

        else:  # in2 and not in1
            if fitter == 2:
                include = True
            elif fitter == 0:
                include = rng.random() < 0.5
            else:
                include = False
            if include:
                conn = copy.deepcopy(net2.connections[inn])
                child_connections[inn] = conn
                node_source.setdefault(conn.source_id, net2)
                node_source.setdefault(conn.target_id, net2)

    # Remove connections that would create cycles in the child.  This can
    # happen when disjoint/excess genes from different parents imply
    # contradictory topological orderings (e.g. A→B from one parent and
    # B→A from the other).
    child_connections = _remove_cyclic_connections(child_connections)

    # Collect required node IDs from inherited connections + all input/output
    # nodes from the fitter parent (or net1 when equal).
    required_node_ids: set[int] = set()
    for conn in child_connections.values():
        required_node_ids.add(conn.source_id)
        required_node_ids.add(conn.target_id)

    # Always include input and output nodes from the fitter parent
    if fitter == 2:
        ref_net = net2
    else:
        ref_net = net1
    for node in ref_net.nodes.values():
        if node.node_type in (NodeType.INPUT, NodeType.OUTPUT):
            required_node_ids.add(node.node_id)

    # Build child nodes — prefer the network that contributed the connection
    child_nodes: dict[int, NodeGene] = {}
    for nid in required_node_ids:
        src_net = node_source.get(nid)
        if src_net is not None and nid in src_net.nodes:
            child_nodes[nid] = copy.deepcopy(src_net.nodes[nid])
        elif nid in net1.nodes:
            child_nodes[nid] = copy.deepcopy(net1.nodes[nid])
        elif nid in net2.nodes:
            child_nodes[nid] = copy.deepcopy(net2.nodes[nid])
        # else: node referenced by connection but missing — skip (shouldn't happen)

    next_node_id = max(child_nodes.keys(), default=-1) + 1

    return CPPNNetwork(
        nodes=child_nodes,
        connections=child_connections,
        next_node_id=next_node_id,
    )


def _remove_cyclic_connections(
    connections: dict[int, ConnectionGene],
) -> dict[int, ConnectionGene]:
    """Drop enabled connections that participate in cycles (Kahn's algorithm).

    Disabled connections are ignored for cycle detection since they don't
    affect evaluation.  Among the connections that form a cycle, the ones
    with the highest innovation numbers (most recently evolved) are removed
    first, preserving older / more established structure.
    """
    enabled = {inn: c for inn, c in connections.items() if c.enabled}

    # Build adjacency and in-degree
    node_ids: set[int] = set()
    in_degree: dict[int, int] = defaultdict(int)
    children: dict[int, list[int]] = defaultdict(list)
    for c in enabled.values():
        node_ids.add(c.source_id)
        node_ids.add(c.target_id)
        children[c.source_id].append(c.target_id)
        in_degree.setdefault(c.source_id, 0)
        in_degree[c.target_id] += 1

    # Kahn's to find acyclic subset
    queue = deque(nid for nid in node_ids if in_degree.get(nid, 0) == 0)
    visited: set[int] = set()
    while queue:
        nid = queue.popleft()
        visited.add(nid)
        for child in children[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(visited) == len(node_ids):
        return connections  # no cycles

    # Nodes still with in_degree > 0 are in cycles.  Remove enabled
    # connections whose target is in a cycle, preferring to drop higher
    # innovation numbers first (newest genes).
    cycle_nodes = node_ids - visited
    to_drop: set[int] = set()
    # Iteratively remove connections until acyclic
    remaining = dict(enabled)
    while True:
        # Rebuild in-degree for remaining
        in_deg: dict[int, int] = defaultdict(int)
        ch: dict[int, list[tuple[int, int]]] = defaultdict(list)  # node -> [(inn, target)]
        r_nodes: set[int] = set()
        for inn, c in remaining.items():
            r_nodes.add(c.source_id)
            r_nodes.add(c.target_id)
            ch[c.source_id].append((inn, c.target_id))
            in_deg.setdefault(c.source_id, 0)
            in_deg[c.target_id] += 1

        q = deque(n for n in r_nodes if in_deg.get(n, 0) == 0)
        vis: set[int] = set()
        while q:
            n = q.popleft()
            vis.add(n)
            for inn, tgt in ch[n]:
                in_deg[tgt] -= 1
                if in_deg[tgt] == 0:
                    q.append(tgt)

        if len(vis) == len(r_nodes):
            break  # acyclic now

        # Drop the highest-innovation enabled connection in the cycle
        cycle_conns = [
            inn for inn, c in remaining.items()
            if c.source_id not in vis or c.target_id not in vis
        ]
        if not cycle_conns:
            break
        drop = max(cycle_conns)
        to_drop.add(drop)
        del remaining[drop]

    # Build result: keep all original connections except dropped ones
    return {inn: c for inn, c in connections.items() if inn not in to_drop}
