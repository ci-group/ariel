"""Validation utilities for tree genomes.

Functions here check required fields and enforce allowed faces/rotations.
"""
from __future__ import annotations

from typing import Any

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    IDX_OF_CORE,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)


def _to_int_node_id(node_id: int | str) -> int:
    return int(node_id) if isinstance(node_id, str) else node_id


def is_single_connected_tree(genome: dict[str, Any]) -> bool:
    """Verify that the genome forms a single connected tree with core as root.

    In this domain, the core (IDX_OF_CORE) must be the root and all nodes
    must be reachable from it.
    """
    nodes = genome.get("nodes", {})
    edges = genome.get("edges", [])

    if not nodes:
        return True  # Empty genome is trivially connected

    # normalize node IDs to ints so dicts with string keys (JSON) work correctly
    node_ids = {_to_int_node_id(k) for k in nodes.keys()}

    # Core must exist and be the root (no incoming edges)
    if IDX_OF_CORE not in node_ids:
        return False

    # Build adjacency list and check core has no parents
    adj_list = {nid: [] for nid in node_ids}
    has_parent = {nid: False for nid in node_ids}

    for e in edges:
        parent = _to_int_node_id(e["parent"])
        child = _to_int_node_id(e["child"])
        if parent in adj_list and child in adj_list:
            adj_list[parent].append(child)
            has_parent[child] = True

    # Core must have no parent
    if has_parent[IDX_OF_CORE]:
        return False

    # Check connectivity: all nodes reachable from core
    visited = set()
    stack = [IDX_OF_CORE]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(child for child in adj_list[node] if child not in visited)

    # All nodes must be visited
    return len(visited) == len(node_ids)


def validate_genome_dict(genome: dict[str, Any]) -> None:
    nodes = genome.get("nodes", {})
    edges = genome.get("edges", [])

    # nodes must include core at IDX_OF_CORE
    if str(IDX_OF_CORE) not in nodes and IDX_OF_CORE not in nodes:
        msg = f"Genome must contain core node with index {IDX_OF_CORE}"
        raise ValueError(msg)

    # Validate node fields
    for k, v in list(nodes.items()):
        nid = int(k) if isinstance(k, str) else k
        t = v.get("type")
        rot = v.get("rotation")
        if t not in ModuleType.__members__:
            msg = f"Node {nid} has invalid type '{t}'"
            raise ValueError(msg)
        if rot not in ModuleRotationsIdx.__members__:
            msg = f"Node {nid} has invalid rotation '{rot}'"
            raise ValueError(msg)

    # Validate edges
    occupied = {}  # (parent, face) -> child
    for e in edges:
        parent = e["parent"]
        child = e["child"]
        face = e["face"]
        # face must be valid
        if face not in ModuleFaces.__members__:
            msg = f"Edge parent={parent} child={child} has invalid face '{face}'"
            raise ValueError(msg)
        # skip edges whose parent node no longer exists (shouldn't happen
        # normally, but may after swaps); treat as invalid structure rather than
        # raising KeyError.
        if str(parent) not in nodes and parent not in nodes:
            msg = f"Edge has missing parent node {parent}"
            raise ValueError(msg)
        # parent type must allow that face
        ptype = nodes[str(parent)]["type"] if str(parent) in nodes else nodes[parent]["type"]
        allowed = [f.name for f in ALLOWED_FACES[ModuleType[ptype]]]
        if face not in allowed:
            msg = f"Face '{face}' not allowed for parent type '{ptype}'"
            raise ValueError(msg)
        # ensure a face is not occupied twice
        key = (parent, face)
        if key in occupied:
            msg = f"Parent {parent} already has child at face '{face}'"
            raise ValueError(msg)
        occupied[key] = child

    # Validate connectivity: must form a single connected tree with core as root
    if not is_single_connected_tree(genome):
        msg = "Genome does not form a single connected tree with core as root"
        raise ValueError(msg)
