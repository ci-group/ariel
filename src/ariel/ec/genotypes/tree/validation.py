"""Validation utilities for tree genomes.

Functions here check required fields and enforce allowed faces/rotations.
"""
from __future__ import annotations

from typing import Dict, List

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
    IDX_OF_CORE,
)


def validate_genome_dict(genome: Dict) -> None:
    nodes = genome.get("nodes", {})
    edges = genome.get("edges", [])

    # nodes must include core at IDX_OF_CORE
    if str(IDX_OF_CORE) not in nodes and IDX_OF_CORE not in nodes:
        raise ValueError(f"Genome must contain core node with index {IDX_OF_CORE}")

    # Validate node fields
    for k, v in list(nodes.items()):
        nid = int(k) if isinstance(k, str) else k
        t = v.get("type")
        rot = v.get("rotation")
        if t not in ModuleType.__members__:
            raise ValueError(f"Node {nid} has invalid type '{t}'")
        if rot not in ModuleRotationsIdx.__members__:
            raise ValueError(f"Node {nid} has invalid rotation '{rot}'")

    # Validate edges
    occupied = {}  # (parent, face) -> child
    for e in edges:
        parent = e["parent"]
        child = e["child"]
        face = e["face"]
        # face must be valid
        if face not in ModuleFaces.__members__:
            raise ValueError(f"Edge parent={parent} child={child} has invalid face '{face}'")
        # skip edges whose parent node no longer exists (shouldn't happen
        # normally, but may after swaps); treat as invalid structure rather than
        # raising KeyError.
        if str(parent) not in nodes and parent not in nodes:
            raise ValueError(f"Edge has missing parent node {parent}")
        # parent type must allow that face
        ptype = nodes[str(parent)]["type"] if str(parent) in nodes else nodes[parent]["type"]
        allowed = [f.name for f in ALLOWED_FACES[ModuleType[ptype]]]
        if face not in allowed:
            raise ValueError(f"Face '{face}' not allowed for parent type '{ptype}'")
        # ensure a face is not occupied twice
        key = (parent, face)
        if key in occupied:
            raise ValueError(f"Parent {parent} already has child at face '{face}'")
        occupied[key] = child


def is_single_connected_tree(genome: Dict) -> bool:
    """Verify that the genome forms a single connected tree with core as root.

    In this domain, the core (IDX_OF_CORE) must be the root and all nodes
    must be reachable from it.
    """
    nodes = genome.get("nodes", {})
    edges = genome.get("edges", [])

    if not nodes:
        return True  # Empty genome is trivially connected

    # Core must exist and be the root (no incoming edges)
    if IDX_OF_CORE not in nodes:
        return False

    # Build adjacency list and check core has no parents
    adj_list = {int(k) if isinstance(k, str) else k: [] for k in nodes.keys()}
    has_parent = {int(k) if isinstance(k, str) else k: False for k in nodes.keys()}

    for e in edges:
        parent = e["parent"]
        child = e["child"]
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
        for child in adj_list[node]:
            if child not in visited:
                stack.append(child)

    # All nodes must be visited
    return len(visited) == len(nodes)


def validate_genome_dict(genome: Dict) -> None:
    nodes = genome.get("nodes", {})
    edges = genome.get("edges", [])

    # nodes must include core at IDX_OF_CORE
    if str(IDX_OF_CORE) not in nodes and IDX_OF_CORE not in nodes:
        raise ValueError(f"Genome must contain core node with index {IDX_OF_CORE}")

    # Validate node fields
    for k, v in list(nodes.items()):
        nid = int(k) if isinstance(k, str) else k
        t = v.get("type")
        rot = v.get("rotation")
        if t not in ModuleType.__members__:
            raise ValueError(f"Node {nid} has invalid type '{t}'")
        if rot not in ModuleRotationsIdx.__members__:
            raise ValueError(f"Node {nid} has invalid rotation '{rot}'")

    # Validate edges
    occupied = {}  # (parent, face) -> child
    for e in edges:
        parent = e["parent"]
        child = e["child"]
        face = e["face"]
        # face must be valid
        if face not in ModuleFaces.__members__:
            raise ValueError(f"Edge parent={parent} child={child} has invalid face '{face}'")
        # skip edges whose parent node no longer exists (shouldn't happen
        # normally, but may after swaps); treat as invalid structure rather than
        # raising KeyError.
        if str(parent) not in nodes and parent not in nodes:
            raise ValueError(f"Edge has missing parent node {parent}")
        # parent type must allow that face
        ptype = nodes[str(parent)]["type"] if str(parent) in nodes else nodes[parent]["type"]
        allowed = [f.name for f in ALLOWED_FACES[ModuleType[ptype]]]
        if face not in allowed:
            raise ValueError(f"Face '{face}' not allowed for parent type '{ptype}'")
        # ensure a face is not occupied twice
        key = (parent, face)
        if key in occupied:
            raise ValueError(f"Parent {parent} already has child at face '{face}'")
        occupied[key] = child

    # Validate connectivity: must form a single connected tree with core as root
    if not is_single_connected_tree(genome):
        raise ValueError("Genome does not form a single connected tree with core as root")
