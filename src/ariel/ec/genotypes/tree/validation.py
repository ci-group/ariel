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
