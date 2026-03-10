"""I/O helpers for tree genomes (JSON load/save and networkx conversion).
"""
from __future__ import annotations

from typing import Dict, Any
from .tree_genome import TreeGenome


def load_genome(path: str) -> TreeGenome:
    return TreeGenome.load_json(path)


def save_genome(genome: TreeGenome, path: str) -> None:
    genome.save_json(path)


def genome_to_networkx_dict(genome: TreeGenome) -> dict[str, Any]:
    return genome.to_dict()
