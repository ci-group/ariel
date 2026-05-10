"""I/O helpers for tree genomes (JSON load/save and networkx conversion)."""
from __future__ import annotations

from typing import Any

from .tree_genome import TreeGenome


def load_genome(path: str) -> TreeGenome:
    """Load a TreeGenome from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file containing the genome.

    Returns
    -------
    TreeGenome
        The loaded TreeGenome instance.
    """
    return TreeGenome.load_json(path)


def save_genome(genome: TreeGenome, path: str) -> None:
    """Save a TreeGenome to a JSON file.

    Parameters
    ----------
    genome : TreeGenome
        The TreeGenome instance to save.
    genome.save_json(path)
    """


def genome_to_networkx_dict(genome: TreeGenome) -> dict[str, Any]:
    """Convert a TreeGenome to a dictionary format compatible with networkx.

    Parameters
    ----------
    genome : TreeGenome
        The TreeGenome instance to convert.

    Returns
    -------
    dict[str, Any]
        A dictionary representation of the genome suitable for networkx.
    """
    return genome.to_dict()
