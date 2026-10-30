"""Test: TreeGenome dataclass — construction, conversion, serialization."""

import json
import tempfile
from pathlib import Path

import networkx as nx
import pytest

from ariel.ec.genotypes.tree.tree_genome import TreeGenome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_genome() -> TreeGenome:
    """Core + one hinge child."""
    g = TreeGenome()
    g.nodes = {
        0: {"type": "CORE", "rotation": "DEG_0"},
        1: {"type": "HINGE", "rotation": "DEG_0"},
    }
    g.edges = [{"parent": 0, "child": 1, "face": "FRONT"}]
    return g


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_tree_genome_default_empty() -> None:
    """Default TreeGenome has empty nodes and edges."""
    g = TreeGenome()
    assert g.nodes == {}
    assert g.edges == []


def test_tree_genome_with_data() -> None:
    """TreeGenome stores nodes and edges correctly."""
    g = _simple_genome()
    assert 0 in g.nodes
    assert 1 in g.nodes
    assert len(g.edges) == 1


# ---------------------------------------------------------------------------
# to_networkx
# ---------------------------------------------------------------------------


def test_to_networkx_returns_digraph() -> None:
    """to_networkx returns a NetworkX DiGraph."""
    g = _simple_genome()
    nxg = g.to_networkx()
    assert isinstance(nxg, nx.DiGraph)


def test_to_networkx_node_count() -> None:
    """DiGraph has the same number of nodes as TreeGenome."""
    g = _simple_genome()
    nxg = g.to_networkx()
    assert len(nxg.nodes) == 2


def test_to_networkx_edge_data() -> None:
    """Edge data carries the face attribute."""
    g = _simple_genome()
    nxg = g.to_networkx()
    assert nxg[0][1]["face"] == "FRONT"


def test_to_networkx_node_attributes() -> None:
    """Node attributes (type, rotation) are preserved."""
    g = _simple_genome()
    nxg = g.to_networkx()
    assert nxg.nodes[0]["type"] == "CORE"
    assert nxg.nodes[1]["rotation"] == "DEG_0"


# ---------------------------------------------------------------------------
# to_dict / from_dict
# ---------------------------------------------------------------------------


def test_to_dict_keys() -> None:
    """to_dict returns a dict with 'nodes' and 'edges' keys."""
    d = _simple_genome().to_dict()
    assert "nodes" in d
    assert "edges" in d


def test_to_dict_nodes_string_keyed() -> None:
    """to_dict converts integer node IDs to strings."""
    d = _simple_genome().to_dict()
    assert "0" in d["nodes"]
    assert "1" in d["nodes"]


def test_from_dict_roundtrip() -> None:
    """from_dict(to_dict()) reproduces the original genome."""
    original = _simple_genome()
    reconstructed = TreeGenome.from_dict(original.to_dict())
    assert reconstructed.nodes == original.nodes
    assert reconstructed.edges == original.edges


def test_from_dict_string_keys() -> None:
    """from_dict handles string-keyed nodes (JSON style)."""
    data = {
        "nodes": {
            "0": {"type": "CORE", "rotation": "DEG_0"},
            "2": {"type": "BRICK", "rotation": "DEG_45"},
        },
        "edges": [{"parent": 0, "child": 2, "face": "BACK"}],
    }
    g = TreeGenome.from_dict(data)
    assert 0 in g.nodes
    assert 2 in g.nodes


def test_from_dict_list_nodes() -> None:
    """from_dict handles the list-of-dicts node format."""
    data = {
        "nodes": [
            {"id": 0, "type": "CORE", "rotation": "DEG_0"},
            {"id": 1, "type": "HINGE", "rotation": "DEG_90"},
        ],
        "edges": [],
    }
    g = TreeGenome.from_dict(data)
    assert g.nodes[0]["type"] == "CORE"
    assert g.nodes[1]["type"] == "HINGE"


# ---------------------------------------------------------------------------
# save_json / load_json
# ---------------------------------------------------------------------------


def test_save_and_load_json_roundtrip() -> None:
    """save_json followed by load_json reproduces the genome."""
    original = _simple_genome()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "genome.json")
        original.save_json(path)
        loaded = TreeGenome.load_json(path)
    assert loaded.nodes == original.nodes
    assert loaded.edges == original.edges


def test_save_json_valid_json() -> None:
    """save_json writes valid JSON to disk."""
    genome = _simple_genome()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "genome.json")
        genome.save_json(path)
        with open(path) as fh:
            data = json.load(fh)
    assert "nodes" in data
