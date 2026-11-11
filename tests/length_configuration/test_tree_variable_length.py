"""Focused tests for tree-genome preservation of variable brick attributes."""

import networkx as nx
import pytest

from ariel.body_phenotypes.robogen_lite.config import (
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome


def _graph_with_length(
    length: float,
) -> nx.DiGraph:
    graph = nx.DiGraph()

    graph.add_node(
        0,
        type=ModuleType.CORE.name,
        rotation=ModuleRotationsIdx.DEG_0.name,
    )

    graph.add_node(
        1,
        type=ModuleType.BRICK.name,
        rotation=ModuleRotationsIdx.DEG_0.name,
        length=length,
        custom_marker="preserve-me",
    )

    graph.add_edge(
        0,
        1,
        face=ModuleFaces.FRONT.name,
    )

    return graph


def test_tree_genome_round_trip_preserves_length() -> None:
    """TreeGenome conversion must not discard the brick length attribute."""
    original = _graph_with_length(
        0.180
    )

    genome = TreeGenome.from_dict(
        nx.node_link_data(
            original,
            edges="links",
        )
    )

    recovered = genome.to_networkx()

    assert recovered.nodes[
        1
    ]["length"] == pytest.approx(
        0.180
    )


def test_tree_genome_round_trip_preserves_unrelated_attributes() -> None:
    """The variable-length change should preserve generic node metadata."""
    original = _graph_with_length(
        0.180
    )

    genome = TreeGenome.from_dict(
        nx.node_link_data(
            original,
            edges="links",
        )    
    )

    recovered = genome.to_networkx()

    assert (
        recovered.nodes[
            1
        ]["custom_marker"]
        == "preserve-me"
    )
