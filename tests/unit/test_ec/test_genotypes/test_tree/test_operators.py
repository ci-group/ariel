"""Test: genetic operators for tree genomes."""

import copy

import pytest

from ariel.ec.genotypes.tree.operators import (
    add_node,
    crossover_subtree,
    get_top_ancestor,
    get_tree_depth,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
    remove_subtree,
    subtree_swap,
    validate_tree_depth,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _core_only() -> TreeGenome:
    g = TreeGenome()
    g.nodes = {0: {"type": "CORE", "rotation": "DEG_0"}}
    g.edges = []
    return g


def _core_hinge() -> TreeGenome:
    g = TreeGenome()
    g.nodes = {
        0: {"type": "CORE", "rotation": "DEG_0"},
        1: {"type": "HINGE", "rotation": "DEG_0"},
    }
    g.edges = [{"parent": 0, "child": 1, "face": "FRONT"}]
    return g


def _core_hinge_brick() -> TreeGenome:
    g = TreeGenome()
    g.nodes = {
        0: {"type": "CORE", "rotation": "DEG_0"},
        1: {"type": "HINGE", "rotation": "DEG_0"},
        2: {"type": "BRICK", "rotation": "DEG_0"},
    }
    g.edges = [
        {"parent": 0, "child": 1, "face": "FRONT"},
        {"parent": 1, "child": 2, "face": "FRONT"},
    ]
    return g


# ---------------------------------------------------------------------------
# add_node
# ---------------------------------------------------------------------------


def test_add_node_appends_node() -> None:
    """add_node inserts a new node into the genome."""
    g = _core_only()
    add_node(g, parent=0, face="FRONT", node_id=1, mtype="HINGE", rotation="DEG_0")
    assert 1 in g.nodes
    assert g.nodes[1] == {"type": "HINGE", "rotation": "DEG_0"}


def test_add_node_appends_edge() -> None:
    """add_node appends a directed edge."""
    g = _core_only()
    add_node(g, parent=0, face="BACK", node_id=1, mtype="BRICK", rotation="DEG_0")
    assert {"parent": 0, "child": 1, "face": "BACK"} in g.edges


# ---------------------------------------------------------------------------
# remove_subtree
# ---------------------------------------------------------------------------


def test_remove_subtree_removes_node_and_edge() -> None:
    """remove_subtree deletes the target node and its incoming edge."""
    g = _core_hinge()
    remove_subtree(g, node_id=1)
    assert 1 not in g.nodes
    assert all(e["child"] != 1 for e in g.edges)


def test_remove_subtree_removes_descendants() -> None:
    """Removing a non-leaf removes all descendant nodes."""
    g = _core_hinge_brick()
    remove_subtree(g, node_id=1)
    assert 1 not in g.nodes
    assert 2 not in g.nodes


def test_remove_subtree_nonexistent_is_noop() -> None:
    """Removing a node that does not exist is silently ignored."""
    g = _core_hinge()
    remove_subtree(g, node_id=99)
    assert len(g.nodes) == 2


# ---------------------------------------------------------------------------
# get_top_ancestor
# ---------------------------------------------------------------------------


def test_get_top_ancestor_direct_child_of_core() -> None:
    """Direct child of core returns itself as top ancestor."""
    g = _core_hinge()
    assert get_top_ancestor(g, 1) == 1


def test_get_top_ancestor_grandchild() -> None:
    """Grandchild returns its parent (first child of core) as top ancestor."""
    g = _core_hinge_brick()
    result = get_top_ancestor(g, 2)
    assert result == 1


# ---------------------------------------------------------------------------
# random_tree
# ---------------------------------------------------------------------------


def test_random_tree_has_core() -> None:
    """random_tree always contains the core node."""
    g = random_tree(5)
    assert 0 in g.nodes
    assert g.nodes[0]["type"] == "CORE"


def test_random_tree_within_budget() -> None:
    """random_tree produces at most max_modules + 1 nodes (core + budget)."""
    g = random_tree(4)
    assert len(g.nodes) <= 5


def test_random_tree_is_valid() -> None:
    """random_tree produces a genome that passes validation."""
    g = random_tree(6)
    validate_genome_dict(g.to_dict())


def test_random_tree_single_module() -> None:
    """random_tree(1) produces at most 2 nodes."""
    g = random_tree(1)
    assert len(g.nodes) <= 2


# ---------------------------------------------------------------------------
# get_tree_depth / validate_tree_depth
# ---------------------------------------------------------------------------


def test_get_tree_depth_core_only() -> None:
    """Core-only tree has depth 0."""
    assert get_tree_depth(_core_only()) == 0


def test_get_tree_depth_chain() -> None:
    """Core → hinge → brick has depth 2."""
    assert get_tree_depth(_core_hinge_brick()) == 2


def test_validate_tree_depth_within_limit() -> None:
    """Depth within limit returns True."""
    g = _core_hinge_brick()
    assert validate_tree_depth(g, max_depth=5) is True


def test_validate_tree_depth_at_limit() -> None:
    """Depth exactly at limit returns True."""
    g = _core_hinge_brick()
    assert validate_tree_depth(g, max_depth=2) is True


def test_validate_tree_depth_exceeds_limit() -> None:
    """Depth exceeding limit returns False."""
    g = _core_hinge_brick()
    assert validate_tree_depth(g, max_depth=1) is False


# ---------------------------------------------------------------------------
# mutate_replace_node
# ---------------------------------------------------------------------------


def test_mutate_replace_node_preserves_core() -> None:
    """mutate_replace_node never changes the core node."""
    g = _core_hinge()
    for _ in range(20):
        mutate_replace_node(g)
    assert g.nodes[0]["type"] == "CORE"


def test_mutate_replace_node_core_only_is_noop() -> None:
    """mutate_replace_node on a core-only genome is a no-op."""
    g = _core_only()
    original_nodes = copy.deepcopy(g.nodes)
    mutate_replace_node(g)
    assert g.nodes == original_nodes


def test_mutate_replace_node_result_is_valid() -> None:
    """After mutation the genome remains valid."""
    g = random_tree(5)
    for _ in range(10):
        mutate_replace_node(g)
    validate_genome_dict(g.to_dict())


# ---------------------------------------------------------------------------
# mutate_shrink
# ---------------------------------------------------------------------------


def test_mutate_shrink_reduces_or_keeps_size() -> None:
    """mutate_shrink does not increase the node count."""
    g = _core_hinge_brick()
    original_size = len(g.nodes)
    mutate_shrink(g)
    assert len(g.nodes) <= original_size


def test_mutate_shrink_core_only_is_noop() -> None:
    """mutate_shrink on a core-only genome is a no-op."""
    g = _core_only()
    mutate_shrink(g)
    assert len(g.nodes) == 1


def test_mutate_shrink_result_is_valid() -> None:
    """Genome remains valid after shrink mutation."""
    g = random_tree(6)
    mutate_shrink(g)
    validate_genome_dict(g.to_dict())


# ---------------------------------------------------------------------------
# mutate_hoist
# ---------------------------------------------------------------------------


def test_mutate_hoist_core_only_is_noop() -> None:
    """mutate_hoist on a core-only genome is a no-op."""
    g = _core_only()
    mutate_hoist(g)
    assert len(g.nodes) == 1


def test_mutate_hoist_result_is_valid() -> None:
    """Genome remains valid after hoist mutation."""
    g = random_tree(6)
    for _ in range(5):
        mutate_hoist(g)
        validate_genome_dict(g.to_dict())


# ---------------------------------------------------------------------------
# mutate_subtree_replacement
# ---------------------------------------------------------------------------


def test_mutate_subtree_replacement_core_only_is_noop() -> None:
    """mutate_subtree_replacement on a core-only genome is a no-op."""
    g = _core_only()
    mutate_subtree_replacement(g)
    assert len(g.nodes) == 1


def test_mutate_subtree_replacement_result_is_valid() -> None:
    """Genome remains valid after subtree replacement mutation."""
    g = random_tree(5)
    for _ in range(5):
        mutate_subtree_replacement(g)
    validate_genome_dict(g.to_dict())


# ---------------------------------------------------------------------------
# subtree_swap
# ---------------------------------------------------------------------------


def test_subtree_swap_core_preserved_in_both() -> None:
    """Both genomes still contain their core nodes after a subtree swap."""
    a = random_tree(4)
    b = random_tree(4)
    # pick a non-core node from each
    a_nid = next(n for n in a.nodes if n != 0)
    b_nid = next(n for n in b.nodes if n != 0)
    subtree_swap(a, b, a_nid, b_nid)
    assert 0 in a.nodes
    assert 0 in b.nodes


# ---------------------------------------------------------------------------
# crossover_subtree
# ---------------------------------------------------------------------------


def test_crossover_subtree_returns_two_genomes() -> None:
    """crossover_subtree always returns a pair of TreeGenome objects."""
    a = random_tree(5)
    b = random_tree(5)
    c1, c2 = crossover_subtree(a, b)
    assert isinstance(c1, TreeGenome)
    assert isinstance(c2, TreeGenome)


def test_crossover_subtree_children_valid() -> None:
    """Children produced by crossover are valid genomes."""
    for _ in range(10):
        a = random_tree(5)
        b = random_tree(5)
        c1, c2 = crossover_subtree(a, b)
        validate_genome_dict(c1.to_dict())
        validate_genome_dict(c2.to_dict())


def test_crossover_subtree_parents_unchanged() -> None:
    """crossover_subtree does not modify the parent genomes."""
    a = random_tree(4)
    b = random_tree(4)
    a_nodes_before = copy.deepcopy(a.nodes)
    b_nodes_before = copy.deepcopy(b.nodes)
    crossover_subtree(a, b)
    assert a.nodes == a_nodes_before
    assert b.nodes == b_nodes_before


def test_crossover_subtree_core_only_returns_copies() -> None:
    """Core-only genomes (no candidates) return deep copies of parents."""
    a = _core_only()
    b = _core_only()
    c1, c2 = crossover_subtree(a, b)
    assert 0 in c1.nodes
    assert 0 in c2.nodes
