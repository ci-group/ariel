"""Test: tree genome validation utilities."""

import pytest

from ariel.ec.genotypes.tree.validation import (
    is_single_connected_tree,
    validate_genome_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _core_only() -> dict:
    return {
        "nodes": {"0": {"type": "CORE", "rotation": "DEG_0"}},
        "edges": [],
    }


def _core_hinge() -> dict:
    return {
        "nodes": {
            "0": {"type": "CORE", "rotation": "DEG_0"},
            "1": {"type": "HINGE", "rotation": "DEG_0"},
        },
        "edges": [{"parent": 0, "child": 1, "face": "FRONT"}],
    }


def _core_brick_hinge() -> dict:
    return {
        "nodes": {
            "0": {"type": "CORE", "rotation": "DEG_0"},
            "1": {"type": "BRICK", "rotation": "DEG_0"},
            "2": {"type": "HINGE", "rotation": "DEG_0"},
        },
        "edges": [
            {"parent": 0, "child": 1, "face": "FRONT"},
            {"parent": 1, "child": 2, "face": "FRONT"},
        ],
    }


# ---------------------------------------------------------------------------
# is_single_connected_tree
# ---------------------------------------------------------------------------


def test_empty_genome_is_connected() -> None:
    """Empty genome is trivially a connected tree."""
    assert is_single_connected_tree({"nodes": {}, "edges": []}) is True


def test_core_only_is_connected() -> None:
    """Single core node is a valid connected tree."""
    assert is_single_connected_tree(_core_only()) is True


def test_core_hinge_is_connected() -> None:
    """Core + one hinge is connected."""
    assert is_single_connected_tree(_core_hinge()) is True


def test_multi_node_chain_is_connected() -> None:
    """Three-node chain is connected."""
    assert is_single_connected_tree(_core_brick_hinge()) is True


def test_missing_core_not_connected() -> None:
    """Genome without core node is not connected."""
    genome = {
        "nodes": {"1": {"type": "HINGE", "rotation": "DEG_0"}},
        "edges": [],
    }
    assert is_single_connected_tree(genome) is False


def test_disconnected_node_not_connected() -> None:
    """An isolated node unreachable from core is not a connected tree."""
    genome = {
        "nodes": {
            "0": {"type": "CORE", "rotation": "DEG_0"},
            "1": {"type": "HINGE", "rotation": "DEG_0"},
            "2": {"type": "BRICK", "rotation": "DEG_0"},
        },
        "edges": [{"parent": 0, "child": 1, "face": "FRONT"}],
        # node 2 is not connected
    }
    assert is_single_connected_tree(genome) is False


def test_core_has_parent_not_connected() -> None:
    """Core with an incoming edge is not a valid root."""
    genome = {
        "nodes": {
            "0": {"type": "CORE", "rotation": "DEG_0"},
            "1": {"type": "HINGE", "rotation": "DEG_0"},
        },
        "edges": [{"parent": 1, "child": 0, "face": "FRONT"}],
    }
    assert is_single_connected_tree(genome) is False


# ---------------------------------------------------------------------------
# validate_genome_dict
# ---------------------------------------------------------------------------


def test_validate_core_only_passes() -> None:
    """A core-only genome passes validation."""
    validate_genome_dict(_core_only())


def test_validate_core_hinge_passes() -> None:
    """Core + hinge genome passes validation."""
    validate_genome_dict(_core_hinge())


def test_validate_missing_core_raises() -> None:
    """Genome without core node raises ValueError."""
    genome = {
        "nodes": {"1": {"type": "HINGE", "rotation": "DEG_0"}},
        "edges": [],
    }
    with pytest.raises(ValueError, match="core node"):
        validate_genome_dict(genome)


def test_validate_invalid_type_raises() -> None:
    """Genome with an unrecognized module type raises ValueError."""
    genome = {
        "nodes": {
            "0": {"type": "CORE", "rotation": "DEG_0"},
            "1": {"type": "UNKNOWN_TYPE", "rotation": "DEG_0"},
        },
        "edges": [],
    }
    with pytest.raises(ValueError, match="invalid type"):
        validate_genome_dict(genome)


def test_validate_invalid_rotation_raises() -> None:
    """Genome with an invalid rotation raises ValueError."""
    genome = {
        "nodes": {
            "0": {"type": "CORE", "rotation": "DEG_999"},
        },
        "edges": [],
    }
    with pytest.raises(ValueError, match="invalid rotation"):
        validate_genome_dict(genome)


def test_validate_invalid_face_raises() -> None:
    """Edge with an unrecognized face name raises ValueError."""
    genome = {
        "nodes": {
            "0": {"type": "CORE", "rotation": "DEG_0"},
            "1": {"type": "HINGE", "rotation": "DEG_0"},
        },
        "edges": [{"parent": 0, "child": 1, "face": "DIAG"}],
    }
    with pytest.raises(ValueError, match="invalid face"):
        validate_genome_dict(genome)


def test_validate_disallowed_face_for_type_raises() -> None:
    """Edge using a face not allowed for the parent type raises ValueError."""
    # HINGE only allows FRONT — using BACK on a hinge parent should fail
    genome = {
        "nodes": {
            "0": {"type": "CORE", "rotation": "DEG_0"},
            "1": {"type": "HINGE", "rotation": "DEG_0"},
            "2": {"type": "BRICK", "rotation": "DEG_0"},
        },
        "edges": [
            {"parent": 0, "child": 1, "face": "FRONT"},
            {"parent": 1, "child": 2, "face": "BACK"},  # BACK not allowed on HINGE
        ],
    }
    with pytest.raises(ValueError, match="not allowed"):
        validate_genome_dict(genome)


def test_validate_duplicate_face_raises() -> None:
    """Two children on the same parent face raises ValueError."""
    genome = {
        "nodes": {
            "0": {"type": "CORE", "rotation": "DEG_0"},
            "1": {"type": "HINGE", "rotation": "DEG_0"},
            "2": {"type": "BRICK", "rotation": "DEG_0"},
        },
        "edges": [
            {"parent": 0, "child": 1, "face": "FRONT"},
            {"parent": 0, "child": 2, "face": "FRONT"},  # duplicate face
        ],
    }
    with pytest.raises(ValueError, match="already has child"):
        validate_genome_dict(genome)


def test_validate_disconnected_raises() -> None:
    """Genome not forming a connected tree raises ValueError."""
    genome = {
        "nodes": {
            "0": {"type": "CORE", "rotation": "DEG_0"},
            "1": {"type": "HINGE", "rotation": "DEG_0"},
        },
        "edges": [],  # node 1 not connected
    }
    with pytest.raises(ValueError, match="connected tree"):
        validate_genome_dict(genome)
