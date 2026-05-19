"""Test: HighProbabilityDecoder and VectorDecoder."""

import numpy as np
import pytest
from networkx import DiGraph

from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
    ModuleRotationsIdx,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.body_phenotypes.robogen_lite.decoders.vector_decoding import VectorDecoder


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

NUM_MODULES = 4


def _type_space(num_modules: int = NUM_MODULES) -> np.ndarray:
    """Type probability space with core forced at index 0, no NONE elsewhere."""
    space = np.ones((num_modules, NUM_OF_TYPES_OF_MODULES), dtype=np.float32)
    space[0] = 0.0
    space[0, ModuleType.CORE.value] = 1.0
    for i in range(1, num_modules):
        space[i, ModuleType.NONE.value] = 0.0
        space[i, ModuleType.CORE.value] = 0.0
    return space


def _conn_space(num_modules: int = NUM_MODULES) -> np.ndarray:
    return np.ones((num_modules, num_modules, NUM_OF_FACES), dtype=np.float32)


def _rot_space(num_modules: int = NUM_MODULES) -> np.ndarray:
    return np.ones((num_modules, NUM_OF_ROTATIONS), dtype=np.float32)


# ---------------------------------------------------------------------------
# HighProbabilityDecoder — initialization
# ---------------------------------------------------------------------------


def test_hi_prob_decoder_initialization() -> None:
    """HighProbabilityDecoder initializes with the given num_modules."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    assert decoder.num_modules == NUM_MODULES


# ---------------------------------------------------------------------------
# HighProbabilityDecoder — probability_matrices_to_graph
# ---------------------------------------------------------------------------


def test_hi_prob_decoder_returns_digraph() -> None:
    """probability_matrices_to_graph returns a NetworkX DiGraph."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    graph = decoder.probability_matrices_to_graph(
        _type_space(), _conn_space(), _rot_space(),
    )
    assert isinstance(graph, DiGraph)


def test_hi_prob_decoder_core_always_present() -> None:
    """The core node (index 0) always appears in the decoded graph."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    graph = decoder.probability_matrices_to_graph(
        _type_space(), _conn_space(), _rot_space(),
    )
    assert 0 in graph.nodes


def test_hi_prob_decoder_core_type_correct() -> None:
    """The core node carries type='CORE'."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    graph = decoder.probability_matrices_to_graph(
        _type_space(), _conn_space(), _rot_space(),
    )
    assert graph.nodes[0]["type"] == "CORE"


def test_hi_prob_decoder_nodes_have_rotation() -> None:
    """Every node carries a 'rotation' attribute."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    graph = decoder.probability_matrices_to_graph(
        _type_space(), _conn_space(), _rot_space(),
    )
    for node in graph.nodes:
        assert "rotation" in graph.nodes[node]


def test_hi_prob_decoder_no_self_loops() -> None:
    """The decoded graph has no self-loops."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    graph = decoder.probability_matrices_to_graph(
        _type_space(), _conn_space(), _rot_space(),
    )
    assert not any(u == v for u, v in graph.edges)


def test_hi_prob_decoder_zero_conn_stops_gracefully() -> None:
    """All-zero connection space produces a graph with at least the core."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    zero_conn = np.zeros((NUM_MODULES, NUM_MODULES, NUM_OF_FACES), dtype=np.float32)
    graph = decoder.probability_matrices_to_graph(
        _type_space(), zero_conn, _rot_space(),
    )
    assert 0 in graph.nodes


def test_hi_prob_decoder_single_module() -> None:
    """Works correctly when num_modules=1 (core only)."""
    n = 1
    type_s = np.zeros((1, NUM_OF_TYPES_OF_MODULES), dtype=np.float32)
    type_s[0, ModuleType.CORE.value] = 1.0
    conn_s = np.zeros((1, 1, NUM_OF_FACES), dtype=np.float32)
    rot_s = np.ones((1, NUM_OF_ROTATIONS), dtype=np.float32)
    decoder = HighProbabilityDecoder(num_modules=n)
    graph = decoder.probability_matrices_to_graph(type_s, conn_s, rot_s)
    assert 0 in graph.nodes


# ---------------------------------------------------------------------------
# HighProbabilityDecoder — apply_connection_constraints
# ---------------------------------------------------------------------------


def test_hi_prob_apply_constraints_removes_self_connections() -> None:
    """After apply_connection_constraints, all diagonal conn entries are zero."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    decoder.conn_p_space = _conn_space()
    decoder.rot_p_space = _rot_space()
    decoder.type_p_space = _type_space()
    decoder.apply_connection_constraints()
    for face_idx in range(NUM_OF_FACES):
        for i in range(NUM_MODULES):
            assert decoder.conn_p_space[i, i, face_idx] == 0.0


def test_hi_prob_apply_constraints_core_never_child() -> None:
    """After constraints, nothing can connect to core as a child."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    decoder.conn_p_space = _conn_space()
    decoder.rot_p_space = _rot_space()
    decoder.type_p_space = _type_space()
    decoder.apply_connection_constraints()
    assert np.all(decoder.conn_p_space[:, 0, :] == 0.0)


# ---------------------------------------------------------------------------
# HighProbabilityDecoder — set_module_types_and_rotations
# ---------------------------------------------------------------------------


def test_hi_prob_set_types_creates_type_dict() -> None:
    """set_module_types_and_rotations builds type_dict for all modules."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    decoder.conn_p_space = _conn_space()
    decoder.rot_p_space = _rot_space()
    decoder.type_p_space = _type_space()
    decoder.set_module_types_and_rotations()
    assert len(decoder.type_dict) == NUM_MODULES


def test_hi_prob_set_types_creates_rot_dict() -> None:
    """set_module_types_and_rotations builds rot_dict for all modules."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    decoder.conn_p_space = _conn_space()
    decoder.rot_p_space = _rot_space()
    decoder.type_p_space = _type_space()
    decoder.set_module_types_and_rotations()
    assert len(decoder.rot_dict) == NUM_MODULES
    for v in decoder.rot_dict.values():
        assert isinstance(v, ModuleRotationsIdx)


# ---------------------------------------------------------------------------
# HighProbabilityDecoder — decode_probability_to_graph
# ---------------------------------------------------------------------------


def _setup_decoder(num_modules: int = NUM_MODULES) -> HighProbabilityDecoder:
    """Return a decoder fully initialized for decode_probability_to_graph."""
    decoder = HighProbabilityDecoder(num_modules=num_modules)
    decoder.conn_p_space = _conn_space(num_modules)
    decoder.rot_p_space = _rot_space(num_modules)
    decoder.type_p_space = _type_space(num_modules)
    decoder.apply_connection_constraints()
    decoder.set_module_types_and_rotations()
    return decoder


def test_hi_prob_decode_builds_nodes_set() -> None:
    """decode_probability_to_graph populates the nodes set."""
    decoder = _setup_decoder()
    decoder.decode_probability_to_graph()
    assert hasattr(decoder, "nodes")
    assert 0 in decoder.nodes


def test_hi_prob_decode_builds_edges_list() -> None:
    """decode_probability_to_graph populates the edges list."""
    decoder = _setup_decoder()
    decoder.decode_probability_to_graph()
    assert hasattr(decoder, "edges")
    assert isinstance(decoder.edges, list)


# ---------------------------------------------------------------------------
# VectorDecoder — initialization
# ---------------------------------------------------------------------------


def test_vector_decoder_initialization() -> None:
    """VectorDecoder initializes with the given num_modules."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    assert decoder.num_modules == NUM_MODULES


# ---------------------------------------------------------------------------
# VectorDecoder — assign_symbols_from_range
# ---------------------------------------------------------------------------


def test_assign_symbols_basic() -> None:
    """assign_symbols_from_range maps values to the correct symbols."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    result = decoder.assign_symbols_from_range(
        vector=[0.0, 0.5, 1.0],
        symbols=["A", "B", "C"],
    )
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(s in ("A", "B", "C") for s in result)


def test_assign_symbols_out_of_range_raises() -> None:
    """Values outside [0, 1] raise ValueError."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        decoder.assign_symbols_from_range(vector=[1.5], symbols=["A", "B"])


def test_assign_symbols_negative_raises() -> None:
    """Negative values raise ValueError."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        decoder.assign_symbols_from_range(vector=[-0.1], symbols=["A", "B"])


def test_assign_symbols_single_symbol_always_returned() -> None:
    """Single-symbol list is always returned regardless of value."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    result = decoder.assign_symbols_from_range(
        vector=[0.0, 0.5, 1.0],
        symbols=["X"],
    )
    assert all(s == "X" for s in result)


def test_assign_symbols_boundary_zero() -> None:
    """Value 0.0 maps to the first symbol."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    result = decoder.assign_symbols_from_range(
        vector=[0.0],
        symbols=["FIRST", "SECOND", "THIRD"],
    )
    assert result[0] == "FIRST"


def test_assign_symbols_boundary_one() -> None:
    """Value 1.0 maps to the last symbol."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    result = decoder.assign_symbols_from_range(
        vector=[1.0],
        symbols=["FIRST", "SECOND", "THIRD"],
    )
    assert result[0] == "THIRD"


def test_assign_symbols_per_element_symbols() -> None:
    """Per-element symbol lists assign from the correct per-element set."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    result = decoder.assign_symbols_from_range(
        vector=[0.2, 0.8],
        symbols=[["A", "B"], ["C", "D"]],
    )
    assert result[0] in ("A", "B")
    assert result[1] in ("C", "D")


def test_assign_symbols_with_weights() -> None:
    """Uniform weights are equivalent to no weights."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    vector = [0.3, 0.7]
    symbols = ["X", "Y", "Z"]
    result_no_weight = decoder.assign_symbols_from_range(vector=vector, symbols=symbols)
    result_uniform = decoder.assign_symbols_from_range(
        vector=vector, symbols=symbols, weight=[1.0, 1.0, 1.0],
    )
    assert result_no_weight == result_uniform


def test_assign_symbols_numpy_input() -> None:
    """assign_symbols_from_range accepts numpy arrays as vector input."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    vector = np.array([0.1, 0.5, 0.9])
    result = decoder.assign_symbols_from_range(vector=vector, symbols=["A", "B", "C"])
    assert len(result) == 3


def test_assign_symbols_integer_symbols() -> None:
    """Integer symbols can be used as the symbol set."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    result = decoder.assign_symbols_from_range(
        vector=[0.0, 0.5, 1.0],
        symbols=[0, 1, 2],
    )
    assert all(s in (0, 1, 2) for s in result)


# ---------------------------------------------------------------------------
# VectorDecoder — set_module_types_and_rotations
# ---------------------------------------------------------------------------


def test_vector_decoder_set_types_and_rotations() -> None:
    """set_module_types_and_rotations populates type_dict and rot_dict."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    decoder.conn_p_space = _conn_space()
    decoder.rot_p_space = _rot_space()
    decoder.type_p_space = _type_space()
    decoder.set_module_types_and_rotations()
    assert len(decoder.type_dict) == NUM_MODULES
    assert len(decoder.rot_dict) == NUM_MODULES


def test_vector_decoder_set_types_core_at_zero() -> None:
    """After set_module_types_and_rotations, node 0 is typed as CORE."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    decoder.conn_p_space = _conn_space()
    decoder.rot_p_space = _rot_space()
    decoder.type_p_space = _type_space()
    decoder.set_module_types_and_rotations()
    assert decoder.type_dict[0] == ModuleType.CORE


# ---------------------------------------------------------------------------
# VectorDecoder — apply_connection_constraints
# ---------------------------------------------------------------------------


def test_vector_decoder_apply_constraints_no_self_connections() -> None:
    """After apply_connection_constraints, diagonal entries are zero."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    decoder.conn_p_space = _conn_space()
    decoder.rot_p_space = _rot_space()
    decoder.type_p_space = _type_space()
    decoder.apply_connection_constraints()
    for face_idx in range(NUM_OF_FACES):
        for i in range(NUM_MODULES):
            assert decoder.conn_p_space[i, i, face_idx] == 0.0


def test_vector_decoder_apply_constraints_core_never_child() -> None:
    """After constraints, core (index 0) cannot be a child."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    decoder.conn_p_space = _conn_space()
    decoder.rot_p_space = _rot_space()
    decoder.type_p_space = _type_space()
    decoder.apply_connection_constraints()
    assert np.all(decoder.conn_p_space[:, 0, :] == 0.0)


# ---------------------------------------------------------------------------
# VectorDecoder — decode_vector_to_graph
# ---------------------------------------------------------------------------


def _setup_vector_decoder(num_modules: int = NUM_MODULES) -> VectorDecoder:
    """Return a VectorDecoder ready for decode_vector_to_graph."""
    decoder = VectorDecoder(num_modules=num_modules)
    decoder.conn_p_space = _conn_space(num_modules)
    decoder.rot_p_space = _rot_space(num_modules)
    decoder.type_p_space = _type_space(num_modules)
    decoder.apply_connection_constraints()
    decoder.set_module_types_and_rotations()
    return decoder


def test_vector_decoder_decode_populates_nodes() -> None:
    """decode_vector_to_graph creates a nodes set containing the core."""
    decoder = _setup_vector_decoder()
    decoder.decode_vector_to_graph()
    assert 0 in decoder.nodes


def test_vector_decoder_decode_populates_edges() -> None:
    """decode_vector_to_graph creates an edges list."""
    decoder = _setup_vector_decoder()
    decoder.decode_vector_to_graph()
    assert isinstance(decoder.edges, list)


def test_vector_decoder_decode_zero_conn_only_core() -> None:
    """All-zero connection space after constraints leaves only the core."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    decoder.conn_p_space = np.zeros((NUM_MODULES, NUM_MODULES, NUM_OF_FACES), dtype=np.float32)
    decoder.rot_p_space = _rot_space()
    decoder.type_p_space = _type_space()
    decoder.set_module_types_and_rotations()
    decoder.decode_vector_to_graph()
    assert 0 in decoder.nodes
