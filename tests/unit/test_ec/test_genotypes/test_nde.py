"""Test: NeuralDevelopmentalEncoding — init, forward shapes, and value range."""

import numpy as np
import pytest
import torch

from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
)
from ariel.ec.genotypes.nde.nde import NeuralDevelopmentalEncoding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GENOTYPE_SIZE = 64


def _nde(n: int = 5) -> NeuralDevelopmentalEncoding:
    return NeuralDevelopmentalEncoding(number_of_modules=n)


def _genotype(n: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Three float32 chromosomes (type, connection, rotation), each of length GENOTYPE_SIZE."""
    rng = np.random.default_rng(seed)
    return [rng.random(GENOTYPE_SIZE).astype(np.float32) for _ in range(3)]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_nde_initialization() -> None:
    """NeuralDevelopmentalEncoding initializes without error."""
    nde = _nde(n=4)
    assert nde is not None


def test_nde_is_nn_module() -> None:
    """NeuralDevelopmentalEncoding is a torch.nn.Module."""
    nde = _nde()
    assert isinstance(nde, torch.nn.Module)


def test_nde_no_grad_params() -> None:
    """All parameters have requires_grad=False (neuroevolution mode)."""
    nde = _nde()
    for param in nde.parameters():
        assert param.requires_grad is False


def test_nde_output_layers_count() -> None:
    """There are exactly 3 output layers (type, connection, rotation)."""
    nde = _nde()
    assert len(nde.output_layers) == 3


def test_nde_output_shapes_count() -> None:
    """There are exactly 3 expected output shapes."""
    nde = _nde()
    assert len(nde.output_shapes) == 3


def test_nde_type_shape() -> None:
    """Type probability output shape is (n, NUM_OF_TYPES_OF_MODULES)."""
    n = 6
    nde = _nde(n)
    assert nde.type_p_shape == (n, NUM_OF_TYPES_OF_MODULES)


def test_nde_connection_shape() -> None:
    """Connection probability output shape is (n, n, NUM_OF_FACES)."""
    n = 6
    nde = _nde(n)
    assert nde.conn_p_shape == (n, n, NUM_OF_FACES)


def test_nde_rotation_shape() -> None:
    """Rotation probability output shape is (n, NUM_OF_ROTATIONS)."""
    n = 6
    nde = _nde(n)
    assert nde.rot_p_shape == (n, NUM_OF_ROTATIONS)


# ---------------------------------------------------------------------------
# Forward — output structure
# ---------------------------------------------------------------------------


def test_nde_forward_returns_list() -> None:
    """forward() returns a Python list."""
    nde = _nde()
    outputs = nde.forward(_genotype())
    assert isinstance(outputs, list)


def test_nde_forward_three_outputs() -> None:
    """forward() returns exactly 3 arrays (type, connection, rotation)."""
    nde = _nde()
    outputs = nde.forward(_genotype())
    assert len(outputs) == 3


def test_nde_forward_type_shape() -> None:
    """First output has shape (n, NUM_OF_TYPES_OF_MODULES)."""
    n = 5
    nde = _nde(n)
    outputs = nde.forward(_genotype(n))
    assert outputs[0].shape == (n, NUM_OF_TYPES_OF_MODULES)


def test_nde_forward_connection_shape() -> None:
    """Second output has shape (n, n, NUM_OF_FACES)."""
    n = 5
    nde = _nde(n)
    outputs = nde.forward(_genotype(n))
    assert outputs[1].shape == (n, n, NUM_OF_FACES)


def test_nde_forward_rotation_shape() -> None:
    """Third output has shape (n, NUM_OF_ROTATIONS)."""
    n = 5
    nde = _nde(n)
    outputs = nde.forward(_genotype(n))
    assert outputs[2].shape == (n, NUM_OF_ROTATIONS)


# ---------------------------------------------------------------------------
# Forward — value properties (sigmoid output → [0, 1])
# ---------------------------------------------------------------------------


def test_nde_forward_outputs_numpy_arrays() -> None:
    """All outputs are numpy ndarrays."""
    nde = _nde()
    for arr in nde.forward(_genotype()):
        assert isinstance(arr, np.ndarray)


def test_nde_forward_type_in_range() -> None:
    """Type probabilities are in [0, 1] (sigmoid activation)."""
    nde = _nde()
    out = nde.forward(_genotype())[0]
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_nde_forward_connection_in_range() -> None:
    """Connection probabilities are in [0, 1]."""
    nde = _nde()
    out = nde.forward(_genotype())[1]
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_nde_forward_rotation_in_range() -> None:
    """Rotation probabilities are in [0, 1]."""
    nde = _nde()
    out = nde.forward(_genotype())[2]
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_nde_forward_no_nan() -> None:
    """forward() never produces NaN values."""
    nde = _nde()
    for arr in nde.forward(_genotype()):
        assert not np.any(np.isnan(arr))


# ---------------------------------------------------------------------------
# Forward — determinism
# ---------------------------------------------------------------------------


def test_nde_forward_deterministic_same_input() -> None:
    """Same genotype always produces the same outputs."""
    nde = _nde(seed_unused := 4)
    g = _genotype()
    out1 = nde.forward(g)
    out2 = nde.forward(g)
    for a1, a2 in zip(out1, out2):
        assert np.allclose(a1, a2)


def test_nde_forward_different_inputs_differ() -> None:
    """Different genotypes produce different outputs."""
    nde = _nde()
    out1 = nde.forward(_genotype(seed=0))
    out2 = nde.forward(_genotype(seed=99))
    # At least one output array must differ
    assert not all(np.allclose(a, b) for a, b in zip(out1, out2))


# ---------------------------------------------------------------------------
# Forward — different module counts
# ---------------------------------------------------------------------------


def test_nde_forward_single_module() -> None:
    """NDE works with n=1 module."""
    nde = _nde(n=1)
    outputs = nde.forward(_genotype(n=1))
    assert outputs[0].shape == (1, NUM_OF_TYPES_OF_MODULES)
    assert outputs[1].shape == (1, 1, NUM_OF_FACES)
    assert outputs[2].shape == (1, NUM_OF_ROTATIONS)


def test_nde_forward_large_module_count() -> None:
    """NDE scales correctly to a larger module count."""
    n = 20
    nde = _nde(n=n)
    outputs = nde.forward(_genotype(n=n))
    assert outputs[0].shape == (n, NUM_OF_TYPES_OF_MODULES)
    assert outputs[1].shape == (n, n, NUM_OF_FACES)
    assert outputs[2].shape == (n, NUM_OF_ROTATIONS)


# ---------------------------------------------------------------------------
# Integration: NDE → HighProbabilityDecoder
# ---------------------------------------------------------------------------


def test_nde_output_feeds_hi_prob_decoder() -> None:
    """NDE outputs can be directly fed into HighProbabilityDecoder."""
    from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
        HighProbabilityDecoder,
    )
    from networkx import DiGraph

    n = 5
    nde = NeuralDevelopmentalEncoding(number_of_modules=n)
    type_p, conn_p, rot_p = nde.forward(_genotype(n))

    decoder = HighProbabilityDecoder(num_modules=n)
    graph = decoder.probability_matrices_to_graph(type_p, conn_p, rot_p)
    assert isinstance(graph, DiGraph)
    assert 0 in graph.nodes
