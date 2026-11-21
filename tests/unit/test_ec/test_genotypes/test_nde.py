"""Test: NeuralDevelopmentalEncoding — init, forward shapes, and value range."""

import numpy as np
import pytest
import torch

from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
    ModuleType,
)
from ariel.ec.genotypes.nde.nde import NeuralDevelopmentalEncoding


GENOTYPE_SIZE = 64


def _nde(n: int = 5) -> NeuralDevelopmentalEncoding:
    return NeuralDevelopmentalEncoding(number_of_modules=n)


def _genotype(
    seed: int = 0,
    *,
    include_length: bool = False,
) -> list[np.ndarray]:
    """Float32 chromosomes, each of length GENOTYPE_SIZE."""
    rng = np.random.default_rng(seed)
    chromosome_count = 4 if include_length else 3
    return [
        rng.random(GENOTYPE_SIZE).astype(np.float32)
        for _ in range(chromosome_count)
    ]


def test_nde_initialization() -> None:
    """NeuralDevelopmentalEncoding initializes without error."""
    nde = _nde(n=4)
    assert nde is not None


def test_nde_is_nn_module() -> None:
    """NeuralDevelopmentalEncoding is a torch.nn.Module."""
    nde = _nde()
    assert isinstance(nde, torch.nn.Module)


def test_nde_no_grad_params() -> None:
    """All parameters have requires_grad=False."""
    nde = _nde()
    for param in nde.parameters():
        assert param.requires_grad is False


def test_nde_output_layers_count() -> None:
    """There are three neural output heads."""
    nde = _nde()
    assert len(nde.output_layers) == 3


def test_nde_output_shapes_count() -> None:
    """There are three neural output shapes."""
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


def test_nde_forward_returns_list() -> None:
    """forward() returns a Python list."""
    nde = _nde()
    outputs = nde.forward(_genotype())
    assert isinstance(outputs, list)


def test_nde_forward_three_outputs_is_backward_compatible() -> None:
    """Three chromosomes still return type, connection, and rotation."""
    nde = _nde()
    outputs = nde.forward(_genotype())
    assert len(outputs) == 3


def test_nde_forward_four_outputs_with_length_chromosome() -> None:
    """A fourth chromosome produces the variable-length output."""
    nde = _nde()
    outputs = nde.forward(_genotype(include_length=True))
    assert len(outputs) == 4


def test_nde_forward_type_shape() -> None:
    """First output has shape (n, NUM_OF_TYPES_OF_MODULES)."""
    n = 5
    nde = _nde(n)
    outputs = nde.forward(_genotype())
    assert outputs[0].shape == (n, NUM_OF_TYPES_OF_MODULES)


def test_nde_forward_connection_shape() -> None:
    """Second output has shape (n, n, NUM_OF_FACES)."""
    n = 5
    nde = _nde(n)
    outputs = nde.forward(_genotype())
    assert outputs[1].shape == (n, n, NUM_OF_FACES)


def test_nde_forward_rotation_shape() -> None:
    """Third output has shape (n, NUM_OF_ROTATIONS)."""
    n = 5
    nde = _nde(n)
    outputs = nde.forward(_genotype())
    assert outputs[2].shape == (n, NUM_OF_ROTATIONS)


def test_nde_forward_length_shape() -> None:
    """Fourth output has shape (n,)."""
    n = 5
    nde = _nde(n)
    outputs = nde.forward(_genotype(include_length=True))
    assert outputs[3].shape == (n,)


def test_nde_forward_outputs_numpy_arrays() -> None:
    """All outputs are numpy ndarrays."""
    nde = _nde()
    for arr in nde.forward(_genotype(include_length=True)):
        assert isinstance(arr, np.ndarray)


@pytest.mark.parametrize("output_idx", [0, 1, 2, 3])
def test_nde_forward_outputs_in_range(output_idx: int) -> None:
    """All outputs are in [0, 1]."""
    nde = _nde()
    out = nde.forward(_genotype(include_length=True))[output_idx]
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_nde_forward_no_nan() -> None:
    """forward() never produces NaN values."""
    nde = _nde()
    for arr in nde.forward(_genotype(include_length=True)):
        assert not np.any(np.isnan(arr))


def test_nde_forward_deterministic_same_input() -> None:
    """Same genotype always produces the same outputs."""
    nde = _nde(4)
    genotype = _genotype(include_length=True)

    out1 = nde.forward(genotype)
    out2 = nde.forward(genotype)

    for a1, a2 in zip(out1, out2):
        assert np.allclose(a1, a2)


def test_nde_forward_different_inputs_differ() -> None:
    """Different genotypes produce different outputs."""
    nde = _nde()

    out1 = nde.forward(
        _genotype(seed=0, include_length=True)
    )
    out2 = nde.forward(
        _genotype(seed=99, include_length=True)
    )

    assert not all(
        np.allclose(a, b)
        for a, b in zip(out1, out2)
    )


def test_nde_forward_single_module() -> None:
    """NDE works with n=1 module."""
    nde = _nde(n=1)
    outputs = nde.forward(_genotype(include_length=True))

    assert outputs[0].shape == (1, NUM_OF_TYPES_OF_MODULES)
    assert outputs[1].shape == (1, 1, NUM_OF_FACES)
    assert outputs[2].shape == (1, NUM_OF_ROTATIONS)
    assert outputs[3].shape == (1,)


def test_nde_forward_large_module_count() -> None:
    """NDE scales correctly to a larger module count."""
    n = 20
    nde = _nde(n=n)
    outputs = nde.forward(_genotype(include_length=True))

    assert outputs[0].shape == (n, NUM_OF_TYPES_OF_MODULES)
    assert outputs[1].shape == (n, n, NUM_OF_FACES)
    assert outputs[2].shape == (n, NUM_OF_ROTATIONS)
    assert outputs[3].shape == (n,)


def test_nde_rejects_too_many_chromosomes() -> None:
    """More than four chromosomes is invalid."""
    nde = _nde()
    rng = np.random.default_rng(0)

    genotype = [
        rng.random(GENOTYPE_SIZE).astype(np.float32)
        for _ in range(5)
    ]

    with pytest.raises(ValueError):
        nde.forward(genotype)


def test_nde_rejects_too_few_chromosomes() -> None:
    """Fewer than three chromosomes is invalid."""
    nde = _nde()

    with pytest.raises(ValueError):
        nde.forward(_genotype()[:2])


def test_nde_rejects_wrong_chromosome_size() -> None:
    """Neural chromosomes must match genotype_size."""
    nde = _nde()

    genotype = _genotype()
    genotype[0] = np.zeros(
        GENOTYPE_SIZE - 1,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        nde.forward(genotype)


def test_nde_rejects_wrong_length_chromosome_size() -> None:
    """Length chromosome must match genotype_size."""
    nde = _nde()

    genotype = _genotype(include_length=True)
    genotype[3] = np.zeros(
        GENOTYPE_SIZE - 1,
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        nde.forward(genotype)


def test_nde_rejects_more_modules_than_genotype_size() -> None:
    """Direct length encoding requires one available gene per module."""
    with pytest.raises(ValueError):
        NeuralDevelopmentalEncoding(
            number_of_modules=GENOTYPE_SIZE + 1,
            genotype_size=GENOTYPE_SIZE,
        )


def test_direct_length_encoding_uses_first_module_genes() -> None:
    """Length output should directly use the first n length genes."""
    n = 4
    nde = _nde(n)

    genotype = _genotype(include_length=True)

    genotype[3][:n] = np.array(
        [
            0.0,
            0.25,
            0.5,
            1.0,
        ],
        dtype=np.float32,
    )

    outputs = nde.forward(genotype)

    assert np.allclose(
        outputs[3],
        [
            0.0,
            0.25,
            0.5,
            1.0,
        ],
    )


def test_direct_length_encoding_reflects_values() -> None:
    """Length values outside [0, 1] should reflect back into range."""
    n = 6
    nde = _nde(n)

    genotype = _genotype(include_length=True)

    genotype[3][:n] = np.array(
        [
            -0.2,
            0.25,
            0.75,
            1.2,
            2.2,
            -1.2,
        ],
        dtype=np.float32,
    )

    outputs = nde.forward(genotype)

    assert np.allclose(
        outputs[3],
        [
            0.2,
            0.25,
            0.75,
            0.8,
            0.2,
            0.8,
        ],
    )
    
def test_nde_output_feeds_hi_prob_decoder_without_lengths() -> None:
    """Existing three-output NDE decoding remains supported."""
    from networkx import DiGraph

    from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
        HighProbabilityDecoder,
    )

    n = 5

    nde = NeuralDevelopmentalEncoding(
        number_of_modules=n
    )

    type_p, conn_p, rot_p = nde.forward(
        _genotype()
    )

    decoder = HighProbabilityDecoder(
        num_modules=n
    )

    graph = decoder.probability_matrices_to_graph(
        type_p,
        conn_p,
        rot_p,
    )

    assert isinstance(graph, DiGraph)
    assert 0 in graph.nodes


def test_nde_variable_lengths_feed_hi_prob_decoder() -> None:
    """Fourth NDE output is decoded into physical BRICK lengths."""
    from networkx import DiGraph

    from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
        HighProbabilityDecoder,
    )
    from ariel.parameters.ariel_modules import ArielModulesConfig

    n = 8
    config = ArielModulesConfig()

    nde = NeuralDevelopmentalEncoding(
        number_of_modules=n
    )

    type_p, conn_p, rot_p, length_p = nde.forward(
        _genotype(include_length=True),
    )

    type_p[1:, :] = 0.0
    type_p[1:, ModuleType.BRICK.value] = 1.0

    decoder = HighProbabilityDecoder(
        num_modules=n
    )

    graph = decoder.probability_matrices_to_graph(
        type_p,
        conn_p,
        rot_p,
        length_p,
    )

    assert isinstance(graph, DiGraph)

    brick_nodes = [
        node
        for node, data in graph.nodes(data=True)
        if data["type"] == ModuleType.BRICK.name
    ]

    assert brick_nodes

    for _, data in graph.nodes(data=True):
        if data["type"] == ModuleType.BRICK.name:
            assert "length" in data
            assert (
                config.BRICK_LENGTH_MIN
                <= data["length"]
                <= config.BRICK_LENGTH_MAX
            )
        else:
            assert "length" not in data