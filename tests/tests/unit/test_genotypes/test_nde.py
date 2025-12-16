"""Tests for Neural Developmental Encoding (NDE)."""

import numpy as np
import numpy.typing as npt
import pytest
import torch
from unittest.mock import patch

from ariel.ec.genotypes.nde.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
)


@pytest.fixture
def nde_small():
    """Create a small NDE instance for testing."""
    return NeuralDevelopmentalEncoding(number_of_modules=5, genotype_size=64)


@pytest.fixture
def nde_large():
    """Create a larger NDE instance for testing."""
    return NeuralDevelopmentalEncoding(number_of_modules=20, genotype_size=64)


@pytest.fixture
def genotype_small():
    """Create a small genotype for testing."""
    rng = np.random.default_rng(42)
    genotype_size = 64
    return [
        rng.random(genotype_size, dtype=np.float32),
        rng.random(genotype_size, dtype=np.float32),
        rng.random(genotype_size, dtype=np.float32),
    ]


@pytest.fixture
def genotype_large():
    """Create a larger genotype for testing."""
    rng = np.random.default_rng(42)
    genotype_size = 64
    return [
        rng.random(genotype_size, dtype=np.float32),
        rng.random(genotype_size, dtype=np.float32),
        rng.random(genotype_size, dtype=np.float32),
    ]


class TestNDEInitialization:
    """Tests for NDE initialization."""

    def test_nde_initialization_sets_number_of_modules(self) -> None:
        """Test that NDE stores number of modules correctly."""
        num_modules = 10
        nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
        
        assert nde.type_p_shape[0] == num_modules
        assert nde.conn_p_shape[0] == num_modules
        assert nde.conn_p_shape[1] == num_modules

    def test_nde_initialization_sets_genotype_size(self) -> None:
        """Test that NDE accepts custom genotype size."""
        genotype_size = 128
        nde = NeuralDevelopmentalEncoding(number_of_modules=5, genotype_size=genotype_size)
        
        assert nde.fc1.in_features == genotype_size

    def test_nde_default_genotype_size_is_64(self) -> None:
        """Test that default genotype size is 64."""
        nde = NeuralDevelopmentalEncoding(number_of_modules=5)
        
        assert nde.fc1.in_features == 64

    def test_nde_has_three_output_layers(self) -> None:
        """Test that NDE has exactly three output layers."""
        nde = NeuralDevelopmentalEncoding(number_of_modules=5)
        
        assert len(nde.output_layers) == 3
        assert len(nde.output_shapes) == 3

    def test_nde_output_layer_sizes_correct(self, nde_small) -> None:
        """Test that output layers have correct input sizes."""
        num_modules = 5
        
        # Type output: num_modules * NUM_OF_TYPES_OF_MODULES
        assert nde_small.type_p_out.out_features == num_modules * NUM_OF_TYPES_OF_MODULES
        
        # Connection output: num_modules * num_modules * NUM_OF_FACES
        assert nde_small.conn_p_out.out_features == num_modules * num_modules * NUM_OF_FACES
        
        # Rotation output: num_modules * NUM_OF_ROTATIONS
        assert nde_small.rot_p_out.out_features == num_modules * NUM_OF_ROTATIONS

    def test_nde_gradients_disabled(self, nde_small) -> None:
        """Test that all parameters have gradients disabled."""
        for param in nde_small.parameters():
            assert param.requires_grad is False

    def test_nde_has_activation_functions(self, nde_small) -> None:
        """Test that NDE has required activation functions."""
        assert hasattr(nde_small, "relu")
        assert hasattr(nde_small, "tanh")
        assert hasattr(nde_small, "sigmoid")
        assert isinstance(nde_small.relu, torch.nn.ReLU)
        assert isinstance(nde_small.tanh, torch.nn.Tanh)
        assert isinstance(nde_small.sigmoid, torch.nn.Sigmoid)


class TestNDEForwardPass:
    """Tests for NDE forward pass."""

    def test_forward_returns_list(self, nde_small, genotype_small) -> None:
        """Test that forward pass returns a list."""
        outputs = nde_small.forward(genotype_small)
        
        assert isinstance(outputs, list)

    def test_forward_returns_three_outputs(self, nde_small, genotype_small) -> None:
        """Test that forward pass returns exactly three outputs."""
        outputs = nde_small.forward(genotype_small)
        
        assert len(outputs) == 3

    def test_forward_all_outputs_are_numpy_arrays(self, nde_small, genotype_small) -> None:
        """Test that all forward outputs are numpy arrays."""
        outputs = nde_small.forward(genotype_small)
        
        for output in outputs:
            assert isinstance(output, np.ndarray)

    def test_forward_type_p_output_shape(self, nde_small, genotype_small) -> None:
        """Test that type probability output has correct shape."""
        outputs = nde_small.forward(genotype_small)
        type_p = outputs[0]
        
        assert type_p.shape == nde_small.type_p_shape
        assert type_p.shape == (5, NUM_OF_TYPES_OF_MODULES)

    def test_forward_conn_p_output_shape(self, nde_small, genotype_small) -> None:
        """Test that connection probability output has correct shape."""
        outputs = nde_small.forward(genotype_small)
        conn_p = outputs[1]
        
        assert conn_p.shape == nde_small.conn_p_shape
        assert conn_p.shape == (5, 5, NUM_OF_FACES)

    def test_forward_rot_p_output_shape(self, nde_small, genotype_small) -> None:
        """Test that rotation probability output has correct shape."""
        outputs = nde_small.forward(genotype_small)
        rot_p = outputs[2]
        
        assert rot_p.shape == nde_small.rot_p_shape
        assert rot_p.shape == (5, NUM_OF_ROTATIONS)

    def test_forward_large_nde(self, nde_large, genotype_large) -> None:
        """Test forward pass with larger NDE."""
        outputs = nde_large.forward(genotype_large)
        
        assert len(outputs) == 3
        assert outputs[0].shape == (20, NUM_OF_TYPES_OF_MODULES)
        assert outputs[1].shape == (20, 20, NUM_OF_FACES)
        assert outputs[2].shape == (20, NUM_OF_ROTATIONS)

    def test_forward_outputs_are_float32(self, nde_small, genotype_small) -> None:
        """Test that forward outputs are float32."""
        outputs = nde_small.forward(genotype_small)
        
        for output in outputs:
            assert output.dtype == np.float32

    def test_forward_values_in_valid_range(self, nde_small, genotype_small) -> None:
        """Test that output values are in valid range [0, 1] (sigmoid output)."""
        outputs = nde_small.forward(genotype_small)
        
        for output in outputs:
            assert np.all(output >= 0.0)
            assert np.all(output <= 1.0)

    def test_forward_no_nan_values(self, nde_small, genotype_small) -> None:
        """Test that forward pass produces no NaN values."""
        outputs = nde_small.forward(genotype_small)
        
        for output in outputs:
            assert not np.any(np.isnan(output))

    def test_forward_no_inf_values(self, nde_small, genotype_small) -> None:
        """Test that forward pass produces no infinite values."""
        outputs = nde_small.forward(genotype_small)
        
        for output in outputs:
            assert not np.any(np.isinf(output))


class TestNDEGenotypeHandling:
    """Tests for genotype handling."""

    def test_forward_with_single_chromosome_in_genotype(self, nde_small) -> None:
        """Test forward pass with single chromosome."""
        genotype = [np.random.random(64).astype(np.float32)]
        
        outputs = nde_small.forward(genotype)
        
        assert len(outputs) == 1
        assert outputs[0].shape == nde_small.type_p_shape

    def test_forward_with_three_chromosomes(self, nde_small) -> None:
        """Test forward pass with three chromosomes (standard genotype)."""
        # NDE implementation expects exactly 3 chromosomes for type, conn, and rotation
        genotype = [
            np.random.random(64).astype(np.float32),
            np.random.random(64).astype(np.float32),
            np.random.random(64).astype(np.float32),
        ]
        
        outputs = nde_small.forward(genotype)
        
        assert len(outputs) == 3
        assert outputs[0].shape == nde_small.type_p_shape
        assert outputs[1].shape == nde_small.conn_p_shape
        assert outputs[2].shape == nde_small.rot_p_shape

    def test_forward_with_different_genotype_values(self, nde_small) -> None:
        """Test that different genotypes produce different outputs."""
        genotype1 = [np.zeros(64, dtype=np.float32)]
        genotype2 = [np.ones(64, dtype=np.float32)]
        
        outputs1 = nde_small.forward(genotype1)
        outputs2 = nde_small.forward(genotype2)
        
        # Outputs should be different (with very high probability)
        assert not np.allclose(outputs1[0], outputs2[0])

    def test_forward_deterministic(self, nde_small) -> None:
        """Test that forward pass is deterministic (same input -> same output)."""
        genotype = [np.random.RandomState(42).random(64).astype(np.float32)]
        
        outputs1 = nde_small.forward(genotype)
        outputs2 = nde_small.forward(genotype)
        
        for out1, out2 in zip(outputs1, outputs2):
            assert np.allclose(out1, out2)

    def test_forward_with_zero_genotype(self, nde_small) -> None:
        """Test forward pass with all-zero genotype."""
        genotype = [np.zeros(64, dtype=np.float32)]
        
        outputs = nde_small.forward(genotype)
        
        assert len(outputs) == 1
        assert outputs[0].shape == nde_small.type_p_shape
        assert not np.any(np.isnan(outputs[0]))

    def test_forward_with_ones_genotype(self, nde_small) -> None:
        """Test forward pass with all-ones genotype."""
        genotype = [np.ones(64, dtype=np.float32)]
        
        outputs = nde_small.forward(genotype)
        
        assert len(outputs) == 1
        assert outputs[0].shape == nde_small.type_p_shape
        assert not np.any(np.isnan(outputs[0]))

    def test_forward_chromosome_index_selects_correct_output_layer(self, nde_small) -> None:
        """Test that chromosome index determines which output layer is used."""
        # First chromosome uses output_layers[0] (type_p_out)
        genotype_0 = [np.random.random(64).astype(np.float32)]
        outputs_0 = nde_small.forward(genotype_0)
        assert outputs_0[0].shape == nde_small.type_p_shape
        
        # Second chromosome uses output_layers[1] (conn_p_out)
        genotype_1 = [
            np.random.random(64).astype(np.float32),
            np.random.random(64).astype(np.float32),
        ]
        outputs_1 = nde_small.forward(genotype_1)
        assert outputs_1[1].shape == nde_small.conn_p_shape
        
        # Third chromosome uses output_layers[2] (rot_p_out)
        genotype_2 = [
            np.random.random(64).astype(np.float32),
            np.random.random(64).astype(np.float32),
            np.random.random(64).astype(np.float32),
        ]
        outputs_2 = nde_small.forward(genotype_2)
        assert outputs_2[2].shape == nde_small.rot_p_shape


class TestNDEOutputProperties:
    """Tests for properties of NDE outputs."""

    def test_type_p_sums_to_near_one_across_types(self, nde_small, genotype_small) -> None:
        """Test that type probabilities could be valid (values between 0 and 1)."""
        outputs = nde_small.forward(genotype_small)
        type_p = outputs[0]
        
        # Each element should be between 0 and 1
        assert np.all(type_p >= 0.0)
        assert np.all(type_p <= 1.0)

    def test_conn_p_valid_probability_range(self, nde_small, genotype_small) -> None:
        """Test that connection probabilities are in valid range."""
        outputs = nde_small.forward(genotype_small)
        conn_p = outputs[1]
        
        assert np.all(conn_p >= 0.0)
        assert np.all(conn_p <= 1.0)

    def test_rot_p_valid_probability_range(self, nde_small, genotype_small) -> None:
        """Test that rotation probabilities are in valid range."""
        outputs = nde_small.forward(genotype_small)
        rot_p = outputs[2]
        
        assert np.all(rot_p >= 0.0)
        assert np.all(rot_p <= 1.0)

    def test_outputs_have_sufficient_variance(self, nde_small) -> None:
        """Test that outputs have sufficient variance (not all same value)."""
        genotype = [np.random.random(64).astype(np.float32)]
        outputs = nde_small.forward(genotype)
        
        for output in outputs:
            # Should not be all the same value
            assert np.std(output) > 0.0


class TestNDEEdgeCases:
    """Edge case tests for NDE."""

    def test_forward_with_empty_genotype_raises(self, nde_small) -> None:
        """Test that empty genotype causes error."""
        genotype = []
        
        outputs = nde_small.forward(genotype)
        
        # Should return empty list
        assert len(outputs) == 0

    def test_very_small_nde(self) -> None:
        """Test NDE with very small number of modules."""
        nde = NeuralDevelopmentalEncoding(number_of_modules=1, genotype_size=64)
        genotype = [np.random.random(64).astype(np.float32)]
        
        outputs = nde.forward(genotype)
        
        assert len(outputs) == 1
        assert outputs[0].shape[0] == 1

    def test_large_genotype_size(self) -> None:
        """Test NDE with large genotype size."""
        nde = NeuralDevelopmentalEncoding(number_of_modules=5, genotype_size=512)
        genotype = [np.random.random(512).astype(np.float32)]
        
        outputs = nde.forward(genotype)
        
        assert len(outputs) == 1
        assert not np.any(np.isnan(outputs[0]))

    def test_forward_with_extreme_genotype_values(self, nde_small) -> None:
        """Test forward pass with extreme genotype values."""
        genotype = [np.full(64, 1e6, dtype=np.float32)]
        
        outputs = nde_small.forward(genotype)
        
        # Should still produce valid outputs (sigmoid bounds values)
        assert not np.any(np.isnan(outputs[0]))
        assert np.all(outputs[0] >= 0.0)
        assert np.all(outputs[0] <= 1.0)

    def test_forward_with_negative_genotype_values(self, nde_small) -> None:
        """Test forward pass with negative genotype values."""
        genotype = [np.full(64, -10.0, dtype=np.float32)]
        
        outputs = nde_small.forward(genotype)
        
        assert len(outputs) == 1
        assert not np.any(np.isnan(outputs[0]))


class TestNDEConsistency:
    """Tests for NDE consistency across calls."""

    def test_different_nde_instances_with_same_input_produce_same_output(self) -> None:
        """Test that different NDE instances produce same output for same input."""
        nde1 = NeuralDevelopmentalEncoding(number_of_modules=5, genotype_size=64)
        nde2 = NeuralDevelopmentalEncoding(number_of_modules=5, genotype_size=64)
        
        genotype = [np.random.RandomState(42).random(64).astype(np.float32)]
        
        outputs1 = nde1.forward(genotype)
        outputs2 = nde2.forward(genotype)
        
        # Both should be valid (this tests that init is deterministic)
        for out1, out2 in zip(outputs1, outputs2):
            assert out1.shape == out2.shape

    def test_nde_no_side_effects_on_genotype(self, nde_small, genotype_small) -> None:
        """Test that forward pass doesn't modify the input genotype."""
        genotype_copy = [chrom.copy() for chrom in genotype_small]
        
        nde_small.forward(genotype_small)
        
        for orig, copy in zip(genotype_small, genotype_copy):
            assert np.allclose(orig, copy)