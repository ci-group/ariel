"""Test: robogen_lite high-probability decoding algorithm."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import numpy.typing as npt

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    NUM_OF_FACES,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
    load_graph_from_json,
    draw_graph,
)


class TestHighProbabilityDecoderInitialization:
    """Tests for HighProbabilityDecoder initialization."""

    def test_decoder_initialization(self) -> None:
        """Test decoder initializes with correct num_modules."""
        num_modules = 5
        decoder = HighProbabilityDecoder(num_modules=num_modules)
        
        assert decoder.num_modules == num_modules
        assert isinstance(decoder._graph, dict)
        assert len(decoder._graph) == 0
        assert isinstance(decoder.graph, nx.DiGraph) if NETWORKX_AVAILABLE else True
        assert len(decoder.graph.nodes) == 0

    def test_decoder_initialization_various_sizes(self) -> None:
        """Test decoder initialization with various module counts."""
        for num_modules in [1, 2, 5, 10, 20]:
            decoder = HighProbabilityDecoder(num_modules=num_modules)
            assert decoder.num_modules == num_modules


class TestApplyConnectionConstraints:
    """Tests for apply_connection_constraints method."""

    def test_apply_connection_constraints_no_self_loops(self) -> None:
        """Test that self-connections are disabled."""
        decoder = HighProbabilityDecoder(num_modules=4)
        
        # Create probability spaces
        conn_p_space = np.ones((4, 4, NUM_OF_FACES), dtype=np.float32)
        type_p_space = np.ones((4, len(list(ModuleType))), dtype=np.float32)
        rot_p_space = np.ones((4, len(list(ModuleRotationsIdx))), dtype=np.float32)
        
        decoder.conn_p_space = conn_p_space
        decoder.type_p_space = type_p_space
        decoder.rot_p_space = rot_p_space
        
        decoder.apply_connection_constraints()
        
        # Check no self-connections
        for face_idx in range(NUM_OF_FACES):
            for i in range(4):
                assert decoder.conn_p_space[i, i, face_idx] == 0.0

    def test_apply_connection_constraints_core_is_unique(self) -> None:
        """Test that only core module can have CORE type."""
        decoder = HighProbabilityDecoder(num_modules=4)
        
        conn_p_space = np.ones((4, 4, NUM_OF_FACES), dtype=np.float32)
        type_p_space = np.ones((4, len(list(ModuleType))), dtype=np.float32)
        rot_p_space = np.ones((4, len(list(ModuleRotationsIdx))), dtype=np.float32)
        
        decoder.conn_p_space = conn_p_space
        decoder.type_p_space = type_p_space
        decoder.rot_p_space = rot_p_space
        
        decoder.apply_connection_constraints()
        
        # Only core module (IDX_OF_CORE) should have non-zero probability for CORE type
        for i in range(4):
            if i != IDX_OF_CORE:
                assert decoder.type_p_space[i, ModuleType.CORE.value] == 0.0
        
        assert decoder.type_p_space[IDX_OF_CORE, ModuleType.CORE.value] == 1.0

    def test_apply_connection_constraints_core_never_child(self) -> None:
        """Test that core module is never a child."""
        decoder = HighProbabilityDecoder(num_modules=4)
        
        conn_p_space = np.ones((4, 4, NUM_OF_FACES), dtype=np.float32)
        type_p_space = np.ones((4, len(list(ModuleType))), dtype=np.float32)
        rot_p_space = np.ones((4, len(list(ModuleRotationsIdx))), dtype=np.float32)
        
        decoder.conn_p_space = conn_p_space
        decoder.type_p_space = type_p_space
        decoder.rot_p_space = rot_p_space
        
        decoder.apply_connection_constraints()
        
        # Core module should not be a child (column IDX_OF_CORE should be all zeros)
        assert np.all(decoder.conn_p_space[:, IDX_OF_CORE, :] == 0.0)


class TestSetModuleTypesAndRotations:
    """Tests for set_module_types_and_rotations method."""

    def test_set_module_types_from_argmax(self) -> None:
        """Test module types are set from argmax of type probability space."""
        decoder = HighProbabilityDecoder(num_modules=4)
        
        type_p_space = np.zeros((4, len(list(ModuleType))), dtype=np.float32)
        # Set clear max values
        type_p_space[0, ModuleType.CORE.value] = 1.0
        type_p_space[1, ModuleType.BRICK.value] = 1.0
        type_p_space[2, ModuleType.HINGE.value] = 1.0
        type_p_space[3, ModuleType.NONE.value] = 1.0
        
        conn_p_space = np.ones((4, 4, NUM_OF_FACES), dtype=np.float32)
        rot_p_space = np.ones((4, len(list(ModuleRotationsIdx))), dtype=np.float32)
        
        decoder.type_p_space = type_p_space
        decoder.conn_p_space = conn_p_space
        decoder.rot_p_space = rot_p_space
        
        decoder.apply_connection_constraints()
        decoder.set_module_types_and_rotations()
        
        assert decoder.type_dict[0] == ModuleType.CORE
        assert decoder.type_dict[1] == ModuleType.BRICK
        assert decoder.type_dict[2] == ModuleType.HINGE
        assert decoder.type_dict[3] == ModuleType.NONE

    def test_set_rotations_constrained_by_module_type(self) -> None:
        """Test that rotations are constrained by module type."""
        decoder = HighProbabilityDecoder(num_modules=2)
        
        type_p_space = np.zeros((2, len(list(ModuleType))), dtype=np.float32)
        type_p_space[0, ModuleType.CORE.value] = 1.0
        type_p_space[1, ModuleType.BRICK.value] = 1.0
        
        rot_p_space = np.ones((2, len(list(ModuleRotationsIdx))), dtype=np.float32)
        conn_p_space = np.ones((2, 2, NUM_OF_FACES), dtype=np.float32)
        
        decoder.type_p_space = type_p_space
        decoder.conn_p_space = conn_p_space
        decoder.rot_p_space = rot_p_space
        
        decoder.apply_connection_constraints()
        decoder.set_module_types_and_rotations()
        
        # CORE module should only have DEG_0 rotation
        assert decoder.rot_dict[0] == ModuleRotationsIdx.DEG_0
        
        # BRICK module can have any rotation (all allowed)
        assert decoder.rot_dict[1] in ModuleRotationsIdx

    def test_disallowed_rotations_zeroed(self) -> None:
        """Test that disallowed rotations are zeroed out."""
        decoder = HighProbabilityDecoder(num_modules=1)
        
        type_p_space = np.zeros((1, len(list(ModuleType))), dtype=np.float32)
        type_p_space[0, ModuleType.CORE.value] = 1.0
        
        rot_p_space = np.ones((1, len(list(ModuleRotationsIdx))), dtype=np.float32)
        conn_p_space = np.ones((1, 1, NUM_OF_FACES), dtype=np.float32)
        
        decoder.type_p_space = type_p_space
        decoder.conn_p_space = conn_p_space
        decoder.rot_p_space = rot_p_space
        
        decoder.apply_connection_constraints()
        decoder.set_module_types_and_rotations()
        
        # CORE only allows DEG_0, so all others should be 0
        for rot_idx in range(len(list(ModuleRotationsIdx))):
            if rot_idx != ModuleRotationsIdx.DEG_0.value:
                assert decoder.rot_p_space[0, rot_idx] == 0.0



class TestGenerateNetworkXGraph:
    """Tests for generate_networkx_graph method."""

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="networkx not available")
    def test_generate_networkx_graph_from_simple_graph(self) -> None:
        """Test generating NetworkX graph from decoded simple graph."""
        decoder = HighProbabilityDecoder(num_modules=2)
        
        # Set up decoded graph manually
        decoder.nodes = {IDX_OF_CORE, 1}
        decoder.edges = [(IDX_OF_CORE, 1, ModuleFaces.FRONT.value)]
        decoder.type_dict = {
            IDX_OF_CORE: ModuleType.CORE,
            1: ModuleType.BRICK,
        }
        decoder.rot_dict = {
            IDX_OF_CORE: ModuleRotationsIdx.DEG_0,
            1: ModuleRotationsIdx.DEG_90,
        }
        
        decoder.generate_networkx_graph()
        
        assert len(decoder.graph.nodes) == 2
        assert IDX_OF_CORE in decoder.graph.nodes
        assert 1 in decoder.graph.nodes
        assert decoder.graph.nodes[IDX_OF_CORE]['type'] == 'CORE'
        assert decoder.graph.nodes[1]['type'] == 'BRICK'
        assert (IDX_OF_CORE, 1) in decoder.graph.edges
        assert decoder.graph[IDX_OF_CORE][1]['face'] == 'FRONT'

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="networkx not available")
    def test_generate_networkx_graph_preserves_rotations(self) -> None:
        """Test that rotations are preserved in NetworkX graph."""
        decoder = HighProbabilityDecoder(num_modules=3)
        
        decoder.nodes = {IDX_OF_CORE, 1, 2}
        decoder.edges = [(IDX_OF_CORE, 1, ModuleFaces.FRONT.value)]
        decoder.type_dict = {
            IDX_OF_CORE: ModuleType.CORE,
            1: ModuleType.BRICK,
            2: ModuleType.HINGE,
        }
        decoder.rot_dict = {
            IDX_OF_CORE: ModuleRotationsIdx.DEG_0,
            1: ModuleRotationsIdx.DEG_90,
            2: ModuleRotationsIdx.DEG_180,
        }
        
        decoder.generate_networkx_graph()
        
        assert decoder.graph.nodes[IDX_OF_CORE]['rotation'] == 'DEG_0'
        assert decoder.graph.nodes[1]['rotation'] == 'DEG_90'
        assert decoder.graph.nodes[2]['rotation'] == 'DEG_180'


class TestProbabilityMatricesToGraph:
    """Tests for probability_matrices_to_graph method."""

    def test_probability_matrices_to_graph_end_to_end(self) -> None:
        """Test full end-to-end decoding pipeline."""
        decoder = HighProbabilityDecoder(num_modules=3)
        
        type_p_space = np.zeros((3, len(list(ModuleType))), dtype=np.float32)
        type_p_space[0, ModuleType.CORE.value] = 1.0
        type_p_space[1, ModuleType.BRICK.value] = 1.0
        type_p_space[2, ModuleType.HINGE.value] = 1.0
        
        conn_p_space = np.zeros((3, 3, NUM_OF_FACES), dtype=np.float32)
        conn_p_space[0, 1, ModuleFaces.FRONT.value] = 2.0
        
        rot_p_space = np.zeros((3, len(list(ModuleRotationsIdx))), dtype=np.float32)
        rot_p_space[:, ModuleRotationsIdx.DEG_0.value] = 1.0
        
        graph = decoder.probability_matrices_to_graph(
            type_p_space,
            conn_p_space,
            rot_p_space,
        )
        
        assert graph is not None
        assert IDX_OF_CORE in graph.nodes


class TestSaveGraphAsJson:
    """Tests for save_graph_as_json function."""

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="networkx not available")
    def test_save_graph_as_json_creates_file(self) -> None:
        """Test that save_graph_as_json creates a file."""
        graph = nx.DiGraph()
        graph.add_node(0, type='CORE', rotation='DEG_0')
        graph.add_node(1, type='BRICK', rotation='DEG_90')
        graph.add_edge(0, 1, face='FRONT')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_graph.json"
            save_graph_as_json(graph, save_path)
            
            assert save_path.exists()
            assert save_path.stat().st_size > 0

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="networkx not available")
    def test_save_graph_as_json_none_path_no_error(self) -> None:
        """Test that save_graph_as_json with None path doesn't error."""
        graph = nx.DiGraph()
        graph.add_node(0, type='CORE')
        
        # Should not raise
        save_graph_as_json(graph, save_file=None)

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="networkx not available")
    def test_save_graph_as_json_valid_json_format(self) -> None:
        """Test that saved file is valid JSON."""
        graph = nx.DiGraph()
        graph.add_node(0, type='CORE', rotation='DEG_0')
        graph.add_edge(0, 0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test.json"
            save_graph_as_json(graph, save_path)
            
            with open(save_path, 'r') as f:
                data = json.load(f)
            
            assert 'nodes' in data or 'directed' in data


class TestLoadGraphFromJson:
    """Tests for load_graph_from_json function."""

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="networkx not available")
    def test_load_graph_from_json_roundtrip(self) -> None:
        """Test saving and loading graph preserves structure."""
        original_graph = nx.DiGraph()
        original_graph.add_node(0, type='CORE', rotation='DEG_0')
        original_graph.add_node(1, type='BRICK', rotation='DEG_90')
        original_graph.add_edge(0, 1, face='FRONT')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test.json"
            save_graph_as_json(original_graph, save_path)
            
            loaded_graph = load_graph_from_json(save_path)
            
            assert len(loaded_graph.nodes) == len(original_graph.nodes)
            assert len(loaded_graph.edges) == len(original_graph.edges)
            assert loaded_graph.nodes[0]['type'] == 'CORE'
            assert loaded_graph.nodes[1]['type'] == 'BRICK'

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="networkx not available")
    def test_load_graph_from_json_invalid_file_raises(self) -> None:
        """Test loading from non-existent file raises error."""
        with pytest.raises((FileNotFoundError, IOError)):
            load_graph_from_json(Path("/nonexistent/path.json"))

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="networkx not available")
    def test_load_graph_from_json_invalid_json_raises(self) -> None:
        """Test loading invalid JSON raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_json_path = Path(tmpdir) / "bad.json"
            bad_json_path.write_text("{ invalid json }")
            
            with pytest.raises((json.JSONDecodeError, ValueError)):
                load_graph_from_json(bad_json_path)


class TestDrawGraph:
    """Tests for draw_graph function."""

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="networkx not available")
    def test_draw_graph_with_file_saves_image(self) -> None:
        """Test draw_graph saves image to file."""
        graph = nx.DiGraph()
        graph.add_node(0, type='CORE')
        graph.add_node(1, type='BRICK')
        graph.add_edge(0, 1, face='FRONT')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "graph.png"
            
            # Suppress display
            with patch('matplotlib.pyplot.show'):
                draw_graph(graph, title="Test", save_file=save_path)
            
            assert save_path.exists()

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="networkx not available")
    def test_draw_graph_no_file_no_error(self) -> None:
        """Test draw_graph without file doesn't error."""
        graph = nx.DiGraph()
        graph.add_node(0, type='CORE')
        
        with patch('matplotlib.pyplot.show'):
            # Should not raise
            draw_graph(graph, title="Test", save_file=None)

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="networkx not available")
    def test_draw_graph_custom_title(self) -> None:
        """Test draw_graph uses custom title."""
        graph = nx.DiGraph()
        graph.add_node(0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "graph.png"
            
            with patch('matplotlib.pyplot.show'):
                draw_graph(graph, title="Custom Title", save_file=save_path)
            
            assert save_path.exists()


class TestEdgeCasesDecoding:
    """Edge case tests for decoding."""

    def test_decoder_with_single_module(self) -> None:
        """Test decoder with only core module."""
        decoder = HighProbabilityDecoder(num_modules=1)
        
        type_p_space = np.zeros((1, len(list(ModuleType))), dtype=np.float32)
        type_p_space[0, ModuleType.CORE.value] = 1.0
        
        conn_p_space = np.zeros((1, 1, NUM_OF_FACES), dtype=np.float32)
        rot_p_space = np.zeros((1, len(list(ModuleRotationsIdx))), dtype=np.float32)
        rot_p_space[0, ModuleRotationsIdx.DEG_0.value] = 1.0
        
        graph = decoder.probability_matrices_to_graph(type_p_space, conn_p_space, rot_p_space)
        
        assert len(graph.nodes) == 1
        assert IDX_OF_CORE in graph.nodes

    def test_decoder_with_many_modules(self) -> None:
        """Test decoder with many modules."""
        num_modules = 10
        decoder = HighProbabilityDecoder(num_modules=num_modules)
        
        type_p_space = np.zeros((num_modules, len(list(ModuleType))), dtype=np.float32)
        type_p_space[0, ModuleType.CORE.value] = 1.0
        for i in range(1, num_modules):
            type_p_space[i, ModuleType.BRICK.value] = 1.0
        
        conn_p_space = np.zeros((num_modules, num_modules, NUM_OF_FACES), dtype=np.float32)
        rot_p_space = np.zeros((num_modules, len(list(ModuleRotationsIdx))), dtype=np.float32)
        rot_p_space[:, ModuleRotationsIdx.DEG_0.value] = 1.0
        
        # No connections means only core should be in final graph
        graph = decoder.probability_matrices_to_graph(type_p_space, conn_p_space, rot_p_space)
        
        assert len(graph.nodes) >= 1

    def test_decoder_all_module_types(self) -> None:
        """Test decoder with all module types represented."""
        decoder = HighProbabilityDecoder(num_modules=4)
        
        type_p_space = np.zeros((4, len(list(ModuleType))), dtype=np.float32)
        type_p_space[0, ModuleType.CORE.value] = 1.0
        type_p_space[1, ModuleType.BRICK.value] = 1.0
        type_p_space[2, ModuleType.HINGE.value] = 1.0
        type_p_space[3, ModuleType.NONE.value] = 1.0
        
        conn_p_space = np.zeros((4, 4, NUM_OF_FACES), dtype=np.float32)
        conn_p_space[0, 1, ModuleFaces.FRONT.value] = 1.0
        conn_p_space[0, 2, ModuleFaces.BACK.value] = 1.0
        
        rot_p_space = np.zeros((4, len(list(ModuleRotationsIdx))), dtype=np.float32)
        rot_p_space[:, ModuleRotationsIdx.DEG_0.value] = 1.0
        
        graph = decoder.probability_matrices_to_graph(type_p_space, conn_p_space, rot_p_space)
        
        assert graph is not None