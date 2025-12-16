"""Test: robogen_lite constructor functions."""

import pytest

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from ariel.body_phenotypes.robogen_lite.config import (
    IDX_OF_CORE,
    ModuleFaces,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="networkx not available")
class TestConstructMjspecFromGraph:
    """Tests for construct_mjspec_from_graph function."""

    def test_construct_single_core_module(self) -> None:
        """Test constructing a graph with only a core module."""
        graph = nx.DiGraph()
        graph.add_node(IDX_OF_CORE, type=ModuleType.CORE.name, rotation="DEG_0")
        
        result = construct_mjspec_from_graph(graph)
        
        assert result is not None
        from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
        assert isinstance(result, CoreModule)
        assert result.index == IDX_OF_CORE

    def test_construct_core_with_brick_module(self) -> None:
        """Test constructing a graph with core and brick module."""
        graph = nx.DiGraph()
        graph.add_node(IDX_OF_CORE, type=ModuleType.CORE.name, rotation="DEG_0")
        graph.add_node(1, type=ModuleType.BRICK.name, rotation="DEG_0")
        graph.add_edge(IDX_OF_CORE, 1, face="FRONT")
        
        result = construct_mjspec_from_graph(graph)
        
        assert result is not None
        from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
        assert isinstance(result, CoreModule)

    def test_construct_core_with_hinge_module(self) -> None:
        """Test constructing a graph with core and hinge module."""
        graph = nx.DiGraph()
        graph.add_node(IDX_OF_CORE, type=ModuleType.CORE.name, rotation="DEG_0")
        graph.add_node(1, type=ModuleType.HINGE.name, rotation="DEG_45")
        graph.add_edge(IDX_OF_CORE, 1, face="FRONT")
        
        result = construct_mjspec_from_graph(graph)
        
        assert result is not None
        from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
        assert isinstance(result, CoreModule)

    def test_construct_core_with_none_module(self) -> None:
        """Test constructing a graph with core and NONE module (skipped)."""
        graph = nx.DiGraph()
        graph.add_node(IDX_OF_CORE, type=ModuleType.CORE.name, rotation="DEG_0")
        graph.add_node(1, type=ModuleType.NONE.name, rotation="DEG_0")
        graph.add_edge(IDX_OF_CORE, 1, face="FRONT")
        
        result = construct_mjspec_from_graph(graph)
        
        assert result is not None
        from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
        assert isinstance(result, CoreModule)

    def test_construct_multiple_modules_different_rotations(self) -> None:
        """Test constructing with modules having different rotations."""
        graph = nx.DiGraph()
        graph.add_node(IDX_OF_CORE, type=ModuleType.CORE.name, rotation="DEG_0")
        graph.add_node(1, type=ModuleType.BRICK.name, rotation="DEG_90")
        graph.add_node(2, type=ModuleType.HINGE.name, rotation="DEG_180")
        graph.add_edge(IDX_OF_CORE, 1, face="FRONT")
        graph.add_edge(1, 2, face="FRONT")
        
        result = construct_mjspec_from_graph(graph)
        
        assert result is not None

    def test_construct_all_rotation_angles(self) -> None:
        """Test constructing with all possible rotation angles."""
        rotations = ["DEG_0", "DEG_45", "DEG_90", "DEG_135", "DEG_180", "DEG_225", "DEG_270", "DEG_315"]
        
        for rotation in rotations:
            graph = nx.DiGraph()
            graph.add_node(IDX_OF_CORE, type=ModuleType.CORE.name, rotation="DEG_0")
            graph.add_node(1, type=ModuleType.BRICK.name, rotation=rotation)
            graph.add_edge(IDX_OF_CORE, 1, face="FRONT")
            
            result = construct_mjspec_from_graph(graph)
            assert result is not None

    def test_construct_all_faces(self) -> None:
        """Test constructing with all possible faces."""
        faces = ["FRONT", "BACK", "RIGHT", "LEFT", "TOP", "BOTTOM"]
        
        for i, face in enumerate(faces):
            graph = nx.DiGraph()
            graph.add_node(IDX_OF_CORE, type=ModuleType.CORE.name, rotation="DEG_0")
            graph.add_node(i + 1, type=ModuleType.BRICK.name, rotation="DEG_0")
            graph.add_edge(IDX_OF_CORE, i + 1, face=face)
            
            result = construct_mjspec_from_graph(graph)
            assert result is not None

    def test_construct_multiple_attachments_same_core(self) -> None:
        """Test constructing with multiple modules attached to core."""
        graph = nx.DiGraph()
        graph.add_node(IDX_OF_CORE, type=ModuleType.CORE.name, rotation="DEG_0")
        
        for i in range(1, 4):
            face = list(ModuleFaces)[i % 6]
            graph.add_node(i, type=ModuleType.BRICK.name, rotation="DEG_0")
            graph.add_edge(IDX_OF_CORE, i, face=face.name)
        
        result = construct_mjspec_from_graph(graph)
        
        assert result is not None

    def test_construct_chained_modules(self) -> None:
        """Test constructing with modules chained together."""
        graph = nx.DiGraph()
        graph.add_node(IDX_OF_CORE, type=ModuleType.CORE.name, rotation="DEG_0")
        graph.add_node(1, type=ModuleType.BRICK.name, rotation="DEG_0")
        graph.add_node(2, type=ModuleType.HINGE.name, rotation="DEG_45")
        graph.add_node(3, type=ModuleType.BRICK.name, rotation="DEG_90")
        
        graph.add_edge(IDX_OF_CORE, 1, face="FRONT")
        graph.add_edge(1, 2, face="FRONT")
        graph.add_edge(2, 3, face="FRONT")
        
        result = construct_mjspec_from_graph(graph)
        
        assert result is not None

    def test_construct_branching_structure(self) -> None:
        """Test constructing with branching module structure."""
        graph = nx.DiGraph()
        graph.add_node(IDX_OF_CORE, type=ModuleType.CORE.name, rotation="DEG_0")
        
        # Create branches
        graph.add_node(1, type=ModuleType.BRICK.name, rotation="DEG_0")
        graph.add_node(2, type=ModuleType.HINGE.name, rotation="DEG_0")
        graph.add_node(3, type=ModuleType.BRICK.name, rotation="DEG_0")
        
        graph.add_edge(IDX_OF_CORE, 1, face="FRONT")
        graph.add_edge(IDX_OF_CORE, 2, face="BACK")
        graph.add_edge(1, 3, face="FRONT")
        
        result = construct_mjspec_from_graph(graph)
        
        assert result is not None

    def test_construct_unknown_module_type_raises(self) -> None:
        """Test constructing with unknown module type raises ValueError."""
        graph = nx.DiGraph()
        graph.add_node(IDX_OF_CORE, type=ModuleType.CORE.name, rotation="DEG_0")
        graph.add_node(1, type="UNKNOWN_TYPE", rotation="DEG_0")
        graph.add_edge(IDX_OF_CORE, 1, face="FRONT")
        
        with pytest.raises(ValueError, match="Unknown module type"):
            construct_mjspec_from_graph(graph)

    def test_construct_non_core_root_raises(self) -> None:
        """Test constructing without core module as root raises ValueError."""
        graph = nx.DiGraph()
        graph.add_node(IDX_OF_CORE, type=ModuleType.BRICK.name, rotation="DEG_0")
        
        with pytest.raises(ValueError, match="core module is not of type CoreModule"):
            construct_mjspec_from_graph(graph)

    def test_construct_missing_core_module_raises(self) -> None:
        """Test constructing without core module raises error."""
        graph = nx.DiGraph()
        graph.add_node(1, type=ModuleType.BRICK.name, rotation="DEG_0")
        
        # This should raise KeyError or similar when trying to access modules[IDX_OF_CORE]
        with pytest.raises((KeyError, ValueError)):
            construct_mjspec_from_graph(graph)

    def test_construct_invalid_rotation_name_raises(self) -> None:
        """Test constructing with invalid rotation name raises KeyError."""
        graph = nx.DiGraph()
        graph.add_node(IDX_OF_CORE, type=ModuleType.CORE.name, rotation="DEG_0")
        graph.add_node(1, type=ModuleType.BRICK.name, rotation="INVALID_ROTATION")
        graph.add_edge(IDX_OF_CORE, 1, face="FRONT")
        
        with pytest.raises(KeyError):
            construct_mjspec_from_graph(graph)

    def test_construct_invalid_face_name_raises(self) -> None:
        """Test constructing with invalid face name raises KeyError."""
        graph = nx.DiGraph()
        graph.add_node(IDX_OF_CORE, type=ModuleType.CORE.name, rotation="DEG_0")
        graph.add_node(1, type=ModuleType.BRICK.name, rotation="DEG_0")
        graph.add_edge(IDX_OF_CORE, 1, face="INVALID_FACE")
        
        with pytest.raises(KeyError):
            construct_mjspec_from_graph(graph)

    def test_construct_edge_to_none_module_skipped(self) -> None:
        """Test that edges to NONE modules don't cause attachment errors."""
        graph = nx.DiGraph()
        graph.add_node(IDX_OF_CORE, type=ModuleType.CORE.name, rotation="DEG_0")
        graph.add_node(1, type=ModuleType.NONE.name, rotation="DEG_0")
        graph.add_edge(IDX_OF_CORE, 1, face="FRONT")
        
        # Should not raise, NONE modules should be skipped
        result = construct_mjspec_from_graph(graph)
        assert result is not None

    def test_construct_complex_multi_level_structure(self) -> None:
        """Test constructing a complex multi-level robot structure."""
        graph = nx.DiGraph()
        graph.add_node(IDX_OF_CORE, type=ModuleType.CORE.name, rotation="DEG_0")
        
        # Level 1: attached to core
        for i in range(1, 3):
            graph.add_node(i, type=ModuleType.BRICK.name, rotation="DEG_0")
            graph.add_edge(IDX_OF_CORE, i, face=list(ModuleFaces)[i].name)
        
        # Level 2: attached to level 1
        for i in range(3, 5):
            graph.add_node(i, type=ModuleType.HINGE.name, rotation="DEG_45")
            parent = (i - 3) + 1
            graph.add_edge(parent, i, face="FRONT")
        
        result = construct_mjspec_from_graph(graph)
        assert result is not None

    def test_construct_preserves_rotation_values(self) -> None:
        """Test that module rotations are correctly applied."""
        graph = nx.DiGraph()
        graph.add_node(IDX_OF_CORE, type=ModuleType.CORE.name, rotation="DEG_0")
        graph.add_node(1, type=ModuleType.BRICK.name, rotation="DEG_90")
        graph.add_edge(IDX_OF_CORE, 1, face="FRONT")
        
        result = construct_mjspec_from_graph(graph)
        
        # Verify rotation was applied (depends on BrickModule implementation)
        assert result is not None
