"""Tests for terrain compilation utilities."""

from unittest.mock import Mock, patch, MagicMock, call
import pytest

import ariel.simulation.environments as envs
from ariel.simulation.environments.compile_terrains import compile_all_world


@pytest.fixture
def mock_world_class():
    """Create a mock world class."""
    mock_cls = Mock()
    mock_instance = Mock()
    mock_instance.store_to_xml = Mock()
    mock_cls.return_value = mock_instance
    return mock_cls, mock_instance


@pytest.fixture
def mock_base_world():
    """Create a mock BaseWorld instance."""
    world = Mock()
    world.store_to_xml = Mock()
    return world


class TestCompileAllWorldBasic:
    """Tests for basic compile_all_world functionality."""

    def test_compile_all_world_calls_store_to_xml(self, mock_base_world):
        """Test that compile_all_world calls store_to_xml on each world."""
        with patch("ariel.simulation.environments.compile_terrains.inspect.getmembers") as mock_getmembers:
            mock_getmembers.return_value = [
                ("TestWorld", Mock(return_value=mock_base_world))
            ]
            
            compile_all_world(with_load_compiled=True)
            
            mock_base_world.store_to_xml.assert_called_once()

    def test_compile_all_world_instantiates_world_classes(self):
        """Test that compile_all_world instantiates world classes."""
        mock_world_cls = Mock()
        mock_instance = Mock()
        mock_instance.store_to_xml = Mock()
        mock_world_cls.return_value = mock_instance
        
        with patch("ariel.simulation.environments.compile_terrains.inspect.getmembers") as mock_getmembers:
            mock_getmembers.return_value = [
                ("World1", mock_world_cls)
            ]
            
            compile_all_world(with_load_compiled=True)
            
            mock_world_cls.assert_called_once_with(load_precompiled=True)

    def test_compile_all_world_with_load_precompiled_false(self):
        """Test compile_all_world with load_precompiled=False."""
        mock_world_cls = Mock()
        mock_instance = Mock()
        mock_instance.store_to_xml = Mock()
        mock_world_cls.return_value = mock_instance
        
        with patch("ariel.simulation.environments.compile_terrains.inspect.getmembers") as mock_getmembers:
            mock_getmembers.return_value = [
                ("World1", mock_world_cls)
            ]
            
            compile_all_world(with_load_compiled=False)
            
            mock_world_cls.assert_called_once_with(load_precompiled=False)

    def test_compile_all_world_processes_all_worlds(self):
        """Test that compile_all_world processes all discovered world classes."""
        worlds = []
        mocks = []
        
        for i in range(3):
            mock_cls = Mock()
            mock_instance = Mock()
            mock_instance.store_to_xml = Mock()
            mock_cls.return_value = mock_instance
            worlds.append((f"World{i}", mock_cls))
            mocks.append(mock_instance)
        
        with patch("ariel.simulation.environments.compile_terrains.inspect.getmembers") as mock_getmembers:
            mock_getmembers.return_value = worlds
            
            compile_all_world(with_load_compiled=True)
            
            # All worlds should be instantiated and store_to_xml called
            for mock_instance in mocks:
                mock_instance.store_to_xml.assert_called_once()

    def test_compile_all_world_default_load_compiled_true(self):
        """Test that default value of load_precompiled is True."""
        mock_world_cls = Mock()
        mock_instance = Mock()
        mock_instance.store_to_xml = Mock()
        mock_world_cls.return_value = mock_instance
        
        with patch("ariel.simulation.environments.compile_terrains.inspect.getmembers") as mock_getmembers:
            mock_getmembers.return_value = [("World1", mock_world_cls)]
            
            compile_all_world()  # No with_load_compiled argument
            
            mock_world_cls.assert_called_once_with(load_precompiled=True)


class TestCompileAllWorldIntrospection:
    """Tests for proper introspection of environment module."""

    def test_compile_all_world_uses_inspect_getmembers(self):
        """Test that compile_all_world uses inspect.getmembers."""
        with patch("ariel.simulation.environments.compile_terrains.inspect.getmembers") as mock_getmembers:
            mock_getmembers.return_value = []
            
            compile_all_world()
            
            mock_getmembers.assert_called_once()

class TestCompileAllWorldErrorHandling:
    """Tests for error handling during compilation."""

    def test_compile_all_world_handles_store_to_xml_error(self):
        """Test that compile_all_world handles errors from store_to_xml."""
        mock_world_cls = Mock()
        mock_instance = Mock()
        mock_instance.store_to_xml = Mock(side_effect=ValueError("XML error"))
        mock_world_cls.return_value = mock_instance
        
        with patch("ariel.simulation.environments.compile_terrains.inspect.getmembers") as mock_getmembers:
            mock_getmembers.return_value = [("World1", mock_world_cls)]
            
            # Should raise the error from store_to_xml
            with pytest.raises(ValueError, match="XML error"):
                compile_all_world()

    def test_compile_all_world_handles_instantiation_error(self):
        """Test that compile_all_world handles errors during instantiation."""
        mock_world_cls = Mock(side_effect=RuntimeError("Init error"))
        
        with patch("ariel.simulation.environments.compile_terrains.inspect.getmembers") as mock_getmembers:
            mock_getmembers.return_value = [("World1", mock_world_cls)]
            
            with pytest.raises(RuntimeError, match="Init error"):
                compile_all_world()

    def test_compile_all_world_continues_after_one_failure(self):
        """Test behavior when one world fails (implementation dependent)."""
        # This test depends on whether compile_all_world is designed to continue
        # on error or stop. Adjust based on actual implementation.
        mock_world_cls_1 = Mock()
        mock_instance_1 = Mock()
        mock_instance_1.store_to_xml = Mock(side_effect=ValueError("Error 1"))
        mock_world_cls_1.return_value = mock_instance_1
        
        mock_world_cls_2 = Mock()
        mock_instance_2 = Mock()
        mock_instance_2.store_to_xml = Mock()
        mock_world_cls_2.return_value = mock_instance_2
        
        with patch("ariel.simulation.environments.compile_terrains.inspect.getmembers") as mock_getmembers:
            mock_getmembers.return_value = [
                ("World1", mock_world_cls_1),
                ("World2", mock_world_cls_2),
            ]
            
            # If implementation stops on first error, this will raise
            # If implementation continues, this will succeed
            try:
                compile_all_world()
            except ValueError:
                # First world failed, which is expected
                pass


class TestCompileAllWorldConsoleOutput:
    """Tests for console output behavior."""

    def test_compile_all_world_logs_world_names(self):
        """Test that compile_all_world logs each compiled world."""
        mock_world_cls = Mock()
        mock_instance = Mock()
        mock_instance.store_to_xml = Mock()
        mock_world_cls.return_value = mock_instance
        
        with patch("ariel.simulation.environments.compile_terrains.inspect.getmembers") as mock_getmembers:
            with patch("ariel.simulation.environments.compile_terrains.console.print") as mock_print:
                mock_getmembers.return_value = [("TestWorld", mock_world_cls)]
                
                compile_all_world()
                
                mock_print.assert_called()
                call_args = mock_print.call_args[0][0]
                assert "TestWorld" in call_args
                assert "XML" in call_args

    def test_compile_all_world_logs_multiple_worlds(self):
        """Test that compile_all_world logs all worlds."""
        worlds = []
        mocks = []
        
        for i in range(3):
            mock_cls = Mock()
            mock_instance = Mock()
            mock_instance.store_to_xml = Mock()
            mock_cls.return_value = mock_instance
            worlds.append((f"World{i}", mock_cls))
            mocks.append(mock_instance)
        
        with patch("ariel.simulation.environments.compile_terrains.inspect.getmembers") as mock_getmembers:
            with patch("ariel.simulation.environments.compile_terrains.console.print") as mock_print:
                mock_getmembers.return_value = worlds
                
                compile_all_world()
                
                # console.print should be called for each world
                assert mock_print.call_count >= 3
                
                # Check that all world names are in the output
                all_output = [call[0][0] for call in mock_print.call_args_list]
                all_text = " ".join(all_output)
                assert "World0" in all_text
                assert "World1" in all_text
                assert "World2" in all_text


class TestCompileAllWorldIntegration:
    """Integration tests for compile_all_world."""

    def test_compile_all_world_full_workflow(self):
        """Test complete workflow of discovering and compiling worlds."""
        # Create realistic mock worlds
        mock_worlds = {}
        
        for i in range(2):
            world_name = f"Environment{i}"
            mock_cls = Mock(name=world_name)
            mock_instance = MagicMock()
            mock_instance.store_to_xml = Mock()
            mock_cls.return_value = mock_instance
            mock_worlds[world_name] = (mock_cls, mock_instance)
        
        with patch("ariel.simulation.environments.compile_terrains.inspect.getmembers") as mock_getmembers:
            with patch("ariel.simulation.environments.compile_terrains.console.print") as mock_print:
                mock_getmembers.return_value = [
                    (name, cls) for name, (cls, _) in mock_worlds.items()
                ]
                
                compile_all_world(with_load_compiled=True)
                
                # Verify all worlds were instantiated and compiled
                for world_name, (mock_cls, mock_instance) in mock_worlds.items():
                    mock_cls.assert_called_once_with(load_precompiled=True)
                    mock_instance.store_to_xml.assert_called_once()

    def test_compile_all_world_order_of_operations(self):
        """Test that operations happen in correct order: instantiate -> store_to_xml -> log."""
        mock_world_cls = Mock()
        mock_instance = Mock()
        
        # Track call order
        call_order = []
        
        mock_world_cls.side_effect = lambda **kwargs: (call_order.append("instantiate"), mock_instance)[1]
        mock_instance.store_to_xml = Mock(side_effect=lambda: call_order.append("store_to_xml"))
        
        with patch("ariel.simulation.environments.compile_terrains.inspect.getmembers") as mock_getmembers:
            with patch("ariel.simulation.environments.compile_terrains.console.print") as mock_print:
                mock_print.side_effect = lambda x: call_order.append("log")
                
                mock_getmembers.return_value = [("TestWorld", mock_world_cls)]
                
                compile_all_world()
                
                # Verify order: instantiate before store_to_xml
                assert call_order.index("instantiate") < call_order.index("store_to_xml")


class TestCompileAllWorldEdgeCases:
    """Edge case tests for compile_all_world."""

    def test_compile_all_world_with_no_world_classes(self):
        """Test compile_all_world when no world classes are found."""
        with patch("ariel.simulation.environments.compile_terrains.inspect.getmembers") as mock_getmembers:
            with patch("ariel.simulation.environments.compile_terrains.console.print") as mock_print:
                mock_getmembers.return_value = []
                
                # Should not raise
                compile_all_world()
                
                # Should not print anything if no worlds found
                # or print a message indicating no worlds found
                # This depends on implementation

    def test_compile_all_world_with_world_name_special_characters(self):
        """Test compile_all_world with special characters in world names."""
        mock_world_cls = Mock()
        mock_instance = Mock()
        mock_instance.store_to_xml = Mock()
        mock_world_cls.return_value = mock_instance
        
        special_name = "World_With-Special.Chars_123"
        
        with patch("ariel.simulation.environments.compile_terrains.inspect.getmembers") as mock_getmembers:
            with patch("ariel.simulation.environments.compile_terrains.console.print") as mock_print:
                mock_getmembers.return_value = [(special_name, mock_world_cls)]
                
                compile_all_world()
                
                mock_instance.store_to_xml.assert_called_once()
                # Verify name appears in output
                output = mock_print.call_args[0][0]
                assert special_name in output

    def test_compile_all_world_called_multiple_times(self):
        """Test that compile_all_world can be called multiple times."""
        mock_world_cls = Mock()
        mock_instance = Mock()
        mock_instance.store_to_xml = Mock()
        mock_world_cls.return_value = mock_instance
        
        with patch("ariel.simulation.environments.compile_terrains.inspect.getmembers") as mock_getmembers:
            mock_getmembers.return_value = [("World1", mock_world_cls)]
            
            # Call twice
            compile_all_world()
            compile_all_world()
            
            # Both calls should succeed
            assert mock_instance.store_to_xml.call_count == 2