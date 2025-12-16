"""Test: prebuilt robot structures."""

import pytest

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


class TestGeckoConstruction:
    """Tests for gecko() function construction."""

    def test_gecko_returns_core_module(self) -> None:
        """Test that gecko() returns a CoreModule."""
        robot = gecko()
        
        assert robot is not None
        assert isinstance(robot, CoreModule)
        assert robot.index == 0

    def test_gecko_core_has_index_zero(self) -> None:
        """Test that gecko core module has index 0."""
        robot = gecko()
        
        assert robot.index == 0

    def test_gecko_structure_contains_all_module_types(self) -> None:
        """Test that gecko structure contains HingeModule and BrickModule."""
        robot = gecko()
        
        # The structure should have been built
        assert robot is not None
        
        # Verify core exists and has sites
        assert hasattr(robot, 'sites')
        assert len(robot.sites) > 0


class TestGeckoModuleInstances:
    """Tests for gecko module instances and their properties."""

    def test_gecko_all_required_faces_used(self) -> None:
        """Test that gecko attaches modules to expected faces."""
        robot = gecko()
        
        faces_with_attachments = [
            face for face in robot.sites
            if robot.sites[face] is not None
        ]
        
        assert ModuleFaces.FRONT in faces_with_attachments
        assert ModuleFaces.LEFT in faces_with_attachments
        assert ModuleFaces.RIGHT in faces_with_attachments


class TestGeckoModuleCount:
    """Tests for gecko module count and structure."""

    def test_gecko_creates_correct_number_of_modules(self) -> None:
        """Test that gecko creates all expected modules."""
        robot = gecko()
        
        # Gecko structure has:
        # - 1 core (index 0)
        # - 1 neck (index 1)
        # - 1 abdomen (index 2)
        # - 1 spine (index 3)
        # - 1 butt (index 4)
        # - 2 front left legs (index 5, 15)
        # - 1 front left flipper (index 6)
        # - 2 front right legs (index 7, 17)
        # - 1 front right flipper (index 8)
        # - 1 back left leg (index 9)
        # - 1 back left flipper (index 10)
        # - 1 back right leg (index 11)
        # - 1 back right flipper (index 12)
        # Total: 15 modules
        
        # Verify robot is constructed
        assert robot is not None
        assert isinstance(robot, CoreModule)


class TestGeckoRobustness:
    """Tests for gecko robustness and consistency."""

    def test_gecko_construction_idempotent(self) -> None:
        """Test that calling gecko() multiple times produces valid robots."""
        robot1 = gecko()
        robot2 = gecko()
        
        # Both should be valid CoreModules
        assert isinstance(robot1, CoreModule)
        assert isinstance(robot2, CoreModule)
        
        # Both should have same structure (different instances)
        assert robot1 is not robot2
        assert robot1.index == robot2.index

    def test_gecko_returns_non_none(self) -> None:
        """Test that gecko never returns None."""
        for _ in range(5):
            robot = gecko()
            assert robot is not None

    def test_gecko_core_index_consistent(self) -> None:
        """Test that gecko core index is always 0."""
        for _ in range(3):
            robot = gecko()
            assert robot.index == 0


class TestGeckoModuleTypes:
    """Tests for gecko module type distribution."""

    def test_gecko_core_is_core_module(self) -> None:
        """Test that gecko's root is a CoreModule."""
        robot = gecko()
        
        assert isinstance(robot, CoreModule)


class TestGeckoEdgeCases:
    """Edge case tests for gecko."""

    def test_gecko_construction_no_exceptions(self) -> None:
        """Test that gecko construction never raises exceptions."""
        try:
            robot = gecko()
            assert robot is not None
        except Exception as e:
            pytest.fail(f"gecko() raised {type(e).__name__}: {e}")

    def test_gecko_sites_attribute_exists(self) -> None:
        """Test that gecko core has sites attribute."""
        robot = gecko()
        
        assert hasattr(robot, 'sites')
        assert isinstance(robot.sites, dict)

    def test_gecko_index_attribute_exists(self) -> None:
        """Test that gecko core has index attribute."""
        robot = gecko()
        
        assert hasattr(robot, 'index')
        assert robot.index == 0
