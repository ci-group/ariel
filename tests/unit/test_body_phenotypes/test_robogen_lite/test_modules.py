"""Test: initialization of robogen_lite module classes."""

import mujoco
import pytest

from ariel.body_phenotypes.robogen_lite.config import IDX_OF_CORE, ModuleType
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule


def test_core_module_initialization() -> None:
    """Simply instantiate the CoreModule with the correct index."""
    core = CoreModule(index=IDX_OF_CORE)
    assert core.index == IDX_OF_CORE
    assert core.module_type == ModuleType.CORE
    del core


def test_core_module_has_spec() -> None:
    """CoreModule exposes a MjSpec that can be compiled."""
    core = CoreModule(index=IDX_OF_CORE)
    model = core.spec.compile()
    data = mujoco.MjData(model)
    del core, model, data


def test_core_module_has_all_sites() -> None:
    """CoreModule sites dict contains all six faces."""
    from ariel.body_phenotypes.robogen_lite.config import ModuleFaces

    core = CoreModule(index=IDX_OF_CORE)
    expected_faces = {
        ModuleFaces.FRONT,
        ModuleFaces.BACK,
        ModuleFaces.LEFT,
        ModuleFaces.RIGHT,
        ModuleFaces.TOP,
        ModuleFaces.BOTTOM,
    }
    assert set(core.sites.keys()) == expected_faces
    del core


def test_core_module_wrong_index_raises() -> None:
    """CoreModule raises ValueError if initialized with a non-zero index."""
    with pytest.raises(ValueError):
        CoreModule(index=1)


def test_core_module_rotation_zero_is_noop() -> None:
    """CoreModule.rotate(0) does not raise."""
    core = CoreModule(index=IDX_OF_CORE)
    core.rotate(0)
    del core


def test_core_module_nonzero_rotation_raises() -> None:
    """CoreModule.rotate with non-zero angle raises AttributeError."""
    core = CoreModule(index=IDX_OF_CORE)
    with pytest.raises(AttributeError):
        core.rotate(90)
    del core


def test_hinge_module_initialization() -> None:
    """Simply instantiate the HingeModule."""
    hinge = HingeModule(index=1)
    assert hinge.index == 1
    assert hinge.module_type == ModuleType.HINGE
    del hinge


def test_hinge_module_has_front_site() -> None:
    """HingeModule sites dict contains the FRONT face."""
    from ariel.body_phenotypes.robogen_lite.config import ModuleFaces

    hinge = HingeModule(index=1)
    assert ModuleFaces.FRONT in hinge.sites
    del hinge


def test_hinge_module_rotate_degrees() -> None:
    """HingeModule.rotate accepts 0, 45, and 90 without error."""
    for angle in (0, 45, 90):
        hinge = HingeModule(index=1)
        hinge.rotate(angle)
        del hinge


def test_brick_module_initialization() -> None:
    """Simply instantiate the BrickModule."""
    brick = BrickModule(index=2)
    assert brick.index == 2
    assert brick.module_type == ModuleType.BRICK
    del brick


def test_brick_module_has_spec() -> None:
    """BrickModule exposes a MjSpec that can be compiled."""
    brick = BrickModule(index=2)
    model = brick.spec.compile()
    data = mujoco.MjData(model)
    del brick, model, data
