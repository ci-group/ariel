"""Test: prebuilt robot bodies compile to valid MuJoCo models."""

import mujoco

from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider import spider


def test_spider_returns_core_module() -> None:
    """spider() returns a CoreModule."""
    core = spider()
    assert isinstance(core, CoreModule)
    del core


def test_spider_compiles() -> None:
    """Spider body compiles to a valid MuJoCo model."""
    core = spider()
    model = core.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    del core, model, data


def test_gecko_returns_core_module() -> None:
    """gecko() returns a CoreModule."""
    core = gecko()
    assert isinstance(core, CoreModule)
    del core


def test_gecko_compiles() -> None:
    """Gecko body compiles to a valid MuJoCo model."""
    core = gecko()
    model = core.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    del core, model, data
