"""Test: construct_mjspec_from_graph builds valid robot specs from graphs."""

import mujoco
import networkx as nx

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule


def _make_core_only_graph() -> nx.DiGraph:
    """Single-node graph — just a core module."""
    g = nx.DiGraph()
    g.add_node(0, type="CORE", rotation="DEG_0")
    return g


def _make_core_with_hinge_graph() -> nx.DiGraph:
    """Core with one hinge attached to the FRONT face."""
    g = nx.DiGraph()
    g.add_node(0, type="CORE", rotation="DEG_0")
    g.add_node(1, type="HINGE", rotation="DEG_0")
    g.add_edge(0, 1, face="FRONT")
    return g


def _make_core_with_brick_graph() -> nx.DiGraph:
    """Core with one brick attached to the BACK face."""
    g = nx.DiGraph()
    g.add_node(0, type="CORE", rotation="DEG_0")
    g.add_node(1, type="BRICK", rotation="DEG_0")
    g.add_edge(0, 1, face="BACK")
    return g


def _make_core_with_none_graph() -> nx.DiGraph:
    """Core with a NONE-type slot on FRONT — should be skipped."""
    g = nx.DiGraph()
    g.add_node(0, type="CORE", rotation="DEG_0")
    g.add_node(1, type="NONE", rotation="DEG_0")
    g.add_edge(0, 1, face="FRONT")
    return g


def test_core_only_graph_returns_core_module() -> None:
    """A graph with only a core node returns a CoreModule."""
    core = construct_mjspec_from_graph(_make_core_only_graph())
    assert isinstance(core, CoreModule)
    del core


def test_core_only_graph_compiles() -> None:
    """The spec produced from a core-only graph can be compiled by MuJoCo."""
    core = construct_mjspec_from_graph(_make_core_only_graph())
    model = core.spec.compile()
    data = mujoco.MjData(model)
    del core, model, data


def test_core_with_hinge_compiles() -> None:
    """Core + hinge graph produces a compilable MuJoCo spec."""
    core = construct_mjspec_from_graph(_make_core_with_hinge_graph())
    model = core.spec.compile()
    data = mujoco.MjData(model)
    del core, model, data


def test_core_with_brick_compiles() -> None:
    """Core + brick graph produces a compilable MuJoCo spec."""
    core = construct_mjspec_from_graph(_make_core_with_brick_graph())
    model = core.spec.compile()
    data = mujoco.MjData(model)
    del core, model, data


def test_core_with_none_module_compiles() -> None:
    """NONE-type nodes are silently skipped; resulting spec still compiles."""
    core = construct_mjspec_from_graph(_make_core_with_none_graph())
    model = core.spec.compile()
    data = mujoco.MjData(model)
    del core, model, data


def test_unknown_module_type_raises() -> None:
    """A graph node with an unknown type raises ValueError."""
    import pytest

    g = nx.DiGraph()
    g.add_node(0, type="CORE", rotation="DEG_0")
    g.add_node(1, type="UNKNOWN_TYPE", rotation="DEG_0")
    g.add_edge(0, 1, face="FRONT")
    with pytest.raises(ValueError, match="Unknown module type"):
        construct_mjspec_from_graph(g)
