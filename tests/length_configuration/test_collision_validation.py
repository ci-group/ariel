"""Tests for static morphology self-collision validation.

These tests deliberately cover only the initial decoded morphology pose.
Runtime articulation collisions belong to simulation/controller tests rather
than morphology decoding validation.
"""

from unittest.mock import patch

import mujoco
import networkx as nx

from ariel.body_phenotypes.robogen_lite.collision_validation import (
    has_self_intersection,
    is_physically_valid,
)
from ariel.body_phenotypes.robogen_lite.config import (
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)


class FakeSpec:
    """Minimal ARIEL-like spec wrapper used for isolated MuJoCo tests."""

    def __init__(
        self,
        xml: str,
    ) -> None:
        self.xml = xml

    def compile(
        self,
    ) -> mujoco.MjModel:
        return mujoco.MjModel.from_xml_string(
            self.xml
        )


class FakeRobot:
    """Minimal constructor result used by the collision validator."""

    def __init__(
        self,
        xml: str,
    ) -> None:
        self.spec = FakeSpec(
            xml
        )


def _empty_graph() -> nx.DiGraph:
    return nx.DiGraph()


def _run_with_xml(
    xml: str,
    **kwargs,
) -> bool:
    with patch(
        "ariel.body_phenotypes.robogen_lite."
        "collision_validation."
        "construct_mjspec_from_graph",
        return_value=FakeRobot(
            xml
        ),
    ):
        return has_self_intersection(
            _empty_graph(),
            **kwargs,
        )


def test_separate_sibling_boxes_are_valid() -> None:
    """Clearly separated robot bodies should not self-intersect."""
    xml = """
    <mujoco>
        <worldbody>
            <body name="body1" pos="0 0 0">
                <geom
                    name="geom1"
                    type="box"
                    size="0.1 0.1 0.1"
                />
            </body>

            <body name="body2" pos="1 0 0">
                <geom
                    name="geom2"
                    type="box"
                    size="0.1 0.1 0.1"
                />
            </body>
        </worldbody>
    </mujoco>
    """

    assert _run_with_xml(
        xml
    ) is False


def test_overlapping_sibling_boxes_are_rejected() -> None:
    """Non-adjacent penetrating bodies must be rejected."""
    xml = """
    <mujoco>
        <worldbody>
            <body name="body1" pos="0 0 0">
                <geom
                    name="geom1"
                    type="box"
                    size="0.2 0.2 0.2"
                />
            </body>

            <body name="body2" pos="0.1 0 0">
                <geom
                    name="geom2"
                    type="box"
                    size="0.2 0.2 0.2"
                />
            </body>
        </worldbody>
    </mujoco>
    """

    assert _run_with_xml(
        xml
    ) is True


def test_large_parent_child_penetration_is_rejected() -> None:
    """A child deeply embedded in its parent is not valid attachment contact."""
    xml = """
    <mujoco>
        <worldbody>
            <body name="parent" pos="0 0 0">
                <geom
                    name="parent_geom"
                    type="box"
                    size="0.2 0.2 0.2"
                />

                <body name="child" pos="0.1 0 0">
                    <geom
                        name="child_geom"
                        type="box"
                        size="0.2 0.2 0.2"
                    />
                </body>
            </body>
        </worldbody>
    </mujoco>
    """

    assert _run_with_xml(
        xml,
        parent_child_tolerance=1e-3,
    ) is True


def test_small_parent_child_penetration_is_allowed() -> None:
    """Tiny attachment penetration can be tolerated numerically."""
    xml = """
    <mujoco>
        <worldbody>
            <body name="parent" pos="0 0 0">
                <geom
                    name="parent_geom"
                    type="box"
                    size="0.1 0.1 0.1"
                />

                <body name="child" pos="0.1995 0 0">
                    <geom
                        name="child_geom"
                        type="box"
                        size="0.1 0.1 0.1"
                    />
                </body>
            </body>
        </worldbody>
    </mujoco>
    """

    assert _run_with_xml(
        xml,
        parent_child_tolerance=1e-3,
    ) is False


def test_is_physically_valid_is_inverse() -> None:
    """Public validity helper should invert self-intersection."""
    xml = """
    <mujoco>
        <worldbody>
            <body name="body1" pos="0 0 0">
                <geom
                    type="box"
                    size="0.1 0.1 0.1"
                />
            </body>

            <body name="body2" pos="1 0 0">
                <geom
                    type="box"
                    size="0.1 0.1 0.1"
                />
            </body>
        </worldbody>
    </mujoco>
    """

    with patch(
        "ariel.body_phenotypes.robogen_lite."
        "collision_validation."
        "construct_mjspec_from_graph",
        return_value=FakeRobot(
            xml
        ),
    ):
        assert is_physically_valid(
            _empty_graph()
        ) is True


def test_compile_failure_is_rejected() -> None:
    """Collision validation should fail closed."""

    class BrokenSpec:
        def compile(
            self,
        ):
            raise RuntimeError(
                "Intentional compile failure"
            )

    class BrokenRobot:
        spec = BrokenSpec()

    with patch(
        "ariel.body_phenotypes.robogen_lite."
        "collision_validation."
        "construct_mjspec_from_graph",
        return_value=BrokenRobot(),
    ):
        assert has_self_intersection(
            _empty_graph()
        ) is True

        assert is_physically_valid(
            _empty_graph()
        ) is False


def _straight_long_chain() -> nx.DiGraph:
    graph = nx.DiGraph()

    graph.add_node(
        0,
        type=ModuleType.CORE.name,
        rotation=ModuleRotationsIdx.DEG_0.name,
    )

    graph.add_node(
        1,
        type=ModuleType.BRICK.name,
        rotation=ModuleRotationsIdx.DEG_0.name,
        length=0.225,
    )

    graph.add_node(
        2,
        type=ModuleType.BRICK.name,
        rotation=ModuleRotationsIdx.DEG_0.name,
        length=0.225,
    )

    graph.add_edge(
        0,
        1,
        face=ModuleFaces.FRONT.name,
    )

    graph.add_edge(
        1,
        2,
        face=ModuleFaces.FRONT.name,
    )

    return graph


def test_real_long_straight_robot_is_valid() -> None:
    """Two maximum-length bricks in a straight chain should be valid."""
    assert is_physically_valid(
        _straight_long_chain()
    ) is True


def test_validator_clears_global_control_callback() -> None:
    """Static collision checking must not leak MuJoCo controller state."""
    callback_calls = []

    def dummy_callback(
        model,
        data,
    ) -> None:
        callback_calls.append(
            (
                model.nq,
                data.time,
            )
        )

    mujoco.set_mjcb_control(
        dummy_callback
    )

    xml = """
    <mujoco>
        <worldbody>
            <body name="body1" pos="0 0 0">
                <geom
                    name="geom1"
                    type="box"
                    size="0.1 0.1 0.1"
                />
            </body>
        </worldbody>
    </mujoco>
    """

    assert _run_with_xml(
        xml
    ) is False

    # The validator should have disabled the stale callback before mj_forward.
    assert callback_calls == []

    # Clean up explicitly in case a future implementation changes behavior.
    mujoco.set_mjcb_control(
        None
    )
