"""Tests for variable-length brick scaling and physical collision validation."""

# Standard library
from unittest.mock import patch

# Third-party libraries
import mujoco
import networkx as nx
import pytest

# Local libraries
from ariel.body_phenotypes.robogen_lite.collision_validation import (
    has_self_intersection,
    is_physically_valid,
)
from ariel.body_phenotypes.robogen_lite.config import (
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import (
    scale_brick_length,
)
from ariel.parameters.ariel_modules import ArielModulesConfig


ariel_modules_config = ArielModulesConfig()


# ============================================================================
# CPPN LENGTH SCALING TESTS
# ============================================================================


def test_scale_brick_length_minimum() -> None:
    """A CPPN output of 0.0 should produce the minimum brick length."""
    length = scale_brick_length(
        0.0
    )

    assert length == pytest.approx(
        ariel_modules_config.BRICK_LENGTH_MIN
    )

    print()
    print(
        "0.0 ->",
        length,
        "m",
    )


def test_scale_brick_length_middle() -> None:
    """A CPPN output of 0.5 should produce the midpoint brick length."""
    length = scale_brick_length(
        0.5
    )

    expected = (
        ariel_modules_config.BRICK_LENGTH_MIN
        + ariel_modules_config.BRICK_LENGTH_MAX
    ) / 2.0

    assert length == pytest.approx(
        expected
    )

    print()
    print(
        "0.5 ->",
        length,
        "m",
    )


def test_scale_brick_length_maximum() -> None:
    """A CPPN output of 1.0 should produce the maximum brick length."""
    length = scale_brick_length(
        1.0
    )

    assert length == pytest.approx(
        ariel_modules_config.BRICK_LENGTH_MAX
    )

    print()
    print(
        "1.0 ->",
        length,
        "m",
    )


def test_scale_brick_length_clamps_below_range() -> None:
    """Values below zero should be clamped to the minimum."""
    length = scale_brick_length(
        -100.0
    )

    assert length == pytest.approx(
        ariel_modules_config.BRICK_LENGTH_MIN
    )


def test_scale_brick_length_clamps_above_range() -> None:
    """Values above one should be clamped to the maximum."""
    length = scale_brick_length(
        100.0
    )

    assert length == pytest.approx(
        ariel_modules_config.BRICK_LENGTH_MAX
    )


# ============================================================================
# SYNTHETIC MUJOCO COLLISION TESTS
# ============================================================================


class FakeSpec:
    """Small wrapper that behaves like an ARIEL robot spec."""

    def __init__(
        self,
        xml: str,
    ) -> None:
        self.xml = xml

    def compile(
        self,
    ) -> mujoco.MjModel:
        """Compile the stored XML into a MuJoCo model."""
        return mujoco.MjModel.from_xml_string(
            self.xml
        )


class FakeRobot:
    """Minimal object matching the constructor return interface."""

    def __init__(
        self,
        xml: str,
    ) -> None:
        self.spec = FakeSpec(
            xml
        )


def empty_graph() -> nx.DiGraph:
    """Return a placeholder graph for mocked collision tests."""
    return nx.DiGraph()


def test_two_separate_boxes_do_not_collide() -> None:
    """Two clearly separated sibling bodies should be physically valid."""
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

    fake_robot = FakeRobot(
        xml
    )

    with patch(
        "ariel.body_phenotypes.robogen_lite."
        "collision_validation."
        "construct_mjspec_from_graph",
        return_value=fake_robot,
    ):
        result = has_self_intersection(
            empty_graph()
        )

    assert result is False


def test_two_overlapping_boxes_are_detected() -> None:
    """Two overlapping sibling bodies should be detected."""
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

    fake_robot = FakeRobot(
        xml
    )

    with patch(
        "ariel.body_phenotypes.robogen_lite."
        "collision_validation."
        "construct_mjspec_from_graph",
        return_value=fake_robot,
    ):
        result = has_self_intersection(
            empty_graph()
        )

    assert result is True


def test_is_physically_valid_is_inverse() -> None:
    """is_physically_valid should invert the collision result."""
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

    fake_robot = FakeRobot(
        xml
    )

    with patch(
        "ariel.body_phenotypes.robogen_lite."
        "collision_validation."
        "construct_mjspec_from_graph",
        return_value=fake_robot,
    ):
        assert (
            is_physically_valid(
                empty_graph()
            )
            is True
        )


def test_compile_failure_is_rejected() -> None:
    """An unconstructable morphology should be considered invalid."""

    class BrokenSpec:
        def compile(
            self,
        ):
            raise RuntimeError(
                "Intentional test failure"
            )

    class BrokenRobot:
        spec = BrokenSpec()

    with patch(
        "ariel.body_phenotypes.robogen_lite."
        "collision_validation."
        "construct_mjspec_from_graph",
        return_value=BrokenRobot(),
    ):
        assert (
            has_self_intersection(
                empty_graph()
            )
            is True
        )

        assert (
            is_physically_valid(
                empty_graph()
            )
            is False
        )


# ============================================================================
# PARENT-CHILD BEHAVIOUR
# ============================================================================


def test_direct_parent_child_contact_is_ignored() -> None:
    """Directly connected bodies are intentionally excluded from checking.

    This test documents the current collision-validation policy.
    """
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

    fake_robot = FakeRobot(
        xml
    )

    with patch(
        "ariel.body_phenotypes.robogen_lite."
        "collision_validation."
        "construct_mjspec_from_graph",
        return_value=fake_robot,
    ):
        result = has_self_intersection(
            empty_graph()
        )

    # These geoms overlap heavily, but the bodies
    # are directly connected, so the validator
    # deliberately ignores them.
    assert result is False


# ============================================================================
# REAL ARIEL MORPHOLOGY TESTS
# ============================================================================


def create_short_chain() -> nx.DiGraph:
    """Create a simple core -> short brick morphology."""
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
        length=0.075,
    )

    graph.add_edge(
        0,
        1,
        face=ModuleFaces.FRONT.name,
    )

    return graph


def create_long_chain() -> nx.DiGraph:
    """Create a straight chain containing two maximum-length bricks."""
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


def test_real_short_ariel_robot_is_valid() -> None:
    """A simple standard-size brick robot should compile and be valid."""
    graph = create_short_chain()

    assert (
        is_physically_valid(
            graph
        )
        is True
    )


def test_real_long_straight_robot_is_valid() -> None:
    """A straight chain of long bricks should not self-intersect."""
    graph = create_long_chain()

    assert (
        is_physically_valid(
            graph
        )
        is True
    )


if __name__ == "__main__":
    test_scale_brick_length_minimum()
    test_scale_brick_length_middle()
    test_scale_brick_length_maximum()
    test_scale_brick_length_clamps_below_range()
    test_scale_brick_length_clamps_above_range()

    test_two_separate_boxes_do_not_collide()
    test_two_overlapping_boxes_are_detected()
    test_is_physically_valid_is_inverse()
    test_compile_failure_is_rejected()
    test_direct_parent_child_contact_is_ignored()

    test_real_short_ariel_robot_is_valid()
    test_real_long_straight_robot_is_valid()

    print()
    print("=" * 70)
    print(
        "PASSED: VARIABLE-LENGTH AND "
        "COLLISION VALIDATION TESTS"
    )
    print("=" * 70)