"""Integration tests for passing brick length through the ARIEL constructor."""

import mujoco
import networkx as nx
import pytest

from ariel.body_phenotypes.robogen_lite.config import (
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.parameters.ariel_modules import ArielModulesConfig


config = ArielModulesConfig()


def _core_with_brick(
    *,
    length: float | None,
) -> nx.DiGraph:
    graph = nx.DiGraph()

    graph.add_node(
        0,
        type=ModuleType.CORE.name,
        rotation=ModuleRotationsIdx.DEG_0.name,
    )

    brick_data = {
        "type": ModuleType.BRICK.name,
        "rotation": ModuleRotationsIdx.DEG_0.name,
    }

    if length is not None:
        brick_data["length"] = length

    graph.add_node(
        1,
        **brick_data,
    )

    graph.add_edge(
        0,
        1,
        face=ModuleFaces.FRONT.name,
    )

    return graph


def _find_brick_geom_id(
    model: mujoco.MjModel,
) -> int:
    """Find the attached brick geom in a compiled robot."""
    for geom_id in range(
        model.ngeom
    ):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )

        if (
            name is not None
            and name.endswith(
                "brick"
            )
        ):
            return geom_id

    raise AssertionError(
        "No brick geom found."
    )


def test_constructor_preserves_explicit_brick_length() -> None:
    """Graph node length should reach the compiled MuJoCo brick geometry."""
    requested_length = 0.180

    robot = construct_mjspec_from_graph(
        _core_with_brick(
            length=requested_length,
        )
    )
    model = robot.spec.compile()

    brick_geom = _find_brick_geom_id(
        model
    )

    full_length = (
        2.0
        * float(
            model.geom_size[
                brick_geom,
                1,
            ]
        )
    )

    assert full_length == pytest.approx(
        requested_length
    )


def test_constructor_uses_default_when_length_missing() -> None:
    """Old morphology graphs without a length attribute remain compatible."""
    robot = construct_mjspec_from_graph(
        _core_with_brick(
            length=None,
        )
    )
    model = robot.spec.compile()

    brick_geom = _find_brick_geom_id(
        model
    )

    full_length = (
        2.0
        * float(
            model.geom_size[
                brick_geom,
                1,
            ]
        )
    )

    assert full_length == pytest.approx(
        config.BRICK_LENGTH_DEFAULT
    )


def test_constructor_rejects_invalid_brick_length() -> None:
    """An invalid graph length should be rejected during construction."""
    graph = _core_with_brick(
        length=0.500,
    )

    with pytest.raises(
        ValueError
    ):
        construct_mjspec_from_graph(
            graph
        )
