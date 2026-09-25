"""Small end-to-end smoke tests for the variable-length morphology feature."""

import networkx as nx
import pytest

from ariel.body_phenotypes.robogen_lite.collision_validation import (
    is_physically_valid,
)
from ariel.body_phenotypes.robogen_lite.config import (
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)


@pytest.mark.parametrize(
    "length",
    [
        0.075,
        0.150,
        0.225,
    ],
)
def test_core_brick_hinge_compiles_and_validates(
    length: float,
) -> None:
    """Representative variable-length robots should pass the full static path."""
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
        length=length,
    )

    graph.add_node(
        2,
        type=ModuleType.HINGE.name,
        rotation=ModuleRotationsIdx.DEG_0.name,
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

    assert is_physically_valid(
        graph
    ) is True

    robot = construct_mjspec_from_graph(
        graph
    )
    model = robot.spec.compile()

    assert model.nbody > 0
    assert model.ngeom > 0
    assert model.nu == 1
