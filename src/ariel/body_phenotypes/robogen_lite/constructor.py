"""TODO(jmdm): description of script."""

# Standard library
from typing import TYPE_CHECKING

# Third-party libraries
from networkx import Graph

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (
    IDX_OF_CORE,
    ModuleFaces,
    ModuleRotationsTheta,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

# Type checking
if TYPE_CHECKING:
    from ariel.body_phenotypes.robogen_lite.modules.module import Module


def construct_mjspec_from_graph(graph: Graph) -> CoreModule:
    """
    Construct a MuJoCo specification from a graph representation. Can be used for
    constructing the body of a robot after crossover using the graph representation.

    Parameters
    ----------
    graph : Graph
        A graph representation of the robot's structure.

    Returns
    -------
    CoreModule
        The core module of the robot, which contains all other modules.

    Raises
    ------
    ValueError
        If the graph contains unknown module types.
    """
    modules: dict[int, Module] = {}
    for node in graph.nodes:
        # Extract module type and rotation from the graph node
        module_type = graph.nodes[node]["type"]
        module_rotation = graph.nodes[node]["rotation"]

        # Create the module based on its type
        match module_type:
            case ModuleType.CORE.name:
                module = CoreModule(index=IDX_OF_CORE)
            case ModuleType.HINGE.name:
                module = HingeModule(index=node)
            case ModuleType.BRICK.name:
                module = BrickModule(index=node)
            case ModuleType.NONE.name:
                module = None
            case _:
                msg = f"Unknown module type: {module_type}"
                raise ValueError(msg)

        # Check that the module is not None
        if module:
            rotation_angle = ModuleRotationsTheta[module_rotation].value
            module.rotate(rotation_angle)
            modules[node] = module
        else:
            modules[node] = None

    # Attach bodies to modules based on the graph edges
    for edge in graph.edges:
        # Extract the from and to modules and the face from the edge
        from_module = edge[0]
        to_module = edge[1]
        face = graph.edges[edge]["face"]

        # Check if both modules exist (not 'None')
        if modules[to_module]:
            modules[from_module].sites[ModuleFaces[face]].attach_body(
                body=modules[to_module].body,
                prefix=f"{modules[from_module].index}-{modules[to_module].index}-{ModuleFaces[face].value}-",
            )

    return modules[IDX_OF_CORE]
