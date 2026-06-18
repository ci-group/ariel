"""Gecko-specific topology helper for the social-learning examples.

``gecko_graph()`` returns the DiGraph representation of the prebuilt gecko
morphology, suitable for passing to ``MorphologyAdapter.from_graph()``.

The reusable ``MorphologyAdapter`` class now lives in the library:
    ``ariel.simulation.controllers.morphology_adapter``

# Environment: uv (main ariel venv, Python >=3.12)
"""

from __future__ import annotations

import networkx as nx

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType

# Re-export so existing example scripts can still do:
#   from morphology_adapter import MorphologyAdapter, gecko_graph
from ariel.simulation.controllers.morphology_adapter import MorphologyAdapter  # noqa: F401

if False:  # TYPE_CHECKING
    from networkx import DiGraph


def gecko_graph() -> DiGraph:
    """Return the DiGraph corresponding to the prebuilt gecko morphology.

    Node and edge attributes match the convention used by
    ``construct_mjspec_from_graph`` (and the decoder pipeline), so
    ``MorphologyAdapter.from_graph(gecko_graph())`` works correctly.

    Module indices:
        0=core, 1=neck, 2=abdomen, 3=spine, 4=butt,
        5=fl_leg, 15=fl_leg2, 6=fl_flipper,
        7=fr_leg, 17=fr_leg2, 8=fr_flipper,
        9=bl_leg, 10=bl_flipper, 11=br_leg, 12=br_flipper
    """
    g: DiGraph = nx.DiGraph()

    nodes = [
        (0,  ModuleType.CORE),
        (1,  ModuleType.HINGE),
        (2,  ModuleType.BRICK),
        (3,  ModuleType.HINGE),
        (4,  ModuleType.BRICK),
        (5,  ModuleType.HINGE),
        (6,  ModuleType.BRICK),
        (7,  ModuleType.HINGE),
        (8,  ModuleType.BRICK),
        (9,  ModuleType.HINGE),
        (10, ModuleType.BRICK),
        (11, ModuleType.HINGE),
        (12, ModuleType.BRICK),
        (15, ModuleType.HINGE),
        (17, ModuleType.HINGE),
    ]
    for idx, mtype in nodes:
        g.add_node(idx, type=mtype.name, rotation="DEG_0")

    edges = [
        (0,  1,  ModuleFaces.FRONT),   # core  → neck
        (1,  2,  ModuleFaces.FRONT),   # neck  → abdomen
        (2,  3,  ModuleFaces.FRONT),   # abdomen → spine
        (3,  4,  ModuleFaces.FRONT),   # spine → butt
        (0,  5,  ModuleFaces.LEFT),    # core  → fl_leg
        (5,  15, ModuleFaces.FRONT),   # fl_leg → fl_leg2
        (15, 6,  ModuleFaces.FRONT),   # fl_leg2 → fl_flipper
        (0,  7,  ModuleFaces.RIGHT),   # core  → fr_leg
        (7,  17, ModuleFaces.FRONT),   # fr_leg → fr_leg2
        (17, 8,  ModuleFaces.FRONT),   # fr_leg2 → fr_flipper
        (4,  9,  ModuleFaces.LEFT),    # butt  → bl_leg
        (9,  10, ModuleFaces.FRONT),   # bl_leg → bl_flipper
        (4,  11, ModuleFaces.RIGHT),   # butt  → br_leg
        (11, 12, ModuleFaces.FRONT),   # br_leg → br_flipper
    ]
    for parent, child, face in edges:
        g.add_edge(parent, child, face=face.name)

    return g
