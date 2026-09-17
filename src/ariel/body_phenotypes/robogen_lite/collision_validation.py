"""Collision validation utilities for robot morphologies."""

# Third-party libraries
import mujoco
import networkx as nx
import numpy as np

# Local libraries
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)


def has_self_intersection(
    graph: nx.DiGraph,
    tolerance: float = 1e-5,
) -> bool:
    """Check whether a decoded robot contains physical self-intersections.

    Directly connected bodies are ignored because they are intentionally
    attached at their connection sites.

    Parameters
    ----------
    graph
        NetworkX morphology graph.
    tolerance
        Allowed penetration tolerance in meters.

    Returns
    -------
    bool
        True if non-adjacent robot parts physically intersect.
    """

    # MuJoCo's control callback is global.
    #
    # A locomotion simulation may have installed a
    # controller for a previously evaluated robot.
    # mj_forward() invokes the global callback, so
    # allowing that callback to remain active here
    # can run an old controller against this new
    # morphology and incorrectly make validation fail.
    #
    # Collision checking does not require a controller.
    mujoco.set_mjcb_control(
        None
    )

    try:
        robot = construct_mjspec_from_graph(
            graph
        )

        model = robot.spec.compile()

        data = mujoco.MjData(
            model
        )

        mujoco.mj_forward(
            model,
            data,
        )

        for geom1 in range(
            model.ngeom
        ):
            body1 = int(
                model.geom_bodyid[
                    geom1
                ]
            )

            for geom2 in range(
                geom1 + 1,
                model.ngeom,
            ):
                body2 = int(
                    model.geom_bodyid[
                        geom2
                    ]
                )

                # Ignore geoms on the same body.
                if body1 == body2:
                    continue

                # Ignore direct parent-child body
                # pairs because they are expected
                # to meet at their attachment site.
                if (
                    model.body_parentid[
                        body1
                    ]
                    == body2
                    or model.body_parentid[
                        body2
                    ]
                    == body1
                ):
                    continue

                fromto = np.zeros(
                    6,
                    dtype=float,
                )

                distance = (
                    mujoco.mj_geomDistance(
                        model,
                        data,
                        geom1,
                        geom2,
                        0.0,
                        fromto,
                    )
                )

                if distance < -tolerance:
                    return True

        return False

    except Exception:
        return True


def is_physically_valid(
    graph: nx.DiGraph,
) -> bool:
    """Check whether a morphology is physically valid."""
    return not has_self_intersection(
        graph
    )