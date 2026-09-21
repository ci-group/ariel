"""Collision validation utilities for robot morphologies."""

# Third-party libraries
import mujoco
import networkx as nx
import numpy as np

# Local libraries
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)


def _is_direct_parent_child(
    model: mujoco.MjModel,
    body1: int,
    body2: int,
) -> bool:
    """Return whether two MuJoCo bodies are directly connected.

    Parameters
    ----------
    model
        Compiled MuJoCo model.
    body1
        First body ID.
    body2
        Second body ID.

    Returns
    -------
    bool
        True when either body is the direct parent of the other.
    """
    return (
        int(
            model.body_parentid[
                body1
            ]
        )
        == body2
        or int(
            model.body_parentid[
                body2
            ]
        )
        == body1
    )


def _geom_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom1: int,
    geom2: int,
) -> float:
    """Return signed distance between two MuJoCo geometries.

    A negative value means the geometries penetrate each other.
    Zero means touching.
    A positive value means they are separated.

    Parameters
    ----------
    model
        Compiled MuJoCo model.
    data
        MuJoCo data associated with ``model``.
    geom1
        First geometry ID.
    geom2
        Second geometry ID.

    Returns
    -------
    float
        Signed geometry distance in meters.
    """
    fromto = np.zeros(
        6,
        dtype=float,
    )

    return float(
        mujoco.mj_geomDistance(
            model,
            data,
            geom1,
            geom2,

            # We only need to know whether the
            # geometries touch / penetrate.
            0.0,

            fromto,
        )
    )


def has_self_intersection(
    graph: nx.DiGraph,
    tolerance: float = 1e-5,
    parent_child_tolerance: float = 1e-3,
) -> bool:
    """Check whether a decoded robot contains physical self-intersections.

    The validator uses MuJoCo's geometry-distance query after compiling the
    actual ARIEL phenotype.

    Non-adjacent body pairs are rejected whenever penetration exceeds
    ``tolerance``.

    Directly connected parent-child bodies are allowed a slightly larger
    penetration tolerance because modules intentionally meet at their
    attachment interface. However, large parent-child overlap is rejected.

    Parameters
    ----------
    graph
        NetworkX morphology graph.
    tolerance
        Maximum allowed penetration between non-adjacent bodies, in meters.
        Default is 1e-5 m.
    parent_child_tolerance
        Maximum allowed penetration between directly attached parent-child
        bodies, in meters. Default is 1e-3 m (1 mm).

    Returns
    -------
    bool
        True if the morphology contains an invalid physical intersection.
        False if it is physically valid.

    Notes
    -----
    Validation is fail-closed: if construction, compilation, or MuJoCo
    evaluation raises an exception, the morphology is treated as invalid.
    """

    # MuJoCo's control callback is global.
    #
    # A locomotion simulation may have installed a controller for a
    # previously evaluated robot. mj_forward() invokes the global callback,
    # so leaving a stale callback active can make collision validation fail
    # for reasons unrelated to geometry.
    #
    # Collision checking does not need a controller.
    mujoco.set_mjcb_control(
        None
    )

    try:
        # ------------------------------------------------------------------ #
        # 1. CONSTRUCT THE REAL ARIEL PHENOTYPE
        # ------------------------------------------------------------------ #

        robot = construct_mjspec_from_graph(
            graph
        )

        # ------------------------------------------------------------------ #
        # 2. COMPILE AND FORWARD THE MODEL
        # ------------------------------------------------------------------ #

        model = robot.spec.compile()

        data = mujoco.MjData(
            model
        )

        mujoco.mj_forward(
            model,
            data,
        )

        # ------------------------------------------------------------------ #
        # 3. CHECK EVERY GEOMETRY PAIR
        # ------------------------------------------------------------------ #

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

                # Geometries belonging to the same rigid body are expected
                # to overlap / touch as part of that module's construction.
                if body1 == body2:
                    continue

                distance = _geom_distance(
                    model,
                    data,
                    geom1,
                    geom2,
                )

                is_parent_child = (
                    _is_direct_parent_child(
                        model,
                        body1,
                        body2,
                    )
                )

                # ---------------------------------------------------------- #
                # DIRECT PARENT-CHILD PAIRS
                # ---------------------------------------------------------- #
                #
                # Connected modules are expected to meet at their attachment
                # sites, so tiny penetration can be acceptable.
                #
                # However, we no longer ignore these pairs completely.
                # A deeply embedded child module is an invalid morphology.
                # ---------------------------------------------------------- #

                if is_parent_child:
                    if (
                        distance
                        < -parent_child_tolerance
                    ):
                        return True

                    continue

                # ---------------------------------------------------------- #
                # NON-ADJACENT BODY PAIRS
                # ---------------------------------------------------------- #
                #
                # Any penetration larger than the tiny numerical tolerance
                # is treated as a self-intersection.
                # ---------------------------------------------------------- #

                if (
                    distance
                    < -tolerance
                ):
                    return True

        return False

    except Exception:
        # Fail closed.
        #
        # A morphology that cannot be constructed, compiled, or evaluated
        # safely should not enter the evolutionary population.
        return True

    finally:
        # Collision validation should never leave a MuJoCo controller callback
        # installed for subsequent models.
        mujoco.set_mjcb_control(
            None
        )


def is_physically_valid(
    graph: nx.DiGraph,
    tolerance: float = 1e-5,
    parent_child_tolerance: float = 1e-3,
) -> bool:
    """Return whether a morphology is physically valid.

    Parameters
    ----------
    graph
        NetworkX morphology graph.
    tolerance
        Maximum allowed penetration between non-adjacent bodies, in meters.
    parent_child_tolerance
        Maximum allowed penetration between directly attached bodies, in
        meters.

    Returns
    -------
    bool
        True when no invalid self-intersection is detected.
    """
    return not has_self_intersection(
        graph,
        tolerance=tolerance,
        parent_child_tolerance=parent_child_tolerance,
    )
