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
    parent_of_body1 = int(
        model.body_parentid[body1]
    )

    parent_of_body2 = int(
        model.body_parentid[body2]
    )

    return (
        parent_of_body1 == body2
        or parent_of_body2 == body1
    )


def _geom_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom1: int,
    geom2: int,
) -> float:
    """Return signed distance between two MuJoCo geometries.

    Negative values indicate penetration.
    Zero means touching.
    Positive values indicate separation.

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
        Signed geometry distance in metres.
    """
    fromto = np.zeros(
        6,
        dtype=float,
    )

    distance = mujoco.mj_geomDistance(
        model,
        data,
        geom1,
        geom2,
        0.0,
        fromto,
    )

    return float(distance)


def _get_current_control_callback():
    """Return the currently installed MuJoCo control callback when possible.

    Some MuJoCo Python releases expose ``get_mjcb_control`` while others only
    expose ``set_mjcb_control``. Returning ``None`` on versions without the
    getter preserves compatibility.

    Returns
    -------
    object | None
        Existing MuJoCo control callback, when retrievable.
    """
    getter = getattr(
        mujoco,
        "get_mjcb_control",
        None,
    )

    if getter is None:
        return None

    try:
        return getter()
    except Exception:
        return None


def _restore_control_callback(
    callback,
) -> None:
    """Restore the MuJoCo control callback.

    Parameters
    ----------
    callback
        Previously installed callback or ``None``.
    """
    mujoco.set_mjcb_control(
        callback
    )


def has_self_intersection(
    graph: nx.DiGraph,
    tolerance: float = 1e-5,
    parent_child_tolerance: float = 1e-3,
) -> bool:
    """Check whether a decoded robot contains physical self-intersections.

    The graph is first converted into the real ARIEL MuJoCo phenotype. The
    compiled geometry is then inspected using ``mj_geomDistance``.

    Non-adjacent bodies are considered invalid when their penetration exceeds
    ``tolerance``.

    Directly attached parent-child modules are allowed slightly more
    penetration because their geometries may intentionally meet at attachment
    interfaces. Penetration larger than ``parent_child_tolerance`` is still
    rejected.

    Parameters
    ----------
    graph
        NetworkX morphology graph.
    tolerance
        Maximum allowed penetration for non-adjacent bodies, in metres.
    parent_child_tolerance
        Maximum allowed penetration for directly connected bodies, in metres.

    Returns
    -------
    bool
        True when an invalid physical intersection exists.

    Notes
    -----
    Validation is fail-closed: if construction, compilation, or MuJoCo
    evaluation fails, the morphology is treated as invalid.

    MuJoCo's control callback is process-global. Collision checking temporarily
    disables it so ``mj_forward`` cannot invoke a controller belonging to a
    different robot. The previous callback is restored afterward whenever the
    installed MuJoCo Python API permits retrieving it.
    """
    previous_callback = _get_current_control_callback()

    # Collision checking is purely geometric and must not invoke an unrelated
    # simulation controller.
    mujoco.set_mjcb_control(
        None
    )

    try:
        # ==============================================================
        # 1. CONSTRUCT THE ACTUAL ARIEL PHENOTYPE
        # ==============================================================

        robot = construct_mjspec_from_graph(
            graph
        )

        # ==============================================================
        # 2. COMPILE THE MODEL
        # ==============================================================

        model = robot.spec.compile()

        data = mujoco.MjData(
            model
        )

        # Compute world-space transforms for all geometries.
        mujoco.mj_forward(
            model,
            data,
        )

        # ==============================================================
        # 3. CHECK GEOMETRY PAIRS
        # ==============================================================

        for geom1 in range(model.ngeom):
            body1 = int(
                model.geom_bodyid[geom1]
            )

            for geom2 in range(
                geom1 + 1,
                model.ngeom,
            ):
                body2 = int(
                    model.geom_bodyid[geom2]
                )

                # Multiple geometries belonging to the same rigid body may
                # intentionally overlap as part of one module.
                if body1 == body2:
                    continue

                distance = _geom_distance(
                    model=model,
                    data=data,
                    geom1=geom1,
                    geom2=geom2,
                )

                is_parent_child = _is_direct_parent_child(
                    model=model,
                    body1=body1,
                    body2=body2,
                )

                # ------------------------------------------------------
                # DIRECT PARENT / CHILD
                # ------------------------------------------------------

                if is_parent_child:
                    if (
                        distance
                        < -parent_child_tolerance
                    ):
                        return True

                    continue

                # ------------------------------------------------------
                # NON-ADJACENT BODIES
                # ------------------------------------------------------

                if (
                    distance
                    < -tolerance
                ):
                    return True

        return False

    except Exception:
        # Fail closed. If a morphology cannot be built and verified safely,
        # it should not enter the evolutionary population.
        return True

    finally:
        _restore_control_callback(
            previous_callback
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
        Maximum allowed penetration between non-adjacent bodies, in metres.
    parent_child_tolerance
        Maximum allowed penetration between directly connected bodies, in
        metres.

    Returns
    -------
    bool
        True when no invalid self-intersection is found.
    """
    return not has_self_intersection(
        graph=graph,
        tolerance=tolerance,
        parent_child_tolerance=parent_child_tolerance,
    )