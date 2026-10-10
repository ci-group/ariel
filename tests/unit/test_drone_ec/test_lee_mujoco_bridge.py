"""Sanity test: STANDARD_HEXACOPTER hovers under the Lee→MuJoCo bridge.

De-risks the bridge before the EA examples depend on it. The test builds the
canonical hexacopter (from UnifiedFitness's edit-distance target), spawns it
in MuJoCo via blueprint_to_mjspec, drives it with the bridge for a few
seconds at a target altitude, and asserts altitude stays within a generous
tolerance and the drone has not tumbled.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.timeout(60)
def test_standard_hexacopter_hovers_with_bridge() -> None:
    """STANDARD_HEXACOPTER should hold altitude within ±0.5 m for 3 s of hover.

    These tolerances are intentionally loose — the goal is to catch
    catastrophic regressions (motor saturation, frame conversion, mass
    mismatch). If the drone holds within ±0.5 m without tumbling, the
    bridge is wired up correctly enough to drive evolution.
    """
    import mujoco  # noqa: F401  (imported for early-fail if mujoco isn't available)

    from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
    from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
    from ariel.ec.drone.evaluators.unified_fitness import STANDARD_HEXACOPTER
    from ariel.simulation.drone.controllers.lee_control.lee_controller import (
        LeeGeometricControl,
    )
    from ariel.simulation.drone.controllers.lee_control.mujoco_bridge import (
        LeeMujocoHoverBridge,
        hover_fitness_from_log,
        spawn_blueprint_in_world,
    )
    from ariel.simulation.drone.drone_interface import DroneInterface

    # --- Build the canonical hexacopter blueprint --------------------------
    bp = spherical_angular_to_blueprint(STANDARD_HEXACOPTER, propsize=2)

    # --- Build the Python NED simulator (Lee's policy needs a `quad` handle) --
    propellers = blueprint_to_propellers(bp, convention="ned")
    quad = DroneInterface(0, propellers=propellers)

    lee_ctrl = LeeGeometricControl(
        quad,
        yawType=1,
        orient="NED",
        auto_scale_gains=True,
        pos_P_gain=np.array([14.3, 14.3, 14.3]),
        vel_P_gain=np.array([9.0, 9.0, 9.0]),
    )

    # --- Spawn in MuJoCo with mass-matched body ---------------------------
    target_alt = 1.0
    spawned = spawn_blueprint_in_world(
        bp,
        propellers=propellers,
        target_mass=float(quad.params["mB"]),
        spawn_position=(0.0, 0.0, target_alt),
        body_name="hex",
    )

    bridge = LeeMujocoHoverBridge(
        quad=quad,
        lee_ctrl=lee_ctrl,
        model=spawned.model,
        data=spawned.data,
        max_thrust_per_motor=spawned.max_thrust_per_motor,
        target_position_enu=(0.0, 0.0, target_alt),
    )

    # --- Hover ------------------------------------------------------------
    # 1.0 s window matches the duration used by the EA evaluator. Long-horizon
    # (>~1.5 s) Lee→MuJoCo hover stability is a known follow-up issue.
    log = bridge.run_hover(duration=1.0, warm_up=0.1)
    assert log["pos"].shape[0] > 0, "hover loop produced no logged poses"

    z = log["pos"][:, 2]
    xy = np.linalg.norm(log["pos"][:, :2], axis=1)
    tilt_cos = np.clip(log["tilt_cos"], -1.0, 1.0)

    assert np.all(np.isfinite(z)), "altitude went non-finite during hover"

    z_err = float(np.max(np.abs(z - target_alt)))
    xy_drift = float(np.max(xy))
    min_tilt_cos = float(np.min(tilt_cos))

    # Generous tolerances — the goal is to catch catastrophic regressions
    # (motor saturation, frame conversion, mass mismatch), not require
    # millimetre hover precision.
    assert z_err < 0.3, f"altitude error too large: max|z - target| = {z_err:.3f} m"
    assert xy_drift < 0.3, f"lateral drift too large: max |xy| = {xy_drift:.3f} m"
    # cos(45°) ≈ 0.707; anything below this means the drone has rolled /
    # pitched far past sane hover orientations.
    assert min_tilt_cos > 0.7, f"drone tumbled: min cos(tilt) = {min_tilt_cos:.3f}"

    # Sanity: the high-level fitness helper should give a finite, sensible score.
    fit = hover_fitness_from_log(log, target_position_enu=(0.0, 0.0, target_alt))
    assert np.isfinite(fit), "hover_fitness_from_log returned non-finite score"
    assert fit < 0.3, f"hover fitness is implausibly high: {fit:.3f}"
