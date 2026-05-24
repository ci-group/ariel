"""DroneBlueprint → phenotype backends.

Each backend consumes a DroneBlueprint and emits something a simulator (or
the real world) can instantiate. v1 ships:

  * ``blueprint_to_propellers`` — list[dict] consumable by
    ``ariel.simulation.drone.DroneSimulator`` / ``DroneConfiguration``.
  * ``blueprint_to_mjspec``     — MuJoCo ``mjSpec`` (compiles to MJCF /
    ``MjModel``); the same blueprint can drive both the Python physics
    stack and a full MuJoCo simulation.

Stubs sketched for future backends:

  * ``blueprint_to_urdf``       — URDF file; can be fed to Isaac Lab's
    ``UrdfConverter`` to produce a USD asset.
  * ``blueprint_to_usd``        — USD prim hierarchy for Isaac Lab
    (direct, no URDF intermediate).
"""
from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import numpy as np

from .blueprint import (
    DroneBlueprint,
    ArmNode,
    MotorNode,
    RotorNode,
    CorePlateNode,
    CylindricalCrossSection,
    HollowTubeCrossSection,
    RectangularCrossSection,
)

if TYPE_CHECKING:
    import mujoco


def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def blueprint_to_propellers(
    bp: DroneBlueprint,
    *,
    convention: str = "z_up",
) -> list[dict[str, Any]]:
    """Flatten the blueprint tree to the propellers-list format consumed by
    ``DroneSimulator(propellers=...)``.

    Args:
        bp: the DroneBlueprint to flatten.
        convention: ``"z_up"`` (default, mathematical / +Z is sky) or
            ``"ned"`` (used by ``DroneSimulator`` / Lee controller, +Z is
            ground). When ``"ned"``, the position Z and thrust-normal Z
            components are inverted.

    Each entry:
        {"loc": [x, y, z],                 # position of motor
         "dir": [nx, ny, nz, "ccw"|"cw"], # thrust normal + spin
         "propsize": int}                  # inches
    """
    if convention not in {"z_up", "ned"}:
        raise ValueError(f"convention must be 'z_up' or 'ned', got {convention!r}")
    z_sign = -1.0 if convention == "ned" else 1.0
    propellers: list[dict[str, Any]] = []

    # Core-relative pose; we descend Arm → Motor → Rotor and accumulate the
    # transform of each Arm (anchored at the core), then the Motor on top.
    for arm_id in bp.children(bp.root_id):  # type: ignore[arg-type]
        arm = bp.payload(arm_id)
        if not isinstance(arm, ArmNode):
            continue
        R_arm = _rpy_to_R(*arm.pose.rpy)
        arm_origin = np.array(arm.pose.xyz)

        for motor_id in bp.children(arm_id):
            motor = bp.payload(motor_id)
            if not isinstance(motor, MotorNode):
                continue
            # Motor offset along the arm's local +X by `length`, plus its own
            # local offset; pose was authored with xyz=(length,0,0) already.
            motor_world = arm_origin + R_arm @ np.array(motor.pose.xyz)
            R_motor = R_arm @ _rpy_to_R(*motor.pose.rpy)
            # Thrust direction = motor's local +Z in world frame.
            # (NED convention used by DroneSimulator: thrust normal points in
            #  the rotor's spin-axis direction; sign is handled by DroneConfig.)
            thrust_normal = R_motor @ np.array([0.0, 0.0, 1.0])

            propellers.append({
                "loc": [
                    float(motor_world[0]),
                    float(motor_world[1]),
                    float(motor_world[2]) * z_sign,
                ],
                "dir": [
                    float(thrust_normal[0]),
                    float(thrust_normal[1]),
                    float(thrust_normal[2]) * z_sign,
                    motor.spin,
                ],
                "propsize": int(motor.propsize),
            })

            # (Rotor geometry is currently absorbed into propsize lookup; an
            #  MJCF/USD backend would consume the RotorNode separately.)
            for _rotor_id in bp.children(motor_id):
                rotor = bp.payload(_rotor_id)
                assert isinstance(rotor, RotorNode)  # type-only

    return propellers


# ---------- MuJoCo backend (blueprint → mjSpec) ----------

def blueprint_to_mjspec(
    bp: DroneBlueprint,
    *,
    core_mass_override: float | None = None,
    max_thrust: float = 5.0,
    body_name: str = "drone",
) -> "mujoco.MjSpec":
    """Compile a DroneBlueprint into a MuJoCo ``MjSpec`` describing a
    free-flying drone with one site-attached thrust actuator per motor.

    Conventions:
      * Z-up (the MuJoCo default). No NED inversion is performed here —
        gravity points in -Z, thrust points along each motor site's local
        +Z. Use this backend when integrating with the MuJoCo-native
        stack (``SimpleFlatWorld``, ``video_renderer``), not the Lee
        controller pipeline (which is NED).
      * The root body has a freejoint so it can fly.
      * Arms attach rigidly to the core (no joint between Arm and Core);
        Motors attach rigidly to Arm tips.
      * One actuator per Motor; ``ctrl`` ∈ [0, 1] maps linearly to thrust
        in [0, ``max_thrust``] Newtons along the rotor's spin axis.

    Physical parameters are read from the blueprint nodes themselves:
    arm mass / inertia from ``arm.mass`` / ``arm.cross_section``; motor
    mass and visual dimensions from ``motor.mass`` / ``motor.radius`` /
    ``motor.thickness`` (sourced from ``propeller_data.PROPELLER_LIBRARY``
    via ``motor.propsize``).

    Args:
        bp: the blueprint to compile.
        core_mass_override: if given, overrides ``core.mass`` for testing.
        max_thrust: maximum thrust per motor in Newtons.
        body_name: root body name.

    Returns:
        A ``mujoco.MjSpec`` containing only the drone (compile it directly
        with ``spec.compile()`` for a standalone model, or hand to
        ``BaseWorld.spawn()`` to drop it into a world).
    """
    import mujoco  # local — avoid hard dependency at import time

    spec = mujoco.MjSpec()

    core = bp.payload(bp.root_id)
    if not isinstance(core, CorePlateNode):
        raise ValueError("Blueprint root must be a CorePlateNode.")

    # --- root body ---
    # Note: no freejoint here. ``BaseWorld.spawn()`` attaches the body and
    # adds the freejoint itself. For standalone use, call ``add_freejoint()``
    # on the returned spec's root body before compiling.
    root = spec.worldbody.add_body(name=body_name, pos=[0.0, 0.0, 0.0])
    core_mass = core_mass_override if core_mass_override is not None else core.mass
    root.add_geom(
        name=f"{body_name}_core",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[core.radius, core.thickness / 2.0, 0.0],
        mass=core_mass,
        rgba=(0.2, 0.4, 0.8, 1.0),
    )

    # --- arms + motors ---
    motor_index = 0
    for arm_id in bp.children(bp.root_id):  # type: ignore[arg-type]
        arm = bp.payload(arm_id)
        if not isinstance(arm, ArmNode):
            continue
        arm_yaw = arm.pose.rpy[2]
        arm_pitch = arm.pose.rpy[1]

        # Compute arm tip in core-local frame: rotate (length, 0, 0) by (-pitch, yaw)
        # using ZYX intrinsic order matching how the decoder authored rpy.
        cy, sy = math.cos(arm_yaw), math.sin(arm_yaw)
        cp, sp = math.cos(arm_pitch), math.sin(arm_pitch)
        tip_local = np.array([cp * cy * arm.length,
                              cp * sy * arm.length,
                              -sp * arm.length])

        # Arm child body (rigid offset; no joint = welded to core)
        arm_body = root.add_body(
            name=f"{body_name}_arm_{arm_id}",
            pos=[0.0, 0.0, 0.0],
        )
        # Geometry dispatch on cross-section type. Mass and inertia
        # come from arm.mass / arm.cross_section.principal_inertia.
        cs = arm.cross_section
        if isinstance(cs, RectangularCrossSection):
            # Box geom centered along the arm, oriented to match tip_local.
            arm_body.add_geom(
                pos=(tip_local / 2.0).tolist(),
                quat=_rpy_to_quat(0.0, arm_pitch, arm_yaw),
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[arm.length / 2.0, cs.width / 2.0, cs.thickness / 2.0],
                mass=arm.mass,
                rgba=(0.3, 0.3, 0.3, 1.0),
            )
        else:
            # CylindricalCrossSection (solid) or HollowTubeCrossSection
            # — visualised as a capsule. Outer radius drives the visual.
            visual_radius = (
                cs.outer_radius if isinstance(cs, HollowTubeCrossSection) else cs.radius
            )
            arm_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[0.0, 0.0, 0.0, *tip_local.tolist()],
                size=[visual_radius, 0.0, 0.0],
                mass=arm.mass,
                rgba=(0.3, 0.3, 0.3, 1.0),
            )

        for motor_id in bp.children(arm_id):
            motor = bp.payload(motor_id)
            if not isinstance(motor, MotorNode):
                continue

            # The motor's site frame: thrust axis is the site's local +Z.
            # The decoder authored motor pose as (xyz=(length, 0, 0), rpy)
            # in the arm's local frame, where the rpy encodes the relative
            # thrust orientation; we want the world-frame thrust to point
            # along world +Z (motor_pitch=0, motor_az=0 → straight up), so
            # we leave the site's orientation matching the parent arm tip
            # for the canonical zero-pitch case.
            #
            # To keep things robust for tilted thrusters, we compose the
            # motor's rpy on top of the arm's orientation:
            mr_roll, mr_pitch, mr_yaw = motor.pose.rpy
            site_quat = _rpy_to_quat(mr_roll,
                                     mr_pitch + arm_pitch,  # cancel arm tilt
                                     mr_yaw + arm_yaw)

            motor_body = arm_body.add_body(
                name=f"{body_name}_motor_{motor_id}",
                pos=tip_local.tolist(),
                quat=site_quat,
            )
            motor_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[motor.radius, motor.thickness, 0.0],
                mass=motor.mass,
                rgba=(1.0, 0.2, 0.2, 1.0)
                     if motor.spin == "cw"
                     else (0.2, 0.8, 0.2, 1.0),
            )

            # Rotor visualisation (thin disc above motor)
            rotor_radius = 0.05  # default; updated if a RotorNode is present
            for rotor_id in bp.children(motor_id):
                rotor = bp.payload(rotor_id)
                if isinstance(rotor, RotorNode):
                    rotor_radius = rotor.radius
            motor_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[0.0, 0.0, motor.thickness + 0.002],
                size=[rotor_radius, 0.001, 0.0],
                mass=0.0,
                rgba=(0.8, 0.8, 0.8, 0.5),
            )

            # Thrust site (force applied along local +Z)
            thrust_site = motor_body.add_site(
                name=f"{body_name}_thrust_{motor_index}",
                pos=[0.0, 0.0, motor.thickness],
                size=[0.005, 0.005, 0.005],
            )

            spec.add_actuator(
                name=f"{body_name}_motor_{motor_index}",
                trntype=mujoco.mjtTrn.mjTRN_SITE,
                target=thrust_site.name,
                gear=[0.0, 0.0, max_thrust, 0.0, 0.0, 0.0],
                ctrlrange=[0.0, 1.0],
                ctrllimited=True,
            )
            motor_index += 1

    return spec


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    """ZYX intrinsic Euler → (w, x, y, z) quaternion (MuJoCo order)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,   # w
        sr * cp * cy - cr * sp * sy,   # x
        cr * sp * cy + sr * cp * sy,   # y
        cr * cp * sy - sr * sp * cy,   # z
    ]


# ---------- future backends (signatures only) ----------

def blueprint_to_urdf(
    bp: DroneBlueprint,
    out_path: str,
    *,
    core_mass_override: float | None = None,
    robot_name: str = "drone",
) -> str:
    """Compile a DroneBlueprint into a URDF file.

    Intended as the intermediate step for Isaac Lab / Isaac Sim: the
    emitted ``.urdf`` is consumed by
    ``isaaclab.sim.converters.UrdfConverter`` to produce a USD asset
    (see ``soft_airframe_optimization/scripts/convert_xconfig_urdf.py``
    for the conversion call pattern).

    Physical parameters are read from the blueprint nodes themselves
    (``arm.mass``, ``arm.inertia_diag``, ``motor.mass``, ``motor.radius``,
    ``motor.thickness``), so this signature only needs URDF-mechanical
    options.

    Planned mapping (mirrors ``blueprint_to_mjspec``):
      * One ``<link>`` per blueprint node (CorePlate, Arm, Motor) with
        an ``<inertial>`` block populated from the node's derived
        ``mass`` and ``inertia_diag``.
      * Arm ``<geometry>`` dispatches on ``arm.cross_section``:
        ``<cylinder>`` for solid / hollow-tube cross sections,
        ``<box>`` for rectangular.
      * One ``<joint>`` per parent-child edge. Default is
        ``type="fixed"`` (rigid drone). A future ``CompliantJoint``
        annotation on ``ArmNode`` would emit ``type="revolute"`` with
        ``<dynamics damping="0"/>``; the non-linear ``τ(θ)`` law is
        stashed as sidecar attributes for a runtime controller to apply
        (same two-layer pattern soft_airframe uses with ``morphy:*``
        USD attributes).

    Not yet implemented.
    """
    raise NotImplementedError(
        "blueprint_to_urdf is not yet implemented. "
        "Use blueprint_to_mjspec for MuJoCo, or blueprint_to_propellers "
        "for the Python physics stack."
    )


def blueprint_to_usd(bp: DroneBlueprint, out_path: str) -> str:
    """Compile a DroneBlueprint into a USD file for Isaac Lab.

    Not yet implemented — would emit a USD prim hierarchy (`Xform` per node,
    `RigidBodyAPI` on links, `ArticulationRootAPI` on the core), so the
    file can be loaded by ``isaaclab.sim.UsdFileCfg(usd_path=...)``.
    """
    raise NotImplementedError("blueprint_to_usd is the consortium-deferred backend.")
