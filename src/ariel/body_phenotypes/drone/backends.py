"""DroneBlueprint → phenotype backends.

Each backend consumes a DroneBlueprint and emits something a simulator (or
the real world) can instantiate:

  * ``blueprint_to_propellers`` — list[dict] consumable by
    ``ariel.simulation.drone.DroneSimulator`` / ``DroneConfiguration``.
  * ``blueprint_to_mjspec``     — MuJoCo ``mjSpec`` (compiles to MJCF /
    ``MjModel``); the same blueprint can drive both the Python physics
    stack and a full MuJoCo simulation.
  * ``blueprint_to_urdf``       — URDF file (rigid drone, fixed joints).
    Intermediate for Isaac Lab via ``isaaclab.sim.converters.UrdfConverter``.

Stubs sketched for future backends:

  * ``blueprint_to_usd``        — USD prim hierarchy for Isaac Lab
    (direct, no URDF intermediate).
"""
from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING
from xml.dom import minidom

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


# ---------- URDF backend (blueprint → .urdf) ----------

def _urdf_fmt(x: float) -> str:
    return f"{float(x):.6g}"


def _urdf_fmt_vec(v: Sequence[float]) -> str:
    return " ".join(_urdf_fmt(c) for c in v)


def _urdf_add_origin(parent: ET.Element, xyz: Sequence[float], rpy: Sequence[float]) -> None:
    ET.SubElement(parent, "origin", attrib={
        "xyz": _urdf_fmt_vec(xyz),
        "rpy": _urdf_fmt_vec(rpy),
    })


def _urdf_add_inertial(
    link: ET.Element,
    *,
    com_xyz: Sequence[float],
    mass: float,
    inertia_diag: Sequence[float],
) -> None:
    """Append an ``<inertial>`` block with diagonal inertia tensor.
    ``inertia_diag`` is ``(Ixx, Iyy, Izz)`` about the COM."""
    inertial = ET.SubElement(link, "inertial")
    _urdf_add_origin(inertial, com_xyz, (0.0, 0.0, 0.0))
    ET.SubElement(inertial, "mass", attrib={"value": _urdf_fmt(mass)})
    Ixx, Iyy, Izz = inertia_diag
    ET.SubElement(inertial, "inertia", attrib={
        "ixx": _urdf_fmt(Ixx), "ixy": "0", "ixz": "0",
        "iyy": _urdf_fmt(Iyy), "iyz": "0",
        "izz": _urdf_fmt(Izz),
    })


def _urdf_add_visual_collision(
    link: ET.Element,
    *,
    xyz: Sequence[float],
    rpy: Sequence[float],
    geometry: ET.Element,
    rgba: Sequence[float] | None = None,
    include_collision: bool = True,
) -> None:
    """Append matching ``<visual>`` (+ optional ``<collision>``) blocks
    with the given geometry. ``geometry`` is a freshly-built ``<geometry>``
    element; we deep-copy it so visual and collision are independent."""
    visual = ET.SubElement(link, "visual")
    _urdf_add_origin(visual, xyz, rpy)
    visual.append(copy.deepcopy(geometry))
    if rgba is not None:
        mat = ET.SubElement(visual, "material", attrib={"name": ""})
        ET.SubElement(mat, "color", attrib={"rgba": _urdf_fmt_vec(rgba)})
    if include_collision:
        collision = ET.SubElement(link, "collision")
        _urdf_add_origin(collision, xyz, rpy)
        collision.append(copy.deepcopy(geometry))


def _urdf_cylinder_geom(radius: float, length: float) -> ET.Element:
    geom = ET.Element("geometry")
    ET.SubElement(geom, "cylinder", attrib={
        "radius": _urdf_fmt(radius),
        "length": _urdf_fmt(length),
    })
    return geom


def _urdf_box_geom(size: Sequence[float]) -> ET.Element:
    geom = ET.Element("geometry")
    ET.SubElement(geom, "box", attrib={"size": _urdf_fmt_vec(size)})
    return geom


def _urdf_add_fixed_joint(
    robot: ET.Element,
    *,
    name: str,
    parent: str,
    child: str,
    xyz: Sequence[float],
    rpy: Sequence[float],
) -> None:
    joint = ET.SubElement(robot, "joint", attrib={"name": name, "type": "fixed"})
    _urdf_add_origin(joint, xyz, rpy)
    ET.SubElement(joint, "parent", attrib={"link": parent})
    ET.SubElement(joint, "child", attrib={"link": child})


def _urdf_core_inertia_diag(core: CorePlateNode, mass: float) -> tuple[float, float, float]:
    """Solid-cylinder inertia about COM, disc axis = link's local +Z."""
    r, h = core.radius, core.thickness
    Ixx = Iyy = mass * (3.0 * r * r + h * h) / 12.0
    Izz = 0.5 * mass * r * r
    return (Ixx, Iyy, Izz)


def _urdf_add_core_link(robot: ET.Element, name: str, core: CorePlateNode, mass: float) -> None:
    link = ET.SubElement(robot, "link", attrib={"name": name})
    _urdf_add_inertial(
        link,
        com_xyz=(0.0, 0.0, 0.0),
        mass=mass,
        inertia_diag=_urdf_core_inertia_diag(core, mass),
    )
    _urdf_add_visual_collision(
        link,
        xyz=(0.0, 0.0, 0.0),
        rpy=(0.0, 0.0, 0.0),
        geometry=_urdf_cylinder_geom(core.radius, core.thickness),
        rgba=(0.2, 0.4, 0.8, 1.0),
    )


def _urdf_add_arm_link(robot: ET.Element, name: str, arm: ArmNode) -> None:
    """Arm link frame is at the joint-to-parent; arm extends along local +X.
    Visual / collision geometry is centered at midpoint along the arm."""
    link = ET.SubElement(robot, "link", attrib={"name": name})
    com_xyz = (arm.length / 2.0, 0.0, 0.0)
    _urdf_add_inertial(link, com_xyz=com_xyz, mass=arm.mass, inertia_diag=arm.inertia_diag)

    cs = arm.cross_section
    if isinstance(cs, RectangularCrossSection):
        _urdf_add_visual_collision(
            link,
            xyz=com_xyz,
            rpy=(0.0, 0.0, 0.0),
            geometry=_urdf_box_geom((arm.length, cs.width, cs.thickness)),
            rgba=(0.3, 0.3, 0.3, 1.0),
        )
    else:
        # Cylindrical (solid) or HollowTube — URDF has no hollow cylinder,
        # render solid using outer radius. URDF cylinders point along the
        # link's local +Z; rotate by π/2 about +Y so the cylinder axis
        # aligns with the arm's local +X.
        visual_radius = (
            cs.outer_radius if isinstance(cs, HollowTubeCrossSection) else cs.radius
        )
        _urdf_add_visual_collision(
            link,
            xyz=com_xyz,
            rpy=(0.0, math.pi / 2.0, 0.0),
            geometry=_urdf_cylinder_geom(visual_radius, arm.length),
            rgba=(0.3, 0.3, 0.3, 1.0),
        )


def _urdf_add_motor_link(
    robot: ET.Element,
    name: str,
    motor: MotorNode,
    rotor: RotorNode | None,
) -> None:
    """Motor link: cylinder along local +Z (the thrust axis), centered at
    the link origin. An optional rotor visual disc sits above it."""
    link = ET.SubElement(robot, "link", attrib={"name": name})
    full_height = 2.0 * motor.thickness
    rgba = (1.0, 0.2, 0.2, 1.0) if motor.spin == "cw" else (0.2, 0.8, 0.2, 1.0)
    _urdf_add_inertial(
        link,
        com_xyz=(0.0, 0.0, 0.0),
        mass=motor.mass,
        inertia_diag=motor.inertia_diag,
    )
    _urdf_add_visual_collision(
        link,
        xyz=(0.0, 0.0, 0.0),
        rpy=(0.0, 0.0, 0.0),
        geometry=_urdf_cylinder_geom(motor.radius, full_height),
        rgba=rgba,
    )
    if rotor is not None:
        # Visual-only thin disc above the motor; no collision, no mass
        # (motor.mass already lumps motor + propeller).
        visual = ET.SubElement(link, "visual")
        _urdf_add_origin(visual, (0.0, 0.0, motor.thickness + 0.002), (0.0, 0.0, 0.0))
        visual.append(_urdf_cylinder_geom(rotor.radius, 0.002))
        mat = ET.SubElement(visual, "material", attrib={"name": ""})
        ET.SubElement(mat, "color", attrib={"rgba": "0.8 0.8 0.8 0.5"})


def blueprint_to_urdf(
    bp: DroneBlueprint,
    out_path: str,
    *,
    core_mass_override: float | None = None,
    robot_name: str = "drone",
) -> str:
    """Compile a DroneBlueprint into a URDF file (rigid drone, fixed joints).

    Intended as the intermediate step for Isaac Lab / Isaac Sim: the
    emitted ``.urdf`` is consumed by
    ``isaaclab.sim.converters.UrdfConverter`` to produce a USD asset
    (see ``soft_airframe_optimization/scripts/convert_xconfig_urdf.py``
    for the conversion call pattern).

    Mapping (mirrors ``blueprint_to_mjspec``):
      * One ``<link>`` per CorePlate / Arm / Motor node, with an
        ``<inertial>`` block populated from the node's derived ``mass``
        and ``inertia_diag``. Visual + collision geometry per link.
        Rotor visuals are folded into the parent motor link (no separate
        rotor link; rotor mass is already lumped into ``motor.mass``).
      * One ``<joint type="fixed">`` per parent-child edge. Drone is
        rigid in v1; compliant-joint support is planned (zero-stiffness
        revolute + sidecar ``ariel:*`` torque attributes, mirroring
        soft_airframe's two-layer pattern).
      * Arm ``<geometry>`` dispatches on ``arm.cross_section``:
        ``<cylinder>`` for solid / hollow-tube cross sections,
        ``<box>`` for rectangular.
      * No actuators: Isaac Lab applies thrust at runtime by force on
        the motor link's local +Z axis.

    Conventions:
      * Z-up, metres, radians. URDF cylinders are along link-local +Z;
        arm cylinders are rotated 90° about +Y so the cylinder axis
        matches the arm's local +X.
      * Each link's frame is at the joint to its parent; joint
        ``<origin>`` carries the child frame's pose in the parent.

    Args:
        bp: blueprint to compile.
        out_path: destination ``.urdf`` path.
        core_mass_override: if given, replaces ``core.mass`` (matches
            ``blueprint_to_mjspec`` semantics).
        robot_name: ``<robot name="...">`` attribute.

    Returns:
        Absolute path to the written URDF, as a string.
    """
    core = bp.payload(bp.root_id)
    if not isinstance(core, CorePlateNode):
        raise ValueError("Blueprint root must be a CorePlateNode.")
    core_mass = core_mass_override if core_mass_override is not None else core.mass

    robot = ET.Element("robot", attrib={"name": robot_name})

    core_link = "base_link"
    _urdf_add_core_link(robot, core_link, core, core_mass)

    for arm_id in bp.children(bp.root_id):  # type: ignore[arg-type]
        arm = bp.payload(arm_id)
        if not isinstance(arm, ArmNode):
            continue
        arm_link = f"arm_{arm_id}"
        _urdf_add_arm_link(robot, arm_link, arm)
        _urdf_add_fixed_joint(
            robot,
            name=f"core_to_arm_{arm_id}",
            parent=core_link,
            child=arm_link,
            xyz=arm.pose.xyz,
            rpy=arm.pose.rpy,
        )
        for motor_id in bp.children(arm_id):
            motor = bp.payload(motor_id)
            if not isinstance(motor, MotorNode):
                continue
            rotor: RotorNode | None = None
            for rotor_id in bp.children(motor_id):
                cand = bp.payload(rotor_id)
                if isinstance(cand, RotorNode):
                    rotor = cand
                    break
            motor_link = f"motor_{motor_id}"
            _urdf_add_motor_link(robot, motor_link, motor, rotor)
            _urdf_add_fixed_joint(
                robot,
                name=f"arm_{arm_id}_to_motor_{motor_id}",
                parent=arm_link,
                child=motor_link,
                xyz=motor.pose.xyz,
                rpy=motor.pose.rpy,
            )

    raw = ET.tostring(robot, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    out = Path(out_path).resolve()
    out.write_text(pretty)
    return str(out)


# ---------- future backends (signatures only) ----------

def blueprint_to_usd(bp: DroneBlueprint, out_path: str) -> str:
    """Compile a DroneBlueprint into a USD file for Isaac Lab.

    Not yet implemented — would emit a USD prim hierarchy (`Xform` per node,
    `RigidBodyAPI` on links, `ArticulationRootAPI` on the core), so the
    file can be loaded by ``isaaclab.sim.UsdFileCfg(usd_path=...)``.
    """
    raise NotImplementedError("blueprint_to_usd is the consortium-deferred backend.")
