"""DroneBlueprint → phenotype backends.

Each backend consumes a DroneBlueprint and emits something a simulator (or
the real world) can instantiate. v1 ships:

  * ``blueprint_to_propellers`` — list[dict] consumable by
    ``ariel.simulation.drone.DroneSimulator`` / ``DroneConfiguration``.
  * ``blueprint_to_mjspec``     — MuJoCo ``mjSpec`` (compiles to MJCF /
    ``MjModel``); the same blueprint can drive both the Python physics
    stack and a full MuJoCo simulation.
  * ``blueprint_to_urdf``       — URDF XML string for Isaac Lab (all fixed
    joints; thrust applied externally via apply_external_force_torque).

Stubs sketched for future backends:

  * ``blueprint_to_usd``        — USD prim hierarchy for Isaac Lab
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Any, TYPE_CHECKING

import numpy as np

from .blueprint import DroneBlueprint, ArmNode, MotorNode, RotorNode, CorePlateNode

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
    motor_mass: float = 0.05,
    arm_mass: float = 0.01,
    core_mass_override: float | None = None,
    arm_radius: float = 0.005,
    motor_radius: float = 0.015,
    motor_thickness: float = 0.008,
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

    Args:
        bp: the blueprint to compile.
        motor_mass: kg per motor+rotor assembly (lumped).
        arm_radius: capsule radius for arm visuals (m).
        motor_radius: cylinder radius for motor visuals (m).
        motor_thickness: cylinder half-length for motor visuals (m).
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
        arm_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0.0, 0.0, 0.0, *tip_local.tolist()],
            size=[arm_radius, 0.0, 0.0],
            mass=arm_mass,
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
                size=[motor_radius, motor_thickness, 0.0],
                mass=motor_mass,
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
                pos=[0.0, 0.0, motor_thickness + 0.002],
                size=[rotor_radius, 0.001, 0.0],
                mass=0.0,
                rgba=(0.8, 0.8, 0.8, 0.5),
            )

            # Thrust site (force applied along local +Z)
            thrust_site = motor_body.add_site(
                name=f"{body_name}_thrust_{motor_index}",
                pos=[0.0, 0.0, motor_thickness],
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


# ---------- URDF backend (blueprint → URDF XML string) ----------

def blueprint_to_urdf(
    bp: DroneBlueprint,
    *,
    robot_name: str = "drone",
    motor_mass: float = 0.05,
    arm_mass: float = 0.01,
    rotor_mass: float = 0.005,
    core_mass_override: float | None = None,
    arm_radius: float = 0.005,
    motor_radius: float = 0.015,
    motor_thickness: float = 0.008,
) -> str:
    """Compile a DroneBlueprint into a URDF XML string for Isaac Lab.

    All joints are fixed — the drone is a rigid body tree. Thrust forces are
    applied externally via Isaac Lab's ``apply_external_force_torque`` API.
    Isaac Lab's URDF converter adds the floating-base freejoint.

    Conventions:
      * Z-up, matching ``blueprint_to_mjspec``.
      * Inertia tensors computed from solid-cylinder geometry.
      * Each link has matching visual and collision primitives.

    Args:
        bp: the blueprint to compile.
        robot_name: ``<robot name>`` attribute and link/joint name prefix.
        motor_mass: kg per motor assembly (lumped).
        arm_mass: kg per arm.
        rotor_mass: kg per rotor disc (must be >0 for PhysX).
        core_mass_override: override ``CorePlateNode.mass`` if set.
        arm_radius: arm cylinder radius (m).
        motor_radius: motor cylinder radius (m).
        motor_thickness: motor cylinder half-height (m).

    Returns:
        A URDF XML string (UTF-8, with ``<?xml?>`` declaration).
    """
    core = bp.payload(bp.root_id)
    if not isinstance(core, CorePlateNode):
        raise ValueError("Blueprint root must be a CorePlateNode.")

    robot = ET.Element("robot", name=robot_name)

    def _v(*vals: float) -> str:
        return " ".join(f"{v:.6g}" for v in vals)

    def _inertial(
        link_el: ET.Element,
        mass: float,
        ixx: float, iyy: float, izz: float,
        ixy: float = 0.0, ixz: float = 0.0, iyz: float = 0.0,
        com: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        inertial = ET.SubElement(link_el, "inertial")
        ET.SubElement(inertial, "origin", xyz=_v(*com), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{mass:.6g}")
        ET.SubElement(inertial, "inertia",
                      ixx=f"{ixx:.6g}", ixy=f"{ixy:.6g}", ixz=f"{ixz:.6g}",
                      iyy=f"{iyy:.6g}", iyz=f"{iyz:.6g}", izz=f"{izz:.6g}")

    def _cylinder(
        link_el: ET.Element,
        radius: float,
        length: float,
        origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rgba: tuple[float, float, float, float] | None = None,
        tag: str = "visual",
    ) -> None:
        el = ET.SubElement(link_el, tag)
        ET.SubElement(el, "origin", xyz=_v(*origin_xyz), rpy=_v(*origin_rpy))
        geom = ET.SubElement(el, "geometry")
        ET.SubElement(geom, "cylinder", radius=f"{radius:.6g}", length=f"{length:.6g}")
        if tag == "visual" and rgba is not None:
            mat = ET.SubElement(el, "material", name="")
            ET.SubElement(mat, "color", rgba=_v(*rgba))

    def _ixx_iyy_izz_cylinder_z(mass: float, r: float, h: float) -> tuple[float, float, float]:
        """Solid cylinder, axis along Z."""
        ixx = iyy = (mass / 12.0) * (3 * r * r + h * h)
        izz = 0.5 * mass * r * r
        return ixx, iyy, izz

    def _ixx_iyy_izz_cylinder_x(mass: float, r: float, h: float) -> tuple[float, float, float]:
        """Solid cylinder, axis along X (used for arms)."""
        ixx = 0.5 * mass * r * r
        iyy = izz = (mass / 12.0) * (3 * r * r + h * h)
        return ixx, iyy, izz

    # --- core link (cylinder along Z) ---
    core_mass = core_mass_override if core_mass_override is not None else core.mass
    core_link = ET.SubElement(robot, "link", name=f"{robot_name}_core")
    ixx, iyy, izz = _ixx_iyy_izz_cylinder_z(core_mass, core.radius, core.thickness)
    _inertial(core_link, core_mass, ixx, iyy, izz)
    _cylinder(core_link, core.radius, core.thickness,
              rgba=(0.2, 0.4, 0.8, 1.0), tag="visual")
    _cylinder(core_link, core.radius, core.thickness, tag="collision")

    motor_index = 0
    for arm_id in bp.children(bp.root_id):  # type: ignore[arg-type]
        arm = bp.payload(arm_id)
        if not isinstance(arm, ArmNode):
            continue

        arm_link_name = f"{robot_name}_arm_{arm_id}"

        # Arm joint: fixed, child frame = arm attachment pose on core.
        arm_joint = ET.SubElement(robot, "joint",
                                  name=f"{arm_link_name}_joint", type="fixed")
        ET.SubElement(arm_joint, "parent", link=f"{robot_name}_core")
        ET.SubElement(arm_joint, "child", link=arm_link_name)
        ET.SubElement(arm_joint, "origin",
                      xyz=_v(*arm.pose.xyz), rpy=_v(*arm.pose.rpy))

        # Arm link: cylinder along arm's local +X.
        # URDF cylinders default to Z; rotate pi/2 around Y to align Z→X.
        # CoM sits at the arm midpoint (length/2 along +X).
        arm_link = ET.SubElement(robot, "link", name=arm_link_name)
        ixx, iyy, izz = _ixx_iyy_izz_cylinder_x(arm_mass, arm_radius, arm.length)
        _inertial(arm_link, arm_mass, ixx, iyy, izz,
                  com=(arm.length / 2.0, 0.0, 0.0))
        arm_geom_origin = (arm.length / 2.0, 0.0, 0.0)
        arm_geom_rpy = (0.0, math.pi / 2.0, 0.0)
        _cylinder(arm_link, arm_radius, arm.length,
                  origin_xyz=arm_geom_origin, origin_rpy=arm_geom_rpy,
                  rgba=(0.3, 0.3, 0.3, 1.0), tag="visual")
        _cylinder(arm_link, arm_radius, arm.length,
                  origin_xyz=arm_geom_origin, origin_rpy=arm_geom_rpy,
                  tag="collision")

        for motor_id in bp.children(arm_id):
            motor = bp.payload(motor_id)
            if not isinstance(motor, MotorNode):
                continue

            motor_link_name = f"{robot_name}_motor_{motor_index}"
            motor_h = 2.0 * motor_thickness

            # Motor joint: fixed, child frame = motor pose on arm.
            motor_joint = ET.SubElement(robot, "joint",
                                        name=f"{motor_link_name}_joint", type="fixed")
            ET.SubElement(motor_joint, "parent", link=arm_link_name)
            ET.SubElement(motor_joint, "child", link=motor_link_name)
            ET.SubElement(motor_joint, "origin",
                          xyz=_v(*motor.pose.xyz), rpy=_v(*motor.pose.rpy))

            # Motor link: cylinder along Z.
            motor_link = ET.SubElement(robot, "link", name=motor_link_name)
            ixx, iyy, izz = _ixx_iyy_izz_cylinder_z(motor_mass, motor_radius, motor_h)
            _inertial(motor_link, motor_mass, ixx, iyy, izz)
            rgba = (1.0, 0.2, 0.2, 1.0) if motor.spin == "cw" else (0.2, 0.8, 0.2, 1.0)
            _cylinder(motor_link, motor_radius, motor_h, rgba=rgba, tag="visual")
            _cylinder(motor_link, motor_radius, motor_h, tag="collision")

            # Rotor (if present in blueprint).
            for rotor_id in bp.children(motor_id):
                rotor = bp.payload(rotor_id)
                if not isinstance(rotor, RotorNode):
                    continue

                rotor_link_name = f"{robot_name}_rotor_{motor_index}"
                rotor_h = 0.002  # thin disc

                rotor_joint = ET.SubElement(robot, "joint",
                                            name=f"{rotor_link_name}_joint", type="fixed")
                ET.SubElement(rotor_joint, "parent", link=motor_link_name)
                ET.SubElement(rotor_joint, "child", link=rotor_link_name)
                ET.SubElement(rotor_joint, "origin",
                               xyz=_v(0.0, 0.0, motor_thickness + rotor_h / 2.0),
                               rpy="0 0 0")

                rotor_link = ET.SubElement(robot, "link", name=rotor_link_name)
                ixx, iyy, izz = _ixx_iyy_izz_cylinder_z(rotor_mass, rotor.radius, rotor_h)
                _inertial(rotor_link, rotor_mass, ixx, iyy, izz)
                _cylinder(rotor_link, rotor.radius, rotor_h,
                          rgba=(0.8, 0.8, 0.8, 0.5), tag="visual")
                _cylinder(rotor_link, rotor.radius, rotor_h, tag="collision")

            motor_index += 1

    ET.indent(robot, space="  ")
    return '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode")


# ---------- USD backend (blueprint → .usda ASCII file) ----------

def blueprint_to_usd(
    bp: DroneBlueprint,
    out_path: str,
    *,
    robot_name: str = "drone",
    motor_mass: float = 0.05,
    arm_mass: float = 0.01,
    rotor_mass: float = 0.005,
    core_mass_override: float | None = None,
    arm_radius: float = 0.005,
    motor_radius: float = 0.015,
    motor_thickness: float = 0.008,
) -> str:
    """Compile a DroneBlueprint into a USD ASCII (.usda) file for Isaac Lab.

    Writes a ``UsdGeom`` prim hierarchy consumable by
    ``isaaclab.sim.UsdFileCfg(usd_path=...)``:

    * Root ``Xform`` carries ``PhysicsArticulationRootAPI`` +
      ``PhysicsRigidBodyAPI`` so Isaac Lab treats it as a free-floating body.
    * Each link (core, arms, motors, rotors) is a child ``Xform`` with
      ``PhysicsMassAPI`` for mass assignment.
    * Geometry prims (``Cylinder`` / ``Capsule``) carry
      ``PhysicsCollisionAPI`` for PhysX contact.
    * Conventions match ``blueprint_to_mjspec``: Z-up, no NED inversion.

    Does **not** require ``pxr`` / Omniverse at runtime — the file is
    produced as plain text.

    Args:
        bp: the blueprint to compile.
        out_path: destination ``.usda`` file path (created/overwritten).
        robot_name: USD ``defaultPrim`` name and prim-name prefix.
        motor_mass: kg per motor assembly (lumped).
        arm_mass: kg per arm.
        rotor_mass: kg per rotor disc (must be >0 for PhysX).
        core_mass_override: override ``CorePlateNode.mass`` if set.
        arm_radius: arm capsule radius (m).
        motor_radius: motor cylinder radius (m).
        motor_thickness: motor cylinder half-height (m).

    Returns:
        The absolute path to the written ``.usda`` file.
    """
    from pathlib import Path

    core = bp.payload(bp.root_id)
    if not isinstance(core, CorePlateNode):
        raise ValueError("Blueprint root must be a CorePlateNode.")

    core_mass = core_mass_override if core_mass_override is not None else core.mass

    # --- lightweight USDA string builder ---
    lines: list[str] = []
    depth = 0

    def _w(s: str = "") -> None:
        lines.append("    " * depth + s if s else "")

    def _prim_open(prim_def: str, schemas: list[str]) -> None:
        nonlocal depth
        schema_str = ", ".join(f'"{s}"' for s in schemas)
        _w(f"{prim_def} (")
        depth += 1
        _w(f"prepend apiSchemas = [{schema_str}]")
        depth -= 1
        _w(")")
        _w("{")
        depth += 1

    def _meta_open() -> None:
        nonlocal depth
        _w("(")
        depth += 1

    def _close(bracket: str = "}") -> None:
        nonlocal depth
        depth -= 1
        _w(bracket)

    def _quat(rpy: tuple[float, float, float]) -> str:
        w, x, y, z = _rpy_to_quat(*rpy)
        return f"({w:.6g}, {x:.6g}, {y:.6g}, {z:.6g})"

    def _xform_ops(xyz: tuple[float, float, float],
                   rpy: tuple[float, float, float] | None = None) -> None:
        _w(f"double3 xformOp:translate = ({xyz[0]:.6g}, {xyz[1]:.6g}, {xyz[2]:.6g})")
        if rpy is not None and any(v != 0.0 for v in rpy):
            _w(f"quatf xformOp:orient = {_quat(rpy)}")
            _w('uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]')
        else:
            _w('uniform token[] xformOpOrder = ["xformOp:translate"]')

    def _color(r: float, g: float, b: float) -> str:
        return f"color3f[] primvars:displayColor = [({r:.3g}, {g:.3g}, {b:.3g})]"

    # --- file header ---
    _w("#usda 1.0")
    _meta_open()
    _w('upAxis = "Z"')
    _w("metersPerUnit = 1.0")
    _w(f'defaultPrim = "{robot_name}"')
    _w('doc = """DroneBlueprint USD export — generated by ariel.body_phenotypes.drone.backends"""')
    _close(")")
    _w()

    # --- root prim ---
    _prim_open(f'def Xform "{robot_name}"',
               ["PhysicsArticulationRootAPI", "PhysicsRigidBodyAPI"])
    _xform_ops((0.0, 0.0, 0.0))
    _w()

    # core geometry
    _prim_open('def Xform "core"', ["PhysicsMassAPI"])
    _w(f"float physics:mass = {core_mass:.6g}")
    _xform_ops((0.0, 0.0, 0.0))
    _w()
    _prim_open('def Cylinder "geom"', ["PhysicsCollisionAPI"])
    _w(f"double radius = {core.radius:.6g}")
    _w(f"double height = {core.thickness:.6g}")
    _w('token axis = "Z"')
    _w(_color(0.2, 0.4, 0.8))
    _close()  # geom
    _close()  # core Xform
    _w()

    motor_index = 0
    for arm_id in bp.children(bp.root_id):  # type: ignore[arg-type]
        arm = bp.payload(arm_id)
        if not isinstance(arm, ArmNode):
            continue

        _prim_open(f'def Xform "arm_{arm_id}"', ["PhysicsMassAPI"])
        _w(f"float physics:mass = {arm_mass:.6g}")
        _xform_ops(arm.pose.xyz, arm.pose.rpy)
        _w()

        # Capsule along local +X; USD Capsule height = cylindrical-part length
        cap_height = max(0.0, arm.length - 2.0 * arm_radius)
        _prim_open('def Capsule "geom"', ["PhysicsCollisionAPI"])
        _w(f"double radius = {arm_radius:.6g}")
        _w(f"double height = {cap_height:.6g}")
        _w('token axis = "X"')
        _w(f"double3 xformOp:translate = ({arm.length / 2.0:.6g}, 0, 0)")
        _w('uniform token[] xformOpOrder = ["xformOp:translate"]')
        _w(_color(0.3, 0.3, 0.3))
        _close()  # geom
        _w()

        for motor_id in bp.children(arm_id):
            motor = bp.payload(motor_id)
            if not isinstance(motor, MotorNode):
                continue

            _prim_open(f'def Xform "motor_{motor_index}"', ["PhysicsMassAPI"])
            _w(f"float physics:mass = {motor_mass:.6g}")
            _xform_ops(motor.pose.xyz, motor.pose.rpy)
            _w()

            motor_h = 2.0 * motor_thickness
            r, g, b = (1.0, 0.2, 0.2) if motor.spin == "cw" else (0.2, 0.8, 0.2)
            _prim_open('def Cylinder "geom"', ["PhysicsCollisionAPI"])
            _w(f"double radius = {motor_radius:.6g}")
            _w(f"double height = {motor_h:.6g}")
            _w('token axis = "Z"')
            _w(_color(r, g, b))
            _close()  # geom
            _w()

            for rotor_id in bp.children(motor_id):
                rotor = bp.payload(rotor_id)
                if not isinstance(rotor, RotorNode):
                    continue

                rotor_h = 0.002
                rotor_z = motor_thickness + rotor_h / 2.0
                _open(f'def Xform "rotor_{motor_index}" (\n    prepend apiSchemas = ["PhysicsMassAPI"]\n)')
                _w(f"float physics:mass = {rotor_mass:.6g}")
                _xform_ops((0.0, 0.0, rotor_z))
                _w()
                _open('def Cylinder "geom" (\n    prepend apiSchemas = ["PhysicsCollisionAPI"]\n)')
                _w(f"double radius = {rotor.radius:.6g}")
                _w(f"double height = {rotor_h:.6g}")
                _w('token axis = "Z"')
                _w(_color(0.8, 0.8, 0.8))
                _close()  # geom
                _close()  # rotor Xform
                _w()

            _close()  # motor Xform
            motor_index += 1
            _w()

        _close()  # arm Xform
        _w()

    _close()  # root Xform

    usda = "\n".join(lines)
    Path(out_path).write_text(usda, encoding="utf-8")
    return str(Path(out_path).resolve())
