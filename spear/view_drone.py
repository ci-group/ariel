#!/usr/bin/env python3
"""Build a drone from a genome (URDF -> USD) and simulate it in the Isaac Lab viewer.

By default gravity is disabled and the base is held at its spawn pose, so
posing the arms can't push the body around. Pass --gravity to let it drop onto
the ground, and --fix_base with --gravity to keep the base held anyway.

A "Drone joints" panel (Kit window, or the Newton viewer's side panel under
"IsaacLab Options") has one slider per limited joint (shoulder, elbow, motor
tilt), plus group sliders that set every shoulder / elbow / motor tilt at once.
By default the sliders are position targets for implicit PD actuators, so the
joints are physically driven there by the simulation (tune --stiffness /
--damping). --teleport instead writes the slider values straight into the joint
state each step (kinematic posing).

Run with:
    <IsaacLab>/isaaclab.sh -p spear/view_drone.py [--genome my_genome.npy]
"""
import argparse
import math
import re
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher

from drone_urdf import OUTPUT_DIR, load_genome, write_urdf

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--genome", type=Path, help="Genome .npy file (default: built-in sample).")
parser.add_argument("--name", type=str, help="Robot name (default: genome file stem).")
parser.add_argument("--z", type=float, default=1.0, help="Spawn height.")
parser.add_argument("--gravity", action="store_true", help="Enable gravity so the drone falls.")
parser.add_argument("--fix_base", action="store_true",
                    help="Hold the root link at its spawn pose even with --gravity (always on without it).")
parser.add_argument("--teleport", action="store_true",
                    help="Write slider values directly into joint state instead of driving the joints.")
parser.add_argument("--stiffness", type=float, default=20.0, help="PD stiffness of the arm/motor joint drives.")
parser.add_argument("--damping", type=float, default=1.0, help="PD damping of the arm/motor joint drives.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene_data.scene_data_provider import SceneDataProvider
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


# Workaround for an IsaacLab bug with the PhysX backend + Newton viewer: when the PhysX body
# order already matches the Newton shadow model (single-robot scene), create_mapping() returns
# None and get_transforms() "passes through" by rebinding the output's field instead of writing
# into NewtonManager._state_0.body_q, so the viewer keeps drawing the spawn pose forever.
# Forcing a copy makes it write into body_q.
_get_transforms = SceneDataProvider.get_transforms
SceneDataProvider.get_transforms = lambda self, output, mapping=None, allow_passthrough=True: _get_transforms(
    self, output, mapping=mapping, allow_passthrough=False
)


# Genome -> URDF -> USD.
genome, default_name = load_genome(args_cli.genome)
name = args_cli.name or default_name
urdf_path = write_urdf(genome, name)
# The importer writes usd/<name>/<name>.usda and adds _1, _2, ... suffixes instead of overwriting.
shutil.rmtree(OUTPUT_DIR / "usd" / name, ignore_errors=True)
usd_path = UrdfConverter(UrdfConverterCfg(
    asset_path=str(urdf_path),
    usd_dir=str(OUTPUT_DIR / "usd"),
    fix_base=False,
    force_usd_conversion=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        target_type="none",
    ),
)).usd_path
print(f"URDF: {urdf_path}\nUSD:  {usd_path}")


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
sim.set_camera_view([1.2, 1.2, args_cli.z + 0.8], [0.0, 0.0, args_cli.z])
for prim_path, cfg in [
    ("/World/GroundPlane", sim_utils.GroundPlaneCfg()),
    ("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0, color=(0.95, 0.95, 0.95))),
]:
    cfg.func(prim_path, cfg)

drone = Articulation(ArticulationCfg(
    prim_path="/World/drone",
    actuators={} if args_cli.teleport else {
        # Every joint except the rotor spin joints (shoulder, elbow, motor tilt).
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=[r"(?!.*_motor_to_rotor_joint$).*"],
            stiffness=args_cli.stiffness,
            damping=args_cli.damping,
        ),
    },
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=not args_cli.gravity,
            linear_damping=0.2,
            angular_damping=0.2,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            # Not fix_root_link: on converted drone USDs it adds a second articulation root and
            # initialization fails. The base is held kinematically in the loop below instead.
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, args_cli.z)),
))

sim.reset()

hold_base = args_cli.fix_base or not args_cli.gravity
print(f"Bodies ({drone.num_bodies}): {drone.body_names}")
print(f"Joints ({drone.num_joints}): {drone.joint_names}")
print(f"Gravity: {'on' if args_cli.gravity else 'off'}, base held: {hold_base}")
print("Joint mode: " + ("teleport" if args_cli.teleport
      else f"PD drive (stiffness={args_cli.stiffness}, damping={args_cli.damping})"))


limits = drone.data.joint_pos_limits.torch[0].detach().cpu()
# Continuous rotor joints come through with infinite (or float-max) limits; skip them.
posable = [
    (i, name, float(limits[i, 0]), float(limits[i, 1]))
    for i, name in enumerate(drone.joint_names)
    if math.isfinite(float(limits[i, 0])) and math.isfinite(float(limits[i, 1]))
    and float(limits[i, 1]) - float(limits[i, 0]) <= 4.0 * math.pi
]
posable_ids = [i for i, _, _, _ in posable]
joint_values = {i: 0.0 for i in posable_ids}
print(f"Posable joints ({len(posable)}): {[name for _, name, _, _ in posable]}")

# Group name -> (member joints as (id, lo, hi), shared slider range). Shared by both UIs.
JOINT_GROUPS = {
    "All shoulders": re.compile(r"base_to_arm_\d+_shoulder_joint"),
    "All elbows": re.compile(r"arm_\d+_elbow_joint"),
    "All motor tilts": re.compile(r"arm_\d+_arm_to_motor_joint"),
}
groups = {}
for group_name, pattern in JOINT_GROUPS.items():
    members = [(j, lo, hi) for j, name, lo, hi in posable if pattern.fullmatch(name)]
    if members:
        groups[group_name] = (members, max(lo for _, lo, _ in members), min(hi for _, _, hi in members))
group_values = {group_name: 0.0 for group_name in groups}


def set_group(group_name, value):
    group_values[group_name] = value
    for joint_id, lo, hi in groups[group_name][0]:
        joint_values[joint_id] = min(max(value, lo), hi)


def reset_joints():
    for group_name in group_values:
        group_values[group_name] = 0.0
    for joint_id in joint_values:
        joint_values[joint_id] = 0.0


def draw_newton_joint_panel(imgui):
    """Immediate-mode slider panel for Newton's ViewerGL (--viz newton)."""
    imgui.separator()
    imgui.text("Drone joints")
    for group_name, (_, lo, hi) in groups.items():
        changed, value = imgui.slider_float(f"{group_name}##group", group_values[group_name], lo, hi)
        if changed:
            set_group(group_name, value)
    if imgui.button("Reset all to 0"):
        reset_joints()
    imgui.separator()
    for joint_id, name, lo, hi in posable:
        changed, value = imgui.slider_float(f"{name.replace('_joint', '')}##{joint_id}", joint_values[joint_id], lo, hi)
        if changed:
            joint_values[joint_id] = value


for visualizer in getattr(sim, "visualizers", []):
    viewer = getattr(visualizer, "_viewer", None)
    if viewer is not None and hasattr(viewer, "register_ui_callback") and posable:
        viewer.register_ui_callback(draw_newton_joint_panel, position="side")
        print(f"Joint sliders added to {type(visualizer).__name__} (under 'IsaacLab Options').")


try:
    import omni.ui as ui
except ImportError:
    ui = None
    print("omni.ui unavailable (headless?), joint sliders disabled.")

if ui is not None and posable:
    joint_models = {}

    def on_joint_changed(joint_id):
        def _fn(model):
            joint_values[joint_id] = model.get_value_as_float()
        return _fn

    def on_group_changed(group_name):
        def _fn(model):
            set_group(group_name, model.get_value_as_float())
            for joint_id, _, _ in groups[group_name][0]:
                joint_models[joint_id].set_value(joint_values[joint_id])
        return _fn

    def reset_all():
        reset_joints()
        for model in joint_models.values():
            model.set_value(0.0)

    ui_window = ui.Window("Drone joints", width=420, height=600)
    with ui_window.frame:
        with ui.ScrollingFrame():
            with ui.VStack(spacing=4):
                for group_name, (_, lo, hi) in groups.items():
                    with ui.HStack(height=22):
                        ui.Label(group_name, width=160)
                        group_slider = ui.FloatSlider(min=lo, max=hi, step=0.01)
                        group_slider.model.add_value_changed_fn(on_group_changed(group_name))
                ui.Button("Reset all to 0", height=26, clicked_fn=reset_all)
                ui.Separator(height=8)
                for joint_id, name, lo, hi in posable:
                    with ui.HStack(height=22):
                        ui.Label(name, width=220, elided_text=True, tooltip=f"[{lo:.2f}, {hi:.2f}] rad")
                        slider = ui.FloatSlider(min=lo, max=hi, step=0.01)
                        slider.model.add_value_changed_fn(on_joint_changed(joint_id))
                        joint_models[joint_id] = slider.model


dt = sim.get_physics_dt()
spawn_root_pose = drone.data.default_root_pose.torch.clone()
zero_root_velocity = torch.zeros(spawn_root_pose.shape[0], 6, device=spawn_root_pose.device)

while simulation_app.is_running():
    if hold_base:
        drone.write_root_pose_to_sim_index(root_pose=spawn_root_pose)
        drone.write_root_velocity_to_sim_index(root_velocity=zero_root_velocity)
    if posable_ids:
        targets = torch.tensor([[joint_values[i] for i in posable_ids]], device=drone.device)
        if args_cli.teleport:
            drone.write_joint_position_to_sim_index(position=targets, joint_ids=posable_ids)
            drone.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(targets), joint_ids=posable_ids)
        else:
            drone.set_joint_position_target_index(target=targets, joint_ids=posable_ids)
    drone.write_data_to_sim()
    sim.step()
    drone.update(dt)

simulation_app.close()
