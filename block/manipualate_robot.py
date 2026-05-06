#!/usr/bin/env python3
"""Interactive manipulator for the baby robot hinge actuators.

Run this script from the repository root (so `baby_robot.py` is importable):

	python block/manipualate_robot.py

Commands (in the REPL):
  - list
  - set <actuator-name> <degrees>
  - zero
  - quit

The script launches the MuJoCo viewer and registers a short control
callback that reads targets set in the REPL. Targets are in radians and
clipped to [-90, 90] degrees.
"""

from pathlib import Path
import sys
import threading

import numpy as np
import mujoco
from mujoco import viewer

# Ensure repository root is importable (so baby_robot can be imported)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from baby_robot import baby_robot
from ariel.simulation.environments._simple_flat import SimpleFlatWorld

# Shared state between REPL and control callback
manual_targets: dict[str, float] = {}
manual_lock = threading.Lock()


def list_actuators(model: mujoco.MjModel) -> list[tuple[int, str]]:
	arr: list[tuple[int, str]] = []
	for aid in range(model.nu):
		name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or f"actuator_{aid}"
		arr.append((aid, name))
	return arr


def control_callback(model: mujoco.MjModel, data: mujoco.MjData) -> None:
	"""MuJoCo control callback: apply targets from the REPL to actuators."""
	with manual_lock:
		for aid in range(model.nu):
			name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
			target = manual_targets.get(name, 0.0)
			data.ctrl[aid] = float(target)


def repl(model: mujoco.MjModel) -> None:
	print("Interactive REPL — control actuators by name.")
	print("Commands:")
	print("  list                        — list actuators")
	print("  set <actuator-name> <deg>   — set target angle in degrees")
	print("  zero                        — reset targets to 0")
	print("  quit                        — exit REPL (close viewer manually to finish)")
	while True:
		try:
			line = input("cmd> ")
		except EOFError:
			break
		if not line:
			continue
		parts = line.strip().split()
		cmd = parts[0].lower()
		if cmd == "list":
			for aid, name in list_actuators(model):
				print(f"{aid}: {name}")
		elif cmd == "set" and len(parts) >= 3:
			name = parts[1]
			try:
				deg = float(parts[2])
			except ValueError:
				print("Angle must be numeric (degrees).")
				continue
			rad = float(np.deg2rad(deg))
			# Clip to [-pi/2, pi/2]
			rad = max(min(rad, np.pi / 2), -np.pi / 2)
			with manual_lock:
				manual_targets[name] = rad
			print(f"{name} -> {rad:.3f} rad")
		elif cmd in ("zero", "reset"):
			with manual_lock:
				manual_targets.clear()
			print("Cleared targets.")
		elif cmd in ("quit", "exit"):
			print("Exiting REPL. Close/quit the viewer window to finish.")
			break
		else:
			print("Unknown command. Type 'list' or 'set <name> <deg>'.")


def main() -> None:
	core = baby_robot()

	world = SimpleFlatWorld()
	world.spawn(core.spec, correct_collision_with_floor=True)

	model = world.spec.compile()
	data = mujoco.MjData(model)

	print("Available actuators:")
	for aid, name in list_actuators(model):
		print(f"  {aid}: {name}")

	# Register MuJoCo control callback
	mujoco.set_mjcb_control(control_callback)

	# Start REPL thread
	t = threading.Thread(target=repl, args=(model,), daemon=True)
	t.start()

	try:
		# Launch viewer (blocks until window closed)
		viewer.launch(model=model, data=data)
	finally:
		# Unset callback to clean up
		mujoco.set_mjcb_control(None)
		print("Viewer closed. Exiting.")


if __name__ == "__main__":
	main()
