"""Deploy an evolved spider robot on the Robohat hardware platform.

This script loads a trained CPG from a file (saved during simulation) and
runs it on the physical spider robot via the Robohat servo interface.

Run on the Raspberry Pi (with robohatlib installed):
    uv run examples/hardware/run_spider.py --cpg path/to/cpg.pt --duration 30

Servo mapping
-------------
The spider has 8 servos wired to assembly board 1 (channels 0-7):
    ch 0: front leg — yaw hinge   (l1_1)
    ch 1: front leg — pitch hinge (l1_2)
    ch 2: left  leg — yaw hinge   (l2_1)
    ch 3: left  leg — pitch hinge (l2_2)
    ch 4: right leg — yaw hinge   (l3_1)
    ch 5: right leg — pitch hinge (l3_2)
    ch 6: back  leg — yaw hinge   (l4_1)
    ch 7: back  leg — pitch hinge (l4_2)

If your physical wiring differs, pass a custom --servo-map, e.g.:
    --servo-map 0 2 1 3 4 6 5 7

Verifying actuator order
------------------------
Run this on your workstation to print the MuJoCo actuator names in order:

    import mujoco
    from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider import spider
    from ariel.simulation.environments import SimpleFlatWorld

    robot = spider()
    world = SimpleFlatWorld()
    world.spawn(robot)
    model = world.spec.compile()
    for i in range(model.nu):
        print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))

The printed order must match your servo_map.
"""

import argparse
import sys
from pathlib import Path

# ── Robohat config (mirrors testlib/TestConfig.py) ────────────────────────────
try:
    from robohatlib.hal.assemblyboard.PwmPlug import PwmPlug
    from robohatlib.hal.assemblyboard.ServoAssemblyConfig import ServoAssemblyConfig
    from robohatlib.hal.assemblyboard.servo.ServoData import ServoData
except ImportError:
    print(
        "ERROR: robohatlib is not importable.\n"
        "Install it from the repo root: uv pip install -e robohat/",
        file=sys.stderr,
    )
    sys.exit(1)

# ── ARIEL imports ──────────────────────────────────────────────────────────────
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider import spider
from ariel.hardware import HardwareRunConfig, RobogenHardwareRunner
from ariel.simulation.controllers.simple_cpg import (
    SimpleCPG,
    create_fully_connected_adjacency,
)

# ── Hardware constants ─────────────────────────────────────────────────────────
_FORMULA_A = 68.50117096018737
_FORMULA_B = -15.294412847106067

SERVOASSEMBLY_1_CONFIG = ServoAssemblyConfig(
    "servoassembly_1",
    0,                   # sw1 PWM address (dip-switch on assembly board)
    0,                   # sw2 power-good address
    PwmPlug.PWMPLUG_P3,  # connected topboard plug
)

SERVOBOARD_1_DATAS_LIST = [
    ServoData(i, 500, 2500, 0, 180, 0, _FORMULA_A, _FORMULA_B)
    for i in range(16)
]

TOPBOARD_SWITCH = 7

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run evolved spider on Robohat hardware")
parser.add_argument(
    "--cpg",
    type=Path,
    required=True,
    help="Path to a .pt file saved by SimpleCPG.save()",
)
parser.add_argument("--duration", type=float, default=30.0, help="Run duration in seconds")
parser.add_argument("--hz", type=float, default=50.0, help="Control loop frequency (Hz)")
parser.add_argument(
    "--servo-map",
    type=int,
    nargs=8,
    default=list(range(8)),
    metavar="CH",
    help=(
        "8 Robohat channel numbers, one per MuJoCo actuator in order "
        "(default: 0 1 2 3 4 5 6 7)"
    ),
)
args = parser.parse_args()


def main() -> None:
    # 1. Build robot with servo mapping from CLI (or default identity)
    robot = spider(servo_map=args.servo_map)

    # 2. Build CPG with the same number of oscillators as joints
    n_joints = len(robot.servo_map)
    adj = create_fully_connected_adjacency(n_joints)
    cpg = SimpleCPG(adj)

    # 3. Load trained parameters saved from simulation
    cpg.load(args.cpg)

    print(f"Loaded CPG from {args.cpg}")
    print(f"Spider servo map: {robot.servo_map}")
    print(f"Run: {args.duration:.0f} s @ {args.hz:.0f} Hz")

    # 4. Run on hardware
    runner = RobogenHardwareRunner(
        robot=robot,
        cpg=cpg,
        servo_assembly_1_config=SERVOASSEMBLY_1_CONFIG,
        servo_board_1_datas_list=SERVOBOARD_1_DATAS_LIST,
        config=HardwareRunConfig(
            duration=args.duration,
            control_hz=args.hz,
            n_servo_channels=16,
            topboard_switch=TOPBOARD_SWITCH,
        ),
    )

    try:
        runner.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
