"""Real-time hardware runner for robogen-lite robots on the Robohat platform.

Bridges the evolved CPG controller from simulation to physical servos.

Usage
-----
    robot = spider(servo_map=[0, 1, 2, 3, 4, 5, 6, 7])
    cpg   = SimpleCPG(create_fully_connected_adjacency(8))
    cpg.set_flat_params(trained_params)

    runner = RobogenHardwareRunner(
        robot=robot,
        cpg=cpg,
        servo_assembly_1_config=SERVOASSEMBLY_1_CONFIG,
        servo_board_1_datas_list=SERVOBOARD_1_DATAS_LIST,
    )
    runner.run(duration=30.0)
"""

import time
from dataclasses import dataclass, field

import numpy as np

from ariel.hardware.angle_utils import SERVO_NEUTRAL_DEG, batch_mujoco_ctrl_to_degrees


@dataclass
class HardwareRunConfig:
    """Configuration for a hardware deployment run.

    Parameters
    ----------
    duration : float
        Total run duration in seconds.
    control_hz : float
        Control loop frequency in Hz. 50 Hz matches typical hobby servo update rates.
    n_servo_channels : int
        Total servo channels to send to Robohat (16 for one assembly board, 32 for two).
    topboard_switch : int
        Dip-switch value of the Robohat Topboard. Default is 7.
    """

    duration: float = 30.0
    control_hz: float = 50.0
    n_servo_channels: int = 16
    topboard_switch: int = 7


class RobogenHardwareRunner:
    """Runs a trained robogen-lite robot on physical Robohat hardware.

    The runner:
    1. Initialises the Robohat servo drivers.
    2. Moves all servos to the neutral position (90°).
    3. Enters a real-time loop at `control_hz` Hz:
       - Steps the CPG forward in time.
       - Converts joint angles from radians to servo degrees.
       - Routes each joint to the physical servo channel defined by `robot.servo_map`.
       - Sends the full angle array to `robohat.set_servo_multiple_angles`.
    4. On exit (normal or exception), returns all servos to neutral and shuts down cleanly.

    Parameters
    ----------
    robot : HardwareRobot | CoreModule
        A robot descriptor with ``servo_map`` and (for duck-typing) a length
        that equals the CPG oscillator count.  On the physical robot use
        :class:`ariel.hardware.robot.HardwareRobot`; on the workstation the
        full robogen-lite ``CoreModule`` (e.g. ``spider()``) also works.
    cpg : SimpleCPGInference | SimpleCPG
        Trained CPG controller.  On the robot load a `.npz` with
        :class:`ariel.hardware.cpg_inference.SimpleCPGInference`; on the
        workstation the torch ``SimpleCPG`` is also accepted.
    servo_assembly_1_config : ServoAssemblyConfig
        Configuration for the primary servo assembly board (P3 plug).
    servo_board_1_datas_list : list[ServoData]
        List of 16 ServoData objects for assembly board 1.
    servo_assembly_2_config : ServoAssemblyConfig | None
        Optional second assembly board (P4 plug). Pass None if not connected.
    servo_board_2_datas_list : list[ServoData] | None
        List of 16 ServoData objects for assembly board 2. Required if
        servo_assembly_2_config is provided.
    config : HardwareRunConfig | None
        Run configuration. Uses defaults if None.
    """

    def __init__(
        self,
        robot,
        cpg,
        servo_assembly_1_config,
        servo_board_1_datas_list: list,
        servo_assembly_2_config=None,
        servo_board_2_datas_list: list | None = None,
        config: HardwareRunConfig | None = None,
    ) -> None:
        if robot.servo_map is None:
            raise ValueError(
                "robot.servo_map is not set. "
                "Pass servo_map when calling spider() or gecko(), "
                "or set robot.servo_map = list(range(n_joints)) manually."
            )

        n_joints = len(robot.servo_map)
        n_cpg = cpg.n
        if n_joints != n_cpg:
            raise ValueError(
                f"servo_map has {n_joints} entries but CPG has {n_cpg} oscillators. "
                "These must match the number of MuJoCo actuators in the robot."
            )

        max_channel = max(robot.servo_map)
        cfg = config or HardwareRunConfig()
        if max_channel >= cfg.n_servo_channels:
            raise ValueError(
                f"servo_map references channel {max_channel} but n_servo_channels={cfg.n_servo_channels}. "
                "Increase n_servo_channels or fix the servo_map."
            )

        self.robot = robot
        self.cpg = cpg
        self.config = cfg
        self._asm1_config = servo_assembly_1_config
        self._board1_datas = servo_board_1_datas_list
        self._asm2_config = servo_assembly_2_config
        self._board2_datas = servo_board_2_datas_list

    def run(self) -> None:
        """Start the real-time hardware control loop.

        Blocks until the configured duration has elapsed, then returns with
        all servos set back to neutral (90°).

        Raises
        ------
        ImportError
            If the robohatlib package is not installed / not importable.
        KeyboardInterrupt
            Re-raised after a clean shutdown so the caller can handle it.
        """
        try:
            from robohatlib.Robohat import Robohat
        except ImportError as exc:
            raise ImportError(
                "robohatlib is not importable. "
                "Install it from the robohat/ directory: "
                "uv pip install -e robohat/"
            ) from exc

        servo_map: list[int] = self.robot.servo_map
        n_channels: int = self.config.n_servo_channels
        dt: float = 1.0 / self.config.control_hz
        duration: float = self.config.duration

        neutral = [SERVO_NEUTRAL_DEG] * n_channels

        robohat = Robohat(self._asm1_config, self._asm2_config, self.config.topboard_switch)
        robohat.init(self._board1_datas, self._board2_datas or [])
        robohat.start_servo_drivers()

        print(f"[hardware] Servos initialised. Moving to neutral for 1 s …")
        robohat.set_servo_multiple_angles(neutral)
        time.sleep(1.0)

        print(f"[hardware] Starting control loop: {duration:.1f} s @ {self.config.control_hz:.0f} Hz")

        elapsed: float = 0.0
        try:
            while elapsed < duration:
                t0 = time.monotonic()

                # Step CPG and get joint angles in radians.
                # Supports both SimpleCPGInference (returns ndarray) and
                # SimpleCPG (returns torch.Tensor — used when testing on workstation).
                result = self.cpg.forward(elapsed)
                angles_rad: np.ndarray = (
                    result.detach().numpy() if hasattr(result, "detach") else np.asarray(result)
                )

                # Build full servo angle array (all channels start at neutral)
                servo_angles = list(neutral)
                for actuator_idx, deg in enumerate(batch_mujoco_ctrl_to_degrees(angles_rad)):
                    servo_angles[servo_map[actuator_idx]] = deg

                robohat.set_servo_multiple_angles(servo_angles)

                # Busy-sleep for the remainder of the control period
                step_elapsed = time.monotonic() - t0
                remaining = dt - step_elapsed
                if remaining > 0:
                    time.sleep(remaining)

                elapsed += dt

        except KeyboardInterrupt:
            print("\n[hardware] Interrupted by user.")
            raise
        finally:
            print("[hardware] Returning to neutral and shutting down.")
            robohat.set_servo_multiple_angles(neutral)
            time.sleep(0.5)
            robohat.stop_servo_drivers()
            robohat.exit_program()
