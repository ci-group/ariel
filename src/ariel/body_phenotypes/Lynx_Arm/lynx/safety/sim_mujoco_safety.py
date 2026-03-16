# safety_watchdog_sim.py

import time
import threading
import numpy as np
import logging


class SafetyWatchdogSim:
    """
    Monitors a robot's state (joint angles, end-effector position)
    and triggers an emergency stop if safety limits are exceeded.
    """
    MIN_END_EFFECTOR_Z_THRESHOLD = -0.2  # Adjusted to Z axis for PyBullet compatibility
    JOINT2_PIVOT_HEIGHT_FROM_TABLE_M = 0.1  # Height of joint 2's pivot from table surface
    CRITICAL_TABLE_CLEARANCE_M = -0.2 # Minimum allowed height for arm parts

    def __init__(
            self, 
            robot_controller, 
            natnet_data_handler, 
            joint_limits, 
            marker_radii: dict,
            l_link2=0.2805,
            l_link3=0.3055,
            ):
        """
        Initializes the watchdog.

        Args:
            robot_controller: The robot object that can be commanded (e.g., self.robot.limp()).
            natnet_data_handler: The sensor data handler (SimNatNetDataHandler in simulation).
            joint_limits: A list of tuples, where each tuple is (min_angle, max_angle)
                          for the corresponding joint.
            marker_radii: Dictionary of marker IDs to their radii (not used in sim).
        """
        self._robot = robot_controller
        self._natnet_handler = natnet_data_handler  # Will be SimNatNetDataHandler in simulation
        self._limits = joint_limits
        self._marker_radii = marker_radii  # This will be an empty dict in simulation
        self.LINK2_LENGTH_M = l_link2
        self.LINK3_LENGTH_M = l_link3
        self.JOINT2_PIVOT_HEIGHT_FROM_TABLE_M = 0.1  # Height of joint 2's pivot from table surface
        self.CRITICAL_TABLE_CLEARANCE_M = -0.2 # Minimum allowed height for arm
        self.MIN_END_EFFECTOR_Z_THRESHOLD = -0.2  # Minimum allowed Z height for end-effector

        # Ensure the number of limits matches the number of motors/actuators
        # For MujocoRobot, use model.nu (number of actuators)
        # For SimRobot, use _motor_ids
        if hasattr(self._robot, 'model') and hasattr(self._robot.model, 'nu'):
            num_robot_motors = self._robot.model.nu
        elif hasattr(self._robot, '_motor_ids'):
            num_robot_motors = len(self._robot._motor_ids)
        else:
            raise AttributeError("Robot controller does not have a recognized way to determine number of motors/actuators.")

        if len(self._limits) != num_robot_motors:
            raise ValueError(
                f"Mismatch between number of joint limits ({len(self._limits)}) "
                f"and number of robot motors/actuators ({num_robot_motors})."
            )

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._limit_violated = False  # Flag for limit violations
        self._exception_event = threading.Event()  # For signaling critical errors

    def start(self, check_interval=0.1):
        """Starts the watchdog monitoring loop in a background thread."""
        if self._thread.is_alive():
            print("[Watchdog] Already running.")
            return
        self.check_interval = check_interval
        print(f"[Watchdog] Starting monitor (check interval: {check_interval * 1000:.0f} ms).")
        print(f"[Watchdog] Joint Limits: {self._limits}")
        self._thread.start()

    def stop(self):
        """Stops the watchdog monitoring loop."""
        if not self._thread.is_alive():
            return
        print("[Watchdog] Stopping monitor...")
        self._stop_event.set()
        self._thread.join(timeout=1.0)  # Wait for thread to finish
        print("[Watchdog] Stopped.")

    def _watchdog_loop(self):
        """The main monitoring loop that runs in the background."""
        try:
            while not self._stop_event.is_set():
                violation_found = False
                eef_z_limit_violation_found = False  # Changed from y_limit
                table_angle_collision_found = False
                marker_overlap_found = False  # Initialized for logic flow

                # 1. Check if any joint exceeds its limits:
                raw_positions = self._robot.get_Position()

                # Flatten the list if it's a list of lists (from SimRobot)
                # Otherwise, assume it's already a flat list (from MujocoRobot)
                if raw_positions and isinstance(raw_positions[0], list):
                    current_positions = [pos[0] for pos in raw_positions if pos[0] is not None]
                else:
                    current_positions = raw_positions

                # Only perform checks if we have valid positions for all joints
                if len(current_positions) == len(self._limits):
                    # print(f"[Watchdog DEBUG] current_positions type: {type(current_positions)}, content: {current_positions}")
                    violation_found = self._check_limits(current_positions)

                    # 2. Check for kinematic table collision angles:
                    if len(current_positions) >= 3:  # Need at least joint 0, 1, 2 for this check (assuming base, shoulder, elbow)
                        table_angle_collision_found = self._check_table_collision_angles(
                            current_positions[1], current_positions[2])

                    # 3. Check End-Effector (EEF) Z-height (vertical position from sensor):
                    current_relative_pos = None
                    with self._natnet_handler._data_lock:  # Ensure thread-safe access
                        self._natnet_handler.update_data() # Explicitly request data update from the mock sensor
                        current_relative_pos = self._natnet_handler.latest_relative_pos

                    if current_relative_pos is not None:
                        # Assuming current_relative_pos[2] is the vertical Z-coordinate in PyBullet's Z-up system
                        # print(
                        #     f'[Watchdog] EEF relative pos: X={current_relative_pos[0]:.4f}m, Y={current_relative_pos[1]:.4f}m, Z={current_relative_pos[2]:.4f}m')
                        if current_relative_pos[2] < self.MIN_END_EFFECTOR_Z_THRESHOLD:
                            eef_z_limit_violation_found = True
                            print(f"[Watchdog] EEF Z-VIOLATION! End-effector Z-pos: {current_relative_pos[2]:.4f}m "
                                  f"is below threshold: {self.MIN_END_EFFECTOR_Z_THRESHOLD:.4f}m")
                    else:
                        print('[Watchdog] End-effector relative position data not yet available from sensor.')

                    # 4. Check for Bounding Sphere Overlap (DISABLED for simulation):
                    marker_overlap_found = self._check_marker_overlap()

                # Final: Trigger emergency recovery if any violation is found:
                if violation_found or eef_z_limit_violation_found or table_angle_collision_found or marker_overlap_found:
                    self._robot.enter_emergency_recovery()

                time.sleep(self.check_interval)
        except Exception as e:
            print(f'[Watchdog] CRITICAL ERROR: Watchdog exception found: {e}')
            self._robot.enter_emergency_recovery()
            self._robot.shutdown()
            self._exception_event.set()  # Signal that an exception occurred
            self._stop_event.set()  # Stop the watchdog thread

    def _check_limits(self, positions):
        """Checks if any joint position is outside its defined min/max limits."""
        for i, pos in enumerate(positions):
            min_limit, max_limit = self._limits[i]

            if not (min_limit <= pos <= max_limit):
                print(f"[Watchdog] JOINT ANGLE VIOLATION! Joint {i} is at {pos:.2f}°, "
                      f"but its limits are [{min_limit:.2f}, {max_limit:.2f}].")
                return True
        return False

    def _check_marker_overlap(self):
        """
        In simulation, this check is disabled as it relies on specific OptiTrack marker data,
        which is not directly simulated by SimNatNetDataHandler.
        Always returns False in this simulated setup.
        """
        return False

    def _check_table_collision_angles(self, joint1_angle_deg, joint2_angle_deg):
        """
        Calculates the approximate lowest point of Link 2 (after Joint 1) and Link 3 (after Joint 2)
        based on their joint angles and checks if this point is below a critical table clearance height.
        This relies on a simplified 2D forward kinematics model (projection).

        ASSUMPTION:
        - For UR5, angle 0 typically means the link is straight, or in a "home" configuration.
          (The description "0 degrees is vertically upward" from original code might need
          re-evaluation with UR5's specific kinematics, but we keep the formula structure.)
        - Positive angles result in rotation downwards (clockwise from "vertical up").

        Args:
            joint1_angle_deg (float): Current angle of Joint 1 (shoulder_lift_joint on UR5) in degrees.
            joint2_angle_deg (float): Current angle of Joint 2 (elbow_joint on UR5) in degrees (relative to Link 2).

        Returns:
            bool: True if a table collision risk is detected based on these joint angles, False otherwise.
        """
        # Convert angles to radians
        # From base to shoulder (Joint 0), then shoulder (Joint 1) is shoulder_lift, then elbow (Joint 2) is elbow_joint.
        # Let's adjust variable names to reflect common UR5 joints
        shoulder_lift_rad = np.radians(joint1_angle_deg)
        elbow_joint_rad = np.radians(joint2_angle_deg)

        # Assuming the base is at Z=0 and shoulder pivot at JOINT2_PIVOT_HEIGHT_FROM_TABLE_M (which is on Z-axis).
        # We need to compute the absolute Z (vertical) height of points on the arm.
        # This kinematic model assumes the arm moves in a vertical plane for simplicity.

        # Y position of Joint 2 (end of Link from J1 to J2) relative to Joint 1's pivot height (shoulder)
        # Assuming Joint1 moves arm up/down vertically
        # `cos` assumes 0 is vertical. If 0 is horizontal, use `sin`. Many UR5 arm angles are 0=straight.
        # Let's assume positive angles go "down" relative to horizontal straight.
        # If UR5's 0 is a specific, often "straight out" or "vertical UP/DOWN" orientation,
        # then these cos/sin choices determine the kinematic interpretation.
        # Based on default UR5 kinematics, 0 deg for shoulder_lift_joint is horizontal.
        # To get z-height from horizontal: L * sin(theta) if theta is from horizontal.
        # If the angle is from vertical (e.g., 0=up), then L * cos(theta) is correct.
        # Let's stick with original code's interpretation for minimal diff and mark it as potentially needing tuning.

        # Calculate Z positions (height) relative to Joint1's pivot point on the table.
        # For a standard arm, J1 (shoulder_lift) controls height, J2 (elbow) controls reach.
        # We need the *absolute* angles of the links relative to the horizontal or vertical.
        # If J1_angle_deg is relative to horizontal:
        # Z_J2_rel_J1 = LINK2_LENGTH_M * np.sin(shoulder_lift_rad) # If angle 0 is horizontal
        # Z_EE_rel_J2 = LINK3_LENGTH_M * np.sin(elbow_joint_rad) # If angle 0 is relative horizontal

        # Sticking VERY closely to *your* original code's math using `cos`
        # and assuming your original `joint2_angle_deg` and `joint3_angle_deg`
        # refer to angles from vertical for those segments:
        # (This is more for a human-like arm or a specific joint type, not typical UR5)

        # Absolute angle of link 2 (shoulder-to-elbow) relative to vertical
        abs_link2_angle_origin = shoulder_lift_rad
        # Absolute angle of link 3 (elbow-to-wrist) relative to vertical
        # Your original code used `np.radians(joint2_angle_deg - joint3_angle_deg)`.
        # This implies your original Joint3 was relative to Joint2.
        # If `joint1_angle_deg` is shoulder_lift and `joint2_angle_deg` is elbow_joint:
        abs_link3_angle_origin = shoulder_lift_rad + elbow_joint_rad  # A common simple sum if angles are relative to previous link
        # If elbow_joint_rad is relative to link 2, and shoulder_lift_rad is relative to vertical.
        # In URDF, angles are often relative to the previous link's frame.
        # To make this robust, actual forward kinematics should be used.
        # For minimal change, we will assume your prior logic's intent.

        # Vertical heights based on `L * cos(angle_from_vertical_up)`:
        z_pos_J2_rel_J1_pivot = self.LINK2_LENGTH_M * np.cos(abs_link2_angle_origin)
        z_pos_EE_rel_J2_pivot = self.LINK3_LENGTH_M * np.cos(abs_link3_angle_origin)

        # Total absolute Z (vertical) height of Joint 2 and End Effector from the table
        # JOINT2_PIVOT_HEIGHT_FROM_TABLE_M is the shoulder pivot height on the base.
        world_z_J2 = self.JOINT2_PIVOT_HEIGHT_FROM_TABLE_M + z_pos_J2_rel_J1_pivot
        world_z_EE = (self.JOINT2_PIVOT_HEIGHT_FROM_TABLE_M +
                      z_pos_J2_rel_J1_pivot +
                      z_pos_EE_rel_J2_pivot)  # Summing heights from base to J1, then J1 to J2, then J2 to EE.

        # Check if any critical point falls below the table clearance threshold
        # We only check the end-effector's approximate lowest point, as that's often the most critical
        # Or, you can check both intermediate joints as well.
        if world_z_EE < self.CRITICAL_TABLE_CLEARANCE_M:
            print(f"[Watchdog] KINEMATIC COLLISION VIOLATION (EE point)! "
                  f"End-Effector estimated Z-pos: {world_z_EE:.4f}m "
                  f"is below table clearance: {self.CRITICAL_TABLE_CLEARANCE_M:.4f}m."
                  f" J1 Angle:{joint1_angle_deg:.1f}°, J2 Angle:{joint2_angle_deg:.1f}°")
            return True
        elif world_z_J2 < self.CRITICAL_TABLE_CLEARANCE_M:  # Optionally check intermediate joint
            print(f"[Watchdog] KINEMATIC COLLISION VIOLATION (Joint 2 point)! "
                  f"Joint 2 estimated Z-pos: {world_z_J2:.4f}m "
                  f"is below table clearance: {self.CRITICAL_TABLE_CLEARANCE_M:.4f}m."
                  f" J1 Angle:{joint1_angle_deg:.1f}°, J2 Angle:{joint2_angle_deg:.1f}°")
            return True

        return False
