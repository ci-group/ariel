# safety_watchdog.py

import time
import threading
from typing import Optional
import numpy as np
import logging


class SafetyWatchdogReal:
    """
    Safety watchdog for the real robot with OptiTrack.
    """
    MIN_END_EFFECTOR_Y_THRESHOLD = 0.05
    LINK2_LENGTH_M = 0.41  # Example: Length of link 2 (from Joint 2 pivot to Joint 3 pivot)
    LINK3_LENGTH_M = 0.45
    JOINT2_PIVOT_HEIGHT_FROM_TABLE_M = 0.1
    CRITICAL_TABLE_CLEARANCE_M = MIN_END_EFFECTOR_Y_THRESHOLD
    
    def __init__(
        self,
        robot_controller,
        natnet_data_handler,
        joint_limits,  # hard limits for the joints
        rigid_bodies: Optional[dict] = None,  # Rigid body info from OptiTrack (if available
        link_lengths: Optional[dict] = None,  # Link lengths for kinematic checks (if available)
    ):
        self._robot = robot_controller
        self._natnet_handler = natnet_data_handler
        self._limits = joint_limits
        self._rigid_bodies = rigid_bodies
        self._link_lengths = link_lengths
        self.check_interval = 0.1  # init with a default check interval of 100 ms

        # Ensure the number of limits matches the number of motors
        num_robot_motors = len(self._robot._motors) if self._robot._motors else len(self._robot._motor_ids)
        if len(self._limits) != num_robot_motors:
            raise ValueError(
                f"Mismatch between number of joint limits ({len(self._limits)}) "
                f"and number of robot motors ({num_robot_motors})."
            )

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._limit_violated = False
        self._loop_counter = 0 # Initialize loop counter
        self._last_violation_log_time = 0.0 # Initialize timestamp for logging violations
        self._exception_event = threading.Event()

        self._loop_counter = 0 # Initialize loop counter for SafetyWatchdogFast
        self.joint_pos_timestamp_before: float = 0.0 # Initialize timestamp for joint positions
        self.joint_pos_timestamp_after: float = 0.0 # Initialize timestamp for joint positions
        self._last_violation_log_time = 0.0 # Initialize timestamp for logging violations

        # robot_controller.reset()  # try to reset the motors due to the safe mode.
        motor_statuses = robot_controller._query_limit_status()
        for i, status in enumerate(motor_statuses):
            if status == "3": # "3": Temperature limit has been reached
                logging.error(f"[SafetyWatchdogFast] MOTOR {i} TEMPERATURE LIMIT VIOLATION! Triggering emergency recovery.")
                robot_controller.enter_limit_recovery()
                break # No need to check other motors if one is overheating

    def start(self, check_interval=0.1):
        """Starts the watchdog monitoring loop in a background thread."""
        if self._thread.is_alive():
            logging.debug("[Watchdog] Already running.")
            return
        self.check_interval = check_interval
        logging.debug(f"[Watchdog] Starting monitor (check interval: {check_interval * 1000:.0f} ms).")
        logging.debug(f"[Watchdog] Limits: {self._limits}")
        self._thread.start()

    def stop(self):
        """Stops the watchdog monitoring loop."""
        if not self._thread.is_alive():
            return
        logging.info("[Watchdog] Stopping monitor...")
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        logging.info("[Watchdog] Stopped.")

    def _check_limits(self, positions):
        for i, pos in enumerate(positions):
            min_limit, max_limit = self._limits[i]

            if not (min_limit <= pos <= max_limit):
                if (time.time() - self._last_violation_log_time) >= 1.0:
                    logging.warning(f"[Watchdog] JOINT LIMIT VIOLATION! Joint {i} is at {pos:.2f}°, "
                                f"but its limits are [{min_limit:.2f}, {max_limit:.2f}].")
                    self._last_violation_log_time = time.time()
                return True
        return False
    
    def _check_rb_overlap(self):
        try:
            from natnet.protocol.MocapFrameMessage import LabelledMarker, RigidBody
        except ImportError:
            logging.warning("[Watchdog] natnet not available, skipping rigid body overlap check.")
            return False
        
        marker_overlap_detected = False
        active_markers_with_pos = []

        # Define rigid body IDs (these were in optitrack.py)
        END_EFFECTOR_RB_ID = 12
        TABLE_RB_ID = 11

        with self._natnet_handler._data_lock:
            for marker in self._natnet_handler.latest_markers:
                # NEW: Filter out markers that belong to the End-Effector or Table Rigid Bodies
                # We only want to check for collisions with markers not part of these predefined RBs
                if marker.model_id == END_EFFECTOR_RB_ID or marker.model_id == TABLE_RB_ID:
                    continue # Skip this marker, it belongs to one of the excluded Rigid Bodies

                if not (np.array(marker.position) == 0).all() and marker.model_id == 0: # Simple check for (0,0,0) ghost markers
                    active_markers_with_pos.append(marker)

        # If we have at least two markers to check, proceed with pair-wise comparison
        if len(active_markers_with_pos) >= 2:
            num_markers = len(active_markers_with_pos)
            for i in range(num_markers):
                for k in range(i + 1, num_markers): # Compare each marker with subsequent ones to avoid duplicates
                    marker1 = active_markers_with_pos[i]
                    marker2 = active_markers_with_pos[k]

                    pos1 = np.array(marker1.position) # Marker positions are (x,y,z)
                    pos2 = np.array(marker2.position)
                    
                    # r1 = self._marker_radii[marker1.marker_id]
                    # r2 = self._marker_radii[marker2.marker_id]
                    r1 = 0.025
                    r2 = 0.025

                    distance = np.linalg.norm(pos2 - pos1) # Euclidean distance
                    sum_radii = r1 + r2

                    if distance < sum_radii:
                        marker_overlap_detected = True
                        logging.error(f"[Watchdog] MARKER SPHERE OVERLAP VIOLATION! "
                                        f"Markers {marker1.marker_id} (R={r1*100:.1f}cm) and {marker2.marker_id} (R={r2*100:.1f}cm) are overlapping. "
                                        f"Distance: {distance*100:.1f}cm, Sum of Radii: {sum_radii*100:.1f}cm.")
                        return True # Found an overlap, no need to check further

        return False # No overlap detected
    
    def _check_table_collision_angles(self, joint2_angle_deg, joint3_angle_deg):

        joint2_angle_rad = np.radians(joint2_angle_deg)

        abs_link3_angle_rad = np.radians(joint2_angle_deg - joint3_angle_deg)  # These two angles are reverse

        y_pos_J3_rel_J2 = self._link_lengths["LINK2_LENGTH_M"] * np.cos(joint2_angle_rad)

        y_pos_EE_rel_J2 = (self._link_lengths["LINK2_LENGTH_M"] * np.cos(joint2_angle_rad) +
                           self._link_lengths["LINK3_LENGTH_M"] * np.cos(abs_link3_angle_rad))

        world_y_EE = self.JOINT2_PIVOT_HEIGHT_FROM_TABLE_M + y_pos_EE_rel_J2

        if world_y_EE < self.CRITICAL_TABLE_CLEARANCE_M:
            if (time.time() - self._last_violation_log_time) >= 1.0:
                logging.warning(f"[Watchdog] KINEMATIC TABLE COLLISION VIOLATION! "
                        # f"Point J2 (end of Link 1) Z-pos: {world_y_J3:.4f}m or"
                        f"EE Z-pos: {world_y_EE:.4f}m "
                        f"is below table clearance: {self.CRITICAL_TABLE_CLEARANCE_M:.4f}m."
                        f" J1 Angle:{joint2_angle_deg:.1f}°, J2 Angle:{joint3_angle_deg:.1f}°")
                self._last_violation_log_time = time.time()
            return True
        
        return False
    
    def _watchdog_loop(self):
        """
        The main monitoring loop that runs in the background, with OptiTrack checks disabled.
        """
        try:
            while not self._stop_event.is_set():
                self._loop_counter += 1 # Increment loop counter

                violation_found = False
                z_limit_violation_found = False
                table_angle_collision_found = False
                marker_overlap_found = False  # TODO: the markers' IDs are not yet clarified, and the distances are usually violated

                current_positions = self._robot._current_joint_angles_deg
                logging.debug(f"[Watchdog] Current positions: {current_positions}")

                # self._robot.get_IMU()

                # Ensure we have positions for all joints before checking
                if len(current_positions) != len(self._limits):
                    logging.error("[Watchdog] CRITICAL ERROR: Number of tracked joint positions doesn't match defined limits. Forcing emergency recovery.")
                    self._robot.enter_emergency_recovery() # Force emergency if state is truly inconsistent
                    logging.info("[Watchdog] Emergency recovery initiated due to position mismatch.")
                    time.sleep(self.check_interval * 5) # Longer pause for critical error
                    continue
                violation_found = self._check_limits(current_positions)
                logging.debug(f"[Watchdog] Joint limit violation found: {violation_found}")
                
                # 3. Check for KINEMATIC table collision angles:
                if len(current_positions) > 3: # Need at least 4 joints (0, 1, 2, 3) to access joint 2 and 3
                    table_angle_collision_found = self._check_table_collision_angles(
                        current_positions[1], current_positions[2])
                logging.debug(f"[Watchdog] Table angle collision found: {table_angle_collision_found}")
                
                # --------------------------------------
                # 4. OptiTrack limits check is included:
                # --------------------------------------
                current_relative_pos = self._natnet_handler.latest_relative_pos # Correctly get the value
                if current_relative_pos is not None:
                    logging.debug(f"z-pos: {current_relative_pos}")
                    logging.debug(f"[Watchdog] EEF Z-pos: {current_relative_pos[2]:.4f}m, Threshold: {self.MIN_END_EFFECTOR_Y_THRESHOLD:.4f}m")
                    if current_relative_pos[2] < self.MIN_END_EFFECTOR_Y_THRESHOLD:
                        if (time.time() - self._last_violation_log_time) >= 1.0:
                            logging.warning(f"[Watchdog] Z-LIMIT VIOLATION !!.")
                            self._last_violation_log_time = time.time()
                        z_limit_violation_found = True
                        # logging.error(f"[Watchdog] EEF Z-limit violation found: {z_limit_violation_found}")
                else:
                    logging.debug('[Watchdog] End-effector relative position data not yet available from OptiTrack.')

                if violation_found or z_limit_violation_found or table_angle_collision_found or marker_overlap_found:
                    # if (time.time() - self._last_violation_log_time) >= 1.0:
                    #     logging.info("[Watchdog] Triggering emergency recovery due to safety violation.")
                    #     self._last_violation_log_time = time.time()
                    self._robot.enter_emergency_recovery()

                time.sleep(self.check_interval)  # shoould not be too fast, otherwise the robot will stuck in failing to send commands
        except Exception as e:
            logging.error(f'[Watchdog] CRITICAL ERROR: watchdog exception found: {e}')
            logging.info("[Watchdog] Initiating emergency recovery and shutdown due to watchdog exception.")
            self._robot.enter_emergency_recovery()
            self._robot.shutdown()
            self._exception_event.set()
            self._stop_event.set()

