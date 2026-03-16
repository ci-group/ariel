# real_robot.py

import os
import time
import serial
import threading
import numpy as np
import atexit
import subprocess
from datetime import datetime, timedelta
import logging

from .motors.motor_cmd import LSS_DefaultBaud, LSS_BroadcastID
from .motors.real_motor import RealMotor

class RealRobot:
    """Real Lynx robot controller for physical hardware."""

    def __init__(self, portname="/dev/ttyACM0"):
        self._bus = None
        self._bus_lock = threading.Lock()  # the bus lock is the only one used for all motors of this robot
        self._motor_ids = [0, 1, 2, 3, 4, 5]
        self._portname = portname
        self._motors = []
        atexit.register(self.close_bus)

        # Emergency state attributes
        self._in_emergency_state = False
        self._emergency_thread = None
        self._in_limit_recovery_state = False # New attribute for limit recovery state
        self._limit_recovery_thread = None # New thread for limit recovery
        self._limit_recovery_cooldown_seconds = 40 # 5 minutes cool-down for temperature limits
        self._limit_recovery_reset_seconds = 20
        self._reset_seconds = 20
        self.home_position = [0, 0, 0, 0, 0, 0]  # Define a safe home position

        # Store the last read joint angles (in degrees)
        self._last_joint_angles_deg = [0.0] * len(self._motor_ids)
        self._current_joint_angles_deg = [0.0] * len(self._motor_ids)
        self._current_joint_velocities_deg = [0.0] * len(self._motor_ids)
        self.tolerance = 2.0

        self._last_action = np.zeros(len(self._motor_ids), dtype=np.float32)  # Last action taken
        self._second_last_action = np.zeros(len(self._motor_ids), dtype=np.float32)  # Second last action taken
        self._IMU_options = ["X", "Y", "Z", "A", "B", "G"]  # IMU axes options
        self._current_IMU_data = [{key: 0.0 for key in self._IMU_options} for _ in range(len(self._motor_ids))]
        self._current_IMU_data_raw = np.zeros((len(self._motor_ids), 6))  # Raw IMU data storage
        self._current_IMU_data_partial = np.zeros(8)  # j1(x,y), j4(x,y,z), j6(x,y,z)
        self._last_abs_pos = np.zeros(len(self._motor_ids), dtype=np.float32)
        self._second_last_abs_pos = np.zeros(len(self._motor_ids), dtype=np.float32)
        self._status = []

        self._current_joint_angles_timestamp_before = time.time()
        self._current_joint_angles_timestamp_after = time.time()

    def init_bus(self):
        logging.info(f"[Lynx] Attempting to open serial port: {self._portname}")
        try:
            if not os.path.exists(self._portname):
                logging.error(f"[Lynx] Error: Serial port '{self._portname}' does not exist. Is the device connected?")
                self._bus = None
                return False

            self._bus = serial.Serial(self._portname, LSS_DefaultBaud)
            self._bus.timeout = 0.1
            logging.info("[Lynx] Serial bus opened successfully.")
            return True

        except serial.SerialException as e:
            if "Permission denied" in str(e) and "[Errno 13]" in str(e):
                logging.warning(f"\n[Lynx] Warning: Permission denied for serial port '{self._portname}'.")
                logging.warning("Attempting to grant temporary permissions via sudo. You may be prompted for your password.")
                try:
                    subprocess.run(["sudo", "chmod", "666", self._portname], check=True)
                    logging.info(f"Permissions for {self._portname} temporarily updated. Retrying bus initialization...")
                    self._bus = serial.Serial(self._portname, LSS_DefaultBaud)
                    self._bus.timeout = 0.1
                    logging.info("[Lynx] Serial bus opened successfully after permission attempt.")
                    return True
                except subprocess.CalledProcessError as sub_e:
                    logging.error(f"\n[Lynx] Error: Failed to change permissions using sudo: {sub_e}")
                    logging.error("Please ensure your user is in the sudoers file and try again.")
                except serial.SerialException as retry_e:
                    logging.error(f"\n[Lynx] Error: Failed to open port even after attempting permission change: {retry_e}")
                except Exception as other_e:
                    logging.error(f"\n[Lynx] Error: An unexpected error occurred during permission attempt: {other_e}")

                logging.info("\n**RECOMMENDED SOLUTION (Permanent & Secure):**")
                logging.info(f"  1. Add your user to the 'dialout' group (or 'uucp'):")
                logging.info(f"     sudo usermod -a -G dialout {os.getlogin()}")
                logging.info(f"  2. **Log out and log back in** (or reboot).")
                logging.info(f"Original error: {e}\n")

            else:
                logging.error(f"\n[Lynx] Error: Could not open serial port '{self._portname}'.")
                logging.error(f"Reason: {e}\n")
            self._bus = None
            return False
        except Exception as e:
            logging.error(f"\n[Lynx] Error: An unexpected error occurred during bus initialization: {e}\n")
            self._bus = None
            return False

    def close_bus(self):
        if self._bus is not None:
            logging.info("[Lynx] Closing the bus...")
            try:
                self._bus.close()
                logging.info("[Lynx] Bus closed successfully.")
            except Exception as e:
                logging.error(f"[Lynx] Error closing bus: {e}")
            finally:
                self._bus = None

    def init_motors(self):
        if self._bus is None:
            return None
        
        self._motors = []
        for idx in self._motor_ids:
            m = RealMotor(id=idx, bus=self._bus, bus_lock=self._bus_lock)
            self._motors.append(m)

    def reset(self):
        with self._bus_lock:
            for i, m in enumerate(self._motors):
                logging.info(f"[Lynx] resetting motor {m._id} ...")
                m.reset()
            logging.info(f"[Lynx] reset the motors of the robot and wait {self._reset_seconds} seconds")
            time.sleep(self._reset_seconds) # Use _reset_seconds for consistency

            # --- Robot Re-initialization of bus and motors ---
            # Close the current bus if it's open, to ensure a clean re-initialization
            if self._bus:
                self.close_bus()

            connected = False
            # Define a list of potential serial ports to try
            potential_ports = [self._portname, '/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2', '/dev/ttyUSB0', '/dev/ttyACM3', '/dev/ttyUSB1']
            
            for port in potential_ports:
                logging.info(f"[Lynx] Attempting to re-connect to robot on port: {port}")
                # Temporarily set the portname for init_bus
                original_portname = self._portname
                self._portname = port
                try:
                    if self.init_bus():
                        self.init_motors()
                        connected = True
                        logging.info(f"[Lynx] Successfully re-connected to robot on port: {port}")
                        break
                    else:
                        logging.warning(f"[Lynx] Failed to re-connect to robot on port: {port}. Trying next port...")
                finally:
                    # Restore original portname regardless of success or failure
                    self._portname = original_portname

            if not connected:
                raise RuntimeError("Failed to re-initialize robot serial bus after trying all potential ports. Please check connections/connections/permissions.")

    def get_Position(self):
        """
        Queries each motor for its position (in centi-degrees), converts
        to degrees (float), and returns a list of [pos] or [None] if reading failed.
        Also updates the internal _current_joint_angles_deg attribute.
        """
        # Create a temporary list to store the new joint angles.
        # Initialize it with the current values to retain them if a read fails.
        temp_joint_angles_deg = self._current_joint_angles_deg[:]
        self._current_joint_angles_timestamp_before = time.time()
        
        for i, m in enumerate(self._motors):
            raw = m.getPosition() # Get raw reading (e.g., 1800, or None)

            if raw is None:
                # If read failed, temp_joint_angles_deg[i] retains its existing value.
                logging.warning(f"[Lynx] Warning: Motor {m._id} returned no data (None). Retaining previous valid angle ({temp_joint_angles_deg[i]:.2f}°).")
                continue # Skip conversion if raw is None

            try:
                # raw is an integer count of centi-degrees
                deg = float(raw) / 100.0
                temp_joint_angles_deg[i] = deg # Update with the new, valid degree
            except (TypeError, ValueError) as e:
                # If conversion failed, temp_joint_angles_deg[i] retains its existing value.
                logging.warning(f"[Lynx] Warning: Motor {m._id} returned invalid data '{raw}': {e}. Retaining previous valid angle ({temp_joint_angles_deg[i]:.2f}°).")
                # No action needed - value already retained

        # Update the internal state _current_joint_angles_deg with the new, processed data
        self._current_joint_angles_deg = temp_joint_angles_deg
        self._current_joint_angles_timestamp_after = time.time()

        # Return value for external functions (e.g., for debugging or other logic)
        # still signals which original reads failed by conforming to the expected [[pos], [None]] format.
        return [[p] if p is not None else [None] for p in self._current_joint_angles_deg]
    
    def get_Position_low_level(self):
        """
        low level version of get_position without querying each motor individually.
        """
        self._last_joint_angles_deg = self._current_joint_angles_deg.copy()  # Store the last joint angles before updating
        with self._bus_lock:
            self._current_joint_angles_timestamp_before = time.time()
            for i in range(len(self._motors)):
                # Get the position of each motor in centi-degrees
                raw_pos = self._motors[i].getPosition_Without_Lock()
                # TODO: record the current time stamp for the motor:
                # self._motor_position_timestamps[i] = time.time()
                if raw_pos is None:
                    logging.warning(f"[Lynx] Warning: Motor {self._motors[i]._id} returned no data (None) for position.")
                else:
                    try:
                        self._current_joint_angles_deg[i] = float(raw_pos) / 100.0  # Convert to degrees
                    except (TypeError, ValueError) as e:
                        logging.warning(f"[Lynx] Warning: Motor {self._motors[i]._id} returned invalid data '{raw_pos}': {e}. Retaining previous valid angle ({self._current_joint_angles_deg[i]:.2f}°).")
                ### Status Commands ###
            self._current_joint_angles_timestamp_after = time.time()

    def get_IMU(self):
        """
        get IMU sensor data
        """
        with self._bus_lock:
            for i in range(len(self._motors)):
                for key in self._IMU_options:
                    raw_imu = self._motors[i].get_IMU(key)
                    if raw_imu is None:
                        logging.warning(f"[Lynx] Warning: Motor {self._motors[i]._id} returned no data (None) for IMU {key}. Retaining previous valid IMU data ({self._current_IMU_data[i][key]:.2f}).")
                    else:
                        try:
                            self._current_IMU_data[i][key] = float(raw_imu) / 100.0
                        except (TypeError, ValueError) as e:
                            logging.warning(f"[Lynx] Warning: Motor {self._motors[i]._id} returned invalid data '{raw_imu}': {e}. Retaining previous valid IMU data ({self._current_IMU_data[i][key]:.2f}).")
        return self._current_IMU_data
    
    def get_IMU_all(self):
        with self._bus_lock:
            for i in range(len(self._motors)):
                raw_imu = self._motors[i].get_IMU_all()
                if raw_imu is None:
                    logging.warning(f"[Lynx] Warning: Motor {self._motors[i]._id} returned no data (None) for IMU. Retaining previous valid IMU data ({self._current_IMU_data[i]}).")
                else:
                    self._current_IMU_data_raw[i] = raw_imu
        return self._current_IMU_data_raw
    
    def get_IMU_partial(self):
        with self._bus_lock:
            self._current_IMU_data_partial[0:2] = self._motors[0].get_IMU_partial(['X', 'Y'])
            self._current_IMU_data_partial[2:5] = self._motors[3].get_IMU_partial(['X', 'Y', 'Z'])
            self._current_IMU_data_partial[5:8] = self._motors[5].get_IMU_partial(['X', 'Y', 'Z'])
        return self._current_IMU_data_partial

    def get_Position_and_Velocity(self):
        """
        Queries each motor for its position (in centi-degrees) and velocity (in centi-degrees per second),
        converts to degrees (float) and degrees per second (float), and returns a list of [pos, vel] or [None, None] if reading failed.
        Also updates the internal _current_joint_angles_deg attribute.
        """
        # Initialize a new list for _current_joint_angles_deg, starting with its current values.
        new_current_joint_angles_deg = self._current_joint_angles_deg[:]
        new_current_joint_velocities_deg = self._current_joint_velocities_deg[:]

        for i, m in enumerate(self._motors):
            raw_pos = m.getPosition()
            raw_vel = m.getVelocity()  # Get raw velocity reading (e.g., 180, or None)
            
            if raw_pos is None:
                logging.warning(f"[Lynx] Warning: Motor {m._id} returned no data (None) for position. Retaining previous valid angle ({new_current_joint_angles_deg[i]:.2f}°).")
            else:
                try:
                    deg = float(raw_pos) / 100.0
                    new_current_joint_angles_deg[i] = deg
                except (TypeError, ValueError) as e:
                    logging.warning(f"[Lynx] Warning: Motor {m._id} returned invalid data '{raw_pos}': {e}. Retaining previous valid angle ({new_current_joint_angles_deg[i]:.2f}°).")

            if raw_vel is None:
                logging.warning(f"[Lynx] Warning: Motor {m._id} returned no data (None) for velocity. Retaining previous valid velocity ({new_current_joint_velocities_deg[i]:.2f}°/s).")
            else:
                try:
                    vel = float(raw_vel) / 100.0
                    new_current_joint_velocities_deg[i] = vel
                except (TypeError, ValueError) as e:
                    logging.warning(f"[Lynx] Warning: Motor {m._id} returned invalid data '{raw_vel}': {e}. Retaining previous valid velocity ({new_current_joint_velocities_deg[i]:.2f}°/s).")

        self._current_joint_angles_deg = new_current_joint_angles_deg
        self._current_joint_velocities_deg = new_current_joint_velocities_deg
        
        # Return a list of [pos, vel] for each motor
        return [[self._current_joint_angles_deg[i], self._current_joint_velocities_deg[i]] for i in range(len(self._motors))]
    
    def move_abs(self, action): # 'action' here is a list of absolute target positions in degrees
        if self._in_emergency_state:
            logging.warning("[Lynx] In emergency state, ignoring move_abs() command.")
            return

        action = list(action)
        assert len(action) == len(self._motors)

        for idx, a in enumerate(action):
            # The low level protocol expects ints (centi-degrees)
            a_cd = int( a * 100) # Convert degrees to centi-degrees
            self._motors[idx].move_abs(a_cd)
        self._second_last_action = self._last_action.copy()  # Store the previous action before updating
        self._last_action = np.array(action, dtype=np.float32)
        self._second_last_abs_pos = self._last_abs_pos
        self._last_abs_pos = np.array(action, dtype=np.float32)

    def move_abs_admin(self, action): # 'action' here is a list of absolute target positions in degrees
        # Skip the emergency state check for admin commands
        action = list(action)
        assert len(action) == len(self._motors)

        for idx, a in enumerate(action):
            # The low level protocol expects ints (centi-degrees)
            a_cd = int( a * 100) # Convert degrees to centi-degrees
            self._motors[idx].move_abs(a_cd)
        self._second_last_action = self._last_action.copy()
        self._last_action = np.array(action, dtype=np.float32)
        self._second_last_abs_pos = self._last_abs_pos
        self._last_abs_pos = np.array(action, dtype=np.float32)

    def move_abs_with_speed(self, action, speed): # 'action' here is a list of absolute target positions in degrees
        if self._in_emergency_state:
            logging.warning("[Lynx] In emergency state, ignoring move_abs() command.")
            return

        action = list(action)
        assert len(action) == len(self._motors)

        for idx, a in enumerate(action):
            # The low level protocol expects ints (centi-degrees)
            a_cd = int( a * 100) # Convert degrees to centi-degrees
            self._motors[idx].move_abs_with_speed(a_cd, speed)
        self._second_last_action = self._last_action.copy()
        self._last_action = np.array(action, dtype=np.float32)  # Store the last action for reference
        self._second_last_abs_pos = self._last_abs_pos
        self._last_abs_pos = np.array(action, dtype=np.float32)

    def move_abs_with_speed_admin(self, action, speed): # 'action' here is a list of absolute target positions in degrees
        action = list(action)
        assert len(action) == len(self._motors)

        for idx, a in enumerate(action):
            # The low level protocol expects ints (centi-degrees)
            a_cd = int( a * 100) # Convert degrees to centi-degrees
            self._motors[idx].move_abs_with_speed(a_cd, speed)
        self._second_last_action = self._last_action.copy()
        self._last_action = np.array(action, dtype=np.float32)
        self._second_last_abs_pos = self._last_abs_pos
        self._last_abs_pos = np.array(action, dtype=np.float32)

    # def _move_and_wait(self, target_position):
    #     """An internal move command that is not blocked by the emergency flag."""
    #     self.move_abs(target_position)

    #     # Wait until the position is reached
    #     while not self._is_at_target(target_position):
    #         print(f"[Lynx] Waiting for position: {target_position} (current: {self._current_joint_angles_deg})")
    #         time.sleep(0.2)  # Check every 200ms
    def _move_and_wait(self, target_position, tolerance=0.5, max_retries=5, timeout_per_attempt_seconds=5.0):
        """
        An internal move command that is not blocked by the emergency flag, for admin use.
        Includes retry logic with timeout to handle motors getting stuck.
        """
        for attempt in range(max_retries):
            logging.info(f"[Lynx] Attempt {attempt + 1}/{max_retries}: Moving to {target_position}...")
            self.move_abs_with_speed(target_position, speed=2000)
            
            start_time = datetime.now()
            
            while not self._is_at_target_direct_query(target_position, tolerance=tolerance):
                if (datetime.now() - start_time).total_seconds() > timeout_per_attempt_seconds:
                    logging.warning(f"[Lynx] Attempt {attempt + 1} timed out after {timeout_per_attempt_seconds} seconds. Retrying...")
                    break # Break from inner while loop, go to next attempt
                
                logging.info(f"[Lynx] Waiting for position: {target_position} (current: {self._current_joint_angles_deg})")
                time.sleep(0.2) # Check every 200ms
            else: # This else belongs to the while loop, executes if loop completes without break
                logging.info(f"[Lynx] Successfully reached target position {target_position} on attempt {attempt + 1}.")
                return True # Movement successful
        
        logging.error(f"[Lynx] Failed to reach target position {target_position} after {max_retries} attempts.")
        return False # Movement failed after all retries

    def _move_and_wait_admin(self, target_position, tolerance=0.5, max_retries=5, timeout_per_attempt_seconds=5.0):
        """
        An internal move command that is not blocked by the emergency flag, for admin use.
        Includes retry logic with timeout to handle motors getting stuck.
        """
        for attempt in range(max_retries):
            logging.info(f"[Lynx] Attempt {attempt + 1}/{max_retries}: Moving to {target_position}...")
            self.move_abs_admin(target_position)
            
            start_time = datetime.now()
            
            while not self._is_at_target_direct_query(target_position, tolerance=tolerance):
                if (datetime.now() - start_time).total_seconds() > timeout_per_attempt_seconds:
                    logging.warning(f"[Lynx] Attempt {attempt + 1} timed out after {timeout_per_attempt_seconds} seconds. Retrying...")
                    break # Break from inner while loop, go to next attempt
                
                logging.info(f"[Lynx] Waiting for position: {target_position} (current: {self._current_joint_angles_deg})")
                time.sleep(0.2) # Check every 200ms
            else: # This else belongs to the while loop, executes if loop completes without break
                logging.info(f"[Lynx] Successfully reached target position {target_position} on attempt {attempt + 1}.")
                return True # Movement successful
        
        logging.error(f"[Lynx] Failed to reach target position {target_position} after {max_retries} attempts.")
        return False # Movement failed after all retries

    def move_rel(self, joint_deltas_degrees, speed=None):
        if self._in_emergency_state:
            logging.warning("[Lynx] In emergency state, ignoring move_rel() command.")
            return
        action = list(joint_deltas_degrees)
        assert len(action) == len(self._motors)

        for idx, a in enumerate(action):
            # The low level protocol expects ints (centi-degrees)
            a_cd = int(a * 100) # Convert degrees to centi-degrees
            self._motors[idx].moveRelative(a_cd)
        self._second_last_action = self._last_action.copy()
        self._last_action = np.array(action, dtype=np.float32)
        self._second_last_abs_pos = self._last_abs_pos
        self._last_abs_pos = np.array(action, dtype=np.float32)
    
    def move_rel_admin(self, joint_deltas_degrees, speed=None):
        # Skip the emergency state check for admin commands
        action = list(joint_deltas_degrees)
        assert len(action) == len(self._motors)

        for idx, a in enumerate(action):
            # The low level protocol expects ints (centi-degrees)
            a_cd = int(a * 100)
            # Convert degrees to centi-degrees
            self._motors[idx].moveRelative(a_cd)
        self._second_last_action = self._last_action.copy()
        self._last_action = np.array(action, dtype=np.float32)
        self._second_last_abs_pos = self._last_abs_pos
        self._last_abs_pos = np.array(action, dtype=np.float32)

    def move_rel_with_speed(self, joint_deltas_degrees, speed):
        if self._in_emergency_state:
            logging.warning("[Lynx] In emergency state, ignoring move_rel() command.")
            return

        action = list(joint_deltas_degrees)
        assert len(action) == len(self._motors)

        for idx, a in enumerate(action):
            # The low level protocol expects ints (centi-degrees)
            a_cd = int(a * 100)
            self._motors[idx].moveRelativeWithSpeed(a_cd, speed)
        self._second_last_action = self._last_action.copy()
        self._last_action = np.array(action, dtype=np.float32)
        self._second_last_abs_pos = self._last_abs_pos
        self._last_abs_pos = np.array(action, dtype=np.float32)

    def move_rel_with_speed_admin(self, joint_deltas_degrees, speed):
        # Skip the emergency state check for admin commands
        action = list(joint_deltas_degrees)
        assert len(action) == len(self._motors)

        for idx, a in enumerate(action):
            # The low level protocol expects ints (centi-degrees)
            a_cd = int(a * 100)
            self._motors[idx].moveRelativeWithSpeed(a_cd, speed)
        self._second_last_action = self._last_action.copy()
        self._last_action = np.array(action, dtype=np.float32)
        self._second_last_abs_pos = self._last_abs_pos
        self._last_abs_pos = np.array(action, dtype=np.float32)

    def move_rel_abs_with_speed(self, action, delta, speed): # 'action' here is a list of absolute target positions in degrees
        if self._in_emergency_state:
            logging.warning("[Lynx] In emergency state, ignoring move_abs() command.")
            return

        action = list(action)
        assert len(action) == len(self._motors)

        for idx, a in enumerate(action):
            # The low level protocol expects ints (centi-degrees)
            a_cd = int( a * 100) # Convert degrees to centi-degrees
            self._motors[idx].move_abs_with_speed(a_cd, speed)
        self._second_last_action = self._last_action.copy()
        self._last_action = np.array(delta, dtype=np.float32)  # Store the last action for reference
        self._second_last_abs_pos = self._last_abs_pos
        self._last_abs_pos = np.array(action, dtype=np.float32)

    def move_rel_abs_with_speed_admin(self, action, delta, speed): # 'action' here is a list of absolute target positions in degrees
        action = list(action)
        assert len(action) == len(self._motors)

        for idx, a in enumerate(action):
            # The low level protocol expects ints (centi-degrees)
            a_cd = int( a * 100) # Convert degrees to centi-degrees
            self._motors[idx].move_abs_with_speed(a_cd, speed)
        self._second_last_action = self._last_action.copy()
        self._last_action = np.array(delta, dtype=np.float32)
        self._second_last_abs_pos = self._last_abs_pos
        self._last_abs_pos = np.array(action, dtype=np.float32)

    def limp(self):
        for m in self._motors:
            logging.info(f'[Lynx] motor {m._id} limping')
            m.limp()

    def hold(self):
        for m in self._motors:
            logging.info(f'[Lynx] motor {m._id} holding')
            m.hold()

    def limp_broadcast(self):
        logging.info("[Lynx] Setting all motors to LIMP.")
        # This uses the broadcast ID #254, which is a good way
        # to override any specific motor commands. We'll create a broadcast motor object.
        broadcast_motor = RealMotor(id=LSS_BroadcastID, bus=self._bus, bus_lock=self._bus_lock)
        broadcast_motor.limp()

    def hold_broadcast(self):
        logging.info("[Lynx] Setting all motors to HOLD.")
        # This uses the broadcast ID #254, which is a good way
        # to override any specific motor commands. We'll create a broadcast motor object.
        broadcast_motor = RealMotor(id=LSS_BroadcastID, bus=self._bus, bus_lock=self._bus_lock)
        broadcast_motor.hold()

    def enter_emergency_recovery(self):
        """Triggers the uninterruptible emergency recovery sequence."""
        # Prevent starting multiple recovery threads
        if self._in_emergency_state or (hasattr(self, '_emergency_thread') and self._emergency_thread is not None and self._emergency_thread.is_alive()):
            return

        self._in_emergency_state = True # Ensure emergency state is active
        # Run the recovery in a separate thread so it doesn't block anything
        self._emergency_thread = threading.Thread(target=self._emergency_recovery_task, daemon=True)
        self._emergency_thread.start()

    def _emergency_recovery_task(self):
        """
        The actual recovery sequence: HOLD, wait, then go home, with retry logic.
        """
        try:
            while self._in_emergency_state:
                logging.info("\n" + "=" * 40)
                logging.info("[Emergency] Starting emergency recovery task.")

                # 1. HOLD all joints immediately
                logging.info("[Emergency] Attempting to HOLD all joints...")
                # Multiple calls to ensure broadcast and individual holds are sent
                self.hold_broadcast()
                self.hold_broadcast()
                self.hold_broadcast()
                self.hold_broadcast()
                self.hold_broadcast()
                self.hold()
                self.hold()
                self.hold()
                self.hold()
                self.hold()
                logging.info("[Emergency] All joints commanded to HOLD.")

                # 2. Wait for 3 seconds
                logging.info("[Emergency] Holding position for 3 seconds before attempting movement...")
                time.sleep(3)

                # Now, move slowly back to the predefined home position
                logging.info(f"[Emergency] Attempting to move to safe home position: {self.home_position}")
                
                # _move_and_wait_admin already has internal retries
                move_successful = self._move_and_wait_admin(self.home_position, tolerance=self.tolerance, max_retries=5, timeout_per_attempt_seconds=5.0)

                if move_successful:
                    logging.info("[Emergency] Home position reached successfully.")
                    
                    # Optional: Add a check to see if motors are responsive after reset
                    # For example, try to read positions and ensure they are not None
                    time.sleep(1) # Give motors a moment to respond after reset
                    current_positions = self.get_Position() # This will print warnings if reads fail
                    all_responsive = all(p[0] is not None for p in current_positions)

                    if all_responsive:
                        logging.info("[Emergency] Motors appear responsive. Recovery successful. Exiting emergency state.")
                        self._in_emergency_state = False # Cancel emergency mode
                        logging.info("[Emergency] Signaled training script to resume.")
                        logging.info("=" * 40 + "\n")
                        return # Exit recovery task
                    else:
                        logging.warning("[Emergency] Motors not fully responsive after movement. Retrying recovery...")
                        self.enter_reset_recovery()
                        # self._in_emergency_state = False # Handled by reset recovery
                else:
                    logging.warning("[Emergency] Failed to reach home position after all attempts. Retrying recovery...")
                    self.enter_reset_recovery()
                    # self._in_emergency_state = False # Handled by reset recovery
            return
            
        except serial.SerialException as e:
            logging.critical(f"[Emergency] CRITICAL SERIAL ERROR during emergency recovery: {e}. Attempting shutdown.")
            self._in_emergency_state = True # Remain in emergency state
            self.shutdown() # Attempt to gracefully shut down
        except Exception as e:
            logging.critical(f"[Emergency] UNEXPECTED ERROR during emergency recovery: {e}. Attempting shutdown.")
            self._in_emergency_state = True # Remain in emergency state
            self.shutdown() # Attempt to gracefully shut down

    def enter_reset_recovery(self):
        """Triggers the uninterruptible reset recovery sequence."""
        # Prevent starting multiple recovery threads
        if self._in_reset_state or (hasattr(self, '_reset_recovery_thread') and self._reset_recovery_thread.is_alive()):
            return
        self._in_reset_state = True
        self._in_emergency_state = True # Reset implies emergency
        self._reset_recovery_thread = threading.Thread(target=self._reset_recovery_task, daemon=True)
        self._reset_recovery_thread.start()

    def _reset_recovery_task(self):
        logging.info(f"[Reset Recovery] Signaled training script to pause.")
        while self._in_reset_state: # Loop while in reset state
            logging.info("\n" + "=" * 50)
            logging.info("\n" + "=" * 50)
            logging.info(f"! ! ! ROBOT ENTERING RESET RECOVERY ! ! !")

            # 1. HOLD all joints immediately
            logging.info("[Reset Recovery] Holding all joints...")
            self.hold_broadcast()
            self.hold_broadcast()
            self.hold_broadcast()
            self.hold()
            self.hold()
            self.hold()

            # 2. Wait for configurable cool-down period
            logging.info(f"[Reset Recovery] Cooling down for {self._limit_recovery_cooldown_seconds} seconds...")
            time.sleep(self._limit_recovery_cooldown_seconds)

            # 3. Reset motors before attempting to move
            logging.info("[Reset Recovery] Resetting motors...")
            self.reset()

            # 4. Move slowly back to the predefined home position
            logging.info(f"[Reset Recovery] Moving to safe home position: {self.home_position}")
            move_successful = self._move_and_wait_admin(self.home_position, tolerance=self.tolerance, max_retries=3, timeout_per_attempt_seconds=10.0)

            if move_successful:
                logging.info("[Reset Recovery] Home position reached. Resuming normal operation.")
                time.sleep(1)
                current_positions = self.get_Position()
                all_responsive = all(p[0] is not None for p in current_positions)

                if all_responsive:
                    logging.info("[Reset Recovery] Motors appear responsive. Recovery successful.")
                    self._in_reset_state = False # Cancel reset recovery mode
                    self._in_emergency_state = False # Also exit emergency state
                    logging.info("[Reset Recovery] Signaled training script to resume.")
                    logging.info("=" * 50 + "\n")
                    logging.info("=" * 50 + "\n")
                    return # Exit recovery task
                else:
                    logging.warning("[Reset Recovery] Motors not fully responsive after movement. Retrying recovery...")
            else:
                logging.warning("[Reset Recovery] Failed to reach home position. Retrying recovery...")

        logging.info("=" * 50 + "\n")
        self._in_emergency_state = True # Ensure emergency state remains True if recovery fails
        logging.info("[Reset Recovery] Signaled training script to resume (recovery failed).")


    def enter_limit_recovery(self):
        """Triggers the uninterruptible limit recovery sequence."""
        if self._in_emergency_state:
            logging.warning("[Lynx] Already in limit or emergency recovery state, ignoring enter_limit_recovery() command.")
            return

        self._in_emergency_state = True
        self._limit_recovery_thread = threading.Thread(target=self._limit_recovery_task, daemon=True)
        self._limit_recovery_thread.start()

    def _limit_recovery_task(self):
        """
        The actual limit recovery sequence: HOLD, cool-down, then go home, with retry logic.
        This is for temperature limits and should be longer.
        """
        logging.info("[Limit Recovery] Signaled training script to pause.")

        while self._in_emergency_state:
            logging.info("\n" + "=" * 40)
            logging.info(f"! ! ! ROBOT ENTERING LIMIT RECOVERY ! ! !")

            # 1. HOLD all joints immediately
            logging.info("[Limit Recovery] Holding all joints...")
            self.hold_broadcast()
            self.hold_broadcast()
            self.hold_broadcast()
            self.hold()
            self.hold()
            self.hold()

            # 2. Wait for configurable cool-down period
            logging.info(f"[Limit Recovery] Cooling down for {self._limit_recovery_cooldown_seconds} seconds...")
            time.sleep(self._limit_recovery_cooldown_seconds)

            # 3. Reset motors before attempting to move
            logging.info("[Limit Recovery] Resetting motors...")
            self.reset()

            # 4. Move slowly back to the predefined home position
            logging.info(f"[Limit Recovery] Moving to safe home position: {self.home_position}")
            move_successful = self._move_and_wait_admin(self.home_position, tolerance=self.tolerance, max_retries=3, timeout_per_attempt_seconds=10.0)

            if move_successful:
                logging.info("[Limit Recovery] Home position reached. Resuming normal operation.")
                time.sleep(1)
                current_positions = self.get_Position()
                all_responsive = all(p[0] is not None for p in current_positions)

                if all_responsive:
                    logging.info("[Limit Recovery] Motors appear responsive. Recovery successful.")
                    self._in_emergency_state = False # Cancel limit recovery mode
                    logging.info("[Limit Recovery] Signaled training script to resume.")
                    logging.info("=" * 40 + "\n")
                    return # Exit recovery task
                else:
                    logging.warning("[Limit Recovery] Motors not fully responsive after movement. Retrying recovery...")
            else:
                logging.warning("[Limit Recovery] Failed to reach home position. Retrying recovery...")
        
        logging.info("=" * 40 + "\n")
        self._in_emergency_state = True # Ensure limit recovery state remains True if recovery fails
        logging.info("[Limit Recovery] Signaled training script to resume (recovery failed).")

    def _is_at_target(self, target_position, tolerance=2.0):
        """Checks if the robot is at the target position within a tolerance."""
        # self.get_Position_low_level()
        current_pos_list = self._current_joint_angles_deg
        current_pos = current_pos_list

        if len(current_pos) != len(target_position):
            return False  # Can't confirm if a joint position is unknown

        for i in range(len(target_position)):
            if abs(current_pos[i] - target_position[i]) > tolerance:
                return False  # At least one joint is not at its target

        return True  # All joints are within the tolerance of their target
    
    def _is_at_target_direct_query(self, target_position, tolerance=2.0):
        """Checks if the robot is at the target position within a tolerance."""
        self.get_Position_low_level()
        current_pos_list = self._current_joint_angles_deg
        current_pos = current_pos_list

        if len(current_pos) != len(target_position):
            return False  # Can't confirm if a joint position is unknown

        for i in range(len(target_position)):
            if abs(current_pos[i] - target_position[i]) > tolerance:
                return False  # At least one joint is not at its target

        return True
    
    def _query_limit_status(self):
        """
        Queries each motor for its limit status and returns a list of status codes.
        Status codes:
        "0": No limits have been passed
        "1": Current limit has been passed
        "2": Input voltage detected is below or above acceptable range
        "3": Temperature limit has been reached
        """
        motor_statuses = []
        try:
            with self._bus_lock:
                for idx in range(len(self._motor_ids)):
                    status = self._motors[idx].getStatus()
                    motor_statuses.append(status)
                    if status == 0:
                        logging.info(f"Queried limit status for motor {idx}: No limits have been passed")
                    elif status == 1:
                        logging.warning(f"Queried limit status for motor {idx}: Current limit has been passed")
                    elif status == 2:
                        logging.warning(f"Queried limit status for motor {idx}: Input voltage detected is below or above acceptable range")
                    elif status == 3:
                        logging.error(f"Queried limit status for motor {idx}: Temperature limit has been reached")
                    else:
                        logging.warning(f"Queried limit status for motor {idx} is unrecognizable: {status}")
                        # Optionally, raise an error or handle unknown status more robustly
        except:
            logging.warning(f"Querying status failed: {motor_statuses}, skip this time.")
        return motor_statuses
        
    def shutdown(self):
        logging.info("[Lynx] Shutting down...")

        # Ensure emergency recovery is initiated and completes before closing the bus
        if not self._in_emergency_state: # Only initiate if not already in emergency
            self.enter_emergency_recovery()
        
        # Always wait for the emergency thread to finish if it exists and is alive
        if hasattr(self, '_emergency_thread') and self._emergency_thread:
            logging.info("[Lynx] Waiting for emergency recovery to complete before closing bus...")
            self._emergency_thread.join() # Wait for the thread to finish
            self._in_emergency_state = False # Ensure state is reset after recovery

        # Stop the spatial watchdog if it exists
        if hasattr(self, 'safety_watchdog') and self.safety_watchdog:
            self.safety_watchdog.stop()

        # Stop the joint watchdog if it exists
        if hasattr(self, 'joint_watchdog') and self.joint_watchdog:
            self.joint_watchdog.stop()

        self.close_bus()
        logging.info("[Lynx] Shutdown complete.")
