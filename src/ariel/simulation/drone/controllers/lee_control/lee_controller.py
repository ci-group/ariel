# -*- coding: utf-8 -*-
"""
Lee Geometric Controller for Arbitrary Drone Configurations

This controller implements the Lee geometric control algorithm and provides
the same interface as GeneralizedControl for seamless integration with the
existing simulation framework.

Based on "Geometric tracking control of a quadrotor UAV on SE(3)" by Lee, Leok, and McClamroch.

author: Implementation compatible with John Bass framework
license: MIT
"""

import numpy as np
from numpy import pi, sin, cos, tan, sqrt
from numpy.linalg import norm, pinv
import ariel.simulation.drone.controllers.utils as utils

from .position_control import LeePositionController
from .attitude_control import LeeAttitudeController
from .velocity_control import LeeVelocityController
from .lee_math_utils import *

rad2deg = 180.0/pi
deg2rad = pi/180.0

class LeeGeometricControl:
    """
    Lee Geometric Controller for Arbitrary Drone Configurations.
    
    This controller implements the Lee geometric control algorithm while maintaining
    the same interface as GeneralizedControl for seamless integration.
    """
    
    def __init__(self, quad, yawType, orient="NED",
                 # Lee control gains - Tuned for direct SO(3) control (not cascaded like PX4).
                 # Defaults are calibrated for a heavier ~1 kg drone; pass auto_scale_gains=True
                 # (or override att/rate gains) for small drones where the legacy defaults
                 # demand torques exceeding motor allocation capacity.
                 pos_P_gain=None,                           # Position gains [kx, ky, kz]
                 vel_P_gain=None,                           # Velocity gains [kvx, kvy, kvz]
                 att_P_gain=None,                           # Attitude gains [kR_roll, kR_pitch, kR_yaw]
                 rate_P_gain=None,                          # Angular rate gains
                 auto_scale_gains=False,                    # Scale att/rate gains by inertia
                 # Interface compatibility (unused parameters for compatibility with PID controllers)
                 **kwargs):                                 # Catches vel_D_gain, vel_I_gain, rate_D_gain, tilt_max, rate_max, vel_max, etc.
        """
        Initialize Lee geometric controller.

        The Lee controller uses a fundamentally different approach than PID cascade controllers.
        It directly computes forces and torques from geometric tracking errors.

        Args:
            quad: ConfigurableQuadcopter instance
            yawType: Yaw control type (0=disabled, 1=enabled)
            orient: Coordinate frame ("NED" or "ENU")

            # Lee Control Gains (the core parameters; pass None for built-in defaults)
            pos_P_gain: Position control gains [kx, ky, kz]
            vel_P_gain: Velocity control gains [kvx, kvy, kvz]
            att_P_gain: Attitude control gains [kR_roll, kR_pitch, kR_yaw]
            rate_P_gain: Angular rate gains [komega_roll, komega_pitch, komega_yaw]
            auto_scale_gains: If True and att/rate gains are not user-provided,
                derive them from the drone's inertia (K_rot = I·ω_n², K_angvel = 2·I·ω_n
                for a critically damped second-order attitude response). Required for
                small drones (e.g. 2-inch quads, Izz ≈ 1e-4) where the heavy-drone
                defaults saturate the motor allocation immediately.

            **kwargs: Unused PID parameters (vel_D_gain, vel_I_gain, rate_D_gain,
                     tilt_max, rate_max, vel_max, aggressiveness, etc.) kept for interface compatibility
        """
        # Store drone reference and configuration
        self.quad = quad
        self.drone_config = quad.drone_sim.config
        self.num_motors = self.drone_config.num_motors
        self.orient = orient

        # Built-in defaults (calibrated for ~1 kg drone with heavy inertia).
        if pos_P_gain is None:
            pos_P_gain = np.array([2.0, 2.0, 3.0])
        if vel_P_gain is None:
            vel_P_gain = np.array([3.0, 3.0, 4.0])

        # Auto-scale att/rate gains from inertia. The Lee controller emits torques
        # directly (N·m), so K_rot must scale with inertia to give a consistent
        # closed-loop bandwidth. Underlying second-order shape:
        #   I·θ̈ + K_angvel·θ̇ + K_rot·θ = 0  →  K_rot = I·ω_n², K_angvel = 2·I·ω_n.
        if auto_scale_gains and att_P_gain is None and rate_P_gain is None:
            I_diag = np.diag(np.asarray(quad.params["IB"]))
            omega_n_att = 12.0  # rad/s; closed-loop attitude natural frequency
            att_P_gain = I_diag * omega_n_att ** 2
            rate_P_gain = 2.0 * I_diag * omega_n_att

        if att_P_gain is None:
            att_P_gain = np.array([0.3, 0.3, 0.1])
        if rate_P_gain is None:
            rate_P_gain = np.array([0.05, 0.05, 0.03])

        # Store control gains (Lee control uses different structure than PID)
        self.pos_P_gain = np.asarray(pos_P_gain, dtype=float).copy()
        self.vel_P_gain = np.asarray(vel_P_gain, dtype=float).copy()
        self.att_P_gain = np.asarray(att_P_gain, dtype=float).copy()
        self.rate_P_gain = np.asarray(rate_P_gain, dtype=float).copy()

        # Extract unused parameters from kwargs for interface compatibility (not enforced)
        self.tilt_max = kwargs.get('tilt_max', 50.0*deg2rad)  # Only used for reporting
        self.rate_max = kwargs.get('rate_max', np.array([200.0*deg2rad, 200.0*deg2rad, 150.0*deg2rad]))  # Only used for reporting
        self.vel_max = kwargs.get('vel_max', np.array([5.0, 5.0, 5.0]))  # Only used for reporting
        self.vel_max_all = kwargs.get('vel_max_all', 5.0)  # Only used for reporting
        self.aggressiveness = kwargs.get('aggressiveness', 1.0)  # Only used for reporting
        
        # Initialize Lee controller components
        self._init_lee_controllers(quad)
        
        # Initialize state variables
        
        # Initialize motor commands
        self.w_cmd = np.ones(self.num_motors) * self._get_hover_speed(quad)
        
        # Setup yaw control
        if (yawType == 0):
            self.att_P_gain[2] = 0
        
        
        # Variables for interface compatibility and logging
        self.sDesCalc = np.zeros(16)        # Calculated desired state (for logging)
        
        # Lee control specific variables
        self.desired_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.wrench_command = np.zeros(6)  # [fx, fy, fz, tx, ty, tz]
        
    def _init_lee_controllers(self, quad):
        """Initialize Lee controller components"""
        # Create a simple config object for the Lee controllers
        class LeeConfig:
            def __init__(self, pos_gains, vel_gains, att_gains, rate_gains):
                self.K_pos_tensor_max = pos_gains
                self.K_pos_tensor_min = pos_gains
                self.K_vel_tensor_max = vel_gains
                self.K_vel_tensor_min = vel_gains
                self.K_rot_tensor_max = att_gains
                self.K_rot_tensor_min = att_gains 
                self.K_angvel_tensor_max = rate_gains 
                self.K_angvel_tensor_min = rate_gains
                self.randomize_params = False
                self.max_yaw_rate = 2.0
        
        config = LeeConfig(self.pos_P_gain, self.vel_P_gain, self.att_P_gain, self.rate_P_gain)
        
        # Get drone properties
        mass = quad.params["mB"]
        inertia = quad.params["IB"]  # Already a 3x3 matrix
        gravity = np.array([0.0, 0.0, quad.params["g"]])  # NED frame
        if self.orient == "ENU":
            gravity = np.array([0.0, 0.0, -quad.params["g"]])  # ENU frame
        
        # Initialize position controller 
        self.position_controller = LeePositionController(config, mass, inertia, gravity, self.orient)
        
        # Initialize velocity controller
        self.velocity_controller = LeeVelocityController(config, mass, inertia, gravity, self.orient)
        
        # Initialize attitude controller
        self.attitude_controller = LeeAttitudeController(config, mass, inertia, gravity, self.orient)
        
    def _get_hover_speed(self, quad):
        """Calculate hover speed for each motor."""
        hover_speeds = []
        hover_thrust_per_motor = (quad.params["mB"] * quad.params["g"]) / self.num_motors
        
        for prop in self.drone_config.propellers:
            k_f, k_m = prop["constants"]
            w_hover = sqrt(hover_thrust_per_motor / k_f)
            hover_speeds.append(w_hover)
        
        return np.array(hover_speeds)
    
    def controller(self, sDes, quad, ctrl_type, Ts):
        """
        Main Lee controller function.
        
        Args:
            sDes: Desired state vector [pos(3), vel(3), acc(3), thrust(3), eul(3), pqr(3), yawRate]
            quad: ConfigurableQuadcopter instance  
            ctrl_type: Control type ("xyz_pos", "xyz_vel", etc.)
            Ts: Time step
        """
        # Extract desired state
        pos_sp = sDes[0:3].copy()
        vel_sp = sDes[3:6].copy()
        thrust_sp = sDes[9:12].copy()
        eul_sp = sDes[12:15].copy()
        yawFF = sDes[18] if len(sDes) > 18 else 0.0
        
        # Update Lee controller state from quadcopter
        self._update_lee_controller_state(quad)
        
        # Apply control based on control type
        if (ctrl_type == "xyz_vel"):
            self._lee_velocity_control(vel_sp, quad, Ts, yawFF)
        elif (ctrl_type == "xyz_pos"):
            self._lee_position_control(pos_sp, eul_sp, quad, Ts)
        
        # Convert Lee control output to motor commands
        self._wrench_to_motor_commands(quad)
        
        # Update calculated desired states for logging (interface compatibility)
        self.sDesCalc[0:3] = pos_sp
        self.sDesCalc[3:6] = vel_sp
        self.sDesCalc[6:9] = thrust_sp
        self.sDesCalc[9:13] = self.desired_orientation
        self.sDesCalc[13:16] = np.zeros(3)  # Rate setpoint (not used in Lee control)
    
    def _update_lee_controller_state(self, quad):
        """Update Lee controller state from quadcopter state"""
        # CRITICAL FIX: Convert quaternion from simulator's [w, x, y, z] format to Lee math utils' [x, y, z, w] format
        quat_wxyz = quad.quat.copy()  # Simulator format: [w, x, y, z]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # Lee format: [x, y, z, w]

        robot_state = {
            "position": quad.pos.copy(),
            "orientation": quat_xyzw,  # Now in correct [x, y, z, w] format for Lee math utils
            "linear_velocity": quad.vel.copy(),
            "angular_velocity": quad.omega.copy(),
            "euler_angles": quad.euler.copy()
        }
        
        # Update all controller states
        self.position_controller.update_state(robot_state)
        self.velocity_controller.update_state(robot_state)
        self.attitude_controller.update_state(robot_state)
    
    def _lee_velocity_control(self, vel_sp, quad, Ts, yaw_rate=0.0):
        """Lee geometric velocity control"""
        
        
        # Update velocity controller state
        self._update_lee_controller_state(quad)
        
        # Create velocity command: [vx, vy, vz, yaw_rate]
        command_actions = np.concatenate([vel_sp, [yaw_rate]])
        
        # Use velocity controller
        self.wrench_command = self.velocity_controller.update(command_actions)
        
        # Store desired orientation for logging
        self.desired_orientation = self.velocity_controller.desired_quat
        
    def _lee_position_control(self, pos_sp, eul_sp, quad, Ts):
        """Lee geometric position control"""
        # Update position controller state
        self._update_lee_controller_state(quad)
        
        # Create position command: [px, py, pz, yaw]
        command_actions = np.concatenate([pos_sp, [eul_sp[2]]])
        
        # Use position controller
        self.wrench_command = self.position_controller.update(command_actions)
        
        # Store desired orientation for logging
        self.desired_orientation = self.position_controller.desired_quat
    
    def _wrench_to_motor_commands(self, quad):
        """Convert Lee control wrench output to motor commands"""
        # Extract force and moment from wrench command
        force_command = self.wrench_command[0:3]
        moment_command = self.wrench_command[3:6]

        # FIXED: Use Z-component like PX4 controller, not magnitude
        # The thrust command represents the motor thrust magnitude (always positive)
        # In NED: force Z is negative for upward, so negate to get positive thrust
        # In ENU: force Z is positive for upward, use directly
        if self.orient == "NED":
            # In NED: negate Z component to convert upward force to positive thrust
            thrust_command = -force_command[2]
            # Debug: remove after confirming fix
            if abs(thrust_command) > 100 or thrust_command < 0:
                print(f"WARNING: Unusual thrust_command = {thrust_command:.2f} N from force = {force_command}")
        else:  # ENU
            # In ENU: Z component is already positive for upward
            thrust_command = force_command[2]
        
        # `compute_body_torque` already emits Lee's M_des in physical body N·m,
        # and the mixerFM rows are physical (T per W², τ per W²), so we pass the
        # moment through directly. (The earlier `torque_scale=-1.0` predated the
        # base-controller sign fix and reversed angular-rate damping.)
        roll_torque = moment_command[0]
        pitch_torque = moment_command[1]
        yaw_torque = moment_command[2]

        # Create command vector [thrust, roll_torque, pitch_torque, yaw_torque]
        t = np.array([thrust_command, roll_torque, pitch_torque, yaw_torque])

        # FIXED: Add safety limits to prevent motor saturation
        max_thrust = quad.params["maxThr"] * 1.0  # Use 80% of max thrust
        min_thrust = quad.params["minThr"] * 1.0  # Allow zero thrust
        thrust_command = np.clip(thrust_command, min_thrust, max_thrust)
        t[0] = thrust_command

        # DEBUG: Print allocation matrix usage (disabled for performance)
        # if not hasattr(self, '_allocation_debug_counter'):
        #     self._allocation_debug_counter = 0
        # self._allocation_debug_counter += 1
        # if self._allocation_debug_counter % 100 == 1:
        #     print(f"\nALLOCATION DEBUG:")
        #     print(f"  Input t (thrust, Mx, My, Mz): {t}")
        #     print(f"  mixerFM (what w²=1 for all motors produces):")
        #     max_capability = np.dot(quad.params["mixerFM"], np.ones(4))
        #     print(f"    {max_capability}")
        #     print(f"  mixerFMinv shape: {quad.params['mixerFMinv'].shape}")

        # Convert to motor commands using allocation matrix (same as GeneralizedControl)
        w_squared_normalized = np.dot(quad.params["mixerFMinv"], t)

        # if self._allocation_debug_counter % 100 == 1:
        #     print(f"  w_squared_normalized: {w_squared_normalized}")
        #     # Verify by multiplying back
        #     verification = np.dot(quad.params["mixerFM"], w_squared_normalized)
        #     print(f"  Verification (should equal t): {verification}")
        
        # Convert from normalized to actual motor speeds
        w_max_values = np.array([prop["wmax"] for prop in quad.drone_sim.config.propellers])
        w_squared_actual = w_squared_normalized * (w_max_values**2)

        # Apply limits and take square root
        self.w_cmd = np.sqrt(np.clip(w_squared_actual,
                                    quad.params["minWmotor"]**2,
                                    quad.params["maxWmotor"]**2))

        # if self._allocation_debug_counter % 100 == 1:
        #     print(f"  w_cmd (motor speeds): {self.w_cmd}")
        #     # Calculate expected thrust
        #     expected_thrust_per_motor = quad.params["kTh"] * self.w_cmd**2
        #     expected_total_thrust = np.sum(expected_thrust_per_motor)
        #     print(f"  Expected thrust per motor: {expected_thrust_per_motor}")
        #     print(f"  Expected total thrust: {expected_total_thrust:.2f} N")
        #     print(f"  Commanded thrust was: {t[0]:.2f} N")
        #     print(f"  Ratio: {expected_total_thrust / t[0]:.2f}x")
    
    
    def get_control_info(self):
        """Get comprehensive control configuration information."""
        info = {
            'num_motors': self.num_motors,
            'controller_type': 'Lee Geometric Control',
            'coordinate_frame': self.orient,
            'control_gains': {
                'position_P': self.pos_P_gain.tolist(),
                'velocity_P': self.vel_P_gain.tolist(),
                'attitude_P': self.att_P_gain.tolist(),
                'rate_P': self.rate_P_gain.tolist()
            },
            'control_limits': {
                'max_velocity': self.vel_max.tolist(),
                'max_velocity_norm': self.vel_max_all,
                'max_tilt_angle': self.tilt_max * rad2deg,  # Reported for compatibility (not enforced)
                'max_rates': (self.rate_max * rad2deg).tolist()  # Reported for compatibility (not enforced)
            },
        }
        return info