"""
DroneInterface - Controller compatibility wrapper for DroneSimulator

This module provides a wrapper class that makes DroneSimulator compatible with
existing controller frameworks while supporting any multi-rotor configuration.
"""

import numpy as np
from .drone_simulator import DroneSimulator


class DroneInterface:
    """
    Wrapper that makes DroneSimulator compatible with existing controller framework.

    Despite the legacy "quadcopter" terminology in variable names, this class works
    with ANY multi-rotor configuration (quad, hex, tri, octo, custom).

    This class adapts the modern DroneSimulator to legacy controller interfaces,
    providing backward compatibility while enabling flexible drone configurations.
    """

    def __init__(self, Ti, propellers=None, drone_type="quad", arm_length=0.11, prop_size=5):
        """
        Initialize drone interface.

        Args:
            Ti: Initial time
            propellers: Custom propeller configuration (optional)
            drone_type: Standard drone type if propellers not specified
            arm_length: Arm length for standard configurations
            prop_size: Propeller size for standard configurations
        """
        # Create drone simulator
        if propellers is not None:
            self.drone_sim = DroneSimulator(propellers=propellers, dt=0.005)
        else:
            self.drone_sim = DroneSimulator.create_standard_drone(
                drone_type, arm_length, prop_size, dt=0.005
            )

        # Get parameters in compatible format
        self.params = self.drone_sim.get_params()

        # Initialize state variables in compatible format
        self._update_state_variables()

        # Initialize extended state variables
        self.extended_state()

        # Store initial time
        self.Ti = Ti

    def _update_state_variables(self):
        """Update all state variables from drone simulator."""
        quad_state = self.drone_sim.get_drone_state()

        # Copy all state variables to be compatible with existing code
        self.state = quad_state['state']
        self.pos = quad_state['pos']
        self.vel = quad_state['vel']
        self.quat = quad_state['quat']
        self.omega = quad_state['omega']
        self.euler = quad_state['euler']
        self.wMotor = quad_state['wMotor']
        self.vel_dot = quad_state['vel_dot']
        self.omega_dot = quad_state['omega_dot']
        self.acc = quad_state['acc']
        self.thr = quad_state['thr']
        self.tor = quad_state['tor']
        self.dcm = quad_state['dcm']

    def extended_state(self):
        """Update extended state variables (quaternion to euler conversion, etc.)."""
        # Import here to avoid circular dependencies
        import ariel.simulation.drone.controllers.utils as utils

        # Update DCM from quaternion
        self.dcm = utils.quat2Dcm(self.quat)

        # Update Euler angles from quaternion
        YPR = utils.quatToYPR_ZYX(self.quat)
        self.euler = YPR[::-1]  # flip YPR so that euler state = phi, theta, psi
        self.psi = YPR[0]
        self.theta = YPR[1]
        self.phi = YPR[2]

    def forces(self):
        """Calculate rotor thrusts and torques."""
        # Read actual motor speeds (rad/s) from sim state. The previous
        # `sqrt(motor_commands) * maxWmotor` form was for the legacy
        # action-in-[0,1] convention; under the reference-form dynamics
        # motor_commands lives in [-1, 1] and `sqrt` of negative values
        # propagates NaN downstream.
        motor_speeds = self.drone_sim._get_actual_motor_speeds()

        # Calculate thrusts and torques
        self.thr = self.params["kTh"] * motor_speeds * motor_speeds
        self.tor = self.params["kTo"] * motor_speeds * motor_speeds

        # Pad to 4 motors for compatibility
        if len(self.thr) < 4:
            self.thr = np.pad(self.thr, (0, 4 - len(self.thr)))
            self.tor = np.pad(self.tor, (0, 4 - len(self.tor)))

    def update(self, t, Ts, cmd, wind):
        """
        Update simulation state.

        Args:
            t: Current time
            Ts: Time step
            cmd: Motor commands
            wind: Wind model
        """
        # Store full motor command for drones with >4 motors
        if hasattr(self, 'w_cmd_full'):
            self.drone_sim.w_cmd_full = self.w_cmd_full

        # Update drone simulator
        new_t = self.drone_sim.update_from_controller(t, Ts, cmd, wind)

        # Update state variables
        self._update_state_variables()
        self.extended_state()
        self.forces()

        return new_t

    def get_configuration_info(self):
        """Get comprehensive configuration information."""
        return self.drone_sim.get_configuration_info()

    def get_propeller_info(self):
        """Get propeller configuration details."""
        return self.drone_sim.get_propeller_info()
