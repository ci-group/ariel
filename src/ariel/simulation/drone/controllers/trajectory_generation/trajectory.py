# -*- coding: utf-8 -*-
"""
Trajectory Generation - B-spline Gate Trajectories

Simplified trajectory module supporting only B-spline trajectories for gate racing.
All waypoint-based and figure-8 trajectory methods have been removed.

author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import pi
from ariel.simulation.drone.controllers.trajectory_generation.bspline_gate_trajectory import BSplineGateTrajectory


class Trajectory:
    """
    Trajectory class for drone control - B-spline gate trajectories only.

    This class generates desired states (position, velocity, acceleration, yaw)
    for the controller to follow B-spline trajectories through racing gates.

    Args:
        quad: Quadcopter instance
        ctrlType: Control type ('xyz_pos', 'xy_vel_z_pos', 'xyz_vel')
        trajSelect: Trajectory selection [xyzType, yawType, averVel]
                   - xyzType: Must be 15 (B-spline gate trajectory)
                   - yawType: Yaw control type (0=none, 1=follow trajectory, 3=velocity-based)
                   - averVel: Not used for B-spline trajectories
        gate_config: Gate configuration object (required for B-spline)
        bspline_params: Dictionary of B-spline parameters (optional)
    """

    def __init__(self, quad, ctrlType, trajSelect, gate_config=None, bspline_params=None):

        self.ctrlType = ctrlType
        self.xyzType = trajSelect[0]
        self.yawType = trajSelect[1]
        self.averVel = trajSelect[2]

        # Validate trajectory type - only B-spline supported
        if self.xyzType != 15:
            raise ValueError(
                f"Only B-spline gate trajectories (xyzType=15) are supported. "
                f"Got xyzType={self.xyzType}. "
                f"Old waypoint-based and figure-8 trajectories have been removed."
            )

        # B-spline trajectory setup (required)
        if gate_config is None:
            raise ValueError("gate_config must be provided for B-spline trajectory (xyzType=15)")

        self.gate_config = gate_config
        self.bspline_trajectory = BSplineGateTrajectory(gate_config)

        # Initialize placeholder waypoint arrays for compatibility
        # (some visualization code may still reference these)
        self.t_wps = np.array([0])
        self.wps = np.array([[0, 0, 0]])
        self.y_wps = np.array([0])
        self.v_wp = 1.0
        self.end_reached = 0

        # Get initial heading from quad
        self.current_heading = quad.psi

        # Initialize trajectory setpoint
        self.desPos = np.zeros(3)      # Desired position (x, y, z)
        self.desVel = np.zeros(3)      # Desired velocity (xdot, ydot, zdot)
        self.desAcc = np.zeros(3)      # Desired acceleration (xdotdot, ydotdot, zdotdot)
        self.desThr = np.zeros(3)      # Desired thrust in N-E-D directions (or E-N-U, if selected)
        self.desEul = np.zeros(3)      # Desired orientation in the world frame (phi, theta, psi)
        self.desPQR = np.zeros(3)      # Desired angular velocity in the body frame (p, q, r)
        self.desYawRate = 0.0          # Desired yaw speed
        self.sDes = np.hstack((self.desPos, self.desVel, self.desAcc, self.desThr,
                               self.desEul, self.desPQR, self.desYawRate)).astype(float)

    def desiredState(self, t, Ts, quad):
        """
        Calculate desired state at time t for B-spline trajectory.

        Args:
            t: Current time
            Ts: Time step
            quad: Quadcopter instance

        Returns:
            sDes: Desired state vector [pos, vel, acc, thr, eul, pqr, yawRate]
        """

        # Reset desired state
        self.desPos = np.zeros(3)
        self.desVel = np.zeros(3)
        self.desAcc = np.zeros(3)
        self.desThr = np.zeros(3)
        self.desEul = np.zeros(3)
        self.desPQR = np.zeros(3)
        self.desYawRate = 0.0

        # Evaluate B-spline gate trajectory
        if self.bspline_trajectory is None:
            raise RuntimeError("B-spline trajectory not initialized")

        # Get trajectory state
        position, velocity, acceleration = self.bspline_trajectory.evaluate(t)

        # Set desired states
        self.desPos = position
        self.desVel = velocity
        self.desAcc = acceleration

        # Handle yaw control
        if self.yawType == 0:
            # No yaw control
            pass

        elif self.yawType == 1:
            # Fixed yaw (maintain initial heading)
            self.desEul[2] = 0.0

        elif self.yawType == 3:
            # Yaw follows velocity direction. At t=0 the spline has zero velocity
            # (quintic startup ramp), so peek forward 0.05s to recover the heading
            # — keeps the yaw setpoint aligned with the drone's initial yaw and
            # avoids a spurious large yaw error on the first controller step.
            if np.linalg.norm(self.desVel[:2]) < 1e-6:
                _, future_vel, _ = self.bspline_trajectory.evaluate(t + 0.05)
                if np.linalg.norm(future_vel[:2]) > 1e-6:
                    self.desEul[2] = np.arctan2(future_vel[1], future_vel[0])
                else:
                    self.desEul[2] = self.current_heading
            else:
                # Calculate desired yaw from velocity vector
                self.desEul[2] = np.arctan2(self.desVel[1], self.desVel[0])

                # Handle wrap-around discontinuity at ±π
                if (np.sign(self.desEul[2]) != np.sign(self.current_heading) and
                    abs(self.desEul[2] - self.current_heading) >= 2*pi - 0.1):
                    self.current_heading = self.current_heading + np.sign(self.desEul[2]) * 2*pi

                # Calculate yaw rate
                delta_psi = self.desEul[2] - self.current_heading
                self.desYawRate = delta_psi / Ts

                # Update current heading for next iteration
                self.current_heading = self.desEul[2]

        # Package desired state
        self.sDes = np.hstack((self.desPos, self.desVel, self.desAcc, self.desThr,
                               self.desEul, self.desPQR, self.desYawRate)).astype(float)

        return self.sDes
