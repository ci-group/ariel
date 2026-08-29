# -*- coding: utf-8 -*-
"""
Generalized Controller for Arbitrary Drone Configurations

This controller extends the original PX4-based controller to work with any drone
configuration by using configurable allocation matrices instead of fixed mixers.

Compatible with the existing configurable drone framework and maintains the same
interface as the original Control class.

author: Enhanced from John Bass original controller
license: MIT
"""

import numpy as np
from numpy import pi, sin, cos, tan, sqrt
from numpy.linalg import norm, pinv
import ariel.simulation.drone.controllers.utils as utils

rad2deg = 180.0/pi
deg2rad = pi/180.0

class GeneralizedControl:
    """
    Generalized Controller for Arbitrary Drone Configurations.
    
    This controller maintains the same interface as the original Control class
    but uses configurable control allocation instead of fixed mixer matrices.
    """
    
    def __init__(self, quad, yawType, orient="NED",
                 pos_P_gain=np.array([1.0, 1.0, 3.0]), 
                 vel_P_gain=np.array([5.0, 5.0, 5.0]), 
                 vel_D_gain=np.array([0.5, 0.5, 1.0]), 
                 vel_I_gain=np.array([5.0, 5.0, 5.0]),
                 att_P_gain=np.array([8.0, 8.0, 1.5]), 
                 rate_P_gain=np.array([1.5, 1.5, 1.5]), 
                 rate_D_gain=np.array([0.04, 0.04, 0.1]),
                 vel_max=np.array([5.0, 5.0, 5.0]), 
                 vel_max_all=5.0, 
                 saturate_vel_separately=False,
                 tilt_max=50.0*deg2rad, 
                 rate_max=np.array([200.0*deg2rad, 200.0*deg2rad, 150.0*deg2rad]),
                 aggressiveness=1.0):
        """
        Initialize generalized controller with configurable parameters and auto-scaling.
        
        Args:
            quad: ConfigurableQuadcopter instance with drone configuration
            yawType: Yaw control type (0 = no yaw control, 1 = yaw control enabled)
            orient: Coordinate frame orientation ("NED" or "ENU")
            
            # Control Gains (base values, will be auto-scaled if enabled)
            pos_P_gain: Position P gains [Px, Py, Pz] for X, Y, Z axes
            vel_P_gain: Velocity P gains [Pxdot, Pydot, Pzdot] 
            vel_D_gain: Velocity D gains [Dxdot, Dydot, Dzdot] for damping
            vel_I_gain: Velocity I gains [Ixdot, Iydot, Izdot] for steady-state error
            att_P_gain: Attitude P gains [Pphi, Ptheta, Ppsi] for roll, pitch, yaw
            rate_P_gain: Rate P gains [Pp, Pq, Pr] for roll, pitch, yaw rates
            rate_D_gain: Rate D gains [Dp, Dq, Dr] for gyroscopic damping
            
            # Control Limits (base values, will be auto-scaled if enabled)
            vel_max: Maximum velocities [uMax, vMax, wMax] in X, Y, Z axes (m/s)
            vel_max_all: Maximum total velocity magnitude (m/s)
            saturate_vel_separately: Whether to saturate each axis independently
            tilt_max: Maximum tilt angle in radians (limits roll/pitch for safety)
            rate_max: Maximum angular rates [pMax, qMax, rMax] in rad/s
            
            # Auto-scaling Parameters
            aggressiveness: Overall scaling factor (0.5=conservative, 2.0=aggressive)
        """
        # Store base control parameters 
        self.base_pos_P_gain = pos_P_gain.copy()
        self.base_vel_P_gain = vel_P_gain.copy()
        self.base_vel_D_gain = vel_D_gain.copy()
        self.base_vel_I_gain = vel_I_gain.copy()
        self.base_att_P_gain = att_P_gain.copy()
        self.base_rate_P_gain = rate_P_gain.copy()
        self.base_rate_D_gain = rate_D_gain.copy()
        self.base_vel_max = vel_max.copy()
        self.base_vel_max_all = vel_max_all
        self.base_tilt_max = tilt_max
        self.base_rate_max = rate_max.copy()
        
        # Store auto-scaling parameters
        self.aggressiveness = aggressiveness
        self.saturate_vel_separately = saturate_vel_separately
        
        # Store coordinate frame orientation
        self.orient = orient
        
        # Get drone configuration first
        self.drone_config = quad.drone_sim.config
        self.num_motors = self.drone_config.num_motors
        
        # Use base values directly
        self.pos_P_gain = self.base_pos_P_gain.copy()
        self.vel_P_gain = self.base_vel_P_gain.copy()
        self.vel_D_gain = self.base_vel_D_gain.copy()
        self.vel_I_gain = self.base_vel_I_gain.copy()
        self.att_P_gain = self.base_att_P_gain.copy()
        self.rate_P_gain = self.base_rate_P_gain.copy()
        self.rate_D_gain = self.base_rate_D_gain.copy()
        self.vel_max = self.base_vel_max.copy()
        self.vel_max_all = self.base_vel_max_all
        self.tilt_max = self.base_tilt_max
        self.rate_max = self.base_rate_max.copy()
        
        # Initialize state variables (same as original)
        self.sDesCalc = np.zeros(16)
        self.thr_int = np.zeros(3)
        
        # Initialize motor commands for arbitrary number of motors
        self.w_cmd = np.ones(self.num_motors) * self._get_hover_speed(quad)
        
        # Setup yaw control
        if (yawType == 0):
            self.att_P_gain[2] = 0
        self.setYawWeight()
        
        # Initialize control variables (same as original)
        self.pos_sp = np.zeros(3)
        self.vel_sp = np.zeros(3)
        self.acc_sp = np.zeros(3)
        self.thrust_sp = np.zeros(3)
        self.eul_sp = np.zeros(3)
        self.pqr_sp = np.zeros(3)
        self.yawFF = np.zeros(3)
    
    def _get_hover_speed(self, quad):
        """Calculate hover speed for each motor."""
        hover_speeds = []
        hover_thrust_per_motor = (quad.params["mB"] * quad.params["g"]) / self.num_motors
        
        for prop in self.drone_config.propellers:
            k_f, k_m = prop["constants"]
            w_hover = sqrt(hover_thrust_per_motor / k_f)
            hover_speeds.append(w_hover)
        
        return np.array(hover_speeds)
    
    def _get_reference_properties(self): # TODO: Not used atm but could be used in future
        """
        Get reference drone properties for gain scaling baseline.
        Based on actual matched configuration from run_3D_simulation_configurable.py:
        0.16m arm quadrotor with matched props - extracted from ConfigurableQuadcopter.
        """
        return {
            'char_length': 0.226274,          # Characteristic length from arm lengths
            'mass': 1.517882,                   # Total mass from matched configuration
            'inertia_trace': 0.125761,        # Inertia trace from matched configuration
            'Ix': 0.031352, 'Iy': 0.031554, 'Iz': 0.062855,  # Inertia components
            'num_motors': 4,               # Standard quadrotor
            'thrust_to_weight': 11.137963,       # Actual T/W ratio
            'max_roll_torque': 6.633960,        # Actual roll authority
            'max_pitch_torque': 6.633960,       # Actual pitch authority
            'max_yaw_torque': 0.620392,         # Actual yaw authority
        }
    
    def controller(self, sDes, quad, ctrl_type, Ts):
        """
        Main controller function.
        
        Args:
            sDes: Desired state vector [pos(3), vel(3), acc(3), thrust(3), eul(3), pqr(3), yawRate]
            quad: ConfigurableQuadcopter instance  
            ctrl_type: Control type ("xyz_pos", "xyz_vel", etc.)
            Ts: Time step
        """
        # Extract desired state
        self.pos_sp[:] = sDes[0:3]
        self.vel_sp[:] = sDes[3:6]
        self.acc_sp[:] = sDes[6:9]
        self.thrust_sp[:] = sDes[9:12]
        self.eul_sp[:] = sDes[12:15]
        self.pqr_sp[:] = sDes[15:18]
        self.yawFF[:] = sDes[18]
        
        # Select Controller based on control type
        if (ctrl_type == "xyz_vel"):
            self.saturateVel()
            self.z_vel_control(quad, Ts)
            self.xy_vel_control(quad, Ts)
            self.thrustToAttitude(quad, Ts)
            self.attitude_control(quad, Ts)
            self.rate_control(quad, Ts)
        elif (ctrl_type == "xy_vel_z_pos"):
            self.z_pos_control(quad, Ts)
            self.saturateVel()
            self.z_vel_control(quad, Ts)
            self.xy_vel_control(quad, Ts)
            self.thrustToAttitude(quad, Ts)
            self.attitude_control(quad, Ts)
            self.rate_control(quad, Ts)
        elif (ctrl_type == "xyz_pos"):
            self.z_pos_control(quad, Ts)
            self.xy_pos_control(quad, Ts)
            self.saturateVel()
            self.z_vel_control(quad, Ts)
            self.xy_vel_control(quad, Ts)
            self.thrustToAttitude(quad, Ts)
            self.attitude_control(quad, Ts)
            self.rate_control(quad, Ts)

        # THRUST DIRECTION FIX: Use Z-component instead of magnitude to preserve direction
        # In NED frame: positive Z is down, motors produce upward thrust (negative Z)
        # So we need to negate the Z-component to match motor thrust direction
        thrust_command = -self.thrust_sp[2]  # Negate to convert NED Z to upward thrust
        t = np.array([thrust_command, self.rateCtrl[0], self.rateCtrl[1], self.rateCtrl[2]])
        
        # Convert thrust and moment to motor speeds using configurable mixer (REVERTED TO ORIGINAL)
        # DEBUG: Add detailed tracing of control values
        if hasattr(self, '_debug_control') and self._debug_control:
            print(f"    Control pipeline debug:")
            print(f"      thrust_sp magnitude: {np.linalg.norm(self.thrust_sp):.3f} N")
            print(f"      thrust_sp vector: [{self.thrust_sp[0]:.3f}, {self.thrust_sp[1]:.3f}, {self.thrust_sp[2]:.3f}]")
            print(f"      thrust_sp[2] (NED Z): {self.thrust_sp[2]:.3f} N")
            print(f"      thrust_command (corrected): {thrust_command:.3f} N")
            print(f"      rateCtrl: [{self.rateCtrl[0]:.6f}, {self.rateCtrl[1]:.6f}, {self.rateCtrl[2]:.6f}] N·m")
            print(f"      t vector: [{t[0]:.3f}, {t[1]:.6f}, {t[2]:.6f}, {t[3]:.6f}]")
            mixer_product = np.dot(quad.params["mixerFMinv"], t)
            print(f"      mixerFMinv @ t: [{mixer_product[0]:.1f}, {mixer_product[1]:.1f}, {mixer_product[2]:.1f}, {mixer_product[3]:.1f}]")
            
        # ==================================================================================
        # CONTROL ALLOCATION - CONVERT CONTROL COMMANDS TO MOTOR SPEEDS
        # ==================================================================================
        #
        # PIPELINE STEP 1: Control Commands → Normalized Motor Commands
        # Input vector t = [thrust_command_N, roll_moment_Nm, pitch_moment_Nm, yaw_moment_Nm]
        # The mixerFMinv matrix maps these physical commands to normalized motor speeds [0,1]
        #
        # WHY NORMALIZED? 
        # - Different motors may have different w_max values (though usually identical)
        # - Allocation matrices are pre-scaled for normalized inputs
        # - Allows consistent control allocation regardless of motor specifications
        #
        w_squared_normalized = np.dot(quad.params["mixerFMinv"], t)
        
        # PIPELINE STEP 2: Normalized Commands → Actual Motor Speeds Squared
        # Convert from normalized range [0,1] to actual motor speed squared values
        # Physical relationship: F_motor = k_f * w² where w is actual motor speed
        #
        # For each motor i: w²_actual[i] = w²_normalized[i] * w_max[i]²
        w_max_values = np.array([prop["wmax"] for prop in quad.drone_sim.config.propellers])
        w_squared_actual = w_squared_normalized * (w_max_values**2)
        
        # PIPELINE STEP 3: Apply Physical Limits and Extract Motor Speeds
        # Clip to ensure motor speeds stay within physical capabilities:
        # - Minimum: quad.params["minWmotor"] (typically 75 rad/s for stability)
        # - Maximum: quad.params["maxWmotor"] (motor/ESC limit, typically w_max)
        #
        # Finally take square root to get actual commanded motor speeds in rad/s
        self.w_cmd = np.sqrt(np.clip(w_squared_actual, 
                                    quad.params["minWmotor"]**2, 
                                    quad.params["maxWmotor"]**2))
        # self.w_cmd = mixerFM(quad, np.linalg.norm(self.thrust_sp), self.rateCtrl)

        # Add calculated Desired States (same as original)
        self.sDesCalc[0:3] = self.pos_sp
        self.sDesCalc[3:6] = self.vel_sp
        self.sDesCalc[6:9] = self.thrust_sp
        self.sDesCalc[9:13] = self.qd
        self.sDesCalc[13:16] = self.rate_sp

    # All the following methods are identical to the original controller
    # (position control, velocity control, attitude control, etc.)
    
    def z_pos_control(self, quad, Ts):
        """Z Position Control (identical to original)."""
        pos_z_error = self.pos_sp[2] - quad.pos[2]
        self.vel_sp[2] += self.pos_P_gain[2]*pos_z_error
        
    def xy_pos_control(self, quad, Ts):
        """XY Position Control (identical to original)."""
        pos_xy_error = (self.pos_sp[0:2] - quad.pos[0:2])
        self.vel_sp[0:2] += self.pos_P_gain[0:2]*pos_xy_error
        
    def saturateVel(self):
        """Saturate Velocity Setpoint (identical to original)."""
        if (self.saturate_vel_separately):
            self.vel_sp = np.clip(self.vel_sp, -self.vel_max, self.vel_max)
        else:
            totalVel_sp = norm(self.vel_sp)
            if (totalVel_sp > self.vel_max_all):
                self.vel_sp = self.vel_sp/totalVel_sp*self.vel_max_all

    def z_vel_control(self, quad, Ts):
        """Z Velocity Control (identical to original)."""
        vel_z_error = self.vel_sp[2] - quad.vel[2]
        if (self.orient == "NED"):
            thrust_z_sp = self.vel_P_gain[2]*vel_z_error - self.vel_D_gain[2]*quad.vel_dot[2] + quad.params["mB"]*(self.acc_sp[2] - quad.params["g"]) + self.thr_int[2]
        elif (self.orient == "ENU"):
            thrust_z_sp = self.vel_P_gain[2]*vel_z_error - self.vel_D_gain[2]*quad.vel_dot[2] + quad.params["mB"]*(self.acc_sp[2] + quad.params["g"]) + self.thr_int[2]
        
        # DEBUG: Check for large Z velocity control values
        if hasattr(self, '_debug_control') and self._debug_control and abs(vel_z_error) > 2.0:
            print(f"    Z VEL CONTROL DEBUG:")
            print(f"      vel_sp[2]: {self.vel_sp[2]:.3f} m/s | quad.vel[2]: {quad.vel[2]:.3f} m/s")
            print(f"      vel_z_error: {vel_z_error:.3f} m/s")
            print(f"      vel_P_gain[2]: {self.vel_P_gain[2]:.3f} | vel_D_gain[2]: {self.vel_D_gain[2]:.3f}")
            print(f"      thrust_z_sp: {thrust_z_sp:.3f} N")
            print(f"      mass*gravity: {quad.params['mB']:.3f} * {quad.params['g']:.3f} = {quad.params['mB']*quad.params['g']:.3f} N")
        
        # Get thrust limits
        if (self.orient == "NED"):
            uMax = -quad.params["minThr"]
            uMin = -quad.params["maxThr"]
        elif (self.orient == "ENU"):
            uMax = quad.params["maxThr"]
            uMin = quad.params["minThr"]

        # Apply Anti-Windup in D-direction
        stop_int_D = (thrust_z_sp >= uMax and vel_z_error >= 0.0) or (thrust_z_sp <= uMin and vel_z_error <= 0.0)

        # Calculate integral part
        if not (stop_int_D):
            self.thr_int[2] += self.vel_I_gain[2]*vel_z_error*Ts * quad.params["useIntergral"]
            self.thr_int[2] = min(abs(self.thr_int[2]), quad.params["maxThr"])*np.sign(self.thr_int[2])

        # Saturate thrust setpoint in D-direction
        self.thrust_sp[2] = np.clip(thrust_z_sp, uMin, uMax)

    def xy_vel_control(self, quad, Ts):
        """XY Velocity Control (identical to original)."""
        vel_xy_error = self.vel_sp[0:2] - quad.vel[0:2]
        thrust_xy_sp = self.vel_P_gain[0:2]*vel_xy_error - self.vel_D_gain[0:2]*quad.vel_dot[0:2] + quad.params["mB"]*(self.acc_sp[0:2]) + self.thr_int[0:2]

        # Max allowed thrust in NE based on tilt and excess thrust
        thrust_max_xy_tilt = abs(self.thrust_sp[2])*np.tan(self.tilt_max)
        thrust_max_xy = sqrt(quad.params["maxThr"]**2 - self.thrust_sp[2]**2)
        thrust_max_xy = min(thrust_max_xy, thrust_max_xy_tilt)

        # Saturate thrust in NE-direction
        self.thrust_sp[0:2] = thrust_xy_sp
        if (np.dot(self.thrust_sp[0:2].T, self.thrust_sp[0:2]) > thrust_max_xy**2):
            mag = norm(self.thrust_sp[0:2])
            self.thrust_sp[0:2] = thrust_xy_sp/mag*thrust_max_xy
        
        # Use tracking Anti-Windup for NE-direction
        arw_gain = 2.0/self.vel_P_gain[0:2]
        vel_err_lim = vel_xy_error - (thrust_xy_sp - self.thrust_sp[0:2])*arw_gain
        self.thr_int[0:2] += self.vel_I_gain[0:2]*vel_err_lim*Ts * quad.params["useIntergral"]
    
    def thrustToAttitude(self, quad, Ts):
        """Thrust to Attitude (identical to original)."""
        yaw_sp = self.eul_sp[2]

        # Desired body_z axis direction
        body_z = -utils.vectNormalize(self.thrust_sp)
        if (self.orient == "ENU"):
            body_z = -body_z
        
        # Vector of desired Yaw direction in XY plane, rotated by pi/2 (fake body_y axis)
        y_C = np.array([-sin(yaw_sp), cos(yaw_sp), 0.0])
        
        # Desired body_x axis direction
        body_x = np.cross(y_C, body_z)
        body_x = utils.vectNormalize(body_x)
        
        # Desired body_y axis direction
        body_y = np.cross(body_z, body_x)

        # Desired rotation matrix
        R_sp = np.array([body_x, body_y, body_z]).T

        # Full desired quaternion
        self.qd_full = utils.RotToQuat(R_sp)
        
    def attitude_control(self, quad, Ts):
        """Attitude Control (identical to original)."""
        # Current thrust orientation e_z and desired thrust orientation e_z_d
        e_z = quad.dcm[:,2]
        e_z_d = -utils.vectNormalize(self.thrust_sp)
        if (self.orient == "ENU"):
            e_z_d = -e_z_d

        # Quaternion error between the 2 vectors
        qe_red = np.zeros(4)
        qe_red[0] = np.dot(e_z, e_z_d) + sqrt(norm(e_z)**2 * norm(e_z_d)**2)
        qe_red[1:4] = np.cross(e_z, e_z_d)
        qe_red = utils.vectNormalize(qe_red)
        
        # Reduced desired quaternion
        self.qd_red = utils.quatMultiply(qe_red, quad.quat)

        # Mixed desired quaternion and resulting desired quaternion qd
        q_mix = utils.quatMultiply(utils.inverse(self.qd_red), self.qd_full)
        q_mix = q_mix*np.sign(q_mix[0])
        q_mix[0] = np.clip(q_mix[0], -1.0, 1.0)
        q_mix[3] = np.clip(q_mix[3], -1.0, 1.0)
        self.qd = utils.quatMultiply(self.qd_red, np.array([cos(self.yaw_w*np.arccos(q_mix[0])), 0, 0, sin(self.yaw_w*np.arcsin(q_mix[3]))]))

        # Resulting error quaternion
        self.qe = utils.quatMultiply(utils.inverse(quad.quat), self.qd)

        # Create rate setpoint from quaternion error
        qe_vector = 2.0*np.sign(self.qe[0])*self.qe[1:4]
        self.rate_sp = qe_vector*self.att_P_gain
        
        # DEBUG: Check for large attitude control values
        if hasattr(self, '_debug_control') and self._debug_control:
            qe_mag = np.linalg.norm(self.qe[1:4])
            rate_sp_mag = np.linalg.norm(self.rate_sp)
            thrust_sp_mag = np.linalg.norm(self.thrust_sp)
            if qe_mag > 0.2 or rate_sp_mag > 2.0 or thrust_sp_mag > 50.0:  # Large attitude errors or commands
                print(f"    ATTITUDE CONTROL DEBUG:")
                print(f"      thrust_sp: [{self.thrust_sp[0]:.3f}, {self.thrust_sp[1]:.3f}, {self.thrust_sp[2]:.3f}] N")
                print(f"      thrust_sp magnitude: {thrust_sp_mag:.3f} N")
                print(f"      qe (quaternion error): [{self.qe[0]:.3f}, {self.qe[1]:.3f}, {self.qe[2]:.3f}, {self.qe[3]:.3f}]")
                print(f"      qe vector magnitude: {qe_mag:.3f}")
                print(f"      qe_vector (2*sign*qe[1:4]): [{qe_vector[0]:.3f}, {qe_vector[1]:.3f}, {qe_vector[2]:.3f}]")
                print(f"      att_P_gain: [{self.att_P_gain[0]:.3f}, {self.att_P_gain[1]:.3f}, {self.att_P_gain[2]:.3f}]")
                print(f"      rate_sp (before limits): [{self.rate_sp[0]:.3f}, {self.rate_sp[1]:.3f}, {self.rate_sp[2]:.3f}] rad/s")
        
        # Limit yawFF
        self.yawFF = np.clip(self.yawFF, -self.rate_max[2], self.rate_max[2])

        # Add Yaw rate feed-forward
        self.rate_sp += utils.quat2Dcm(utils.inverse(quad.quat))[:,2]*self.yawFF

        # Limit rate setpoint
        self.rate_sp = np.clip(self.rate_sp, -self.rate_max, self.rate_max)

    def rate_control(self, quad, Ts):
        """Rate Control (identical to original)."""
        rate_error = self.rate_sp - quad.omega
        self.rateCtrl = self.rate_P_gain*rate_error - self.rate_D_gain*quad.omega_dot
        
        # DEBUG: Check for large rate control values
        if hasattr(self, '_debug_control') and self._debug_control:
            rate_error_mag = np.linalg.norm(rate_error)
            rateCtrl_mag = np.linalg.norm(self.rateCtrl)
            if rate_error_mag > 1.0 or rateCtrl_mag > 0.1:  # Large rate errors or commands
                print(f"    RATE CONTROL DEBUG:")
                print(f"      rate_sp (setpoint): [{self.rate_sp[0]:.3f}, {self.rate_sp[1]:.3f}, {self.rate_sp[2]:.3f}] rad/s")
                print(f"      quad.omega (actual): [{quad.omega[0]:.3f}, {quad.omega[1]:.3f}, {quad.omega[2]:.3f}] rad/s")
                print(f"      rate_error: [{rate_error[0]:.3f}, {rate_error[1]:.3f}, {rate_error[2]:.3f}] rad/s")
                print(f"      rate_P_gain: [{self.rate_P_gain[0]:.3f}, {self.rate_P_gain[1]:.3f}, {self.rate_P_gain[2]:.3f}]")
                print(f"      rate_D_gain: [{self.rate_D_gain[0]:.3f}, {self.rate_D_gain[1]:.3f}, {self.rate_D_gain[2]:.3f}]")
                print(f"      omega_dot: [{quad.omega_dot[0]:.3f}, {quad.omega_dot[1]:.3f}, {quad.omega_dot[2]:.3f}] rad/s²")
                print(f"      rateCtrl output: [{self.rateCtrl[0]:.6f}, {self.rateCtrl[1]:.6f}, {self.rateCtrl[2]:.6f}] N·m")
        
    def setYawWeight(self):
        """Set Yaw Weight (identical to original)."""
        roll_pitch_gain = 0.5*(self.att_P_gain[0] + self.att_P_gain[1])
        self.yaw_w = np.clip(self.att_P_gain[2]/roll_pitch_gain, 0.0, 1.0)
        self.att_P_gain[2] = roll_pitch_gain
    
    def get_control_info(self):
        """Get comprehensive control configuration information including auto-scaling details."""
        info = {
            'num_motors': self.num_motors,
            'motor_mapping': getattr(self, 'motor_mapping_info', {'type': 'unknown'}),
            'motor_validation': getattr(self, 'motor_validation', {}),
            'control_signs_validated': getattr(self, '_control_signs_validated', False),
            'control_gains': {
                'current': {
                    'pos_P_gain': self.pos_P_gain.tolist(),
                    'vel_P_gain': self.vel_P_gain.tolist(),
                    'vel_D_gain': self.vel_D_gain.tolist(),
                    'vel_I_gain': self.vel_I_gain.tolist(),
                    'att_P_gain': self.att_P_gain.tolist(),
                    'rate_P_gain': self.rate_P_gain.tolist(),
                    'rate_D_gain': self.rate_D_gain.tolist(),
                },
                'base': {
                    'pos_P_gain': self.base_pos_P_gain.tolist(),
                    'vel_P_gain': self.base_vel_P_gain.tolist(),
                    'vel_D_gain': self.base_vel_D_gain.tolist(),
                    'vel_I_gain': self.base_vel_I_gain.tolist(),
                    'att_P_gain': self.base_att_P_gain.tolist(),
                    'rate_P_gain': self.base_rate_P_gain.tolist(),
                    'rate_D_gain': self.base_rate_D_gain.tolist(),
                }
            },
            'control_limits': {
                'current': {
                    'vel_max': self.vel_max.tolist(),
                    'vel_max_all': self.vel_max_all,
                    'tilt_max_deg': self.tilt_max * 180.0/pi,
                    'rate_max_deg': (self.rate_max * 180.0/pi).tolist(),
                },
                'base': {
                    'vel_max': self.base_vel_max.tolist(),
                    'vel_max_all': self.base_vel_max_all,
                    'tilt_max_deg': self.base_tilt_max * 180.0/pi,
                    'rate_max_deg': (self.base_rate_max * 180.0/pi).tolist(),
                }
            },
            'other_settings': {
                'saturate_vel_separately': self.saturate_vel_separately,
            }
        }
        
        return info


# Compatibility alias - allows drop-in replacement
Control = GeneralizedControl