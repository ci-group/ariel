import jax.numpy as jnp
from jax import jit

@jit
def trapezoidal_velocity_profile_non_zero_start_jax(q_start, q_end, v_start, v_max, a_max, t):
    """
    Generates a trapezoidal velocity profile for a single joint with a non-zero starting velocity.
    Returns position, velocity, and acceleration at time t.
    This function is JAX-compatible.
    
    Args:
        q_start (float/jnp.ndarray): Starting position.
        q_end (float/jnp.ndarray): Ending position.
        v_start (float/jnp.ndarray): Starting velocity (must be non-negative if moving in positive direction, non-positive if moving in negative direction).
        v_max (float/jnp.ndarray): Maximum absolute velocity.
        a_max (float/jnp.ndarray): Maximum absolute acceleration.
        t (float/jnp.ndarray): Current time.
    
    Returns:
        tuple: (position, velocity, acceleration) at time t.
    """
    delta_q = q_end - q_start
    
    # Calculate direction and handle delta_q == 0 case
    # If delta_q is 0, direction is 1.0 (or arbitrary, as motion is zero anyway)
    direction = jnp.where(delta_q != 0, jnp.sign(delta_q), 1.0)

    # Adjust v_start: if delta_q != 0 AND v_start is against direction, set v_start = 0.0
    v_start_adjusted = jnp.where(
        (delta_q != 0) & (jnp.sign(v_start) != direction),
        0.0,
        v_start
    )
    v_start_abs = jnp.abs(v_start_adjusted)

    # Calculate times for acceleration and deceleration phases (assuming trapezoidal profile)
    
    # Time to accelerate from v_start_abs to v_max
    Ta1 = jnp.where(v_max > v_start_abs, (v_max - v_start_abs) / a_max, 0.0)
    # Distance covered during Ta1
    Sa1 = v_start_abs * Ta1 + 0.5 * a_max * Ta1**2

    # Time to decelerate from v_max to 0
    Ta2 = v_max / a_max
    # Distance covered during Ta2
    Sa2 = 0.5 * a_max * Ta2**2

    # Total distance required for full acceleration to v_max and deceleration from v_max
    S_accel_decel = Sa1 + Sa2

    # Determine if the profile is triangular or trapezoidal
    is_triangular = jnp.abs(delta_q) < S_accel_decel

    # --- Triangular Profile Calculation ---
    
    # v_peak^2 = (2 * a_max * S + v_start^2) / 2 where S = abs(delta_q)
    v_peak_sq = (2 * a_max * jnp.abs(delta_q) + v_start_abs**2) / 2
    v_peak = jnp.sqrt(v_peak_sq)
    
    T_accel_tri = (v_peak - v_start_abs) / a_max
    T_decel_tri = v_peak / a_max
    T_total_tri = T_accel_tri + T_decel_tri
    
    # Time phases for triangular profile
    t_accel_tri = t < T_accel_tri
    t_decel_tri = (t >= T_accel_tri) & (t < T_total_tri)
    t_end_tri = t >= T_total_tri
    
    # Acceleration phase (t < T_accel_tri)
    a_accel_tri = a_max * direction
    v_accel_tri = v_start_adjusted + a_accel_tri * t
    q_accel_tri = q_start + v_start_adjusted * t + 0.5 * a_accel_tri * t**2
    
    # Deceleration phase (t < T_total_tri)
    a_decel_tri = -a_max * direction
    t_rel_decel_tri = t - T_accel_tri
    
    # Calculate position at T_accel_tri
    q_at_T_accel_tri = q_start + v_start_adjusted * T_accel_tri + 0.5 * a_max * direction * T_accel_tri**2
    
    v_decel_tri = v_peak * direction + a_decel_tri * t_rel_decel_tri
    q_decel_tri = q_at_T_accel_tri + v_peak * direction * t_rel_decel_tri + 0.5 * a_decel_tri * t_rel_decel_tri**2
    
    # End of motion (t >= T_total_tri)
    a_end_tri = 0.0
    v_end_tri = 0.0
    q_end_tri = q_end
    
    # Select results for triangular profile
    q_tri = jnp.select(
        [t_accel_tri, t_decel_tri, t_end_tri],
        [q_accel_tri, q_decel_tri, q_end_tri]
    )
    v_tri = jnp.select(
        [t_accel_tri, t_decel_tri, t_end_tri],
        [v_accel_tri, v_decel_tri, v_end_tri]
    )
    a_tri = jnp.select(
        [t_accel_tri, t_decel_tri, t_end_tri],
        [a_accel_tri, a_decel_tri, a_end_tri]
    )

    # --- Trapezoidal Profile Calculation ---
    
    T_const_vel_trap = (jnp.abs(delta_q) - S_accel_decel) / v_max
    T_total_trap = Ta1 + T_const_vel_trap + Ta2
    
    # Time phases for trapezoidal profile
    t_accel_trap = t < Ta1
    t_const_trap = (t >= Ta1) & (t < Ta1 + T_const_vel_trap)
    t_decel_trap = (t >= Ta1 + T_const_vel_trap) & (t < T_total_trap)
    t_end_trap = t >= T_total_trap
    
    # 1. Acceleration phase (t < Ta1)
    a_accel_trap = a_max * direction
    v_accel_trap = v_start_adjusted + a_accel_trap * t
    q_accel_trap = q_start + v_start_adjusted * t + 0.5 * a_accel_trap * t**2
    
    # Calculate position at Ta1
    q_at_Ta1_trap = q_start + v_start_adjusted * Ta1 + 0.5 * a_max * direction * Ta1**2
    
    # 2. Constant velocity phase (t < Ta1 + T_const_vel)
    a_const_trap = 0.0
    v_const_trap = v_max * direction
    q_const_trap = q_at_Ta1_trap + v_max * direction * (t - Ta1)
    
    # 3. Deceleration phase (t < T_total)
    a_decel_trap = -a_max * direction
    t_start_decel_trap = Ta1 + T_const_vel_trap
    t_rel_decel_trap = t - t_start_decel_trap
    
    v_decel_trap = v_max * direction + a_decel_trap * t_rel_decel_trap
    
    q_at_Ta1_T_const_vel_trap = q_at_Ta1_trap + v_max * direction * T_const_vel_trap
    q_decel_trap = q_at_Ta1_T_const_vel_trap + v_max * direction * t_rel_decel_trap + 0.5 * a_decel_trap * t_rel_decel_trap**2
    
    # 4. End of motion (t >= T_total)
    a_end_trap = 0.0
    v_end_trap = 0.0
    q_end_trap = q_end
    
    # Select results for trapezoidal profile
    q_trap = jnp.select(
        [t_accel_trap, t_const_trap, t_decel_trap, t_end_trap],
        [q_accel_trap, q_const_trap, q_decel_trap, q_end_trap]
    )
    v_trap = jnp.select(
        [t_accel_trap, t_const_trap, t_decel_trap, t_end_trap],
        [v_accel_trap, v_const_trap, v_decel_trap, v_end_trap]
    )
    a_trap = jnp.select(
        [t_accel_trap, t_const_trap, t_decel_trap, t_end_trap],
        [a_accel_trap, a_const_trap, a_decel_trap, a_end_trap]
    )
    
    # --- Final Selection ---
    
    q = jnp.where(is_triangular, q_tri, q_trap)
    v = jnp.where(is_triangular, v_tri, v_trap)
    a = jnp.where(is_triangular, a_tri, a_trap)
    
    return q, v, a