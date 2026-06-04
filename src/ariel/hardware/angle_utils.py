"""Angle conversion utilities between MuJoCo control space and Robohat servo degrees."""

import numpy as np

# Servo neutral position in degrees (midpoint of [0°, 180°] range)
SERVO_NEUTRAL_DEG: float = 90.0
SERVO_MIN_DEG: float = 0.0
SERVO_MAX_DEG: float = 180.0


def mujoco_ctrl_to_degrees(ctrl_rad: float) -> float:
    """Convert a MuJoCo actuator control value to a Robohat servo angle in degrees.

    MuJoCo robogen-lite hinges use ctrlrange [-π/2, π/2].
    Robohat servos accept [0°, 180°] with 90° as the mechanical neutral.

    Parameters
    ----------
    ctrl_rad : float
        MuJoCo actuator value in radians, expected in [-π/2, π/2].

    Returns
    -------
    float
        Servo angle in degrees, clamped to [0°, 180°].
    """
    return float(np.clip(SERVO_NEUTRAL_DEG + np.degrees(ctrl_rad), SERVO_MIN_DEG, SERVO_MAX_DEG))


def batch_mujoco_ctrl_to_degrees(ctrl_rads: np.ndarray) -> list[float]:
    """Vectorised version of mujoco_ctrl_to_degrees for an array of joint angles.

    Parameters
    ----------
    ctrl_rads : np.ndarray
        Array of MuJoCo actuator values in radians, shape (n_joints,).

    Returns
    -------
    list[float]
        List of servo angles in degrees, each clamped to [0°, 180°].
    """
    degrees = np.clip(SERVO_NEUTRAL_DEG + np.degrees(ctrl_rads), SERVO_MIN_DEG, SERVO_MAX_DEG)
    return degrees.tolist()
