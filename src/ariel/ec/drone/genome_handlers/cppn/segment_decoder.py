"""Decode a CPPN network into a phenotype array of arm parameters."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .network import CPPNNetwork
from .evaluation import evaluate_cppn


def _map_tanh_to_range(tanh_val: float, low: float, high: float) -> float:
    """Map a value in [-1, 1] to [low, high]."""
    return low + (tanh_val + 1.0) * 0.5 * (high - low)


def decode_cppn_to_phenotype(
    network: CPPNNetwork,
    num_segments: int,
    arm_limit: int,
    parameter_limits: npt.NDArray,
) -> npt.NDArray:
    """Decode a CPPN to phenotype array by evaluating it over azimuthal segments.

    Args:
        network: The CPPN to decode.
        num_segments: Number of azimuthal segments to evaluate.
        arm_limit: Maximum number of arms (rows in output array).
        parameter_limits: Array of shape ``(6, 2)`` with ``[min, max]`` per
            parameter: ``[magnitude, arm_yaw, arm_pitch, motor_pitch,
            motor_yaw, direction]``.

    Returns:
        Array of shape ``(arm_limit, 6)`` with NaN for unused rows.
        Columns: ``[magnitude, arm_yaw, arm_pitch, motor_pitch, motor_yaw,
        direction]``.
    """
    phenotype = np.full((arm_limit, 6), np.nan)
    arms_placed = 0
    segment_angle_width = 2.0 * np.pi / num_segments

    for seg_idx in range(num_segments):
        if arms_placed >= arm_limit:
            break  # Rule 1: arm limit reached

        remaining_segments = num_segments - seg_idx
        remaining_slots = arm_limit - arms_placed

        # Normalise segment index to [-1, 1]
        if num_segments > 1:
            seg_normalized = 2.0 * seg_idx / (num_segments - 1) - 1.0
        else:
            seg_normalized = 0.0

        outputs = evaluate_cppn(network, np.array([seg_normalized, 1.0]))
        # outputs has 7 elements (all tanh-range from output activations)

        # Rule 2: force arm if remaining segments <= remaining slots
        if remaining_segments <= remaining_slots:
            place_arm = True
        else:
            place_arm = outputs[0] >= 0.0

        if place_arm:
            segment_center = seg_idx * segment_angle_width - np.pi
            half_width = segment_angle_width / 2.0

            mag_min, mag_max = parameter_limits[0]
            pitch_min, pitch_max = parameter_limits[2]
            motor_pitch_min, motor_pitch_max = parameter_limits[3]
            motor_yaw_min, motor_yaw_max = parameter_limits[4]

            magnitude = _map_tanh_to_range(outputs[1], mag_min, mag_max)
            arm_yaw = segment_center + outputs[2] * half_width
            arm_pitch = _map_tanh_to_range(outputs[3], pitch_min, pitch_max)
            motor_yaw = _map_tanh_to_range(outputs[4], motor_yaw_min, motor_yaw_max)
            motor_pitch = _map_tanh_to_range(outputs[5], motor_pitch_min, motor_pitch_max)
            direction = 0.0 if outputs[6] < 0.0 else 1.0

            phenotype[arms_placed] = [
                magnitude, arm_yaw, arm_pitch,
                motor_pitch, motor_yaw, direction,
            ]
            arms_placed += 1

    return phenotype
