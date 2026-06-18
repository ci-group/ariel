# -*- coding: utf-8 -*-
"""
Gate Configurations for Drone Racing/Navigation

This module provides predefined gate configurations for drone racing and
navigation tasks. Gates are defined by their positions and orientations (yaw angles).

Usage:
    from ariel.simulation.drone.controllers.utils.gate_configs import GATE_CONFIGS

    # Get figure-8 configuration
    config = GATE_CONFIGS['figure8']
    gate_positions = config.gate_pos
    gate_yaws = config.gate_yaw
"""

import numpy as np


class GateConfig:
    """Base class for gate configurations"""
    gate_pos = None
    gate_yaw = None
    gate_size = 1.0  # Gate size in meters
    x_bounds = [-10, 10]
    y_bounds = [-10, 10]
    z_bounds = [-2, 2]
    starting_pos = None  # Initial drone position
    periodic = True  # Whether the trajectory loops back to gate 0


class Figure8Gates(GateConfig):
    """
    Figure-8 gate configuration

    This configuration matches the gate setup used in gate_train.py for the figure8 task.
    Gates are ordered according to the trajectory path starting from [0.0, -1.5, 0.0].
    The trajectory follows a figure-8 pattern and encounters gates in this sequence.
    """
    # NED frame: +z is down. Gates lifted to z=-1.0 (1m altitude) so the
    # full 1.5m gate cube sits above the floor — previously z=0.0 placed
    # half of every gate below ground, making the task unreachable.
    gate_pos = np.array([
        [  1.5, -1.5, -1.0],  # Gate 0: Lower right
        [  3.0,  0.0, -1.0],  # Gate 1: Far right
        [  1.5,  1.5, -1.0],  # Gate 2: Upper right
        [  0.0,  0.0, -1.0],  # Gate 3: Center
        [ -1.5, -1.5, -1.0],  # Gate 4: Lower left
        [ -3.0,  0.0, -1.0],  # Gate 5: Far left
        [ -1.5,  1.5, -1.0],  # Gate 6: Upper left
        [  0.0,  0.0, -1.0],  # Gate 7: Return to center
    ], dtype=np.float64)
    gate_yaw = np.array([0, -1, 0, 1, 2, -1, 2, 1], dtype=np.float64) * np.pi / 2
    x_bounds = np.array([-4, 4], dtype=np.float64)
    y_bounds = np.array([-2.5, 2.5], dtype=np.float64)
    z_bounds = np.array([-2.5, 0.5], dtype=np.float64)
    starting_pos = np.array([0.0, -1.5, -1.0], dtype=np.float64)


class CircleGates(GateConfig):
    """Circular gate configuration.

    Gates lifted to z=-1.0 NED (1m altitude) so the 1.5m gate cube clears
    the floor; previously z=0 placed half the gate below ground.
    """
    gate_pos = np.array([
        [ 0.0, -1.5, -1.0],
        [ 1.5,  0.0, -1.0],
        [ 0.0,  1.5, -1.0],
        [-1.5,  0.0, -1.0]
    ], dtype=np.float64)
    gate_yaw = np.array([0, 1, 2, 3], dtype=np.float64) * np.pi / 2
    x_bounds = np.array([-3, 3], dtype=np.float64)
    y_bounds = np.array([-3, 3], dtype=np.float64)
    z_bounds = np.array([-2.5, 0.5], dtype=np.float64)
    starting_pos = np.array([-1.5, -1.5, -1.0], dtype=np.float64)


class SlalomGates(GateConfig):
    """Slalom gate configuration.

    Gates at z=-1.0 NED (1m altitude) so the 1.5m gate cube clears the floor.
    """
    num_gates = 100
    gate_pos = np.array([[x, (i % 2) * (1 if i % 4 == 1 else -1), -1.0]
                         for i, x in enumerate(range(0, num_gates*2, 2))],
                        dtype=np.float64)
    gate_yaw = np.tile([1, 0, -1, 0], num_gates) * np.pi / 2
    x_bounds = np.array([-2, num_gates*2+2], dtype=np.float64)
    y_bounds = np.array([-3, 3], dtype=np.float64)
    z_bounds = np.array([-2.5, 0.5], dtype=np.float64)
    starting_pos = np.array([0, -1, -1.0], dtype=np.float64)
    periodic = False  # Slalom is a one-way linear path


class BackAndForthGates(GateConfig):
    """Back and forth gate configuration.

    Gates at z=-1.0 NED (1m altitude) so the 1.5m gate cube clears the floor.
    """
    gate_pos = np.array([
        [ 2.0,  0.0, -1.0],
        [ 8.0,  0.0, -1.0],
        [ 8.0,  0.0, -1.0],
        [ 2.0,  0.0, -1.0],
    ], dtype=np.float64)
    gate_yaw = np.array([0, 0, 2, 2], dtype=np.float64) * np.pi / 2
    x_bounds = np.array([-1, 11], dtype=np.float64)
    y_bounds = np.array([-1, 1], dtype=np.float64)
    z_bounds = np.array([-2.5, 0.5], dtype=np.float64)
    starting_pos = np.array([0.0, 0.0, -1.0], dtype=np.float64)


# Dictionary mapping gate configuration names to classes
GATE_CONFIGS = {
    'figure8': Figure8Gates,
    'circle': CircleGates,
    'slalom': SlalomGates,
    'backandforth': BackAndForthGates
}
