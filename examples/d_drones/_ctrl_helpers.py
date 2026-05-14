"""Shared helpers for drone simulation and controller-tuning examples.

Extracted from airevolve's tuning example so that examples 4 (simulate) and
5 (tune) can import a common drone configuration without needing sys.path
hacks between sibling directories.
"""

from __future__ import annotations

import numpy as np

from airevolve.simulator.simulation import DroneInterface

# 2-inch quadrotor constants (60 mm arm, 2-inch propellers, ~0.08 kg total)
ARM_LENGTH: float = 0.06   # metres
PROP_SIZE: int = 2         # inches


def create_2inch_quad() -> DroneInterface:
    """Instantiate a standard 2-inch X-configuration quadrotor."""
    propellers = [
        {"loc": [ ARM_LENGTH,  ARM_LENGTH, 0], "dir": [0, 0, -1, "ccw"], "propsize": PROP_SIZE},
        {"loc": [-ARM_LENGTH,  ARM_LENGTH, 0], "dir": [0, 0, -1, "cw"],  "propsize": PROP_SIZE},
        {"loc": [-ARM_LENGTH, -ARM_LENGTH, 0], "dir": [0, 0, -1, "ccw"], "propsize": PROP_SIZE},
        {"loc": [ ARM_LENGTH, -ARM_LENGTH, 0], "dir": [0, 0, -1, "cw"],  "propsize": PROP_SIZE},
    ]
    return DroneInterface(0, propellers=propellers)


class GateChecker:
    """Detect when the drone passes through a sequence of gates.

    A gate pass is recorded when the drone crosses the gate plane within the
    gate's half-size radius (measured in the gate's local frame).
    """

    def __init__(
        self,
        gate_pos: np.ndarray,
        gate_yaw: np.ndarray,
        gate_size: float = 1.0,
        max_gate_distance: float = 10.0,
    ) -> None:
        self.gate_pos = np.asarray(gate_pos)
        self.gate_yaw = np.asarray(gate_yaw)
        self.gate_size = gate_size
        self.max_gate_distance = max_gate_distance
        self.num_gates = len(gate_pos)
        self.reset()

    def reset(self) -> None:
        self.gates_passed = 0
        self._next_gate = 0
        self._prev_signed_dist: float | None = None

    def check_gate_passing(self, pos: np.ndarray) -> bool:
        """Return True if *pos* has just crossed the next gate."""
        if self._next_gate >= self.num_gates:
            return False

        gate_p = self.gate_pos[self._next_gate]
        gate_y = self.gate_yaw[self._next_gate]
        normal = np.array([np.cos(gate_y), np.sin(gate_y), 0.0])

        signed_dist = float(np.dot(pos - gate_p, normal))

        crossed = (
            self._prev_signed_dist is not None
            and self._prev_signed_dist < 0.0
            and signed_dist >= 0.0
        )
        self._prev_signed_dist = signed_dist

        if crossed:
            lateral_err = np.linalg.norm(pos[:2] - gate_p[:2]
                                         - signed_dist * normal[:2])
            if lateral_err <= self.gate_size / 2.0:
                self.gates_passed += 1
                self._next_gate = (self._next_gate + 1) % self.num_gates
                self._prev_signed_dist = None
                return True
        return False

    def get_normalized_distance_to_next_gate(self, pos: np.ndarray) -> float:
        """Return a [0,1] proximity bonus to the next gate (1 = at gate, 0 = far)."""
        if self._next_gate >= self.num_gates:
            return 0.0
        distance = float(np.linalg.norm(pos - self.gate_pos[self._next_gate]))
        return max(0.0, min(1.0, 1.0 - distance / self.max_gate_distance))
