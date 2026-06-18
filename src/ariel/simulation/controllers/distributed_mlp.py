"""Distributed and standard MLP brain architectures for morphology-agnostic control.

``DistributedMLP`` uses a single shared MLP across all actuator nodes.  Each
node receives its own local observation (self + K zero-padded face-neighbours)
so the parameter count depends only on K and the feature size — not on
morphology size.  This lets the same θ transfer between any two morphologies.

``StandardMLP`` is the centralised baseline: it takes the full robot state as a
single vector, so its parameter count scales with the number of actuators.

Both classes expose ``set_theta`` / ``get_theta`` for use with black-box
optimisers (e.g. ``CMAESLearner``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class NodeObservation:
    """Per-node observation passed to ``DistributedMLP``.

    Parameters
    ----------
    vel_x:
        X-axis velocity of the node (or its representative point).
    vel_y:
        Y-axis velocity of the node (or its representative point).
    state:
        Normalised scalar state — volume fraction for EvoGym voxels,
        joint angle (normalised by π/2) for ARIEL hinge modules.
    type_onehot:
        5-element one-hot encoding of the module/voxel type.
    """

    vel_x: float
    vel_y: float
    state: float
    type_onehot: np.ndarray  # shape (5,)

    def to_vector(self) -> np.ndarray:
        return np.array(
            [self.vel_x, self.vel_y, self.state, *self.type_onehot],
            dtype=np.float32,
        )


EMPTY_NODE = NodeObservation(0.0, 0.0, 0.0, np.zeros(5, dtype=np.float32))


class DistributedMLP:
    """Single MLP shared across all actuator nodes.

    Each actuator feeds its own local observation (self + K zero-padded
    neighbours) through the same weight vector θ.  Because θ size depends only
    on K and the feature dimension (not morphology size), the same θ transfers
    between any two morphologies in the same domain.

    Parameters
    ----------
    n_neighbors:
        Number of neighbour slots (K).  Must match the topology used when
        building node inputs (e.g. 8 for EvoGym Moore neighbourhood, 6 for
        ARIEL face-based neighbourhood).
    features_per_node:
        Dimensionality of each ``NodeObservation.to_vector()`` output.
    hidden:
        Hidden layer width.
    """

    def __init__(
        self,
        n_neighbors: int,
        features_per_node: int = 8,
        hidden: int = 32,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.features_per_node = features_per_node
        self.hidden = hidden
        input_size = (1 + n_neighbors) * features_per_node + 1  # +1 for time signal
        self._net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self._net.parameters())

    def get_theta(self) -> np.ndarray:
        return np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in self._net.parameters()]
        ).astype(np.float64)

    def set_theta(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=np.float32)
        offset = 0
        for p in self._net.parameters():
            size = p.numel()
            p.data.copy_(
                torch.from_numpy(theta[offset : offset + size].reshape(p.shape))
            )
            offset += size

    def forward_single(
        self,
        self_obs: NodeObservation,
        neighbor_obs: list[NodeObservation],
        time_signal: float,
    ) -> float:
        """Return raw tanh output in [-1, 1] for one actuator node."""
        padded = list(neighbor_obs) + [EMPTY_NODE] * (self.n_neighbors - len(neighbor_obs))
        padded = padded[: self.n_neighbors]

        parts = [self_obs.to_vector()]
        for nb in padded:
            parts.append(nb.to_vector())
        parts.append(np.array([time_signal], dtype=np.float32))

        x = np.concatenate(parts)
        t = torch.from_numpy(x).unsqueeze(0)
        with torch.no_grad():
            out = self._net(t)
        return float(out.squeeze())

    def forward_all(
        self,
        node_inputs: list[tuple[NodeObservation, list[NodeObservation]]],
        time_signal: float,
    ) -> np.ndarray:
        """Return shape (n_actuators,) array of tanh outputs in [-1, 1]."""
        return np.array(
            [self.forward_single(s, nb, time_signal) for s, nb in node_inputs],
            dtype=np.float32,
        )


class StandardMLP:
    """Centralised MLP baseline taking full robot state as input.

    Input size is ``2 * n_actuators + 9`` (joint pos, joint vel, body state).
    Output size is ``n_actuators``, passed through Tanh.  θ size scales with
    morphology, so cross-morphology transfer is ill-defined — this is the
    baseline used to motivate the distributed architecture.

    Parameters
    ----------
    n_actuators:
        Number of actuated joints / voxels (``nu``).
    """

    def __init__(self, n_actuators: int) -> None:
        self.n_actuators = n_actuators
        input_size = 2 * n_actuators + 9
        self._net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, n_actuators),
            nn.Tanh(),
        )

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self._net.parameters())

    def get_theta(self) -> np.ndarray:
        return np.concatenate(
            [p.detach().cpu().numpy().ravel() for p in self._net.parameters()]
        ).astype(np.float64)

    def set_theta(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=np.float32)
        offset = 0
        for p in self._net.parameters():
            size = p.numel()
            p.data.copy_(
                torch.from_numpy(theta[offset : offset + size].reshape(p.shape))
            )
            offset += size

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Return shape (n_actuators,) array in [-1, 1].

        Parameters
        ----------
        obs:
            Shape ``(2 * n_actuators + 9,)`` flat state vector.
        """
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = self._net(x)
        return out.squeeze(0).numpy()
