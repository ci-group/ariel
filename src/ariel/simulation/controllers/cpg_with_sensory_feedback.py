"""CPG model with attached sensory feedback.

Sources:
-------
    [1] DOI: 10.5772/59186

Notes
-----
    *

Todo
----
    [ ] Add targeted locomotion

"""

from typing import Any

import numpy as np
from rich.progress import track

SEED = 42
RNG = np.random.default_rng(seed=SEED)


class CPGSensoryFeedback:
    def __init__(
        self,
        num_neurons: int,
        dt: float = 0.002,
        coupling_weights: np.ndarray | None = None,
        sensory_term: float = 0.0,
        _lambda: float = 0.1,
    ) -> None:
        self.num_neurons = num_neurons
        self.dt = dt
        self._lambda = _lambda
        self.sensory_term = sensory_term

        # State variables
        self.x = np.zeros(self.num_neurons)
        self.y = np.zeros(self.num_neurons)

        # Hyper-parameters
        self.omega = np.ones(self.num_neurons)
        self.amplitude = np.ones(self.num_neurons)

        # Coupling matrix:
        if coupling_weights is not None:
            self.c = coupling_weights
        else:
            self.c = np.zeros((num_neurons, num_neurons))

    def step(
        self,
    ) -> tuple[
        np.ndarray[tuple[int, ...], np.dtype[Any]],
        np.ndarray[tuple[int, ...], np.dtype[Any]],
    ]:
        # Useful simplification
        r_squared = self.x**2 + self.y**2

        # Core Hopf dynamics (without coupling)
        dx = -self.omega * self.y + self.x * (self.amplitude**2 - r_squared)
        dy = self.omega * self.x + self.y * (self.amplitude**2 - r_squared)

        # Coupling terms using the single matrix for both dx and dy
        dx += self.c @ self.y + self.sensory_term
        dy += self.c @ self.x - self.sensory_term

        # Actual update
        self.x += self.dt * dx
        self.y += self.dt * dy

        return self.x.copy(), self.y.copy()

    def reset(self) -> None:
        self.x = RNG.uniform(0.01, 0.01, self.num_neurons)
        self.y = RNG.uniform(0.01, 0.01, self.num_neurons)

    def simulate(
        self,
        steps: int,
    ) -> tuple[
        np.ndarray[tuple[int, ...], np.dtype[Any]],
        np.ndarray[tuple[int, ...], np.dtype[Any]],
    ]:
        # Main simulation
        x_history = np.zeros((steps, self.num_neurons))
        y_history = np.zeros((steps, self.num_neurons))

        for t in track(range(steps)):
            x, y = self.step()
            x_history[t] = x
            y_history[t] = y

        return x_history, y_history
