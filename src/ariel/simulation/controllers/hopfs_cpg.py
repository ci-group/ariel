"""Hopf-based Central Pattern Generator (CPG).

Sources
-------
    [1] https://www.sciencedirect.com/science/article/pii/S2667379722000353#sec4

Notes
-----
    *

Todo
----
    [ ]

"""

# Third-party libraries
import numpy as np
import numpy.typing as npt
from rich.progress import track

# Type Aliases
type ArrayLike = npt.NDArray[np.float64]

# Global constants
SEED = 42

# Global functions
RNG = np.random.default_rng(seed=SEED)


class HopfCPG:
    def __init__(
        self,
        num_neurons: int,
        adjacency_list: dict[int, list[int]],
        dt: float = 0.02,
        h: float = 0.1,
        alpha: float = 1.0,
    ) -> None:
        # --- Inherent parameters --- #
        # Number of neurons
        self.num_neurons = num_neurons

        # Time step
        self.dt = dt

        # Learning rate
        self.alpha = np.ones(num_neurons) * alpha

        # Coupling coefficient
        self.h = h

        # Adjacency list for coupling
        self.adjacency_list = adjacency_list
        if len(adjacency_list) != num_neurons:
            raise ValueError(
                "Adjacency list length must match number of neurons."
            )

        # --- Initialize state variables --- #
        self.init_state = 0.5
        self.x: ArrayLike = RNG.uniform(
            -self.init_state, self.init_state, self.num_neurons
        )
        self.y: ArrayLike = RNG.uniform(
            -self.init_state, self.init_state, self.num_neurons
        )

        # --- Adjustable parameters --- #
        # Angular frequency (1Hz default)
        self.omega = np.ones(num_neurons) * 2 * np.pi

        # Amplitude
        self.A = np.ones(num_neurons) * 1.0

        # Phase differences
        self.phase_diff: ArrayLike = np.zeros((num_neurons, num_neurons))

        # Set up ring topology phase differences (neighbors are π/2 apart)
        for i, conn in adjacency_list.items():
            for j in conn:
                self.phase_diff[i, j] = np.pi / 2

    def reset(self) -> None:
        # Reset state variables
        self.x = RNG.uniform(
            -self.init_state, self.init_state, self.num_neurons
        )
        self.y = RNG.uniform(
            -self.init_state, self.init_state, self.num_neurons
        )
        self.omega = np.ones(self.num_neurons) * 2 * np.pi
        self.A = np.ones(self.num_neurons) * 1.0

    def step(
        self,
    ) -> tuple[ArrayLike, ArrayLike]:
        # Define new state variables
        new_x = np.zeros_like(self.x)
        new_y = np.zeros_like(self.y)

        # Update states using Euler integration
        for i, conn in self.adjacency_list.items():
            r_squared = self.x[i] ** 2 + self.y[i] ** 2

            # Hopf oscillator dynamics
            x_dot = (
                self.alpha[i] * (self.A[i] ** 2 - r_squared) * self.x[i]
                - self.omega[i] * self.y[i]
            )
            y_dot = (
                self.omega[i] * self.x[i]
                + self.alpha[i] * (self.A[i] ** 2 - r_squared) * self.y[i]
            )

            # Coupling terms from neighboring oscillators
            for j in conn:
                # Rotation matrix for phase difference
                rot_matrix = np.array([
                    [
                        np.cos(self.phase_diff[i, j]),
                        -np.sin(self.phase_diff[i, j]),
                    ],
                    [
                        np.sin(self.phase_diff[i, j]),
                        np.cos(self.phase_diff[i, j]),
                    ],
                ])

                # Apply coupling
                coupled = self.h * rot_matrix @ np.array([self.x[j], self.y[j]])
                x_dot += coupled[0]
                y_dot += coupled[1]

            # Update states using Euler integration
            new_x[i] = self.x[i] + x_dot * self.dt
            new_y[i] = self.y[i] + y_dot * self.dt

        self.x, self.y = new_x, new_y
        return self.x.copy(), self.y.copy()

    def simulate(
        self,
        steps: int,
    ) -> tuple[ArrayLike, ArrayLike]:
        # Main simulation
        x_history: ArrayLike = np.zeros((steps, self.num_neurons))
        y_history: ArrayLike = np.zeros((steps, self.num_neurons))

        for t in track(range(steps)):
            x, y = self.step()
            x_history[t] = x
            y_history[t] = y

        return x_history, y_history
