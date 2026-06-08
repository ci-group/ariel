"""Pure-numpy inference engine for a trained SimpleCPG.

Load a `.npz` file exported by `ariel.hardware.export_cpg.export_simple_cpg`
and run the Hopf-oscillator forward pass without torch.

Typical usage on the robot
--------------------------
    from ariel.hardware.cpg_inference import SimpleCPGInference

    cpg = SimpleCPGInference.load("spider_cpg.npz")
    angles = cpg.forward(elapsed)   # np.ndarray of shape (n,)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_E = 1e-9  # small epsilon to avoid division by zero in atan2


@dataclass
class SimpleCPGInference:
    """Numpy inference-only clone of SimpleCPG.

    Parameters mirror the trained SimpleCPG so saved weights can be loaded
    directly.  Only the forward pass is implemented; no gradient machinery.

    Parameters
    ----------
    n : int
        Number of CPG oscillators.
    adjacency_dict : dict[int, list[int]]
        Connectivity graph.  Key i → list of neighbour indices.
    phase : np.ndarray, shape (n,)
    w : np.ndarray, shape (n,)
    amplitudes : np.ndarray, shape (n,)
    ha : np.ndarray, shape (n,)
    b : np.ndarray, shape (n,)
    x0 : np.ndarray, shape (n,)
        Initial x state (copied from the training-time buffer snapshot).
    y0 : np.ndarray, shape (n,)
        Initial y state.
    mu : float
    dt : float
    hard_bounds : tuple[float, float] | None
    """

    n: int
    adjacency_dict: dict[int, list[int]]
    phase: np.ndarray
    w: np.ndarray
    amplitudes: np.ndarray
    ha: np.ndarray
    b: np.ndarray
    x0: np.ndarray
    y0: np.ndarray
    mu: float = 1.0
    dt: float = 0.01
    hard_bounds: tuple[float, float] | None = (-np.pi / 2, np.pi / 2)

    # mutable run-time state — populated by __post_init__
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.x = self.x0.copy()
        self.y = self.y0.copy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save weights and metadata to a `.npz` file."""
        path = Path(path)
        adjacency_json = json.dumps(
            {str(k): v for k, v in self.adjacency_dict.items()}
        )
        np.savez(
            path,
            phase=self.phase,
            w=self.w,
            amplitudes=self.amplitudes,
            ha=self.ha,
            b=self.b,
            x0=self.x0,
            y0=self.y0,
            mu=np.array(self.mu),
            dt=np.array(self.dt),
            hard_bounds=np.array(self.hard_bounds)
            if self.hard_bounds is not None
            else np.array([np.nan, np.nan]),
            has_hard_bounds=np.array(self.hard_bounds is not None),
            adjacency_json=np.array(adjacency_json),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SimpleCPGInference":
        """Load from a `.npz` file written by :meth:`save` or
        :func:`ariel.hardware.export_cpg.export_simple_cpg`."""
        path = Path(path)
        data = np.load(path, allow_pickle=False)

        raw = json.loads(str(data["adjacency_json"]))
        adjacency_dict: dict[int, list[int]] = {int(k): v for k, v in raw.items()}

        hard_bounds: tuple[float, float] | None
        if bool(data["has_hard_bounds"]):
            hb = data["hard_bounds"]
            hard_bounds = (float(hb[0]), float(hb[1]))
        else:
            hard_bounds = None

        return cls(
            n=len(adjacency_dict),
            adjacency_dict=adjacency_dict,
            phase=data["phase"].astype(np.float32),
            w=data["w"].astype(np.float32),
            amplitudes=data["amplitudes"].astype(np.float32),
            ha=data["ha"].astype(np.float32),
            b=data["b"].astype(np.float32),
            x0=data["x0"].astype(np.float32),
            y0=data["y0"].astype(np.float32),
            mu=float(data["mu"]),
            dt=float(data["dt"]),
            hard_bounds=hard_bounds,
        )

    # ------------------------------------------------------------------
    # Run-time interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal oscillator state to the initial snapshot."""
        self.x = self.x0.copy()
        self.y = self.y0.copy()

    def forward(self, time: float | None = None) -> np.ndarray:
        """Step the CPG and return joint angles.

        Parameters
        ----------
        time : float | None
            Current elapsed time.  Passing ``0.0`` resets the oscillator.

        Returns
        -------
        np.ndarray, shape (n,)
            Joint angles in radians.
        """
        if time is not None and np.isclose(time, 0.0):
            self.reset()

        r = np.sqrt(self.x**2 + self.y**2 + _E)
        dx = np.zeros(self.n, dtype=np.float32)
        dy = np.zeros(self.n, dtype=np.float32)

        for i in range(self.n):
            dx[i] = (self.mu - r[i] ** 2) * self.x[i] - self.w[i] * self.y[i]
            dy[i] = (self.mu - r[i] ** 2) * self.y[i] + self.w[i] * self.x[i]

            cs = self.ha[i]
            for j in self.adjacency_dict[i]:
                phase_diff = np.arctan2(self.y[j], self.x[j] + _E) - np.arctan2(
                    self.y[i], self.x[i] + _E
                )
                sin_pd = np.sin(phase_diff)
                dx[i] += cs * sin_pd * (self.x[j] - self.x[i])
                dy[i] += cs * sin_pd * (self.y[j] - self.y[i])

        self.x = self.x + dx * self.dt
        self.y = self.y + dy * self.dt

        angles = self.amplitudes * self.y + self.phase + self.b

        if self.hard_bounds is not None:
            angles = np.clip(angles, self.hard_bounds[0], self.hard_bounds[1])

        if np.isnan(angles).any():
            raise ValueError(f"NaN detected in CPG angles: {angles}")

        return angles.copy()
