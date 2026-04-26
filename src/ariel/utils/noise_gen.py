"""Module for generating Perlin noise using NumPy."""

# Standard library
from dataclasses import dataclass
from typing import Literal

# Third-party libraries
import numpy as np

# Local libraries
from ariel.parameters.ariel_types import (
    ND_FLOAT_PRECISION,
    ND_INT_PRECISION,
    FloatArray,
    IntArray,
)

# Type Aliases
type NormMethod = Literal["linear", "clip", "none"]


@dataclass
class PerlinNoise:
    """
    Vectorized Perlin noise (2D) using NumPy only.

    Examples
    --------
    >>> noise = PerlinNoise(seed=1234)
    >>> width, height = 256, 256
    >>> scale = 50.0
    >>> grid = noise.as_grid(width, height, scale=scale, normalize=True)
    """

    seed: int | None = None
    norm_methods: NormMethod = "linear"

    def __post_init__(self) -> None:
        """Initialize the Perlin noise generator."""
        # Permutation table (size 256, repeated) for hashing lattice coordinates
        rng = np.random.default_rng(self.seed)
        p = np.arange(256)
        rng.shuffle(p)

        # Duplicate list: len == 512 for safe wrap
        self._perm = np.concatenate([p, p])

        # 8 gradient directions (unit vectors)
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)

        # Gradient lookup table
        self._grad_lut = np.stack(
            [np.cos(angles), np.sin(angles)],
            axis=-1,
        )

    @staticmethod
    def _fade(t: FloatArray) -> FloatArray:
        # 6t^5 - 15t^4 + 10t^3 (smoothstep^3), stable & vectorized
        #   https://www.wikiwand.com/en/articles/Smoothstep
        return np.array(((6 * t - 15) * t + 10) * (t * t * t)).astype(
            ND_FLOAT_PRECISION,
        )

    @staticmethod
    def _lerp(a: FloatArray, b: FloatArray, t: FloatArray) -> FloatArray:
        return (a + t * (b - a)).astype(ND_FLOAT_PRECISION)

    def _hash2(self, xi: IntArray, yi: IntArray) -> IntArray:
        """
        Hash (xi, yi) -> [0, 255] via permutation table, vectorized.

        Parameters
        ----------
        xi
            x-coordinate of the lattice point
        yi
            y-coordinate of the lattice point

        Returns
        -------
            array with shape xi.shape + (2,)
        """
        # Ensure non-negative and wrap
        xi &= 255
        yi &= 255
        return self._perm[(self._perm[xi] + yi) & 255]

    def _grad_at_corner(self, xi: IntArray, yi: IntArray) -> FloatArray:
        """
        Lookup 2D unit gradient vector at integer lattice corner (xi, yi).

        Parameters
        ----------
        xi
            x-coordinate of the lattice point
        yi
            y-coordinate of the lattice point

        Returns
        -------
            array with shape xi.shape + (2,)
        """
        h = self._hash2(xi, yi) % self._grad_lut.shape[0]
        return self._grad_lut[h]

    def as_grid(  # noqa: D102, PLR0914
        self,
        width: int,
        height: int,
        scale: float = 64.0,
        normalize: NormMethod = "none",
    ) -> FloatArray:
        """
        Generate a 2D grid of Perlin noise.

        Parameters
        ----------
        width
            Width of the output grid in pixels.
        height
            Height of the output grid in pixels.
        scale
            Scale factor for the noise frequency. Larger values result in
            smoother noise.
        normalize
            Normalization method for the output values:
            - "linear": Scale to [0, 1] linearly.
            - "clip": Clip values to [0, 1].
            - "none": No normalization.

        Returns
        -------
            2D array of shape (height, width) with Perlin noise values.

        Raises
        ------
        ValueError
            If width, height, or scale are not positive.
        """
        # Check that width height, scale are valid
        for arg in (width, height, scale):
            if arg <= 0:
                msg = f"{arg} must be > 0"
                raise ValueError(msg)

        # Pixel coordinate grid
        xs = np.arange(width)
        ys = np.arange(height)
        x_arr, y_arr = np.meshgrid(xs, ys, indexing="xy")

        # Noise-space coordinates
        x = x_arr / scale
        y = y_arr / scale

        # Integer lattice corners
        xi = np.floor(x).astype(ND_INT_PRECISION)
        yi = np.floor(y).astype(ND_INT_PRECISION)

        # Local offsets inside cell
        xf = x - xi
        yf = y - yi

        # Gradients at 4 corners (broadcasts to HxW x 2)
        g00 = self._grad_at_corner(xi, yi)
        g10 = self._grad_at_corner(xi + 1, yi)
        g01 = self._grad_at_corner(xi, yi + 1)
        g11 = self._grad_at_corner(xi + 1, yi + 1)

        # Offset vectors to corners (HxW x 2)
        d00 = np.stack([xf, yf], axis=-1)
        d10 = np.stack([xf - 1.0, yf], axis=-1)
        d01 = np.stack([xf, yf - 1.0], axis=-1)
        d11 = np.stack([xf - 1.0, yf - 1.0], axis=-1)

        # Dot products (HxW)
        n00 = np.sum(g00 * d00, axis=-1)
        n10 = np.sum(g10 * d10, axis=-1)
        n01 = np.sum(g01 * d01, axis=-1)
        n11 = np.sum(g11 * d11, axis=-1)

        # Smooth interpolation
        u = self._fade(xf)
        v = self._fade(yf)

        nx0 = self._lerp(n00, n10, u)  # along x at y0
        nx1 = self._lerp(n01, n11, u)  # along x at y1
        nxy = self._lerp(nx0, nx1, v)  # along y

        # Match normalization method
        match normalize:
            case "linear":
                nxy = nxy * 0.5 + 0.5
            case "clip":
                nxy = np.clip(nxy, 0, 1.0)
            case "none":
                pass
        return nxy.astype(ND_FLOAT_PRECISION)
