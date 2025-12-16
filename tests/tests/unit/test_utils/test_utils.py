"""Test: initialization of utility classes."""

# Standard library
# Third-party libraries
# Local libraries
from ariel.utils.noise_gen import PerlinNoise

# Global constants
# Global functions
# Warning Control
# Type Checking
# Type Aliases


def test_perlin_noise_initialization() -> None:
    """Simply instantiate the PerlinNoise class."""
    _ = PerlinNoise()
