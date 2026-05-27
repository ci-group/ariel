"""Genome handlers for different coordinate representations.

This package intentionally avoids eager imports of every handler module.
Some handlers pull in optional heavy dependencies (for example collision
libraries), and importing them unconditionally breaks light-weight use
cases that only need helper modules under this package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import GenomeHandler

if TYPE_CHECKING:
    from .cartesian_euler_genome_handler import CartesianEulerDroneGenomeHandler
    from .cppn_neat_genome_handler import CPPNNeatDroneGenomeHandler
    from .hybrid_cppn_genome_handler import HybridCPPNDroneGenomeHandler
    from .mlp_genome_handler import MLPDroneGenomeHandler
    from .spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler

__all__ = [
    "GenomeHandler",
    "CartesianEulerDroneGenomeHandler",
    "SphericalAngularDroneGenomeHandler",
    "CPPNNeatDroneGenomeHandler",
    "HybridCPPNDroneGenomeHandler",
    "MLPDroneGenomeHandler",
]


def __getattr__(name: str) -> Any:
    if name == "CartesianEulerDroneGenomeHandler":
        from .cartesian_euler_genome_handler import CartesianEulerDroneGenomeHandler
        return CartesianEulerDroneGenomeHandler
    if name == "SphericalAngularDroneGenomeHandler":
        from .spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler
        return SphericalAngularDroneGenomeHandler
    if name == "CPPNNeatDroneGenomeHandler":
        from .cppn_neat_genome_handler import CPPNNeatDroneGenomeHandler
        return CPPNNeatDroneGenomeHandler
    if name == "HybridCPPNDroneGenomeHandler":
        from .hybrid_cppn_genome_handler import HybridCPPNDroneGenomeHandler
        return HybridCPPNDroneGenomeHandler
    if name == "MLPDroneGenomeHandler":
        from .mlp_genome_handler import MLPDroneGenomeHandler
        return MLPDroneGenomeHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")