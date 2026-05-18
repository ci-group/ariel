# Genome handlers for different coordinate representations

from .base import GenomeHandler
from .cartesian_euler_genome_handler import CartesianEulerDroneGenomeHandler
from .spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler
from .cppn_neat_genome_handler import CPPNNeatDroneGenomeHandler
from .hybrid_cppn_genome_handler import HybridCPPNDroneGenomeHandler

# Optional MLP handler (requires PyTorch)
try:
    from .mlp_genome_handler import MLPDroneGenomeHandler
    MLP_AVAILABLE = True
except ImportError:
    MLP_AVAILABLE = False
    MLPDroneGenomeHandler = None

__all__ = [
    'GenomeHandler',
    'CartesianEulerDroneGenomeHandler',
    'SphericalAngularDroneGenomeHandler',
    'CPPNNeatDroneGenomeHandler',
    'HybridCPPNDroneGenomeHandler',
]

if MLP_AVAILABLE:
    __all__.append('MLPDroneGenomeHandler')