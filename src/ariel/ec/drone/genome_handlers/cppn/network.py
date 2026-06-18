"""CPPN network data structures: nodes, connections, activation functions."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np


class ActivationFunction(Enum):
    """Activation functions available for CPPN nodes."""
    IDENTITY = "identity"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SIN = "sin"
    COS = "cos"
    GAUSSIAN = "gaussian"
    ABS = "abs"
    RELU = "relu"
    STEP = "step"


def _identity(x: np.ndarray) -> np.ndarray:
    return x

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def _sin(x: np.ndarray) -> np.ndarray:
    return np.sin(x)

def _cos(x: np.ndarray) -> np.ndarray:
    return np.cos(x)

def _gaussian(x: np.ndarray) -> np.ndarray:
    return np.exp(-x * x / 2.0)

def _abs(x: np.ndarray) -> np.ndarray:
    return np.abs(x)

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _step(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1.0, 0.0)


ACTIVATION_FUNCTIONS: Dict[ActivationFunction, Callable] = {
    ActivationFunction.IDENTITY: _identity,
    ActivationFunction.SIGMOID: _sigmoid,
    ActivationFunction.TANH: _tanh,
    ActivationFunction.SIN: _sin,
    ActivationFunction.COS: _cos,
    ActivationFunction.GAUSSIAN: _gaussian,
    ActivationFunction.ABS: _abs,
    ActivationFunction.RELU: _relu,
    ActivationFunction.STEP: _step,
}


def apply_activation(activation: ActivationFunction, x: np.ndarray) -> np.ndarray:
    """Apply an activation function to an array."""
    return ACTIVATION_FUNCTIONS[activation](x)


class NodeType(Enum):
    """Types of nodes in a CPPN."""
    INPUT = "input"
    OUTPUT = "output"
    HIDDEN = "hidden"


@dataclass
class NodeGene:
    """A node in the CPPN graph."""
    node_id: int
    node_type: NodeType
    activation: ActivationFunction
    bias: float = 0.0
    output_index: Optional[int] = None  # For output nodes: index in output array
    input_label: Optional[str] = None   # For input nodes: descriptive label


@dataclass
class ConnectionGene:
    """A connection (edge) in the CPPN graph."""
    innovation_number: int
    source_id: int
    target_id: int
    weight: float
    enabled: bool = True


@dataclass
class CPPNNetwork:
    """A CPPN represented as a directed acyclic graph of nodes and connections."""
    nodes: Dict[int, NodeGene] = field(default_factory=dict)
    connections: Dict[int, ConnectionGene] = field(default_factory=dict)  # keyed by innovation number
    next_node_id: int = 0

    def copy(self) -> CPPNNetwork:
        """Deep copy of the network."""
        return copy.deepcopy(self)

    def get_input_nodes(self) -> List[NodeGene]:
        """Return input nodes sorted by node_id."""
        return sorted(
            [n for n in self.nodes.values() if n.node_type == NodeType.INPUT],
            key=lambda n: n.node_id,
        )

    def get_output_nodes(self) -> List[NodeGene]:
        """Return output nodes sorted by output_index."""
        return sorted(
            [n for n in self.nodes.values() if n.node_type == NodeType.OUTPUT],
            key=lambda n: (n.output_index if n.output_index is not None else 0),
        )

    def get_hidden_nodes(self) -> List[NodeGene]:
        """Return hidden nodes sorted by node_id."""
        return sorted(
            [n for n in self.nodes.values() if n.node_type == NodeType.HIDDEN],
            key=lambda n: n.node_id,
        )

    def get_enabled_connections(self) -> List[ConnectionGene]:
        """Return all enabled connections."""
        return [c for c in self.connections.values() if c.enabled]
