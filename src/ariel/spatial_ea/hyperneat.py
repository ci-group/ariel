"""HyperNEAT controllers for the spatial evolutionary algorithm.

Implements Compositional Pattern Producing Networks (CPPNs) and the
substrate-based ANN they generate. A CPPN is queried with the spatial
coordinates of a source and a target neuron and returns the weight of the
connection between them, so a small evolved network indirectly encodes a much
larger controller.

Notes
-----
    * Genomes are plain dictionaries with ``nodes`` and ``connections`` keys so
      they stay JSON/NPZ serialisable without a custom encoder.

References
----------
    [1] Stanley, K. O., D'Ambrosio, D. B., & Gauci, J. (2009). A
        Hypercube-Based Encoding for Evolving Large-Scale Neural Networks.
        Artificial Life, 15(2), 185-212.

"""

# Standard library
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Third-party libraries
import numpy as np

# Local libraries
from ariel.parameters.ariel_types import FloatArray

# Type Aliases
type Coordinate = tuple[float, ...]
type CPPNGenome = dict[str, Any]
type ActivationFunction = Callable[
    [FloatArray],
    FloatArray,
]

# Global constants
SIGMOID_CLIP = 500.0


# -- Activation functions ------------------------------------------------------
def sigmoid(x: FloatArray) -> FloatArray:
    """Logistic activation function.

    Parameters
    ----------
    x
        Input values.

    Returns
    -------
        The logistic sigmoid of ``x``, clipped to avoid overflow.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -SIGMOID_CLIP, SIGMOID_CLIP)))


def tanh_activation(x: FloatArray) -> FloatArray:
    """Hyperbolic tangent activation function.

    Parameters
    ----------
    x
        Input values.

    Returns
    -------
        The hyperbolic tangent of ``x``.
    """
    return np.tanh(x)


def sine(x: FloatArray) -> FloatArray:
    """Sine activation function.

    Parameters
    ----------
    x
        Input values.

    Returns
    -------
        The sine of ``x``.
    """
    return np.sin(x)


def gaussian(x: FloatArray) -> FloatArray:
    """Gaussian activation function.

    Parameters
    ----------
    x
        Input values.

    Returns
    -------
        ``exp(-x ** 2)``.
    """
    return np.exp(-(x**2))


def relu(x: FloatArray) -> FloatArray:
    """Rectified linear activation function.

    Parameters
    ----------
    x
        Input values.

    Returns
    -------
        ``max(0, x)`` element-wise.
    """
    return np.maximum(0, x)


def linear(x: FloatArray) -> FloatArray:
    """Identity activation function.

    Parameters
    ----------
    x
        Input values.

    Returns
    -------
        ``x`` unchanged.
    """
    return x


def abs_activation(x: FloatArray) -> FloatArray:
    """Absolute value activation function.

    Parameters
    ----------
    x
        Input values.

    Returns
    -------
        The absolute value of ``x``.
    """
    return np.abs(x)


ACTIVATION_FUNCTIONS: dict[str, ActivationFunction] = {
    "sigmoid": sigmoid,
    "tanh": tanh_activation,
    "sine": sine,
    "gaussian": gaussian,
    "relu": relu,
    "linear": linear,
    "abs": abs_activation,
}


# -- Genome primitives ---------------------------------------------------------
@dataclass
class CPPNNode:
    """A single node of a CPPN.

    Parameters
    ----------
    node_id
        Unique identifier of the node within its genome.
    activation
        Key into :data:`ACTIVATION_FUNCTIONS`.
    layer
        Feed-forward depth: ``0`` is the input layer, the largest layer value
        in a genome is the output layer.
    """

    node_id: int
    activation: str
    layer: int


@dataclass
class CPPNConnection:
    """A weighted connection between two CPPN nodes.

    Parameters
    ----------
    from_node
        Identifier of the source node.
    to_node
        Identifier of the target node.
    weight
        Connection weight.
    enabled
        Whether the connection participates in evaluation.
    """

    from_node: int
    to_node: int
    weight: float
    enabled: bool = True


# -- Networks ------------------------------------------------------------------
class CPPN:
    """Compositional Pattern Producing Network.

    Maps the concatenated coordinates of two substrate neurons onto the weight
    of the connection between them.
    """

    def __init__(self, genome: CPPNGenome) -> None:
        """Build a CPPN from a genome.

        Parameters
        ----------
        genome
            Mapping with ``nodes`` (list of :class:`CPPNNode`) and
            ``connections`` (list of :class:`CPPNConnection`).
        """
        self.nodes: list[CPPNNode] = genome["nodes"]
        self.connections: list[CPPNConnection] = genome["connections"]

        self.max_layer = max(node.layer for node in self.nodes)
        self.num_inputs = sum(1 for node in self.nodes if node.layer == 0)
        self.num_outputs = sum(
            1 for node in self.nodes if node.layer == self.max_layer
        )

        self.node_activations: dict[int, ActivationFunction] = {
            node.node_id: ACTIVATION_FUNCTIONS[node.activation]
            for node in self.nodes
        }

        self._build_network_structure()

    def _build_network_structure(self) -> None:
        """Index nodes by layer and connections by target for evaluation."""
        self.nodes_by_layer: dict[int, list[CPPNNode]] = {}
        for node in self.nodes:
            self.nodes_by_layer.setdefault(node.layer, []).append(node)

        self.input_nodes = self.nodes_by_layer.get(0, [])
        self.output_nodes = self.nodes_by_layer.get(self.max_layer, [])

        self.incoming_connections: dict[int, list[CPPNConnection]] = {}
        for conn in self.connections:
            if conn.enabled:
                self.incoming_connections.setdefault(conn.to_node, []).append(
                    conn,
                )

    def activate(
        self,
        inputs: FloatArray,
    ) -> FloatArray:
        """Evaluate the CPPN on a coordinate vector.

        Parameters
        ----------
        inputs
            Input values, typically the concatenated coordinates of a source
            and a target substrate neuron.

        Returns
        -------
            The output node values, ordinarily a single connection weight.
        """
        node_values: dict[int, float] = {}

        for i, node in enumerate(self.input_nodes):
            node_values[node.node_id] = (
                float(inputs[i]) if i < len(inputs) else 0.0
            )

        for layer in range(1, self.max_layer + 1):
            for node in self.nodes_by_layer.get(layer, []):
                weighted_sum = 0.0
                for conn in self.incoming_connections.get(node.node_id, []):
                    if conn.from_node in node_values:
                        weighted_sum += (
                            conn.weight * node_values[conn.from_node]
                        )

                activation_fn = self.node_activations[node.node_id]
                node_values[node.node_id] = float(
                    activation_fn(np.array([weighted_sum]))[0],
                )

        return np.array(
            [node_values.get(node.node_id, 0.0) for node in self.output_nodes],
        )


class SubstrateNetwork:
    """Feed-forward network whose weights are painted by a CPPN.

    The substrate fixes where neurons sit in space; the CPPN decides how
    strongly any two of them are connected.
    """

    def __init__(
        self,
        input_coords: list[Coordinate],
        hidden_coords: list[Coordinate] | None,
        output_coords: list[Coordinate],
        cppn: CPPN,
        weight_threshold: float = 0.2,
    ) -> None:
        """Generate the substrate weights from a CPPN.

        Parameters
        ----------
        input_coords
            Spatial coordinates of the input neurons.
        hidden_coords
            Spatial coordinates of the hidden neurons, or ``None`` for a
            directly connected input/output substrate.
        output_coords
            Spatial coordinates of the output neurons.
        cppn
            The CPPN queried for each candidate connection.
        weight_threshold
            Connections whose absolute CPPN weight falls below this value are
            left at zero.
        """
        self.input_coords = input_coords
        self.hidden_coords = hidden_coords or []
        self.output_coords = output_coords
        self.cppn = cppn
        self.weight_threshold = weight_threshold

        self._generate_weights()

    def _query_weights(
        self,
        source_coords: list[Coordinate],
        target_coords: list[Coordinate],
    ) -> FloatArray:
        """Query the CPPN for every source/target coordinate pair.

        Parameters
        ----------
        source_coords
            Coordinates of the presynaptic neurons.
        target_coords
            Coordinates of the postsynaptic neurons.

        Returns
        -------
            Weight matrix of shape ``(len(source_coords), len(target_coords))``
            with sub-threshold entries zeroed out.
        """
        weights = np.zeros((len(source_coords), len(target_coords)))
        for i, source in enumerate(source_coords):
            for j, target in enumerate(target_coords):
                cppn_input = np.array([*source, *target])
                weight = float(self.cppn.activate(cppn_input)[0])
                if abs(weight) >= self.weight_threshold:
                    weights[i, j] = weight
        return weights

    def _generate_weights(self) -> None:
        """Populate the substrate weight matrices from the CPPN."""
        if len(self.hidden_coords) > 0:
            self.weights_input_hidden = self._query_weights(
                self.input_coords,
                self.hidden_coords,
            )
            self.weights_hidden_output = self._query_weights(
                self.hidden_coords,
                self.output_coords,
            )
        else:
            self.weights_input_output = self._query_weights(
                self.input_coords,
                self.output_coords,
            )

    def activate(
        self,
        inputs: FloatArray,
    ) -> FloatArray:
        """Run the substrate network forward.

        Parameters
        ----------
        inputs
            Sensor values, one per input neuron.

        Returns
        -------
            Actuator values, one per output neuron.
        """
        inputs = np.asarray(inputs, dtype=float)

        if len(self.hidden_coords) > 0:
            hidden = np.tanh(np.dot(inputs, self.weights_input_hidden))
            # Linear output, scaled down to keep actuators out of saturation.
            outputs = 0.5 * np.dot(hidden, self.weights_hidden_output)
        else:
            outputs = 0.5 * np.dot(inputs, self.weights_input_output)

        return np.asarray(outputs, dtype=np.float64)


# -- Genome and substrate construction -----------------------------------------
def create_minimal_cppn_genome(
    num_inputs: int = 4,
    num_outputs: int = 1,
    activation: str = "sine",
) -> CPPNGenome:
    """Create a minimal fully connected CPPN genome.

    Parameters
    ----------
    num_inputs
        Number of input nodes, typically four for ``(x1, y1, x2, y2)``.
    num_outputs
        Number of output nodes, typically one connection weight.
    activation
        Activation function assigned to the output nodes.

    Returns
    -------
        A genome with ``nodes`` and ``connections`` entries.
    """
    nodes: list[CPPNNode] = [
        CPPNNode(node_id=i, activation="linear", layer=0)
        for i in range(num_inputs)
    ]

    output_ids = [num_inputs + i for i in range(num_outputs)]
    nodes.extend(
        CPPNNode(node_id=output_id, activation=activation, layer=1)
        for output_id in output_ids
    )

    connections = [
        CPPNConnection(
            from_node=i,
            to_node=output_id,
            weight=float(np.random.randn() * 0.5),
            enabled=True,
        )
        for output_id in output_ids
        for i in range(num_inputs)
    ]

    return {"nodes": nodes, "connections": connections}


def _line_coordinates(count: int, y: float) -> list[Coordinate]:
    """Lay ``count`` neurons out evenly on a horizontal line.

    Parameters
    ----------
    count
        Number of neurons on the line.
    y
        Vertical coordinate shared by every neuron on the line.

    Returns
    -------
        Coordinates spanning ``x`` from ``-1`` to ``1``, or a single centred
        neuron when ``count`` is one.
    """
    if count <= 1:
        return [(0.0, y)] * count
    return [(-1.0 + (2.0 * i / (count - 1)), y) for i in range(count)]


def create_substrate_for_gecko(
    num_joints: int,
    *,
    use_hidden_layer: bool = True,
    hidden_layer_size: int | None = None,
    num_cpg_oscillators: int = 4,
    num_directional_inputs: int = 2,
) -> tuple[list[Coordinate], list[Coordinate] | None, list[Coordinate]]:
    """Create the substrate layout for a gecko robot.

    Inputs are the joint angles, a bank of CPG oscillators, the directional
    inputs and a bias, laid out on a line at ``y = -0.5``. Hidden neurons sit
    at ``y = 0.0`` and the joint actuators at ``y = 0.5``.

    Parameters
    ----------
    num_joints
        Number of controllable joints.
    use_hidden_layer
        Whether to include a hidden layer.
    hidden_layer_size
        Size of the hidden layer, defaulting to ``num_joints``.
    num_cpg_oscillators
        Number of CPG oscillator inputs.
    num_directional_inputs
        Number of directional inputs, two for an ``(x, y)`` heading.

    Returns
    -------
    input_coords
        Coordinates of the input neurons.
    hidden_coords
        Coordinates of the hidden neurons, or ``None``.
    output_coords
        Coordinates of the output neurons.
    """
    num_inputs = (
        num_joints + num_cpg_oscillators + num_directional_inputs + 1
    )  # + bias
    input_coords = _line_coordinates(num_inputs, -0.5)

    hidden_coords: list[Coordinate] | None = None
    if use_hidden_layer:
        hidden_size = hidden_layer_size or num_joints
        hidden_coords = _line_coordinates(hidden_size, 0.0)

    output_coords = _line_coordinates(num_joints, 0.5)

    return input_coords, hidden_coords, output_coords


def substrate_input_size(
    num_joints: int,
    num_cpg_oscillators: int = 4,
    num_directional_inputs: int = 2,
) -> int:
    """Return the number of substrate input neurons for a robot.

    Parameters
    ----------
    num_joints
        Number of controllable joints.
    num_cpg_oscillators
        Number of CPG oscillator inputs.
    num_directional_inputs
        Number of directional inputs.

    Returns
    -------
        Joint sensors plus oscillators plus directional inputs plus bias.
    """
    return num_joints + num_cpg_oscillators + num_directional_inputs + 1
