"""Test: CPPN and substrate networks."""

# Third-party libraries
import numpy as np
import pytest

# Local libraries
from ariel.spatial_ea.hyperneat import (
    ACTIVATION_FUNCTIONS,
    CPPN,
    CPPNConnection,
    CPPNNode,
    SubstrateNetwork,
    create_minimal_cppn_genome,
    create_substrate_for_gecko,
    substrate_input_size,
)


def _constant_weight_cppn(weight: float) -> CPPN:
    """Build a CPPN whose single output is a fixed multiple of its bias."""
    return CPPN({
        "nodes": [
            CPPNNode(node_id=0, activation="linear", layer=0),
            CPPNNode(node_id=1, activation="linear", layer=0),
            CPPNNode(node_id=2, activation="linear", layer=0),
            CPPNNode(node_id=3, activation="linear", layer=0),
            CPPNNode(node_id=4, activation="linear", layer=1),
        ],
        # Only the first input is used, so activate([w, 0, 0, 0]) == w.
        "connections": [
            CPPNConnection(from_node=0, to_node=4, weight=weight),
        ],
    })


@pytest.mark.parametrize("name", sorted(ACTIVATION_FUNCTIONS))
def test_activations_are_finite_and_shape_preserving(name: str) -> None:
    """Every activation should map an array to a finite array of equal shape."""
    values = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    result = ACTIVATION_FUNCTIONS[name](values)

    assert result.shape == values.shape
    assert np.isfinite(result).all()


def test_minimal_genome_activates_to_one_output() -> None:
    """A minimal genome should be directly usable as a CPPN."""
    genome = create_minimal_cppn_genome()
    output = CPPN(genome).activate(np.array([0.5, -0.5, 0.25, -0.25]))

    assert output.shape == (1,)
    assert np.isfinite(output).all()


def test_cppn_propagates_through_hidden_layers() -> None:
    """A value should be transformed by every layer it passes through."""
    genome = {
        "nodes": [
            CPPNNode(node_id=0, activation="linear", layer=0),
            CPPNNode(node_id=1, activation="abs", layer=1),
            CPPNNode(node_id=2, activation="linear", layer=2),
        ],
        "connections": [
            CPPNConnection(from_node=0, to_node=1, weight=2.0),
            CPPNConnection(from_node=1, to_node=2, weight=3.0),
        ],
    }

    # abs(-4 * 2) * 3 == 24
    assert CPPN(genome).activate(np.array([-4.0]))[0] == pytest.approx(24.0)


def test_disabled_connections_are_ignored() -> None:
    """A disabled connection must not contribute to the output."""
    genome = {
        "nodes": [
            CPPNNode(node_id=0, activation="linear", layer=0),
            CPPNNode(node_id=1, activation="linear", layer=1),
        ],
        "connections": [
            CPPNConnection(
                from_node=0,
                to_node=1,
                weight=5.0,
                enabled=False,
            ),
        ],
    }

    assert CPPN(genome).activate(np.array([1.0]))[0] == pytest.approx(0.0)


def test_substrate_thresholds_small_weights_away() -> None:
    """Weights below the threshold should be dropped to zero."""
    input_coords = [(-1.0, -0.5), (1.0, -0.5)]
    output_coords = [(0.0, 0.5)]

    below = SubstrateNetwork(
        input_coords=input_coords,
        hidden_coords=None,
        output_coords=output_coords,
        cppn=_constant_weight_cppn(0.1),
        weight_threshold=0.2,
    )
    above = SubstrateNetwork(
        input_coords=input_coords,
        hidden_coords=None,
        output_coords=output_coords,
        cppn=_constant_weight_cppn(0.9),
        weight_threshold=0.2,
    )

    assert np.count_nonzero(below.weights_input_output) == 0
    assert np.count_nonzero(above.weights_input_output) > 0


def test_substrate_output_matches_actuator_count() -> None:
    """The substrate should emit exactly one value per joint."""
    num_joints = 8
    input_coords, hidden_coords, output_coords = create_substrate_for_gecko(
        num_joints=num_joints,
    )

    substrate = SubstrateNetwork(
        input_coords=input_coords,
        hidden_coords=hidden_coords,
        output_coords=output_coords,
        cppn=CPPN(create_minimal_cppn_genome()),
    )
    outputs = substrate.activate(np.ones(len(input_coords)))

    assert len(output_coords) == num_joints
    assert outputs.shape == (num_joints,)
    assert np.isfinite(outputs).all()


def test_gecko_substrate_layout() -> None:
    """Layers should be separated in space and sized as documented."""
    num_joints = 8
    input_coords, hidden_coords, output_coords = create_substrate_for_gecko(
        num_joints=num_joints,
    )

    assert len(input_coords) == substrate_input_size(num_joints)
    assert len(input_coords) == num_joints + 4 + 2 + 1
    assert hidden_coords is not None
    assert len(hidden_coords) == num_joints

    assert {coord[1] for coord in input_coords} == {-0.5}
    assert {coord[1] for coord in hidden_coords} == {0.0}
    assert {coord[1] for coord in output_coords} == {0.5}


def test_substrate_without_hidden_layer() -> None:
    """Disabling the hidden layer should wire inputs straight to outputs."""
    input_coords, hidden_coords, output_coords = create_substrate_for_gecko(
        num_joints=4,
        use_hidden_layer=False,
    )

    assert hidden_coords is None

    substrate = SubstrateNetwork(
        input_coords=input_coords,
        hidden_coords=hidden_coords,
        output_coords=output_coords,
        cppn=CPPN(create_minimal_cppn_genome()),
    )

    assert substrate.weights_input_output.shape == (len(input_coords), 4)
    assert substrate.activate(np.ones(len(input_coords))).shape == (4,)
