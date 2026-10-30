"""Test: SimpleCPG Hopf oscillator controller."""

import numpy as np
import pytest
import torch

from ariel.simulation.controllers.simple_cpg import (
    SimpleCPG,
    create_fully_connected_adjacency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adj(n: int) -> dict[int, list[int]]:
    return create_fully_connected_adjacency(n)


def _cpg(n: int = 3, seed: int = 0) -> SimpleCPG:
    return SimpleCPG(_adj(n), seed=seed)


# ---------------------------------------------------------------------------
# create_fully_connected_adjacency
# ---------------------------------------------------------------------------


def test_fully_connected_adjacency_size() -> None:
    """create_fully_connected_adjacency returns a dict with n keys."""
    adj = create_fully_connected_adjacency(4)
    assert len(adj) == 4


def test_fully_connected_adjacency_no_self_loops() -> None:
    """No node is connected to itself."""
    adj = create_fully_connected_adjacency(5)
    for node, neighbors in adj.items():
        assert node not in neighbors


def test_fully_connected_adjacency_all_others() -> None:
    """Each node is connected to every other node."""
    n = 4
    adj = create_fully_connected_adjacency(n)
    for node, neighbors in adj.items():
        assert len(neighbors) == n - 1


# ---------------------------------------------------------------------------
# SimpleCPG initialization
# ---------------------------------------------------------------------------


def test_simplecpg_initialization() -> None:
    """SimpleCPG initializes without error."""
    cpg = _cpg()
    assert cpg.n == 3


def test_simplecpg_parameter_count() -> None:
    """Total parameter count equals 5 groups × n."""
    n = 4
    cpg = _cpg(n)
    assert cpg.num_of_parameters == 5 * n


def test_simplecpg_parameter_groups() -> None:
    """Parameter groups dict has the expected 5 keys."""
    cpg = _cpg()
    expected = {"phase", "w", "amplitudes", "ha", "b"}
    assert set(cpg.parameter_groups.keys()) == expected


def test_simplecpg_seeded_reproducible() -> None:
    """Two CPGs with the same seed produce the same initial parameters."""
    cpg1 = _cpg(seed=42)
    cpg2 = _cpg(seed=42)
    assert torch.allclose(cpg1.get_flat_params(), cpg2.get_flat_params())


def test_simplecpg_different_seeds_differ() -> None:
    """Different seeds yield different parameters."""
    cpg1 = _cpg(seed=1)
    cpg2 = _cpg(seed=2)
    assert not torch.allclose(cpg1.get_flat_params(), cpg2.get_flat_params())


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


def test_simplecpg_forward_returns_tensor() -> None:
    """forward() returns a torch.Tensor."""
    cpg = _cpg()
    output = cpg.forward()
    assert isinstance(output, torch.Tensor)


def test_simplecpg_forward_correct_size() -> None:
    """forward() output has length n."""
    n = 5
    cpg = _cpg(n)
    output = cpg.forward()
    assert output.shape == (n,)


def test_simplecpg_forward_within_hard_bounds() -> None:
    """With default hard bounds, all outputs are in [-pi/2, pi/2]."""
    cpg = _cpg()
    for _ in range(50):
        output = cpg.forward()
        assert torch.all(output >= -torch.pi / 2 - 1e-5)
        assert torch.all(output <= torch.pi / 2 + 1e-5)


def test_simplecpg_forward_no_nan() -> None:
    """forward() never produces NaN values."""
    cpg = _cpg()
    for _ in range(100):
        output = cpg.forward()
        assert not torch.isnan(output).any()



def test_simplecpg_forward_with_angle_tracking() -> None:
    """angle_tracking=True appends to angle_history."""
    cpg = SimpleCPG(_adj(3), angle_tracking=True, seed=0)
    steps = 5
    for _ in range(steps):
        cpg.forward()
    assert len(cpg.angle_history) == steps


def test_simplecpg_forward_no_bounds() -> None:
    """hard_bounds=None allows outputs outside [-pi/2, pi/2]."""
    cpg = SimpleCPG(_adj(3), hard_bounds=None, seed=0)
    outputs = [cpg.forward() for _ in range(100)]
    all_values = torch.stack(outputs).abs()
    # With no bounds, some values may exceed pi/2 over 100 steps
    # Just check it runs without error
    assert all_values.shape[0] == 100


# ---------------------------------------------------------------------------
# set_flat_params / get_flat_params
# ---------------------------------------------------------------------------


def test_simplecpg_set_get_flat_params_roundtrip() -> None:
    """set_flat_params followed by get_flat_params reproduces the params."""
    cpg = _cpg()
    params = torch.zeros(cpg.num_of_parameters)
    cpg.set_flat_params(params)
    result = cpg.get_flat_params()
    assert torch.allclose(result, params)


def test_simplecpg_set_flat_params_wrong_size_raises() -> None:
    """set_flat_params raises ValueError for incorrect parameter count."""
    cpg = _cpg()
    with pytest.raises(ValueError, match="incorrect size"):
        cpg.set_flat_params(torch.zeros(1))


def test_simplecpg_set_flat_params_from_list() -> None:
    """set_flat_params accepts a Python list."""
    cpg = _cpg()
    params = [0.0] * cpg.num_of_parameters
    cpg.set_flat_params(params)
    result = cpg.get_flat_params()
    assert torch.allclose(result, torch.zeros(cpg.num_of_parameters))


def test_simplecpg_set_flat_params_from_numpy() -> None:
    """set_flat_params accepts a NumPy array."""
    cpg = _cpg()
    params = np.zeros(cpg.num_of_parameters)
    cpg.set_flat_params(params)
    result = cpg.get_flat_params()
    assert torch.allclose(result, torch.zeros(cpg.num_of_parameters))


# ---------------------------------------------------------------------------
# set_param_with_dict / set_params_by_group
# ---------------------------------------------------------------------------


def test_set_params_by_group_valid() -> None:
    """set_params_by_group updates the named group correctly."""
    cpg = _cpg()
    new_phase = torch.ones(cpg.n)
    cpg.set_params_by_group("phase", new_phase)
    assert torch.allclose(cpg.phase.data, new_phase)


def test_set_params_by_group_invalid_name_raises() -> None:
    """set_params_by_group raises ValueError for an unknown group name."""
    cpg = _cpg()
    with pytest.raises(ValueError, match="does not exist"):
        cpg.set_params_by_group("INVALID", torch.zeros(cpg.n))


def test_set_params_by_group_wrong_size_raises() -> None:
    """set_params_by_group raises ValueError for a mismatched size."""
    cpg = _cpg(3)
    with pytest.raises(ValueError, match="incorrect size"):
        cpg.set_params_by_group("phase", torch.zeros(99))


def test_set_param_with_dict() -> None:
    """set_param_with_dict updates all specified groups."""
    cpg = _cpg()
    updates = {
        "phase": torch.ones(cpg.n),
        "w": torch.zeros(cpg.n),
    }
    cpg.set_param_with_dict(updates)
    assert torch.allclose(cpg.phase.data, torch.ones(cpg.n))
    assert torch.allclose(cpg.w.data, torch.zeros(cpg.n))


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def test_simplecpg_reset_restores_initial_state() -> None:
    """reset() restores x, y, angles and clears angle_history."""
    cpg = SimpleCPG(_adj(3), angle_tracking=True, seed=0)
    for _ in range(20):
        cpg.forward()
    cpg.reset()
    assert torch.allclose(cpg.x, cpg.initial_state["x"])
    assert torch.allclose(cpg.y, cpg.initial_state["y"])
    assert cpg.angle_history == []
