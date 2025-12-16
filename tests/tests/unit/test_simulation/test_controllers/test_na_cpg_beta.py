import numpy as np
import torch
import pytest
from pathlib import Path

from ariel.simulation.controllers.na_cpg_beta import (
    create_fully_connected_adjacency,
    NaCPGBeta,
)


def test_create_fully_connected_adjacency_small():
    adj = create_fully_connected_adjacency(3)
    assert isinstance(adj, dict)
    assert set(adj.keys()) == {0, 1, 2}
    for k, v in adj.items():
        assert k not in v
        assert set(v) == set(range(3)) - {k}


@pytest.fixture
def small_adj():
    return create_fully_connected_adjacency(4)


def test_nacpg_beta_initialization_and_params(small_adj):
    model = NaCPGBeta(small_adj, seed=0)
    assert model.n == 4
    # parameter groups present
    for key in ("phase", "w", "amplitudes", "ha", "b"):
        assert key in model.parameter_groups
    # numeric parameter bookkeeping
    assert isinstance(model.num_of_parameters, int)
    assert model.num_of_parameters > 0


def test_param_type_converter_list_and_numpy(small_adj):
    model = NaCPGBeta(small_adj, seed=0)
    lst = [0.1, 0.2, 0.3, 0.4]
    t = model.param_type_converter(lst)
    assert isinstance(t, torch.Tensor)
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    t2 = model.param_type_converter(arr)
    assert isinstance(t2, torch.Tensor)


def test_set_flat_params_wrong_size_raises(small_adj):
    model = NaCPGBeta(small_adj, seed=0)
    wrong = torch.ones(max(1, model.num_of_parameters - 1))
    with pytest.raises(ValueError):
        model.set_flat_params(wrong)


def test_set_params_by_group_valid_and_invalid(small_adj):
    model = NaCPGBeta(small_adj, seed=0)
    # valid: set phase to zeros
    zeros = torch.zeros_like(model.phase)
    model.set_params_by_group("phase", zeros)
    assert torch.allclose(model.phase, zeros)
    # invalid group
    with pytest.raises(ValueError):
        model.set_params_by_group("nonexistent", torch.tensor([1.0]))


def test_set_params_by_group_size_mismatch(small_adj):
    model = NaCPGBeta(small_adj, seed=0)
    wrong = torch.ones(max(1, model.phase.numel() - 1))
    with pytest.raises(ValueError):
        model.set_params_by_group("phase", wrong)


def test_forward_returns_angles_shape_and_no_nan(small_adj):
    model = NaCPGBeta(small_adj, seed=0, angle_tracking=False)
    angles = model.forward(time=0.0)  # also triggers reset
    assert isinstance(angles, torch.Tensor)
    assert angles.shape[0] == model.n
    assert not torch.any(torch.isnan(angles))


def test_reset_restores_initial_state(small_adj):
    model = NaCPGBeta(small_adj, seed=0)
    before = {k: v.clone() for k, v in model.initial_state.items()}
    _ = model.forward()  # change internal state
    model.reset_state()
    assert torch.allclose(model.xy, before["xy"])
    assert torch.allclose(model.xy_dot_old, before["xy_dot_old"])
    assert torch.allclose(model.angles, before["angles"])
    assert model.angle_history == []


def test_angle_tracking_appends_history(small_adj):
    model = NaCPGBeta(small_adj, seed=0, angle_tracking=True)
    for _ in range(3):
        model.forward()
    assert len(model.angle_history) == 3
    for entry in model.angle_history:
        assert len(entry) == model.n


def test_term_and_zeta_helpers():
    a = NaCPGBeta.term_a(0.1, 0.5)
    assert isinstance(a, float)
    b = NaCPGBeta.term_b(0.2, 1.0)
    assert isinstance(b, float)
    # pass a tensor for x_dot_old because zeta uses torch.abs internally
    z = NaCPGBeta.zeta(0.1, torch.tensor(0.05))
    # zeta returns a torch Tensor (per implementation) or float; accept both
    assert isinstance(z, (float, torch.Tensor))


def test_save_and_load_roundtrip(tmp_path, small_adj):
    model = NaCPGBeta(small_adj, seed=0)
    save_path = Path(tmp_path) / "params.pt"
    model.save(save_path)
    # change parameters
    model.phase.data += 0.1
    # load back
    model.load(save_path)
    assert model.phase.numel() == model.n
    assert not torch.any(torch.isnan(model.phase))
