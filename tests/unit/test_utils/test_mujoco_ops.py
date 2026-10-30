"""Test: MuJoCo utility functions."""

import mujoco
import numpy as np
import pytest

from ariel.utils.mujoco_ops import euler_to_quat_conversion, has_self_collision, mjspec_deep_copy


# ---------------------------------------------------------------------------
# euler_to_quat_conversion
# ---------------------------------------------------------------------------


def test_euler_to_quat_returns_length_4() -> None:
    """Conversion returns a quaternion of length 4."""
    q = euler_to_quat_conversion((0.0, 0.0, 0.0), "XYZ")
    assert len(q) == 4


def test_euler_zero_rotation_is_identity() -> None:
    """Zero Euler angles map to the identity quaternion (0,0,0,1)."""
    q = euler_to_quat_conversion((0.0, 0.0, 0.0), "XYZ")
    # MuJoCo returns (x,y,z,w); identity is w=1, xyz=0
    assert np.abs(q).sum() == pytest.approx(1.0, abs=1e-5)


def test_euler_to_quat_unit_norm() -> None:
    """Output quaternion always has unit norm."""
    q = euler_to_quat_conversion((30.0, 45.0, 60.0), "XYZ")
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-5)


def test_euler_to_quat_90_deg_rotation() -> None:
    """90° rotation around Z produces a non-identity quaternion."""
    q_id = euler_to_quat_conversion((0.0, 0.0, 0.0), "XYZ")
    q_rot = euler_to_quat_conversion((0.0, 0.0, 90.0), "XYZ")
    assert not np.allclose(q_id, q_rot)


# ---------------------------------------------------------------------------
# has_self_collision
# ---------------------------------------------------------------------------


def _simple_spec() -> mujoco.MjSpec:
    """A minimal single-box spec with no self-collision possible."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="box")
    body.add_geom(
        name="box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.1, 0.1, 0.1),
        pos=(0.0, 0.0, 0.0),
    )
    return spec


def test_has_self_collision_no_collision() -> None:
    """A single isolated geom has no self-collision."""
    spec = _simple_spec()
    assert has_self_collision(spec) is False


def test_has_self_collision_returns_bool() -> None:
    """has_self_collision always returns a bool."""
    spec = _simple_spec()
    result = has_self_collision(spec)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# mjspec_deep_copy
# ---------------------------------------------------------------------------


def test_mjspec_deep_copy_returns_mjspec() -> None:
    """mjspec_deep_copy returns a MjSpec instance."""
    spec = _simple_spec()
    copy = mjspec_deep_copy(spec)
    assert isinstance(copy, mujoco.MjSpec)


def test_mjspec_deep_copy_is_independent() -> None:
    """The copy compiles independently to the same XML."""
    spec = _simple_spec()
    copy = mjspec_deep_copy(spec)
    # both should compile without error
    model_orig = spec.compile()
    model_copy = copy.compile()
    assert model_orig.ngeom == model_copy.ngeom
