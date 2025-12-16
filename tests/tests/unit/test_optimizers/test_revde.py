import numpy as np
import pytest

from ariel.utils.optimizers.revde import RevDE


def test_r_matrix_values_for_scaling_factor():
    f = 0.5
    rev = RevDE(scaling_factor=f)

    f2 = f**2
    f3 = f**3
    a = 1 - f2
    b = f + f2
    c = -f + f2 + f3
    d = 1 - (2 * f2) - f3

    expected = np.array([[1, f, -f], [-f, a, b], [b, c, d]])
    assert np.allclose(rev.r_matrix, expected)


def test_mutate_returns_expected_children_with_basis_parents():
    # Use basis parents so expected result is easy to compute from r_matrix rows
    rev = RevDE(scaling_factor=-0.5)
    parent_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    parent_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    parent_c = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

    children = rev.mutate(parent_a, parent_b, parent_c)

    assert isinstance(children, list)
    assert len(children) == 3

    # Expected: each child row corresponds to the corresponding r_matrix row
    # applied to the first three basis columns; fourth column remains zero.
    expected_matrix = rev.r_matrix @ np.vstack((parent_a, parent_b, parent_c))
    expected_children = [expected_matrix[i] for i in range(3)]

    for got, exp in zip(children, expected_children):
        assert np.allclose(got, exp)


def test_mutate_preserves_parent_arrays_and_shape():
    rev = RevDE(scaling_factor=0.2)
    parent_a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    parent_b = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    parent_c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    a_copy = parent_a.copy()
    b_copy = parent_b.copy()
    c_copy = parent_c.copy()

    children = rev.mutate(parent_a, parent_b, parent_c)

    # Parents should remain unchanged
    assert np.allclose(parent_a, a_copy)
    assert np.allclose(parent_b, b_copy)
    assert np.allclose(parent_c, c_copy)

    # Children shapes match parents
    assert len(children) == 3
    for child in children:
        assert isinstance(child, np.ndarray)
        assert child.shape == parent_a.shape


def test_mutate_raises_on_shape_mismatch():
    rev = RevDE(scaling_factor=0.3)
    parent_a = np.zeros(4)
    parent_b = np.zeros(3)  # different shape
    parent_c = np.zeros(4)

    with pytest.raises(ValueError, match="Parents must have the same shape"):
        rev.mutate(parent_a, parent_b, parent_c)