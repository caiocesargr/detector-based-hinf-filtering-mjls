"""Basic expression and margin checks for the CVXPY LMI helpers."""

import cvxpy as cp
import numpy as np
import pytest

from detector_hinf.lmi import (
    her,
    negative_definite_constraint,
    positive_definite_constraint,
)


@pytest.mark.parametrize(
    "helper, sign",
    [(positive_definite_constraint, 1), (negative_definite_constraint, -1)],
)
@pytest.mark.parametrize("eps", [None, 0.25])
def test_definiteness_constraint_margin(helper, sign, eps):
    X = cp.Variable((2, 2), symmetric=True)
    constraint = helper(X) if eps is None else helper(X, eps=eps)
    margin = 1e-8 if eps is None else eps
    assert isinstance(constraint, cp.constraints.PSD)
    assert constraint.shape == (2, 2)
    assert constraint.is_dcp()

    # A feasible matrix, the exact boundary, and a violating eigenvalue.
    for eigenvalues, violation in [
        ([2 * margin, 3 * margin], 0.0),
        ([margin, 2 * margin], 0.0),
        ([0.5 * margin, 2 * margin], 0.5 * margin),
    ]:
        X.value = sign * np.diag(eigenvalues)
        np.testing.assert_allclose(
            constraint.violation(), violation, rtol=1e-10, atol=1e-15
        )


def test_her_affine_expression():
    X = cp.Variable((2, 2))
    expression = her(2 * X + np.eye(2))
    assert isinstance(expression, cp.Expression)
    assert expression.is_affine()
    X.value = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(expression.value, [[6.0, 10.0], [10.0, 18.0]])


@pytest.mark.parametrize(
    "helper", [positive_definite_constraint, negative_definite_constraint]
)
def test_constraint_accepts_affine_expression(helper):
    X = cp.Variable((2, 2))
    constraint = helper(her(X) + np.eye(2))
    assert constraint.is_dcp()
    assert constraint.shape == (2, 2)


@pytest.mark.parametrize(
    "helper", [positive_definite_constraint, negative_definite_constraint]
)
@pytest.mark.parametrize("eps", [0.0, -1e-8, np.nan, np.inf])
def test_invalid_margin(helper, eps):
    with pytest.raises(ValueError, match="eps"):
        helper(cp.Variable((2, 2), symmetric=True), eps=eps)


@pytest.mark.parametrize(
    "helper", [positive_definite_constraint, negative_definite_constraint]
)
@pytest.mark.parametrize("shape", [(2, 3), (2,)])
def test_invalid_matrix_shape(helper, shape):
    with pytest.raises(ValueError, match="square"):
        helper(cp.Variable(shape))
