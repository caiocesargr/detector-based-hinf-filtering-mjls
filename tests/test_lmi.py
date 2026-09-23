"""Basic expression and margin checks for the CVXPY LMI helpers."""

import cvxpy as cp
import numpy as np
import pytest

from detector_hinf.lmi import (
    build_theorem1_problem,
    her,
    negative_definite_constraint,
    positive_definite_constraint,
    recover_filter_matrices,
    solve_theorem1_example1,
)
from detector_hinf.models import example1_emission_matrix, example1_system


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


def _example1_recovery_values(n_sets):
    """Synthetic data to check recovery algebra, not filters from the paper."""
    return {
        "X": [np.arange(9, dtype=float).reshape(3, 3) + ell for ell in range(2)],
        "Y": [np.array([[1.0], [2.0], [3.0]]) + ell for ell in range(2)],
        "S": [np.array([[0.2, 0.3, 0.4]]) + ell for ell in range(2)],
        "T": [np.array([[0.1 + ell]]) for ell in range(2)],
        "Z": [(nu + 1) * np.array([[2., 1., 0.], [0., 3., 1.], [0., 0., 4.]])
              for nu in range(n_sets)],
    }


@pytest.mark.parametrize(
    "rho, expected_phi", [(0.0, [0, 1]), (0.2, [0, 0]), (0.5, [0, 0]), (1.0, [1, 0])]
)
def test_example1_filter_recovery_dimensions_and_equations(rho, expected_phi):
    values = _example1_recovery_values(len(set(expected_phi)))
    recovered = recover_filter_matrices(example1_emission_matrix(rho), values)
    for name, shape in (("Ahat", (3, 3)), ("Bhat", (3, 1)),
                        ("Lhat", (1, 3)), ("Ehat", (1, 1))):
        assert isinstance(recovered[name], list)
        assert len(recovered[name]) == 2
        assert all(matrix.shape == shape for matrix in recovered[name])
    for ell, nu in enumerate(expected_phi):
        np.testing.assert_allclose(values["Z"][nu] @ recovered["Ahat"][ell],
                                   values["X"][ell], atol=1e-12)
        np.testing.assert_allclose(values["Z"][nu] @ recovered["Bhat"][ell],
                                   values["Y"][ell], atol=1e-12)
        np.testing.assert_array_equal(recovered["Lhat"][ell], values["S"][ell])
        np.testing.assert_array_equal(recovered["Ehat"][ell], values["T"][ell])
        assert not np.shares_memory(recovered["Lhat"][ell], values["S"][ell])
        assert not np.shares_memory(recovered["Ehat"][ell], values["T"][ell])


@pytest.mark.parametrize("return_filter", [False, True])
def test_example1_solver_optional_filter(monkeypatch, return_filter):
    # Populate the actual builder's variables without requiring a solver license.
    values = _example1_recovery_values(1)

    def fake_solve(problem, solver, verbose):
        assert solver == "MOSEK" and verbose is False
        for variable in problem.variables():
            if variable.name() == "gamma_sq":
                variable.value = 0.25
            else:
                name, *indices = variable.name().split("_")
                if name in values:
                    variable.value = values[name][int(indices[0])]
        problem._status = cp.OPTIMAL

    monkeypatch.setattr(cp.Problem, "solve", fake_solve)
    result = solve_theorem1_example1(0.2, return_filter=return_filter)
    assert result["status"] == cp.OPTIMAL
    assert result["gamma"] == 0.5 and result["gamma_sq"] == 0.25
    expected_keys = {"status", "gamma", "gamma_sq"}
    if return_filter:
        expected_keys.update(("Ahat", "Bhat", "Lhat", "Ehat"))
        for name, shape in (("Ahat", (3, 3)), ("Bhat", (3, 1)),
                            ("Lhat", (1, 3)), ("Ehat", (1, 1))):
            assert len(result[name]) == 2
            assert all(matrix.shape == shape for matrix in result[name])
    assert set(result) == expected_keys


def test_filter_recovery_rejects_unsolved_values():
    E = np.vstack([np.eye(3)] * 4 + [np.zeros((1, 3))])
    Upsilon = example1_emission_matrix(0.2)
    _, _, variables = build_theorem1_problem(example1_system(), Upsilon, [E, E])
    with pytest.raises(ValueError, match="numerical matrix"):
        recover_filter_matrices(Upsilon, variables)


def test_filter_recovery_rejects_singular_Z():
    values = _example1_recovery_values(1)
    values["Z"][0] = np.zeros((3, 3))
    with pytest.raises(np.linalg.LinAlgError):
        recover_filter_matrices(example1_emission_matrix(0.2), values)


def test_filter_recovery_rejects_wrong_dimensions():
    values = _example1_recovery_values(1)
    values["Y"][1] = np.ones((2, 1))
    with pytest.raises(ValueError, match=r"Y\[1\].*shape"):
        recover_filter_matrices(example1_emission_matrix(0.2), values)


def test_unsuccessful_solve_does_not_recover_filter(monkeypatch):
    def fake_solve(problem, **kwargs):
        problem._status = cp.INFEASIBLE

    monkeypatch.setattr(cp.Problem, "solve", fake_solve)
    result = solve_theorem1_example1(0.2, return_filter=True)
    assert result["status"] == cp.INFEASIBLE
    assert all(result[key] is None for key in ("gamma", "gamma_sq", "Ahat", "Bhat", "Lhat", "Ehat"))
