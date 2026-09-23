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
    solve_theorem1_example2,
)
from detector_hinf.models import (
    example1_emission_matrix,
    example1_system,
    example2_emission_matrix,
    example2_markov_generator,
    example2_uav_system,
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


def test_example2_theorem1_variable_dimensions():
    system, _ = example2_uav_system()
    Ecal0 = np.vstack([np.eye(4)] * 4 + [np.array([[1., 0., 0., 0.]])])
    assert Ecal0.shape == (17, 4)
    problem, gamma_sq, variables = build_theorem1_problem(
        system, example2_emission_matrix(), [Ecal0, Ecal0],
    )
    assert problem.is_dcp()
    assert gamma_sq.shape == () and gamma_sq.attributes["nonneg"]
    assert problem.objective.args[0] is gamma_sq
    for name, count, shape in (
        ("P", 2, (8, 8)), ("Q", 2, (17, 4)),
        ("X", 3, (4, 4)), ("Y", 3, (4, 2)),
        ("S", 3, (1, 4)), ("T", 3, (1, 2)), ("Z", 2, (4, 4)),
    ):
        assert len(variables[name]) == count
        assert all(variable.shape == shape for variable in variables[name])
    for name, shape in (("U", (8, 8)), ("V", (1, 8)), ("W", (1, 1))):
        assert set(variables[name]) == {(0, 0), (1, 1), (1, 2)}
        assert all(variable.shape == shape for variable in variables[name].values())
    for name in ("P", "U", "W"):
        group = variables[name]
        entries = group.values() if isinstance(group, dict) else group
        assert all(variable.attributes["symmetric"] for variable in entries)
    np.testing.assert_array_equal(variables["phi"], [0, 1, 1])
    np.testing.assert_array_equal(variables["psi"], [0, 1])
    assert sorted(c.shape for c in problem.constraints) == sorted(
        [(8, 8)] * 2 + [(17, 17)] * 2 + [(10, 10)] * 3
    )


@pytest.mark.parametrize("return_filter", [False, True])
def test_example2_wrapper_and_filter_dimensions(monkeypatch, return_filter):
    from detector_hinf import lmi

    original_builder = lmi.build_theorem1_problem
    captured = {}

    def capture_builder(system, emission, Ecal):
        assert len(system[2]) == 2  # No C3_reported plant mode.
        np.testing.assert_array_equal(system[-1], example2_markov_generator())
        np.testing.assert_array_equal(emission, example2_emission_matrix())
        assert len(Ecal) == 2
        for E in Ecal:
            assert E.shape == (17, 4)
            np.testing.assert_array_equal(E[:16], np.tile(np.eye(4), (4, 1)))
            np.testing.assert_array_equal(E[16], [1., 0., 0., 0.])
        problem, gamma_sq, variables = original_builder(system, emission, Ecal)
        captured.update(variables)
        return problem, gamma_sq, variables

    def fake_solve(problem, solver, verbose):
        assert solver == "SCS" and verbose is True
        for variable in problem.variables():
            if variable.name() == "gamma_sq":
                variable.value = 0.04
        for nu, Z in enumerate(captured["Z"]):
            Z.value = (nu + 1) * np.eye(4)
        for name in ("X", "Y", "S", "T"):
            for ell, variable in enumerate(captured[name]):
                variable.value = np.full(variable.shape, ell + 1.0)
        problem._status = cp.OPTIMAL

    monkeypatch.setattr(lmi, "build_theorem1_problem", capture_builder)
    monkeypatch.setattr(cp.Problem, "solve", fake_solve)
    result = solve_theorem1_example2(return_filter=return_filter, solver="SCS", verbose=True)
    assert result["status"] == cp.OPTIMAL
    assert result["gamma_sq"] == 0.04 and result["gamma"] == 0.2
    if not return_filter:
        assert set(result) == {"status", "gamma_sq", "gamma"}
        return
    for name, shape in (("Ahat", (4, 4)), ("Bhat", (4, 2)),
                        ("Lhat", (1, 4)), ("Ehat", (1, 2))):
        assert len(result[name]) == 3
        assert all(matrix.shape == shape for matrix in result[name])
    for ell, nu in enumerate([0, 1, 1]):
        np.testing.assert_allclose(captured["Z"][nu].value @ result["Ahat"][ell],
                                   captured["X"][ell].value)
        np.testing.assert_allclose(captured["Z"][nu].value @ result["Bhat"][ell],
                                   captured["Y"][ell].value)
        np.testing.assert_array_equal(result["Lhat"][ell], captured["S"][ell].value)
        np.testing.assert_array_equal(result["Ehat"][ell], captured["T"][ell].value)


def test_example2_unsuccessful_solve(monkeypatch):
    def fake_solve(problem, **kwargs):
        problem._status = cp.INFEASIBLE

    monkeypatch.setattr(cp.Problem, "solve", fake_solve)
    result = solve_theorem1_example2(return_filter=True)
    assert result["status"] == cp.INFEASIBLE
    assert all(result[key] is None for key in ("gamma", "gamma_sq", "Ahat", "Bhat", "Lhat", "Ehat"))
