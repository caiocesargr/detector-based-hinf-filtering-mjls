"""Alternating fixed-matrix optimization in Algorithm 1 of the paper."""

import cvxpy as cp
import numpy as np

from .detector import phi_map, psi_map
from .lmi import (
    _numeric_matrix, _system_data, build_theorem1_problem, her,
    negative_definite_constraint, positive_definite_constraint,
    recover_filter_matrices,
)
from .models import example2_emission_matrix, example2_uav_system


def _build_step2_problem(system, Upsilon, fixed, eps=1e-8):
    """LMIs (19)--(22), with only X, Y, Z fixed and Ecal free.

    This mirrors the fixed-Ecal builder without changing its equations.
    In particular, P, Q, U, V, W, S and T are fresh decision variables.
    """
    system, Upsilon, symbols, (nx, nw, ny, nz) = _system_data(system, Upsilon)
    A, B, C, D, L, F, Lambda = system
    n_modes, n_symbols = Upsilon.shape
    n_aug = 2 * nx
    n_lmi = 4 * nx + nw
    Ecal = [cp.Variable((n_lmi, nx), name=f"Ecal_{i}")
            for i in range(n_modes)]
    psi = psi_map(Upsilon, tol=0.0)
    phi = phi_map(Upsilon, tol=0.0)
    n_sets = int(psi.max()) + 1

    gamma_sq = cp.Variable(nonneg=True, name="gamma_sq")
    P = [cp.Variable((n_aug, n_aug), symmetric=True, name=f"P_{i}")
         for i in range(n_modes)]
    Q = [cp.Variable((n_lmi, nx), name=f"Q_{i}") for i in range(n_modes)]
    pairs = [(i, ell) for i in range(n_modes) for ell in sorted(symbols[i])]
    U = {(i, ell): cp.Variable((n_aug, n_aug), symmetric=True, name=f"U_{i}_{ell}")
         for i, ell in pairs}
    V = {(i, ell): cp.Variable((nw, n_aug), name=f"V_{i}_{ell}")
         for i, ell in pairs}
    W = {(i, ell): cp.Variable((nw, nw), symmetric=True, name=f"W_{i}_{ell}")
         for i, ell in pairs}
    if len(fixed["X"]) != n_symbols:
        raise ValueError("Incorrect number of fixed X matrices.")
    X = [_numeric_matrix(value, "X", (nx, nx)).copy()
         for value in fixed["X"]]
    if len(fixed["Y"]) != n_symbols:
        raise ValueError("Incorrect number of fixed Y matrices.")
    Y = [_numeric_matrix(value, "Y", (nx, ny)).copy()
         for value in fixed["Y"]]
    S = [cp.Variable((nz, nx), name=f"S_{ell}") for ell in range(n_symbols)]
    T = [cp.Variable((nz, ny), name=f"T_{ell}") for ell in range(n_symbols)]
    if len(fixed["Z"]) != n_sets:
        raise ValueError("Incorrect number of fixed Z matrices.")
    Z = [_numeric_matrix(value, "Z", (nx, nx)).copy()
         for value in fixed["Z"]]

    constraints = [positive_definite_constraint(Pi, eps) for Pi in P]
    for i in range(n_modes):
        active = sorted(symbols[i])
        Uavg = sum(Upsilon[i, ell] * U[i, ell] for ell in active)
        Vavg = sum(Upsilon[i, ell] * V[i, ell] for ell in active)
        Wavg = sum(Upsilon[i, ell] * W[i, ell] for ell in active)
        # Equation (20): the averaged U term has a plus sign.
        Sigma = sum(Lambda[i, j] * P[j] for j in range(n_modes)) + Uavg
        Gamma = cp.bmat([
            [Sigma, P[i], Vavg.T],
            [P[i], np.zeros((n_aug, n_aug)), np.zeros((n_aug, nw))],
            [Vavg, np.zeros((nw, n_aug)), Wavg - gamma_sq * np.eye(nw)],
        ])
        Phi = cp.bmat([[
            A[i], np.zeros((nx, nx)), -np.eye(nx), np.zeros((nx, nx)), B[i],
        ]])
        HX = sum(Upsilon[i, ell] * X[ell] for ell in active)
        HY = sum(Upsilon[i, ell] * Y[ell] for ell in active)
        Psi = cp.bmat([[
            HY @ C[i], HX, np.zeros((nx, nx)), -Z[psi[i]], HY @ D[i],
        ]])
        assert Gamma.shape == (n_lmi, n_lmi)
        assert Phi.shape == Psi.shape == (nx, n_lmi)
        # Equation (19a): both Hermitian terms are added.
        lmi = Gamma + her(Q[i] @ Phi) + her(Ecal[i] @ Psi)
        constraints.append(negative_definite_constraint(lmi, eps))

        for ell in active:
            Lbar = cp.bmat([[L[i] - T[ell] @ C[i], -S[ell]]])
            Ebar = F[i] - T[ell] @ D[i]
            Xi = cp.bmat([
                [U[i, ell], V[i, ell].T, Lbar.T],
                [V[i, ell], W[i, ell], Ebar.T],
                [Lbar, Ebar, np.eye(nz)],
            ])
            assert Xi.shape == (n_aug + nw + nz, n_aug + nw + nz)
            constraints.append(positive_definite_constraint(Xi, eps))

    variables = {
        "P": P, "Q": Q, "U": U, "V": V, "W": W,
        "X": X, "Y": Y, "S": S, "T": T, "Z": Z,
        "psi": psi, "phi": phi, "Ecal": Ecal,
    }
    return cp.Problem(cp.Minimize(gamma_sq), constraints), gamma_sq, variables


def _solve(problem, gamma_sq, solver, verbose, iteration, step):
    """Reject unsuccessful solves before reading or reusing variable values."""
    problem.solve(solver=solver, verbose=verbose)
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(
            f"Algorithm 1 iteration {iteration}, Step {step}: {problem.status}"
        )
    value = gamma_sq.value
    if value is None or not np.isfinite(value) or value < 0:
        raise RuntimeError("Solver returned an invalid gamma_sq.")
    return float(value), float(np.sqrt(value))


def solve_algorithm1_example2(tol=1e-3, max_iterations=50, solver="MOSEK",
                              verbose=False, return_filter=False):
    """Run Algorithm 1 with the published Example 2 initialization.

    Both subproblems minimize gamma squared, which has the same minimizers
    as gamma. The stopping test uses gamma itself. History entries contain
    both costs, the fixed X/Y/Z snapshots, and the updated Ecal matrices.
    The strict-LMI margin remains the Theorem 1 default, 1e-8.

    Non-optimal solves (including optimal_inaccurate) raise RuntimeError;
    solver exceptions propagate. At the iteration limit, return the last
    solution with status='max_iterations' and converged=False. Numerical
    increases larger than 1e-6 + 1e-5 * abs(previous cost) raise RuntimeError.
    Smaller increases are reported unchanged, never clipped.
    """
    if not np.isscalar(tol) or not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive.")
    if (isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, (int, np.integer))
            or max_iterations < 1):
        raise ValueError("max_iterations must be a positive integer.")
    system, _ = example2_uav_system()
    Upsilon = example2_emission_matrix()
    initial = np.vstack([np.eye(4)] * 4 + [np.array([[1., 0., 0., 0.]])])
    assert initial.shape == (17, 4)
    Ecal = [initial.copy() for _ in range(2)]
    history = []
    converged = False
    print("phi | gamma_bar | gamma | improvement")

    def check_decrease(previous, current):
        if current > previous + 1e-6 + 1e-5 * abs(previous):
            raise RuntimeError("Algorithm 1 cost increased beyond numerical tolerance.")

    for iteration in range(max_iterations):
        problem, objective, variables = build_theorem1_problem(system, Upsilon, Ecal)
        gamma_sq_bar, gamma_bar = _solve(
            problem, objective, solver, verbose, iteration, 1
        )
        if history:
            check_decrease(history[-1]["gamma"], gamma_bar)
        fixed = {key: [value.value.copy() for value in variables[key]]
                 for key in ("X", "Y", "Z")}
        problem, objective, variables = _build_step2_problem(system, Upsilon, fixed)
        gamma_sq, gamma = _solve(problem, objective, solver, verbose, iteration, 2)
        check_decrease(gamma_bar, gamma)
        Ecal = [value.value.copy() for value in variables["Ecal"]]
        improvement = gamma_bar - gamma
        history.append({
            "phi": iteration, "gamma_bar": gamma_bar, "gamma": gamma,
            "gamma_sq_bar": gamma_sq_bar, "gamma_sq": gamma_sq,
            "improvement": improvement, "Ecal": [value.copy() for value in Ecal],
            **fixed,
        })
        print(f"{iteration} | {gamma_bar:.10g} | {gamma:.10g} | {improvement:.10g}")
        if improvement < tol:
            converged = True
            break

    result = {
        "status": problem.status if converged else "max_iterations",
        "solver_status": problem.status, "converged": converged,
        "gamma": gamma, "gamma_sq": gamma_sq, "iterations": len(history),
        "history": history, "Ecal": Ecal,
    }
    if return_filter:
        result.update(recover_filter_matrices(Upsilon, variables))
    return result
