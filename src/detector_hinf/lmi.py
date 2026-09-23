"""Reusable CVXPY helpers for real-valued linear matrix inequalities."""

import cvxpy as cp
import numpy as np

from .detector import (
    detector_symbol_sets,
    phi_map,
    psi_map,
    weighted_detector_average,
)
from .models import example1_emission_matrix, example1_system


def _strict_lmi_arguments(X, eps):
    """Validate a square expression and a positive numerical margin."""
    X = cp.Expression.cast_to_const(X)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError("X must be a square matrix expression.")
    if not np.isscalar(eps) or not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite, positive scalar.")
    return X, eps * np.eye(X.shape[0])


def positive_definite_constraint(X, eps=1e-8):
    """Return the CVXPY constraint ``X >> eps * I``.

    The positive margin approximates strict positive definiteness with a
    non-strict PSD constraint. Supply a symmetric, real matrix expression;
    CVXPY otherwise constrains only its symmetric part. Solver tolerances
    determine the numerical accuracy with which the margin is enforced.
    """
    X, margin = _strict_lmi_arguments(X, eps)
    return X >> margin


def negative_definite_constraint(X, eps=1e-8):
    """Return ``X << -eps * I``, approximating strict negative definiteness.

    Supply a symmetric, real matrix expression. As with the positive helper,
    the PSD constraint acts on the symmetric part and is subject to solver
    tolerances.
    """
    X, margin = _strict_lmi_arguments(X, eps)
    return X << -margin


def her(X):
    """Return ``X + X.T`` (the unscaled symmetric part of a real matrix)."""
    return X + X.T


def _numeric_matrix(value, name, shape=None):
    """Validate fixed real matrix data shared by the LMI builders."""
    array = np.asarray(value)
    if (array.ndim != 2 or 0 in array.shape
            or array.dtype.kind not in "biuf"
            or not np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be a finite, real numerical matrix.")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    return array.astype(float, copy=False)


def _system_data(system, Upsilon):
    """Validate the plant and detector dimensions for both LMI builders."""
    Upsilon = _numeric_matrix(Upsilon, "Upsilon")
    symbols = detector_symbol_sets(Upsilon, tol=0.0)
    if not np.allclose(Upsilon.sum(axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Upsilon must be row-stochastic.")
    n_modes = Upsilon.shape[0]
    A, B, C, D, L, F, Lambda = system
    plant_groups = (A, B, C, D, L, F)
    if any(len(group) != n_modes for group in plant_groups):
        raise ValueError("Each plant matrix list must contain one entry per mode.")
    nx = _numeric_matrix(A[0], "A[0]").shape[0]
    nw = _numeric_matrix(B[0], "B[0]").shape[1]
    ny = _numeric_matrix(C[0], "C[0]").shape[0]
    nz = _numeric_matrix(L[0], "L[0]").shape[0]
    plant_shapes = ((nx, nx), (nx, nw), (ny, nx), (ny, nw), (nz, nx), (nz, nw))
    A, B, C, D, L, F = [
        [_numeric_matrix(value, f"{name}[{i}]", shape)
         for i, value in enumerate(group)]
        for name, group, shape in zip("ABCDLF", plant_groups, plant_shapes)
    ]
    Lambda = _numeric_matrix(Lambda, "Lambda", (n_modes, n_modes))
    if (not np.allclose(Lambda.sum(axis=1), 0.0, rtol=0, atol=1e-12)
            or np.any(Lambda[~np.eye(n_modes, dtype=bool)] < 0)
            or np.any(np.diag(Lambda) > 0)):
        raise ValueError("Lambda must be a Markov transition-rate matrix.")
    return (A, B, C, D, L, F, Lambda), Upsilon, symbols, (nx, nw, ny, nz)


def build_brl_problem(system, Upsilon, filter_matrices, gamma, eps=1e-8):
    """Build, but do not solve, the Lemma 1 feasibility problem (15)--(16).

    ``system`` is ``(A, B, C, D, L, F, Lambda)``, as returned by
    example1_system. The first six entries contain one matrix per Markov
    mode. ``filter_matrices`` is either ``(Ahat, Bhat, Lhat, Ehat)`` or a
    dictionary with those keys; each entry contains one fixed numerical
    matrix per detector symbol. All indexing is zero-based.

    For plant dimensions nx, nw, ny, nz, filter matrices have shapes
    (nx, nx), (nx, ny), (nz, nx), (nz, ny), respectively. Upsilon must be
    row-stochastic. All data are finite and real, and gamma is a fixed
    positive scalar. Strictly positive emission probabilities are retained,
    including values below the detector helpers' default tolerance.

    Return ``(problem, P)``, where P is a list of symmetric (2*nx, 2*nx)
    decision variables. Each mode contributes P_i >> eps*I and an LMI of
    size 2*nx + nw + nz*len(D_i), constrained below -eps*I. The objective
    is zero; the only decision variables are P. No solver is invoked.
    """
    gamma_value = np.asarray(gamma)
    if (gamma_value.ndim != 0 or gamma_value.dtype.kind not in "biuf"
            or not np.isfinite(gamma_value) or gamma_value <= 0):
        raise ValueError("gamma must be a finite, positive numerical scalar.")
    gamma = float(gamma_value)
    system, Upsilon, symbols, (nx, nw, ny, nz) = _system_data(system, Upsilon)
    n_modes, n_symbols = Upsilon.shape
    A, B, C, D, L, F, Lambda = system

    filter_names = ("Ahat", "Bhat", "Lhat", "Ehat")
    if isinstance(filter_matrices, dict):
        filter_matrices = tuple(filter_matrices[name] for name in filter_names)
    Ahat, Bhat, Lhat, Ehat = filter_matrices
    filter_groups = (Ahat, Bhat, Lhat, Ehat)
    if any(len(group) != n_symbols for group in filter_groups):
        raise ValueError("Each filter matrix list must contain one entry per symbol.")
    filter_shapes = ((nx, nx), (nx, ny), (nz, nx), (nz, ny))
    Ahat, Bhat, Lhat, Ehat = [
        [_numeric_matrix(value, f"{name}[{ell}]", shape)
         for ell, value in enumerate(group)]
        for name, group, shape in zip(filter_names, filter_groups, filter_shapes)
    ]

    P = [cp.Variable((2 * nx, 2 * nx), symmetric=True, name=f"P_{i}")
         for i in range(n_modes)]
    constraints = [positive_definite_constraint(Pi, eps) for Pi in P]
    for i in range(n_modes):
        HA = weighted_detector_average(Upsilon, i, Ahat)
        HB = weighted_detector_average(Upsilon, i, Bhat)
        Abar = cp.bmat([[A[i], np.zeros((nx, nx))], [HB @ C[i], HA]])
        Bbar = cp.bmat([[B[i]], [HB @ D[i]]])
        Lrows, Erows = [], []
        for ell in sorted(symbols[i]):
            weight = np.sqrt(Upsilon[i, ell])
            Lbar = cp.bmat([[L[i] - Ehat[ell] @ C[i], -Lhat[ell]]])
            Ebar = F[i] - Ehat[ell] @ D[i]
            Lrows.append(weight * Lbar)
            Erows.append(weight * Ebar)
        Lstack = cp.vstack(Lrows)
        Estack = cp.vstack(Erows)
        T = her(P[i] @ Abar) + sum(Lambda[i, j] * P[j] for j in range(n_modes))
        PB = P[i] @ Bbar
        lmi = cp.bmat([
            [T, PB, Lstack.T],
            [PB.T, -gamma**2 * np.eye(nw), Estack.T],
            [Lstack, Estack, -np.eye(nz * len(symbols[i]))],
        ])
        constraints.append(negative_definite_constraint(lmi, eps))
    return cp.Problem(cp.Minimize(0), constraints), P


def build_theorem1_problem(system, Upsilon, Ecal, eps=1e-8):
    """Build Theorem 1's SDP for fixed design matrices, without solving it.

    ``system`` is (A, B, C, D, L, F, Lambda), as in build_brl_problem.
    ``Ecal[i]`` is a fixed, finite, real matrix of shape (4*nx + nw, nx).
    Upsilon is row-stochastic; all strictly positive emissions are used.
    Every detector symbol must have positive probability in at least one
    mode. Strict LMIs use the supplied positive margin eps.

    Return ``(problem, gamma_sq, variables)``. The objective minimizes the
    nonnegative scalar gamma_sq, representing gamma squared. The variables
    dictionary contains mode-indexed lists P and Q, dictionaries U, V, W
    keyed by active (i, ell) pairs, symbol-indexed lists X, Y, S, T, and a
    distinguishable-set-indexed list Z. It also includes integer arrays
    ``psi`` and ``phi``. All indices are zero-based. X[ell] and Y[ell]
    represent only X_{ell,phi(ell)} and Y_{ell,phi(ell)}.

    After solving, equation (23) recovers Ahat[ell] and Bhat[ell] by solving
    Z[phi[ell]] @ Ahat[ell] = X[ell] and Z[phi[ell]] @ Bhat[ell] = Y[ell],
    with Lhat[ell] = S[ell] and Ehat[ell] = T[ell], using variable values.
    Recovery requires nonsingular Z values, which must be checked at that
    stage; no nonconvex invertibility constraint is added here.
    """
    system, Upsilon, symbols, (nx, nw, ny, nz) = _system_data(system, Upsilon)
    A, B, C, D, L, F, Lambda = system
    n_modes, n_symbols = Upsilon.shape
    n_aug = 2 * nx
    n_lmi = 4 * nx + nw
    if len(Ecal) != n_modes:
        raise ValueError("Ecal must contain one fixed design matrix per mode.")
    Ecal = [_numeric_matrix(value, f"Ecal[{i}]", (n_lmi, nx))
            for i, value in enumerate(Ecal)]
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
    X = [cp.Variable((nx, nx), name=f"X_{ell}") for ell in range(n_symbols)]
    Y = [cp.Variable((nx, ny), name=f"Y_{ell}") for ell in range(n_symbols)]
    S = [cp.Variable((nz, nx), name=f"S_{ell}") for ell in range(n_symbols)]
    T = [cp.Variable((nz, ny), name=f"T_{ell}") for ell in range(n_symbols)]
    Z = [cp.Variable((nx, nx), name=f"Z_{nu}") for nu in range(n_sets)]

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
        "psi": psi, "phi": phi,
    }
    return cp.Problem(cp.Minimize(gamma_sq), constraints), gamma_sq, variables


def solve_theorem1_example1(rho, solver="MOSEK", verbose=False):
    """Solve Theorem 1 for one Example 1 emission parameter rho.

    Both modes use Ecal_i = vstack(I3, I3, I3, I3, zeros((1, 3))).
    Return a dictionary with the solver status, gamma_sq, and its square
    root gamma. For optimal or optimal_inaccurate status, finite objective
    values are returned as floats (clipped at zero for numerical roundoff).
    Otherwise gamma and gamma_sq are None. Solver or licensing errors
    propagate to the caller; the solver is never silently changed.
    """
    Upsilon = example1_emission_matrix(rho)
    E = np.vstack([np.eye(3), np.eye(3), np.eye(3), np.eye(3), np.zeros((1, 3))])
    problem, gamma_sq, _ = build_theorem1_problem(
        example1_system(), Upsilon, [E.copy(), E.copy()]
    )
    problem.solve(solver=solver, verbose=verbose)
    value = None
    if (problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            and gamma_sq.value is not None and np.isfinite(gamma_sq.value)):
        value = max(0.0, float(gamma_sq.value))
    return {
        "status": problem.status,
        "gamma": None if value is None else float(np.sqrt(value)),
        "gamma_sq": value,
    }
