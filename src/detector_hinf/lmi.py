"""Reusable CVXPY helpers for real-valued linear matrix inequalities."""

import cvxpy as cp
import numpy as np

from .detector import detector_symbol_sets, weighted_detector_average


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
    def matrix(value, name, shape=None):
        array = np.asarray(value)
        if (array.ndim != 2 or 0 in array.shape
                or array.dtype.kind not in "biuf"
                or not np.all(np.isfinite(array))):
            raise ValueError(f"{name} must be a finite, real numerical matrix.")
        if shape is not None and array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
        return array.astype(float, copy=False)

    gamma_value = np.asarray(gamma)
    if (gamma_value.ndim != 0 or gamma_value.dtype.kind not in "biuf"
            or not np.isfinite(gamma_value) or gamma_value <= 0):
        raise ValueError("gamma must be a finite, positive numerical scalar.")
    gamma = float(gamma_value)
    Upsilon = matrix(Upsilon, "Upsilon")
    symbols = detector_symbol_sets(Upsilon, tol=0.0)
    if not np.allclose(Upsilon.sum(axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Upsilon must be row-stochastic.")
    n_modes, n_symbols = Upsilon.shape
    A, B, C, D, L, F, Lambda = system
    plant_groups = (A, B, C, D, L, F)
    if any(len(group) != n_modes for group in plant_groups):
        raise ValueError("Each plant matrix list must contain one entry per mode.")
    nx = matrix(A[0], "A[0]").shape[0]
    nw = matrix(B[0], "B[0]").shape[1]
    ny = matrix(C[0], "C[0]").shape[0]
    nz = matrix(L[0], "L[0]").shape[0]
    plant_shapes = ((nx, nx), (nx, nw), (ny, nx), (ny, nw), (nz, nx), (nz, nw))
    A, B, C, D, L, F = [
        [matrix(value, f"{name}[{i}]", shape) for i, value in enumerate(group)]
        for name, group, shape in zip("ABCDLF", plant_groups, plant_shapes)
    ]
    Lambda = matrix(Lambda, "Lambda", (n_modes, n_modes))
    if (not np.allclose(Lambda.sum(axis=1), 0.0, rtol=0, atol=1e-12)
            or np.any(Lambda[~np.eye(n_modes, dtype=bool)] < 0)
            or np.any(np.diag(Lambda) > 0)):
        raise ValueError("Lambda must be a Markov transition-rate matrix.")

    filter_names = ("Ahat", "Bhat", "Lhat", "Ehat")
    if isinstance(filter_matrices, dict):
        filter_matrices = tuple(filter_matrices[name] for name in filter_names)
    Ahat, Bhat, Lhat, Ehat = filter_matrices
    filter_groups = (Ahat, Bhat, Lhat, Ehat)
    if any(len(group) != n_symbols for group in filter_groups):
        raise ValueError("Each filter matrix list must contain one entry per symbol.")
    filter_shapes = ((nx, nx), (nx, ny), (nz, nx), (nz, ny))
    Ahat, Bhat, Lhat, Ehat = [
        [matrix(value, f"{name}[{ell}]", shape) for ell, value in enumerate(group)]
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
