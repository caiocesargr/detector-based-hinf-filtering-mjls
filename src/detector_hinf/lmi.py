"""Reusable CVXPY helpers for real-valued linear matrix inequalities."""

import cvxpy as cp
import numpy as np


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
