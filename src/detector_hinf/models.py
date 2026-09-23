"""System data for the paper's numerical examples."""

import numpy as np


def example1_system():
    """Return ``(A, B, C, D, L, F, Lambda)`` from Section 4.1.

    Each mode-dependent quantity is a list of two floating-point arrays:
    index 0 corresponds to mathematical mode 1, and index 1 to mode 2.
    All arrays are two-dimensional, including the scalar feedthrough matrices.
    Their shapes are A: (3, 3), B: (3, 1), C and L: (1, 3), and
    D and F: (1, 1).

    Lambda is the (2, 2) continuous-time Markov transition-rate matrix,
    with source modes along rows and destination modes along columns.
    Each call returns fresh lists and arrays.
    """
    A = [
        np.array([[-3.0, 1.0, 0.0],
                  [0.3, -2.5, 1.0],
                  [-0.1, 0.3, -3.8]]),
        np.array([[-2.5, 0.5, -0.1],
                  [0.1, -3.5, 0.3],
                  [-0.1, 1.0, -2.0]]),
    ]
    B = [
        np.array([[1.0], [0.0], [1.0]]),
        np.array([[-0.6], [0.5], [0.0]]),
    ]
    C = [
        np.array([[0.8, 0.3, 0.0]]),
        np.array([[-0.5, 0.2, 0.3]]),
    ]
    D = [np.array([[0.2]]), np.array([[0.5]])]
    L = [
        np.array([[0.5, -0.1, 1.0]]),
        np.array([[0.0, 1.0, 0.6]]),
    ]
    F = [np.array([[-1.0]]), np.array([[-1.0]])]
    Lambda = np.array([[-0.5, 0.5], [0.3, -0.3]])
    return A, B, C, D, L, F, Lambda


def example1_emission_matrix(rho):
    """Return the (2, 2) emission probability matrix in equation (28).

    Rows represent system modes and columns detector modes, both ordered
    as modes 1 and 2. Raise ValueError unless rho is a real scalar in [0, 1].
    """
    value = np.asarray(rho)
    if (value.ndim != 0 or value.dtype.kind not in "biuf"
            or not 0.0 <= value.item() <= 1.0):
        raise ValueError("rho must be a real scalar in [0, 1].")
    rho = float(value)
    return np.array([[1.0 - rho, rho], [rho, 1.0 - rho]])


def example1_detector_generators():
    """Return the two unscaled detector generators used in Example 1.

    List indices 0 and 1 correspond to the Markov modes. Rows and columns
    index detector symbols. Equation (3) divides these rates by epsilon;
    no scaling is applied here. Each call returns fresh arrays.
    """
    return [
        np.array([[-0.009, 0.009], [0.036, -0.036]]),
        np.array([[-0.036, 0.036], [0.009, -0.009]]),
    ]
