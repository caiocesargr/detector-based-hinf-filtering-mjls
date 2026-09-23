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


def example2_markov_generator():
    """Return the Example 2 plant-mode generator, ordered as modes 0 and 1."""
    return np.array([[-2.5, 2.5], [9.0, -9.0]])


def example2_detector_generators():
    """Return the unscaled detector generators in equations (29a)--(29b).

    List indices 0 and 1 correspond to the paper's modes 1 and 2. Each
    (3, 3) array uses detector-symbol order 0, 1, 2. The factor 1e-5 is
    included; equation (3)'s division by epsilon is applied separately.

    Their exact stationary distributions are [1, 1e-5, 1e-5]/(1+2e-5)
    and [2e-5, 0.5, 0.5]/(1+2e-5). These approximate, but do not exactly
    equal, the emission rows reported in equation (30).
    """
    return [
        1e-5 * np.array([[-2.0, 1.0, 1.0],
                         [1e5, -1e5, 0.0],
                         [1e5, 0.0, -1e5]]),
        1e-5 * np.array([[-1e5, 5e4, 5e4],
                         [2.0, -2.0, 0.0],
                         [2.0, 0.0, -2.0]]),
    ]


def example2_emission_matrix():
    """Return the reported (2, 3) emission matrix in equation (30).

    Rows index Markov modes and columns index detector symbols, starting
    at zero. The rows approximate the stationary distributions of the
    finite-rate generators returned by example2_detector_generators.
    """
    return np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]])


def example2_uav_system():
    """Return ``(system, metadata)`` for the Example 2 UAV plant.

    ``system`` is the tuple (A, B, C, D, L, F, Lambda), compatible with the
    LMI builders. Each mode-dependent quantity is a list of two floating-
    point arrays, indexed 0 and 1 for the paper's modes 1 and 2. Dimensions
    are nx=4, nw=1, ny=2, nz=1; scalar F entries remain (1, 1) matrices.

    The paper also prints C3 despite defining only two plant modes.
    It is preserved as metadata["C3_reported"], separately from C, and is
    not used as a plant mode. Pass only ``system`` to the LMI builders.
    All arrays are freshly allocated, with no sharing between plant modes.
    """
    common_A = np.array([
        [-0.3522, 0.2585, -4.2749, -9.4851],
        [-0.6782, -1.8272, 16.4537, -2.4644],
        [0.0948, -0.3649, -0.3392, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    kappa = (0.1, 0.75)
    b0 = np.array([[-0.6759], [2.6014], [-8.4335], [0.0]])
    A = [common_A.copy(), common_A.copy()]
    B = [value * b0 for value in kappa]
    C = [
        np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    ]
    D = [value * np.ones((2, 1)) for value in kappa]
    L = [np.array([[0.0, 0.0, 0.0, 1.0]]) for _ in kappa]
    F = [np.array([[value]]) for value in kappa]
    metadata = {
        "nx": 4, "nw": 1, "ny": 2, "nz": 1,
        "C3_reported": np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
    }
    return (A, B, C, D, L, F, example2_markov_generator()), metadata
