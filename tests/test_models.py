"""Structural and probability checks for the paper's example models."""

import numpy as np
import pytest

from detector_hinf.models import (
    example1_detector_generators,
    example1_emission_matrix,
    example1_system,
    example2_detector_generators,
    example2_emission_matrix,
    example2_markov_generator,
    example2_published_algorithm1_filter,
    example2_uav_system,
)


@pytest.mark.parametrize("key,shape", [
    ("Ahat", (4, 4)), ("Bhat", (4, 2)),
    ("Lhat", (1, 4)), ("Ehat", (1, 2)),
])
def test_example2_published_filter_dimensions(key, shape):
    matrices = example2_published_algorithm1_filter()[key]
    assert isinstance(matrices, list)
    assert len(matrices) == 3
    for matrix in matrices:
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == shape
        assert np.isfinite(matrix).all()


def test_example2_published_filter_output_coefficients():
    reported = example2_published_algorithm1_filter()
    np.testing.assert_array_equal(reported["Lhat"], [
        [[0.0016, 0.0517, 0.0309, -1.0012]],
        [[-0.1044, 0.0370, -0.1884, 0.3614e-6]],
        [[0.1044, 0.0370, -0.1884, 0.3613e-6]],
    ])
    np.testing.assert_array_equal(reported["Ehat"], [
        [[0.0525, 0.0395]], [[0., 1.]], [[0., 1.]],
    ])


def test_example1_matrix_dimensions():
    A, B, C, D, L, F, Lambda = example1_system()

    for matrices, shape in (
        (A, (3, 3)),
        (B, (3, 1)),
        (C, (1, 3)),
        (D, (1, 1)),
        (L, (1, 3)),
        (F, (1, 1)),
    ):
        assert isinstance(matrices, list)
        assert len(matrices) == 2
        for matrix in matrices:
            assert isinstance(matrix, np.ndarray)
            assert matrix.shape == shape

    assert isinstance(Lambda, np.ndarray)
    assert Lambda.shape == (2, 2)


def test_example1_markov_generator():
    *_, Lambda = example1_system()

    assert np.all(np.isfinite(Lambda))
    np.testing.assert_allclose(Lambda.sum(axis=1), 0.0, rtol=0, atol=1e-12)
    assert np.all(np.diag(Lambda) <= 0.0)
    off_diagonal = Lambda[~np.eye(2, dtype=bool)]
    assert np.all(off_diagonal >= 0.0)


@pytest.mark.parametrize("rho", [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0])
def test_example1_emission_matrix_is_row_stochastic(rho):
    emission = example1_emission_matrix(rho)

    assert isinstance(emission, np.ndarray)
    assert emission.shape == (2, 2)
    assert np.all(np.isfinite(emission))
    assert np.all(emission >= 0.0)
    assert np.all(emission <= 1.0)
    np.testing.assert_allclose(emission.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_example1_unscaled_detector_generators():
    generators = example1_detector_generators()
    assert isinstance(generators, list) and len(generators) == 2
    np.testing.assert_array_equal(generators[0], [[-0.009, 0.009], [0.036, -0.036]])
    np.testing.assert_array_equal(generators[1], [[-0.036, 0.036], [0.009, -0.009]])
    for i, generator in enumerate(generators):
        assert generator.shape == (2, 2)
        assert np.all(generator[~np.eye(2, dtype=bool)] >= 0)
        np.testing.assert_allclose(generator.sum(axis=1), 0, atol=1e-12)
        np.testing.assert_allclose(example1_emission_matrix(0.2)[i] @ generator,
                                   0, atol=1e-12)
    generators[0][0, 0] = 0
    assert example1_detector_generators()[0][0, 0] == -0.009


def test_example2_markov_generator():
    generator = example2_markov_generator()
    assert generator.shape == (2, 2)
    np.testing.assert_array_equal(generator, [[-2.5, 2.5], [9.0, -9.0]])
    assert np.all(np.isfinite(generator))
    assert np.all(generator[~np.eye(2, dtype=bool)] >= 0)
    np.testing.assert_allclose(generator.sum(axis=1), 0, rtol=0, atol=1e-12)


def test_example2_detector_generators():
    generators = example2_detector_generators()
    assert isinstance(generators, list) and len(generators) == 2
    expected = [
        [[-2e-5, 1e-5, 1e-5], [1., -1., 0.], [1., 0., -1.]],
        [[-1., 0.5, 0.5], [2e-5, -2e-5, 0.], [2e-5, 0., -2e-5]],
    ]
    for generator, reference in zip(generators, expected):
        assert generator.shape == (3, 3)
        assert np.all(np.isfinite(generator))
        assert np.all(generator[~np.eye(3, dtype=bool)] >= 0)
        assert np.all(np.diag(generator) <= 0)
        np.testing.assert_allclose(generator, reference, rtol=1e-14, atol=0)
        np.testing.assert_allclose(generator.sum(axis=1), 0, rtol=0, atol=1e-12)


def test_example2_emission_matrix():
    emission = example2_emission_matrix()
    assert emission.shape == (2, 3)
    np.testing.assert_array_equal(emission, [[1., 0., 0.], [0., 0.5, 0.5]])
    assert np.all(np.isfinite(emission))
    assert np.all((emission >= 0) & (emission <= 1))
    np.testing.assert_allclose(emission.sum(axis=1), 1, rtol=0, atol=1e-12)


@pytest.mark.parametrize("mode", [0, 1])
def test_example2_stationary_distributions_approximate_reported_emissions(mode):
    generator = example2_detector_generators()[mode]
    # Solve pi @ Q = 0 with sum(pi) = 1 independently of the reported rows.
    coefficients = generator.T.copy()
    coefficients[-1] = 1.0
    stationary = np.linalg.solve(coefficients, [0., 0., 1.])
    exact = np.array([[1., 1e-5, 1e-5], [2e-5, 0.5, 0.5]]) / (1 + 2e-5)
    np.testing.assert_allclose(stationary, exact[mode], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(stationary @ generator, 0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(stationary.sum(), 1, rtol=0, atol=1e-12)
    assert np.all(stationary >= 0)
    # Equation (30) omits small probabilities present for the finite rates
    # in (29). The maximum difference is 2e-5/(1+2e-5), not roundoff.
    reported = example2_emission_matrix()[mode]
    np.testing.assert_allclose(stationary, reported, rtol=0, atol=2e-5)
    expected_residual = np.array([[-2e-5, 1e-5, 1e-5], [2e-5, -1e-5, -1e-5]])
    np.testing.assert_allclose(reported @ generator, expected_residual[mode],
                               rtol=1e-12, atol=1e-15)


def test_example2_uav_dimensions_and_metadata():
    system, metadata = example2_uav_system()
    assert len(system) == 7
    for matrices, shape in zip(system[:-1], [(4, 4), (4, 1), (2, 4), (2, 1), (1, 4), (1, 1)]):
        assert isinstance(matrices, list) and len(matrices) == 2
        for matrix in matrices:
            assert matrix.shape == shape
            assert matrix.dtype.kind == "f"
            assert np.all(np.isfinite(matrix))
    assert system[-1].shape == (2, 2)
    np.testing.assert_array_equal(system[-1], example2_markov_generator())
    assert [metadata[key] for key in ("nx", "nw", "ny", "nz")] == [4, 1, 2, 1]
    np.testing.assert_array_equal(metadata["C3_reported"], [[1, 0, 0, 0], [0, 1, 0, 0]])
    # The reported extra matrix is not a third plant measurement mode.
    assert len(system[2]) == 2
    assert all(not np.array_equal(C, metadata["C3_reported"]) for C in system[2])


def test_example2_uav_matrix_values():
    (A, B, C, D, L, F, _), _ = example2_uav_system()
    expected_A = [
        [-0.3522, 0.2585, -4.2749, -9.4851],
        [-0.6782, -1.8272, 16.4537, -2.4644],
        [0.0948, -0.3649, -0.3392, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    expected_B = [[[-0.06759], [0.26014], [-0.84335], [0.0]],
                  [[-0.506925], [1.95105], [-6.325125], [0.0]]]
    for i, kappa in enumerate((0.1, 0.75)):
        np.testing.assert_array_equal(A[i], expected_A)
        np.testing.assert_allclose(B[i], expected_B[i], rtol=1e-14, atol=0)
        np.testing.assert_array_equal(D[i], [[kappa], [kappa]])
        np.testing.assert_array_equal(L[i], [[0, 0, 0, 1]])
        np.testing.assert_array_equal(F[i], [[kappa]])
    np.testing.assert_array_equal(C[0], [[0, 1, 0, 0], [0, 0, 1, 0]])
    np.testing.assert_array_equal(C[1], [[1, 0, 0, 0], [0, 0, 0, 1]])


def test_example2_uav_returns_independent_arrays():
    system, metadata = example2_uav_system()
    other_system, other_metadata = example2_uav_system()
    for matrices, other_matrices in zip(system[:-1], other_system[:-1]):
        assert not np.shares_memory(matrices[0], matrices[1])
        for matrix, other in zip(matrices, other_matrices):
            assert not np.shares_memory(matrix, other)
    assert not np.shares_memory(system[-1], other_system[-1])
    assert not np.shares_memory(metadata["C3_reported"], other_metadata["C3_reported"])
