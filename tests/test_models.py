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
)


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
