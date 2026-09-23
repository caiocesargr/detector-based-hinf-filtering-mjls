"""Structural and probability checks for the Example 1 system."""

import numpy as np
import pytest

from detector_hinf.models import example1_emission_matrix, example1_system


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
