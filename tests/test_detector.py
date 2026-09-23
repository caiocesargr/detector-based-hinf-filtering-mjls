"""Tests for detector supports, overlap components, and mode mappings."""

import numpy as np
import pytest

from detector_hinf.detector import (
    detector_symbol_sets,
    distinguishable_sets,
    psi_map,
    weighted_detector_average,
)
from detector_hinf.models import example1_emission_matrix


@pytest.mark.parametrize(
    "rho, symbols, partition, psi",
    [
        (0.0, [{0}, {1}], [{0}, {1}], [0, 1]),
        (0.2, [{0, 1}, {0, 1}], [{0, 1}], [0, 0]),
        (0.5, [{0, 1}, {0, 1}], [{0, 1}], [0, 0]),
        (1.0, [{1}, {0}], [{0}, {1}], [0, 1]),
    ],
)
def test_example1_detector(rho, symbols, partition, psi):
    Upsilon = example1_emission_matrix(rho)
    assert detector_symbol_sets(Upsilon) == symbols
    assert distinguishable_sets(Upsilon) == partition
    np.testing.assert_array_equal(psi_map(Upsilon), psi)


def test_transitive_overlaps_and_component_order():
    # Modes 0 and 2 have disjoint supports but are linked through mode 3.
    Upsilon = [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],
    ]
    assert detector_symbol_sets(Upsilon) == [{0}, {2}, {1}, {0, 1}]
    assert distinguishable_sets(Upsilon) == [{0, 2, 3}, {1}]
    np.testing.assert_array_equal(psi_map(Upsilon), [0, 1, 0, 0])


def test_tolerance_is_strict_and_propagated():
    Upsilon = [[0.75, 0.25], [0.0, 1.0]]
    assert detector_symbol_sets(Upsilon, tol=0.25) == [{0}, {1}]
    assert distinguishable_sets(Upsilon, tol=0.25) == [{0}, {1}]
    np.testing.assert_array_equal(psi_map(Upsilon, tol=0.25), [0, 1])
    assert distinguishable_sets(Upsilon, tol=0.0) == [{0, 1}]
    np.testing.assert_array_equal(psi_map(Upsilon, tol=0.0), [0, 0])


def test_empty_thresholded_supports_are_singletons():
    Upsilon = [[0.5, 0.5], [0.5, 0.5]]
    assert detector_symbol_sets(Upsilon, tol=0.5) == [set(), set()]
    assert distinguishable_sets(Upsilon, tol=0.5) == [{0}, {1}]
    np.testing.assert_array_equal(psi_map(Upsilon, tol=0.5), [0, 1])


@pytest.mark.parametrize("Upsilon", [[], [1.0, 0.0], [[np.nan]], [[-0.1]]])
def test_invalid_matrix(Upsilon):
    with pytest.raises(ValueError):
        detector_symbol_sets(Upsilon)


@pytest.mark.parametrize("tol", [-1e-12, np.nan, np.inf, [0.0]])
def test_invalid_tolerance(tol):
    with pytest.raises(ValueError):
        detector_symbol_sets([[1.0]], tol=tol)


@pytest.mark.parametrize(
    "mode, expected",
    [
        (0, [[4, 5, 6], [7, 8, 9]]),
        (1, [[2, 3, 4], [5, 6, 7]]),
    ],
)
def test_weighted_detector_average(mode, expected):
    Upsilon = np.array([[0.25, 0.75], [0.75, 0.25]])
    matrices = [np.array([[1, 2, 3], [4, 5, 6]]),
                np.array([[5, 6, 7], [8, 9, 10]])]
    originals = [matrix.copy() for matrix in matrices]
    result = weighted_detector_average(Upsilon, mode, matrices)
    np.testing.assert_allclose(result, expected)
    assert result.shape == (2, 3)
    for matrix, original in zip(matrices, originals):
        np.testing.assert_array_equal(matrix, original)
        assert not np.shares_memory(result, matrix)


def test_weighted_average_skips_zero_probability_symbols():
    matrices = [np.full((2, 1), np.nan), np.array([[2.0], [-3.0]])]
    result = weighted_detector_average([[0.0, 1.0]], 0, matrices)
    np.testing.assert_array_equal(result, [[2.0], [-3.0]])


def test_weighted_average_includes_tiny_positive_probabilities():
    result = weighted_detector_average(
        [[1e-15, 1.0 - 1e-15]], 0, [np.array([[1e15]]), np.array([[0.0]])]
    )
    np.testing.assert_allclose(result, [[1.0]])


def test_weighted_average_preserves_complex_values():
    result = weighted_detector_average(
        [[0.25, 0.75]], 0, [np.array([[2 + 4j]]), np.array([[6 - 4j]])]
    )
    np.testing.assert_allclose(result, [[5 - 2j]])


def test_weighted_average_of_zero_row():
    result = weighted_detector_average([[0.0]], 0, [np.full((2, 3), np.nan)])
    np.testing.assert_array_equal(result, np.zeros((2, 3)))


@pytest.mark.parametrize("mode", [-1, 1, 0.5])
def test_weighted_average_rejects_invalid_mode(mode):
    with pytest.raises(ValueError, match="mode"):
        weighted_detector_average([[1.0]], mode, [np.eye(2)])


def test_weighted_average_rejects_missing_matrix():
    with pytest.raises(ValueError, match="one matrix"):
        weighted_detector_average([[0.5, 0.5]], 0, [np.eye(2)])


def test_weighted_average_rejects_broadcastable_but_different_shapes():
    with pytest.raises(ValueError, match="same shape"):
        weighted_detector_average([[0.5, 0.5]], 0, [np.ones((2, 2)), np.ones((1, 2))])
