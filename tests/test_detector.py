"""Tests for detector supports, overlap components, and mode mappings."""

import numpy as np
import pytest

from detector_hinf.detector import (
    build_augmented_generator,
    detector_symbol_sets,
    distinguishable_sets,
    phi_map,
    psi_map,
    weighted_detector_average,
)
from detector_hinf.models import (
    example1_detector_generators,
    example1_emission_matrix,
    example1_system,
)


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


@pytest.mark.parametrize(
    "rho, expected",
    [(0.0, [0, 1]), (0.2, [0, 0]), (0.5, [0, 0]), (1.0, [1, 0])],
)
def test_example1_phi_map(rho, expected):
    Upsilon = example1_emission_matrix(rho)
    phi = phi_map(Upsilon)
    assert phi.shape == (2,)
    assert np.issubdtype(phi.dtype, np.integer)
    np.testing.assert_array_equal(phi, expected)


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
    np.testing.assert_array_equal(phi_map(Upsilon), [0, 0, 1])
    phi = phi_map(Upsilon)
    psi = psi_map(Upsilon)
    for mode, symbols in enumerate(detector_symbol_sets(Upsilon)):
        for ell in symbols:
            assert phi[ell] == psi[mode]


def test_tolerance_is_strict_and_propagated():
    Upsilon = [[0.75, 0.25], [0.0, 1.0]]
    assert detector_symbol_sets(Upsilon, tol=0.25) == [{0}, {1}]
    assert distinguishable_sets(Upsilon, tol=0.25) == [{0}, {1}]
    np.testing.assert_array_equal(psi_map(Upsilon, tol=0.25), [0, 1])
    np.testing.assert_array_equal(phi_map(Upsilon, tol=0.25), [0, 1])
    assert distinguishable_sets(Upsilon, tol=0.0) == [{0, 1}]
    np.testing.assert_array_equal(psi_map(Upsilon, tol=0.0), [0, 0])
    np.testing.assert_array_equal(phi_map(Upsilon, tol=0.0), [0, 0])


@pytest.mark.parametrize("probability", [0.0, 1e-12])
def test_phi_map_rejects_symbols_without_thresholded_support(probability):
    with pytest.raises(ValueError, match="Every detector symbol"):
        phi_map([[1.0 - probability, probability]])


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


@pytest.mark.parametrize("epsilon", [1.0, 0.075, 0.001])
def test_example1_augmented_generator(epsilon):
    Lambda = example1_system()[-1]
    detectors = example1_detector_generators()
    Upsilon = example1_emission_matrix(0.2)
    augmented = build_augmented_generator(Lambda, detectors, Upsilon, epsilon)
    assert augmented.shape == (4, 4)
    assert np.all(augmented[~np.eye(4, dtype=bool)] >= 0)
    np.testing.assert_allclose(augmented.sum(axis=1), 0, rtol=0, atol=1e-12)
    # Detector switches scale with epsilon; plant jumps reset the detector
    # according to the destination mode, independently of its previous symbol.
    np.testing.assert_allclose(augmented[:2, 2:], [[0.1, 0.4], [0.1, 0.4]])
    np.testing.assert_allclose(augmented[2:, :2], [[0.24, 0.06], [0.24, 0.06]])
    np.testing.assert_allclose(
        [augmented[0, 1], augmented[1, 0], augmented[2, 3], augmented[3, 2]],
        np.array([0.009, 0.036, 0.036, 0.009]) / epsilon,
    )
    expected_diagonal = np.repeat(np.diag(Lambda), 2) + np.concatenate(
        [np.diag(generator) for generator in detectors]
    ) / epsilon
    np.testing.assert_allclose(np.diag(augmented), expected_diagonal)


def test_example1_augmented_generator_explicit_rates():
    augmented = build_augmented_generator(
        example1_system()[-1], example1_detector_generators(),
        example1_emission_matrix(0.2), 0.075,
    )
    np.testing.assert_allclose(augmented, [
        [-0.62, 0.12, 0.10, 0.40],
        [0.48, -0.98, 0.10, 0.40],
        [0.24, 0.06, -0.78, 0.48],
        [0.24, 0.06, 0.12, -0.42],
    ])


def test_augmented_generator_with_three_symbols_preserves_plant_rates():
    Lambda = np.array([[-2., 2.], [3., -3.]])
    detectors = [np.array([[-1., 1., 0.], [0., -2., 2.], [3., 0., -3.]])] * 2
    Upsilon = np.array([[0.5, 0.5, 0.], [0., 0.25, 0.75]])
    before = [Lambda.copy(), Upsilon.copy(), detectors[0].copy()]
    augmented = build_augmented_generator(Lambda, detectors, Upsilon, 0.5)
    assert augmented.shape == (6, 6)
    assert np.all(augmented[~np.eye(6, dtype=bool)] >= 0)
    np.testing.assert_allclose(augmented.sum(axis=1), 0, atol=1e-12)
    for i in range(2):
        for j in range(2):
            block = augmented[3*i:3*(i+1), 3*j:3*(j+1)]
            np.testing.assert_allclose(block.sum(axis=1), Lambda[i, j], atol=1e-12)
    for actual, original in zip([Lambda, Upsilon, detectors[0]], before):
        np.testing.assert_array_equal(actual, original)


def test_augmented_generator_single_absorbing_state():
    np.testing.assert_array_equal(
        build_augmented_generator([[0.]], [np.zeros((1, 1))], [[1.]], 0.1), [[0.]]
    )


@pytest.mark.parametrize("epsilon", [0, -0.1, np.nan, np.inf, [0.1]])
def test_augmented_generator_rejects_invalid_epsilon(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        build_augmented_generator(example1_system()[-1], example1_detector_generators(),
                                  example1_emission_matrix(0.2), epsilon)


@pytest.mark.parametrize("invalid", ["negative_rate", "row_sum", "shape", "count", "emission"])
def test_augmented_generator_validates_inputs(invalid):
    Lambda = example1_system()[-1]
    detectors = example1_detector_generators()
    Upsilon = example1_emission_matrix(0.2)
    if invalid == "negative_rate":
        Lambda[0] = [0.5, -0.5]
    elif invalid == "row_sum":
        detectors[0][0, 0] = -1.0
    elif invalid == "shape":
        detectors[0] = np.zeros((3, 3))
    elif invalid == "count":
        detectors.pop()
    else:
        Upsilon[0] = [0.1, 0.1]
    with pytest.raises(ValueError):
        build_augmented_generator(Lambda, detectors, Upsilon, 0.075)
