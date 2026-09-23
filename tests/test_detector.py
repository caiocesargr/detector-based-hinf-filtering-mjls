"""Tests for detector supports, overlap components, and mode mappings."""

import numpy as np
import pytest

from detector_hinf.detector import (
    detector_symbol_sets,
    distinguishable_sets,
    psi_map,
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
