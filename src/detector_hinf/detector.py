"""Detector support sets and their induced partition of Markov modes."""

import numpy as np


def detector_symbol_sets(Upsilon, tol=1e-12):
    """Return a list of sets D_i = {ell: Upsilon[i, ell] > tol}.

    Rows denote Markov modes and columns detector symbols; both use
    zero-based indices. Upsilon must be a nonempty, finite, nonnegative
    two-dimensional array. Probabilities are not normalized or modified.
    Thresholding may leave a mode with an empty symbol set.
    """
    Upsilon = np.asarray(Upsilon, dtype=float)
    if Upsilon.ndim != 2 or 0 in Upsilon.shape:
        raise ValueError("Upsilon must be a nonempty two-dimensional matrix.")
    if not np.all(np.isfinite(Upsilon)) or np.any(Upsilon < 0):
        raise ValueError("Upsilon must contain finite, nonnegative entries.")
    if not np.isscalar(tol) or not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a finite, nonnegative scalar.")
    return [set(np.flatnonzero(row > tol).tolist()) for row in Upsilon]


def distinguishable_sets(Upsilon, tol=1e-12):
    """Return the finest partition induced by connected support overlaps.

    Modes sharing a detector symbol belong to the same component, including
    modes linked only through intermediate modes. Return a list of sets of
    zero-based mode indices, ordered by each component's smallest mode.
    A mode with empty support forms a singleton component.
    """
    symbols = detector_symbol_sets(Upsilon, tol)
    remaining = set(range(len(symbols)))
    partition = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        component = {first}
        pending = [first]
        while pending:
            mode = pending.pop()
            neighbors = {j for j in remaining if symbols[mode] & symbols[j]}
            remaining.difference_update(neighbors)
            component.update(neighbors)
            pending.extend(neighbors)
        partition.append(component)
    return partition


def psi_map(Upsilon, tol=1e-12):
    """Return an integer array mapping each mode to its partition index.

    Both modes and distinguishable-set indices are zero-based. Set indices
    follow the deterministic ordering returned by distinguishable_sets.
    """
    partition = distinguishable_sets(Upsilon, tol)
    psi = np.empty(sum(len(component) for component in partition), dtype=int)
    for index, component in enumerate(partition):
        for mode in component:
            psi[mode] = index
    return psi


def phi_map(Upsilon, tol=1e-12):
    """Return an integer array mapping detector symbols to partition indices.

    All indices are zero-based, with the same distinguishable-set ordering
    as psi_map and distinguishable_sets. For every ell in D_i,
    phi[ell] = psi[i]: shared symbols connect their emitting modes, so this
    assignment is unique. Raise ValueError if a symbol has no emission
    probability above tol, since it has no associated distinguishable set.
    """
    symbols = detector_symbol_sets(Upsilon, tol)
    psi = psi_map(Upsilon, tol)
    phi = np.full(np.asarray(Upsilon).shape[1], -1, dtype=int)
    for mode, emitted in enumerate(symbols):
        for ell in emitted:
            phi[ell] = psi[mode]
    if np.any(phi < 0):
        raise ValueError("Every detector symbol must have an emission probability above tol.")
    return phi


def weighted_detector_average(Upsilon, mode, matrices):
    """Return sum_ell Upsilon[mode, ell] * matrices[ell] as a NumPy array.

    Mode and detector-symbol indices are zero-based. Supply one numeric,
    two-dimensional matrix per detector symbol, all with the same shape.
    Only strictly positive probabilities contribute, without thresholding
    or renormalization. An all-zero row returns a zero matrix. Inputs are
    not modified, and zero-probability matrices are not multiplied or added.
    """
    symbols = detector_symbol_sets(Upsilon, tol=0.0)
    Upsilon = np.asarray(Upsilon, dtype=float)
    if not isinstance(mode, (int, np.integer)) or not 0 <= mode < len(symbols):
        raise ValueError("mode must be a valid zero-based integer mode index.")
    if len(matrices) != Upsilon.shape[1]:
        raise ValueError("Supply one matrix per detector symbol.")
    arrays = [np.asarray(matrix) for matrix in matrices]
    shape = arrays[0].shape
    if any(matrix.ndim != 2 or matrix.shape != shape for matrix in arrays):
        raise ValueError("All matrices must be two-dimensional with the same shape.")
    if any(matrix.dtype.kind not in "biufc" for matrix in arrays):
        raise ValueError("Matrices must contain numeric entries.")
    dtype = np.result_type(float, *(matrix.dtype for matrix in arrays))
    average = np.zeros(shape, dtype=dtype)
    for ell in sorted(symbols[mode]):
        average += Upsilon[mode, ell] * arrays[ell]
    return average
