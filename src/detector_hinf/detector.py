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
