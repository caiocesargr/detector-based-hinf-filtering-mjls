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


def build_augmented_generator(Lambda, detector_generators, Upsilon, epsilon):
    """Build the continuous-time generator of (theta, theta_hat) in (3).

    For N Markov modes and M detector symbols, return an (N*M, N*M) array.
    State (i, k) has zero-based index i*M + k; rows are source states and
    columns are destination states. Markov jumps use the destination mode's
    emission probabilities, Lambda[i, j] * Upsilon[j, ell]. Within mode i,
    detector off-diagonal rates are detector_generators[i][k, ell]/epsilon.
    Diagonals are minus the sum of each row's off-diagonal rates.

    Inputs must be finite real matrices: Lambda is an (N, N) generator,
    detector_generators contains N unscaled (M, M) generators, and Upsilon
    is row-stochastic. Epsilon must be a finite positive scalar. This helper
    constructs rates only; it does not simulate paths or impose an invariant
    distribution condition between Upsilon and the detector generators.
    """
    def real_matrix(value, name):
        array = np.asarray(value)
        if (array.ndim != 2 or 0 in array.shape or array.dtype.kind not in "biuf"
                or not np.all(np.isfinite(array))):
            raise ValueError(f"{name} must be a nonempty finite real matrix.")
        return array.astype(float, copy=False)

    def generator(value, size, name):
        array = real_matrix(value, name)
        if array.shape != (size, size):
            raise ValueError(f"{name} must have shape {(size, size)}.")
        off_diagonal = array[~np.eye(size, dtype=bool)]
        if (np.any(off_diagonal < 0) or np.any(np.diag(array) > 0)
                or not np.allclose(array.sum(axis=1), 0, rtol=0, atol=1e-12)):
            raise ValueError(f"{name} must be a Markov generator.")
        return array

    epsilon = np.asarray(epsilon)
    if (epsilon.ndim != 0 or epsilon.dtype.kind not in "biuf"
            or not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("epsilon must be a finite positive scalar.")
    epsilon = float(epsilon)
    Upsilon = real_matrix(Upsilon, "Upsilon")
    if (np.any(Upsilon < 0)
            or not np.allclose(Upsilon.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("Upsilon must be row-stochastic.")
    n_modes, n_symbols = Upsilon.shape
    Lambda = generator(Lambda, n_modes, "Lambda")
    if len(detector_generators) != n_modes:
        raise ValueError("Supply one detector generator per Markov mode.")
    detectors = [generator(value, n_symbols, f"detector_generators[{i}]")
                 for i, value in enumerate(detector_generators)]

    augmented = np.zeros((n_modes * n_symbols, n_modes * n_symbols))
    for i in range(n_modes):
        source = slice(i * n_symbols, (i + 1) * n_symbols)
        for j in range(n_modes):
            destination = slice(j * n_symbols, (j + 1) * n_symbols)
            if i == j:
                augmented[source, destination] = detectors[i] / epsilon
            else:
                augmented[source, destination] = Lambda[i, j] * Upsilon[j]
    np.fill_diagonal(augmented, 0.0)
    np.fill_diagonal(augmented, -augmented.sum(axis=1))
    return augmented
