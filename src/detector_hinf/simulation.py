"""Event-driven simulation utilities for continuous-time Markov chains."""

import numpy as np
from scipy.integrate import solve_ivp

from .detector import weighted_detector_average
from .models import example1_system


def simulate_ctmc(Q, t_final, initial_state, rng):
    """Simulate a finite-state CTMC using exponential holding times.

    Q is a finite real generator with source states along rows. All state
    indices are zero-based. Supply a numpy.random.Generator; its state is
    advanced in place, and no global random state is used.

    Return equally sized arrays (jump_times, states), starting with time
    zero and initial_state. Each later entry records a genuine jump and
    the state immediately after it. Only jumps at or before t_final are
    included; t_final is not appended as an artificial jump. The last state
    persists until t_final. Absorbing states have no subsequent jumps.
    """
    Q = np.asarray(Q)
    if (Q.ndim != 2 or Q.shape[0] == 0 or Q.shape[0] != Q.shape[1]
            or Q.dtype.kind not in "biuf" or not np.all(np.isfinite(Q))):
        raise ValueError("Q must be a nonempty finite real square matrix.")
    Q = Q.astype(float, copy=False)
    rates = Q.copy()
    np.fill_diagonal(rates, 0.0)
    exit_rates = rates.sum(axis=1)
    if (np.any(rates < 0) or np.any(np.diag(Q) > 0)
            or not np.all(np.isfinite(exit_rates))
            or not np.allclose(np.diag(Q), -exit_rates, rtol=1e-10, atol=1e-12)):
        raise ValueError("Q must have nonnegative off-diagonal rates and zero row sums.")
    horizon = np.asarray(t_final)
    if (horizon.ndim != 0 or horizon.dtype.kind not in "biuf"
            or not np.isfinite(horizon) or horizon < 0):
        raise ValueError("t_final must be a finite nonnegative scalar.")
    if (isinstance(initial_state, (bool, np.bool_))
            or not isinstance(initial_state, (int, np.integer))
            or not 0 <= initial_state < Q.shape[0]):
        raise ValueError("initial_state must be a valid zero-based integer state.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    t_final = float(horizon)
    time = 0.0
    state = int(initial_state)
    jump_times, states = [time], [state]
    while time < t_final:
        rate = exit_rates[state]
        if rate == 0.0:
            break
        next_time = time + rng.exponential(scale=1.0 / rate)
        if next_time > t_final:
            break
        if next_time <= time:
            raise FloatingPointError("Holding time is too small to advance the simulation clock.")
        state = int(rng.choice(Q.shape[0], p=rates[state] / rate))
        time = next_time
        jump_times.append(time)
        states.append(state)
    return np.asarray(jump_times, dtype=float), np.asarray(states, dtype=int)


def decode_augmented_states(states, n_symbols):
    """Decode indices i*n_symbols + k into (theta, detector_symbol).

    Accept a nonnegative integer scalar or array, using the ordering from
    build_augmented_generator. Return two integers for scalar input, or
    two integer arrays preserving the input shape. n_symbols is the number
    of detector symbols, not the number of Markov modes.
    """
    if (isinstance(n_symbols, (bool, np.bool_))
            or not isinstance(n_symbols, (int, np.integer)) or n_symbols <= 0):
        raise ValueError("n_symbols must be a positive integer.")
    states = np.asarray(states)
    if states.dtype.kind not in "iu" or np.any(states < 0):
        raise ValueError("states must contain nonnegative integer indices.")
    theta, detector = np.divmod(states, n_symbols)
    if states.ndim == 0:
        return int(theta), int(detector)
    return theta, detector


def simulate_example1_dynamics(time_grid, jump_times, augmented_states,
                               Upsilon, filter_matrices, *, rtol=1e-7, atol=1e-9):
    """Integrate equations (1), (6), and (7) along one supplied CTMC path.

    ``jump_times, augmented_states`` follow simulate_ctmc's convention:
    the first entry is at zero and state (i, ell) has index 2*i + ell.
    The last mode persists through the end of time_grid. Later jumps beyond
    that horizon are ignored. time_grid is a nonempty, strictly increasing,
    nonnegative array; integration always starts at zero, even if its first
    requested sample is later.

    ``filter_matrices`` is a dictionary with Ahat, Bhat, Lhat, Ehat (such as
    the result of solve_theorem1_example1 with return_filter=True), or a
    tuple of those four lists. Supply two numerical matrices of each kind.
    Upsilon is the (2, 2) emission matrix used for the averaged dynamics.

    Integrate the plant and both filters together with solve_ivp, restarting
    at every jump, with x(0) = ones(3) and both filter states initially zero.
    No state resets occur at jumps. Outputs and sampled modes are
    right-continuous: a sample exactly at a jump uses the new mode/symbol.
    Both output estimates use the current detector symbol, including the
    averaged filter of equation (7).

    Return a dictionary with t, w=sin(t), theta, detector_symbol (1-D arrays),
    x, xhat_a, xhat_b (shape (len(time_grid), 3)), and y, z, zhat_a, zhat_b
    (shape (len(time_grid), 1)). The grid controls output sampling only,
    not the CTMC path or the adaptive integrator's internal step sizes.
    """
    def real_array(value, name, ndim):
        array = np.asarray(value)
        if (array.ndim != ndim or 0 in array.shape or array.dtype.kind not in "biuf"
                or not np.all(np.isfinite(array))):
            raise ValueError(f"{name} must be a nonempty finite real {ndim}-D array.")
        return array.astype(float, copy=False)

    time_grid = real_array(time_grid, "time_grid", 1)
    if time_grid[0] < 0 or np.any(np.diff(time_grid) <= 0):
        raise ValueError("time_grid must be nonnegative and strictly increasing.")
    jump_times = real_array(jump_times, "jump_times", 1)
    if jump_times[0] != 0 or np.any(np.diff(jump_times) <= 0):
        raise ValueError("jump_times must start at zero and be strictly increasing.")
    augmented_states = np.asarray(augmented_states)
    if (augmented_states.shape != jump_times.shape
            or augmented_states.dtype.kind not in "iu"
            or np.any(augmented_states < 0) or np.any(augmented_states >= 4)):
        raise ValueError("augmented_states must match jump_times and contain indices 0 through 3.")
    Upsilon = real_array(Upsilon, "Upsilon", 2)
    if (Upsilon.shape != (2, 2) or np.any(Upsilon < 0)
            or not np.allclose(Upsilon.sum(axis=1), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("Upsilon must be a row-stochastic (2, 2) matrix.")
    names = ("Ahat", "Bhat", "Lhat", "Ehat")
    if isinstance(filter_matrices, dict):
        filter_matrices = tuple(filter_matrices[name] for name in names)
    if len(filter_matrices) != 4:
        raise ValueError("Supply Ahat, Bhat, Lhat, and Ehat.")
    filters = []
    for name, group, shape in zip(names, filter_matrices, ((3, 3), (3, 1), (1, 3), (1, 1))):
        if group is None or len(group) != 2:
            raise ValueError(f"{name} must contain two numerical matrices.")
        arrays = [real_array(value, f"{name}[{ell}]", 2) for ell, value in enumerate(group)]
        if any(value.shape != shape for value in arrays):
            raise ValueError(f"Each {name} matrix must have shape {shape}.")
        filters.append(arrays)
    Ahat, Bhat, Lhat, Ehat = filters
    A, B, C, D, L, F, _ = example1_system()
    HA = [weighted_detector_average(Upsilon, i, Ahat) for i in range(2)]
    HB = [weighted_detector_average(Upsilon, i, Bhat) for i in range(2)]
    path_theta, path_detector = decode_augmented_states(augmented_states, 2)

    combined = np.concatenate([np.ones(3), np.zeros(6)])
    trajectory = np.empty((len(time_grid), 9))
    t_final = time_grid[-1]
    for index, start in enumerate(jump_times):
        if start >= t_final:
            break
        stop = min(jump_times[index + 1], t_final) if index + 1 < len(jump_times) else t_final
        i, ell = path_theta[index], path_detector[index]

        def rhs(t, state):
            x, xhat_a, xhat_b = state[:3], state[3:6], state[6:]
            disturbance = np.sin(t)
            y = C[i] @ x + D[i][:, 0] * disturbance
            return np.concatenate([
                A[i] @ x + B[i][:, 0] * disturbance,
                Ahat[ell] @ xhat_a + Bhat[ell] @ y,
                HA[i] @ xhat_b + HB[i] @ y,
            ])

        solution = solve_ivp(rhs, (start, stop), combined, dense_output=True,
                             rtol=rtol, atol=atol)
        if not solution.success:
            raise RuntimeError(f"Integration failed on [{start}, {stop}]: {solution.message}")
        first = np.searchsorted(time_grid, start, side="left")
        last = np.searchsorted(time_grid, stop, side="left")
        if first < last:
            trajectory[first:last] = solution.sol(time_grid[first:last]).T
        combined = solution.y[:, -1]
    # Also covers a zero-length horizon and a jump exactly at the horizon.
    trajectory[-1] = combined
    sampled_path = np.searchsorted(jump_times, time_grid, side="right") - 1
    theta, detector = path_theta[sampled_path], path_detector[sampled_path]
    x, xhat_a, xhat_b = trajectory[:, :3], trajectory[:, 3:6], trajectory[:, 6:]
    w = np.sin(time_grid)
    y, z, zhat_a, zhat_b = [np.empty((len(time_grid), 1)) for _ in range(4)]
    for sample, (i, ell) in enumerate(zip(theta, detector)):
        y[sample] = C[i] @ x[sample] + D[i][:, 0] * w[sample]
        z[sample] = L[i] @ x[sample] + F[i][:, 0] * w[sample]
        zhat_a[sample] = Lhat[ell] @ xhat_a[sample] + Ehat[ell] @ y[sample]
        zhat_b[sample] = Lhat[ell] @ xhat_b[sample] + Ehat[ell] @ y[sample]
    return {
        "t": time_grid.copy(), "w": w, "theta": theta, "detector_symbol": detector,
        "x": x, "xhat_a": xhat_a, "xhat_b": xhat_b,
        "y": y, "z": z, "zhat_a": zhat_a, "zhat_b": zhat_b,
    }
