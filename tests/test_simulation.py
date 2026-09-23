"""Checks for event-driven CTMC paths and augmented-state decoding."""

import numpy as np
import pytest
from scipy.linalg import expm

from detector_hinf.detector import build_augmented_generator
from detector_hinf.models import (
    example1_detector_generators,
    example1_emission_matrix,
    example1_system,
)
from detector_hinf.simulation import (
    decode_augmented_states,
    simulate_ctmc,
    simulate_example1_dynamics,
)


def test_example1_ctmc_reproducibility_and_path_properties():
    Q = build_augmented_generator(
        example1_system()[-1], example1_detector_generators(),
        example1_emission_matrix(0.2), epsilon=0.075,
    )
    original = Q.copy()
    times, states = simulate_ctmc(Q, 20.0, 0, np.random.default_rng(123))
    repeated = simulate_ctmc(Q, 20.0, 0, np.random.default_rng(123))
    np.testing.assert_array_equal(times, repeated[0])
    np.testing.assert_array_equal(states, repeated[1])
    np.testing.assert_array_equal(Q, original)
    assert times.shape == states.shape
    assert len(times) > 1
    assert times[0] == 0.0 and states[0] == 0
    assert np.all(np.diff(times) > 0)
    assert times[-1] <= 20.0
    assert np.all((0 <= states) & (states < 4))
    assert np.all(states[1:] != states[:-1])
    assert np.all(Q[states[:-1], states[1:]] > 0)
    theta, detector = decode_augmented_states(states, 2)
    np.testing.assert_array_equal(2 * theta + detector, states)


def test_exact_exponential_holding_time_and_absorption():
    # The only possible transition is 0 -> 1, where state 1 is absorbing.
    Q = np.array([[-2.0, 2.0], [0.0, 0.0]])
    expected_time = np.random.default_rng(42).exponential(scale=0.5)
    times, states = simulate_ctmc(Q, expected_time + 1.0, 0, np.random.default_rng(42))
    np.testing.assert_array_equal(times, [0.0, expected_time])
    np.testing.assert_array_equal(states, [0, 1])
    # A sampled jump beyond the horizon must not be recorded.
    times, states = simulate_ctmc(Q, expected_time / 2, 0, np.random.default_rng(42))
    np.testing.assert_array_equal(times, [0.0])
    np.testing.assert_array_equal(states, [0])


def test_transition_probabilities_follow_off_diagonal_rates():
    Q = np.array([[-4., 1., 3.], [0., 0., 0.], [0., 0., 0.]])
    expected_rng = np.random.default_rng(12)
    expected_time = expected_rng.exponential(scale=0.25)
    expected_state = expected_rng.choice(3, p=[0.0, 0.25, 0.75])
    times, states = simulate_ctmc(Q, expected_time + 1, 0, np.random.default_rng(12))
    np.testing.assert_array_equal(times, [0.0, expected_time])
    np.testing.assert_array_equal(states, [0, expected_state])


@pytest.mark.parametrize("horizon", [0.0, 100.0])
def test_initial_absorbing_state(horizon):
    times, states = simulate_ctmc(np.zeros((1, 1)), horizon, 0, np.random.default_rng(1))
    np.testing.assert_array_equal(times, [0.0])
    np.testing.assert_array_equal(states, [0])


def test_zero_horizon_does_not_advance_rng():
    rng = np.random.default_rng(5)
    times, states = simulate_ctmc([[-1., 1.], [1., -1.]], 0.0, 1, rng)
    np.testing.assert_array_equal(times, [0.0])
    np.testing.assert_array_equal(states, [1])
    assert rng.random() == np.random.default_rng(5).random()


def test_decode_augmented_states():
    theta, detector = decode_augmented_states(np.arange(6).reshape(2, 3), 3)
    np.testing.assert_array_equal(theta, [[0, 0, 0], [1, 1, 1]])
    np.testing.assert_array_equal(detector, [[0, 1, 2], [0, 1, 2]])
    assert decode_augmented_states(3, 2) == (1, 1)


@pytest.mark.parametrize("Q", [[], [0], [[0, 0]], [[np.nan]], [[1, -1], [0, 0]], [[-1]]])
def test_invalid_generator(Q):
    with pytest.raises(ValueError, match="Q"):
        simulate_ctmc(Q, 1.0, 0, np.random.default_rng(1))


@pytest.mark.parametrize("horizon", [-1, np.inf, np.nan, [1.0]])
def test_invalid_horizon(horizon):
    with pytest.raises(ValueError, match="t_final"):
        simulate_ctmc([[0]], horizon, 0, np.random.default_rng(1))


@pytest.mark.parametrize("state", [-1, 1, 0.5, True])
def test_invalid_initial_state(state):
    with pytest.raises(ValueError, match="initial_state"):
        simulate_ctmc([[0]], 1.0, state, np.random.default_rng(1))


def test_rng_must_be_generator():
    with pytest.raises(TypeError, match="Generator"):
        simulate_ctmc([[0]], 1.0, 0, 123)


@pytest.mark.parametrize("states, n_symbols", [([-1], 2), ([0.5], 2), ([0], 0), ([0], 1.5)])
def test_invalid_augmented_indices(states, n_symbols):
    with pytest.raises(ValueError):
        decode_augmented_states(states, n_symbols)


@pytest.fixture
def artificial_filters():
    """Simple numerical fixtures, not filter coefficients from the paper."""
    return {
        "Ahat": [-np.diag([1., 2., 3.]), -np.diag([2., 3., 4.])],
        "Bhat": [np.array([[1.], [2.], [3.]]), np.array([[3.], [1.], [2.]])],
        "Lhat": [np.array([[1., 0., 2.]]), np.array([[0., 2., 1.]])],
        "Ehat": [np.array([[0.1]]), np.array([[0.4]])],
    }


def _exact_piecewise_states(grid, jumps, states, Upsilon, filters):
    """Independent reference: augment with sin/cos and use matrix exponentials."""
    A, B, C, D, *_ = example1_system()
    reference = []
    # Compute each requested observation independently from time zero.
    for target in grid:
        state = np.concatenate([np.ones(3), np.zeros(6), [0., 1.]])
        for index, start in enumerate(jumps):
            if start >= target:
                break
            stop = min(jumps[index + 1], target) if index + 1 < len(jumps) else target
            i, ell = divmod(states[index], 2)
            HA = Upsilon[i, 0] * filters["Ahat"][0] + Upsilon[i, 1] * filters["Ahat"][1]
            HB = Upsilon[i, 0] * filters["Bhat"][0] + Upsilon[i, 1] * filters["Bhat"][1]
            dynamics = np.zeros((11, 11))
            dynamics[:3, :3] = A[i]
            dynamics[:3, 9:10] = B[i]
            dynamics[3:6, :3] = filters["Bhat"][ell] @ C[i]
            dynamics[3:6, 3:6] = filters["Ahat"][ell]
            dynamics[3:6, 9:10] = filters["Bhat"][ell] @ D[i]
            dynamics[6:9, :3] = HB @ C[i]
            dynamics[6:9, 6:9] = HA
            dynamics[6:9, 9:10] = HB @ D[i]
            dynamics[9, 10] = 1.
            dynamics[10, 9] = -1.
            state = expm(dynamics * (stop - start)) @ state
        reference.append(state[:9])
    return np.asarray(reference)


def test_example1_piecewise_dynamics_against_exact_reference(artificial_filters):
    # No samples fall inside the short interval [0.2, 0.3).
    grid = np.array([0., 0.1, 0.3, 0.65, 0.9, 1.])
    jumps = np.array([0., 0.2, 0.3, 0.9, 1.])
    states = np.array([0, 1, 3, 2, 0])
    Upsilon = example1_emission_matrix(0.2)
    result = simulate_example1_dynamics(grid, jumps, states, Upsilon, artificial_filters,
                                        rtol=1e-10, atol=1e-12)
    expected = _exact_piecewise_states(grid, jumps, states, Upsilon, artificial_filters)
    actual = np.hstack([result["x"], result["xhat_a"], result["xhat_b"]])
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(result["x"][0], np.ones(3))
    np.testing.assert_array_equal(result["xhat_a"][0], np.zeros(3))
    np.testing.assert_array_equal(result["xhat_b"][0], np.zeros(3))
    np.testing.assert_array_equal(result["theta"], [0, 0, 1, 1, 1, 0])
    np.testing.assert_array_equal(result["detector_symbol"], [0, 0, 1, 1, 0, 0])
    _, _, C, D, L, F, _ = example1_system()
    for k, (i, ell) in enumerate(zip(result["theta"], result["detector_symbol"])):
        y = C[i] @ expected[k, :3] + D[i][:, 0] * np.sin(grid[k])
        z = L[i] @ expected[k, :3] + F[i][:, 0] * np.sin(grid[k])
        np.testing.assert_allclose(result["y"][k], y, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(result["z"][k], z, rtol=1e-8, atol=1e-10)
        for label, section in (("a", slice(3, 6)), ("b", slice(6, 9))):
            estimate = artificial_filters["Lhat"][ell] @ expected[k, section]
            estimate += artificial_filters["Ehat"][ell] @ y
            np.testing.assert_allclose(result[f"zhat_{label}"][k], estimate,
                                       rtol=1e-8, atol=1e-10)
    assert not np.allclose(result["xhat_a"], result["xhat_b"])
    assert all(result[key].shape == (len(grid), 1) for key in ("y", "z", "zhat_a", "zhat_b"))


@pytest.mark.parametrize("rho, states", [(0., [0, 3]), (1., [1, 2])])
def test_filters_agree_for_deterministic_emissions(artificial_filters, rho, states):
    result = simulate_example1_dynamics(
        np.linspace(0., 1., 11), [0., 0.45], states,
        example1_emission_matrix(rho), artificial_filters,
    )
    np.testing.assert_allclose(result["xhat_a"], result["xhat_b"], atol=1e-12)
    np.testing.assert_allclose(result["zhat_a"], result["zhat_b"], atol=1e-12)


def test_sampling_grid_does_not_control_integration(artificial_filters):
    args = ([0., 0.15, 0.4], [0, 1, 3], example1_emission_matrix(0.2), artificial_filters)
    dense = simulate_example1_dynamics(np.linspace(0., 1., 11), *args)
    sparse = simulate_example1_dynamics([0.2, 0.5, 1.], *args)
    for name in ("x", "xhat_a", "xhat_b", "y", "z", "zhat_a", "zhat_b"):
        np.testing.assert_allclose(sparse[name], dense[name][[2, 5, 10]], atol=1e-12)


def test_zero_horizon_and_filter_tuple(artificial_filters):
    result = simulate_example1_dynamics(
        [0.], [0.], [0], example1_emission_matrix(0.2),
        tuple(artificial_filters[name] for name in ("Ahat", "Bhat", "Lhat", "Ehat")),
    )
    np.testing.assert_array_equal(result["x"], [[1., 1., 1.]])
    np.testing.assert_array_equal(result["xhat_a"], np.zeros((1, 3)))
    np.testing.assert_array_equal(result["xhat_b"], np.zeros((1, 3)))
    np.testing.assert_allclose(result["zhat_a"], [[0.11]])


def test_integration_failure_is_reported(monkeypatch, artificial_filters):
    from types import SimpleNamespace
    from detector_hinf import simulation

    monkeypatch.setattr(simulation, "solve_ivp", lambda *args, **kwargs:
                        SimpleNamespace(success=False, message="test failure"))
    with pytest.raises(RuntimeError, match="Integration failed.*test failure"):
        simulate_example1_dynamics([0., 1.], [0.], [0],
                                   example1_emission_matrix(0.2), artificial_filters)


@pytest.mark.parametrize("grid, jumps, states", [
    ([], [0.], [0]), ([0., 0.], [0.], [0]), ([-1., 1.], [0.], [0]),
    ([0., 1.], [0.1], [0]), ([0., 1.], [0., 0.], [0, 1]),
    ([0., 1.], [0.], [4]), ([0., 1.], [0., 0.5], [0]),
])
def test_dynamics_rejects_invalid_grid_or_path(artificial_filters, grid, jumps, states):
    with pytest.raises(ValueError):
        simulate_example1_dynamics(grid, jumps, states,
                                   example1_emission_matrix(0.2), artificial_filters)
