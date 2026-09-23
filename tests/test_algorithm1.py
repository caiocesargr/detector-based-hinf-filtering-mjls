"""Tests of the alternating subproblems and gamma-based stopping rule."""

import cvxpy as cp
import numpy as np
import pytest

from detector_hinf import algorithm1 as alg
from detector_hinf.models import example2_emission_matrix, example2_uav_system


def test_step2_matches_theorem1_and_fixes_only_xyz():
    system, _ = example2_uav_system()
    upsilon = example2_emission_matrix()
    rng = np.random.default_rng(42)
    Ecal = [rng.normal(size=(17, 4)) for _ in range(2)]
    first, cost1, values1 = alg.build_theorem1_problem(system, upsilon, Ecal)
    for variable in first.variables():
        value = rng.normal(size=variable.shape)
        if variable.attributes['symmetric']:
            value = value + value.T
        variable.value = abs(value) if variable.attributes['nonneg'] else value
    fixed = {key: [v.value.copy() for v in values1[key]] for key in ('X', 'Y', 'Z')}
    second, cost2, values2 = alg._build_step2_problem(system, upsilon, fixed)
    assert second.is_dcp()
    names = {v.name() for v in second.variables()}
    assert not any(name.startswith(('X_', 'Y_', 'Z_')) for name in names)
    for key in ('P', 'Q', 'U', 'V', 'W', 'S', 'T'):
        assert any(name.startswith(key + '_') for name in names)
    by_name = {v.name(): v.value for v in first.variables()}
    for variable in second.variables():
        variable.value = (Ecal[int(variable.name().split('_')[1])]
                          if variable.name().startswith('Ecal_')
                          else by_name[variable.name()])
    assert cost1.value == cost2.value
    for a, b in zip(first.constraints, second.constraints, strict=True):
        np.testing.assert_allclose(a.args[0].value, b.args[0].value)
    assert all(v.shape == (17, 4) for v in values2['Ecal'])


def mock_solves(monkeypatch, costs):
    costs = iter(costs)

    def solve(problem, **kwargs):
        problem._status = cp.OPTIMAL
        for v in problem.variables():
            if v.name() == 'gamma_sq':
                v.value = next(costs)**2
            elif v.name().startswith('Z_'):
                v.value = np.eye(4)
            else:
                v.value = np.zeros(v.shape)
    monkeypatch.setattr(cp.Problem, 'solve', solve)


def test_gamma_stopping_sequence_and_filter_dimensions(monkeypatch):
    # First squared-cost decrease is .000775 < tol, but gamma decrease
    # is .005 > tol. A gamma-squared stopping test would stop too early.
    mock_solves(monkeypatch, [.08, .075, .074, .0735])
    result = alg.solve_algorithm1_example2(return_filter=True)
    assert result['iterations'] == 2
    assert result['converged']
    assert result['status'] == cp.OPTIMAL
    history = result['history']
    sequence = [v for row in history for v in (row['gamma_bar'], row['gamma'])]
    assert np.all(np.diff(sequence) <= 1e-6)
    assert all(row['gamma'] <= row['gamma_bar'] + 1e-6 for row in history)
    assert result['gamma'] == pytest.approx(np.sqrt(result['gamma_sq']))
    for key, shape in [('Ahat', (4, 4)), ('Bhat', (4, 2)),
                       ('Lhat', (1, 4)), ('Ehat', (1, 2))]:
        assert len(result[key]) == 3
        assert all(m.shape == shape for m in result[key])


def test_iteration_limit(monkeypatch):
    mock_solves(monkeypatch, [.2, .1])
    result = alg.solve_algorithm1_example2(max_iterations=1)
    assert result['status'] == 'max_iterations'
    assert not result['converged']
    assert result['iterations'] == 1


@pytest.mark.parametrize('status', [cp.INFEASIBLE, cp.UNBOUNDED, cp.OPTIMAL_INACCURATE])
def test_unsuccessful_status(monkeypatch, status):
    def solve(problem, **kwargs):
        problem._status = status
    monkeypatch.setattr(cp.Problem, 'solve', solve)
    with pytest.raises(RuntimeError, match='Step 1'):
        alg.solve_algorithm1_example2()


def test_cost_increase_rejected(monkeypatch):
    mock_solves(monkeypatch, [.1, .2])
    with pytest.raises(RuntimeError, match='increased'):
        alg.solve_algorithm1_example2()


@pytest.mark.parametrize('kwargs', [{'tol': 0}, {'tol': np.nan},
                                    {'max_iterations': 0}, {'max_iterations': 1.5}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        alg.solve_algorithm1_example2(**kwargs)


def test_mosek_alternating_sequence():
    """Opt-in numerical regression requiring an available MOSEK license."""
    import os
    if os.environ.get('RUN_MOSEK_TESTS') != '1':
        pytest.skip('Set RUN_MOSEK_TESTS=1 to run licensed MOSEK integration test')
    result = alg.solve_algorithm1_example2(return_filter=True)
    assert result['converged']
    sequence = [v for row in result['history']
                for v in (row['gamma_bar'], row['gamma'])]
    for previous, current in zip(sequence, sequence[1:]):
        assert current <= previous + 1e-6 + 1e-5 * abs(previous)
    assert result['history'][-1]['improvement'] < 1e-3
    for key, shape in [('Ahat', (4, 4)), ('Bhat', (4, 2)),
                       ('Lhat', (1, 4)), ('Ehat', (1, 2))]:
        assert len(result[key]) == 3
        assert all(m.shape == shape and np.isfinite(m).all() for m in result[key])
