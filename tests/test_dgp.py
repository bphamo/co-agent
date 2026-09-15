"""Tests for the synthetic processes used to measure the estimator.

These check that each process has the property it exists to supply. A GARCH
fixture that does not cluster, or a Student-t that is not fat-tailed, would make
the bias study report that the estimator handles a condition it was never shown.
"""

from __future__ import annotations

import numpy as np
import pytest

from co_agent.sim import TerminalAbove, TouchBelow
from co_agent.sim.dgp import (
    DEFAULT_PANEL,
    Garch11,
    IIDNormal,
    RegimeSwitch,
    StudentT,
)
from co_agent.sim.validate import true_probability


def excess_kurtosis(x: np.ndarray) -> float:
    d = x - x.mean()
    return float((d**4).mean() / (d**2).mean() ** 2 - 3)


def abs_lag1(x: np.ndarray) -> float:
    a = np.abs(x)
    return float(np.corrcoef(a[1:], a[:-1])[0, 1])


def test_all_processes_hit_their_target_volatility():
    rng = np.random.default_rng(0)
    for dgp in DEFAULT_PANEL:
        returns, _ = dgp.simulate(rng, 20_000)
        assert 0.005 < returns.std() < 0.05, dgp.name


def test_iid_normal_has_neither_fat_tails_nor_clustering():
    """The baseline. A bias here is a bug, not a limitation of the method."""
    returns, state = IIDNormal().simulate(np.random.default_rng(1), 20_000)
    assert state is None, "an IID process carries no state"
    assert abs(excess_kurtosis(returns)) < 0.3
    assert abs(abs_lag1(returns)) < 0.05


def test_student_t_is_fat_tailed_without_clustering():
    returns, _ = StudentT(df=4).simulate(np.random.default_rng(2), 20_000)
    assert excess_kurtosis(returns) > 2
    assert abs(abs_lag1(returns)) < 0.05


def test_student_t_rejects_a_variance_free_df():
    with pytest.raises(ValueError):
        StudentT(df=2.0)


def test_garch_clusters_and_carries_its_variance_forward():
    dgp = Garch11()
    returns, state = dgp.simulate(np.random.default_rng(3), 20_000)
    assert abs_lag1(returns) > 0.05, "no volatility clustering"
    assert state > 0, "state should be the next conditional variance"
    assert dgp.unconditional_vol() == pytest.approx(0.02, rel=0.01)


def test_garch_rejects_a_nonstationary_parameterisation():
    with pytest.raises(ValueError):
        Garch11(alpha=0.2, beta=0.85)


def test_regime_switch_clusters_hard_and_reports_its_regime():
    returns, state = RegimeSwitch().simulate(np.random.default_rng(4), 20_000)
    assert abs_lag1(returns) > 0.2, "regimes should cluster more than GARCH"
    assert state in (0, 1)


def test_state_dependent_processes_branch_on_their_state():
    """Forward paths from a calm state must differ from a stormy one.

    This is the property that makes truth conditional, and therefore the property
    that makes conditioning on current volatility a real question rather than a
    stylistic one.
    """
    dgp = RegimeSwitch()
    calm = true_probability(dgp, 0, TouchBelow(0.15), 60, np.random.default_rng(5), 20_000)
    storm = true_probability(dgp, 1, TouchBelow(0.15), 60, np.random.default_rng(5), 20_000)
    assert storm > calm + 0.10, f"{calm=} {storm=}"


def test_forward_paths_are_zero_mean_so_truth_is_not_drifting():
    """Every process is driftless, so a terminal-above-zero falsifier is a coin flip."""
    for dgp in DEFAULT_PANEL:
        state = dgp.simulate(np.random.default_rng(6), 500)[1]
        p = true_probability(
            dgp, state, TerminalAbove(0.0), 60, np.random.default_rng(7), 40_000
        )
        assert abs(p - 0.5) < 0.04, f"{dgp.name}: {p}"


def test_forward_paths_have_the_requested_shape():
    for dgp in DEFAULT_PANEL:
        state = dgp.simulate(np.random.default_rng(8), 300)[1]
        paths = dgp.forward_paths(np.random.default_rng(9), state, 30, 100)
        assert paths.shape == (100, 30)
