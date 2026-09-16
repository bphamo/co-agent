"""Tests for the validation harnesses.

These test the instruments, not the findings: that truth is computed from the
process, that the walk-forward cannot see past its origin, that windows do not
overlap, and that every reliability row carries its sample size. The findings
themselves are numbers in co_agent/sim/README.md, reproduced by the study CLI.
"""

from __future__ import annotations

import numpy as np
import pytest

from co_agent.sim import Band, Config, Interval, TerminalAbove, TouchBelow, Verdict
from co_agent.sim.dgp import Garch11, IIDNormal, RegimeSwitch
from co_agent.sim.validate import (
    Observation,
    bias_study,
    non_overlapping_origins,
    reliability,
    true_probability,
    walk_forward,
)

MC = Config(paths=4_000, interval=Interval.MC)


# --------------------------------------------------------------- known truth


def test_true_probability_is_monotone_in_the_threshold():
    rng = np.random.default_rng(1)
    dgp = IIDNormal()
    previous = 1.0
    for drop in (0.05, 0.12, 0.25):
        p = true_probability(dgp, None, TouchBelow(drop), 60, rng, 20_000)
        assert p < previous
        previous = p


def test_true_probability_of_a_coin_flip_is_a_half():
    p = true_probability(IIDNormal(), None, TerminalAbove(0.0), 60, np.random.default_rng(2), 40_000)
    assert abs(p - 0.5) < 0.02


# ---------------------------------------------------------------- bias study


def test_bias_study_reports_every_field_a_finding_needs():
    r = bias_study(IIDNormal(), TouchBelow(0.12), trials=8, truth_paths=4_000, config=MC)
    assert r.n == 8
    assert r.skipped == 0
    assert r.dgp == "iid_normal"
    assert r.dgp_params == {"vol": 0.02}
    assert r.falsifier["kind"] == "touch_below"
    assert 0 < r.mean_truth < 1
    assert r.mae >= abs(r.bias), "mean absolute error cannot be below the mean signed error"
    assert 0 <= r.coverage <= 1


def test_bias_study_varies_the_estimator_seed_across_trials():
    """A shared seed would correlate the Monte Carlo error and hide the spread."""
    r = bias_study(IIDNormal(), TouchBelow(0.12), trials=6, truth_paths=4_000, config=MC)
    assert len({t.estimate for t in r.trials}) > 1


def test_bias_study_counts_refusals_rather_than_dropping_them():
    """A history the estimator rejects is a finding, not a gap in the sample."""
    r = bias_study(
        IIDNormal(),
        TouchBelow(0.12),
        trials=5,
        history_obs=200,  # below min_history
        truth_paths=2_000,
        config=MC,
    )
    assert r.n == 0
    assert r.skipped == 5
    assert np.isnan(r.bias) and np.isnan(r.coverage)


def test_gate_errors_partition_the_trials_by_where_truth_falls():
    r = bias_study(IIDNormal(), TouchBelow(0.12), trials=10, truth_paths=4_000, config=MC)
    errs = r.gate_errors(Band())
    assert errs["n_truth_inside"] + errs["n_truth_outside"] == r.n
    for key in ("false_reject", "false_accept", "indeterminate"):
        v = errs[key]
        assert np.isnan(v) or 0 <= v <= 1


def test_the_monte_carlo_interval_under_covers_on_clustered_data():
    """The finding that drove the interval change, pinned as a regression.

    Monte Carlo error is an order of magnitude smaller than the estimator's real
    error, so a gate judging it is confident about the wrong quantity. If this
    ever passes at nominal coverage, either the estimator improved enormously or
    the harness stopped measuring.
    """
    r = bias_study(
        Garch11(),
        TouchBelow(0.12),
        trials=25,
        truth_paths=10_000,
        config=Config(paths=10_000, interval=Interval.MC),
    )
    assert r.coverage < 0.5, f"coverage={r.coverage}"


def test_the_double_bootstrap_interval_covers_on_iid_data():
    r = bias_study(
        IIDNormal(),
        TouchBelow(0.12),
        trials=15,
        truth_paths=10_000,
        config=Config(paths=4_000, interval=Interval.DOUBLE_BOOTSTRAP, outer_resamples=12,
                      inner_paths=1_500),
    )
    assert r.coverage > 0.6, f"coverage={r.coverage}"


# ------------------------------------------------------------- walk-forward


def test_non_overlapping_origins_do_not_overlap():
    origins = non_overlapping_origins(n_obs=1000, history_obs=400, horizon=60)
    assert origins[0] == 400
    assert all(b - a >= 60 for a, b in zip(origins, origins[1:]))
    assert origins[-1] + 60 <= 1000


def test_non_overlapping_origins_is_empty_when_history_leaves_no_room():
    assert non_overlapping_origins(n_obs=420, history_obs=400, horizon=60) == []


def test_walk_forward_cannot_see_past_its_origin():
    """Enforced by slicing, not by convention.

    A series that is flat before every origin and violent after it must produce
    low predictions and tripped outcomes. If the estimator could see forward, the
    predictions would rise to meet the outcomes.
    """
    rng = np.random.default_rng(3)
    quiet = rng.normal(0, 0.001, 900)
    series = np.concatenate([quiet, np.full(60, -0.01)])  # a steady 45% slide

    obs = walk_forward(
        "X", series, TouchBelow(0.12), horizon_days=60, history_obs=840, config=MC
    )
    assert obs, "expected at least one origin"
    last = obs[-1]
    assert last.tripped is True
    assert last.predicted < 0.05, "a calm history should not predict the crash it precedes"


def test_walk_forward_resolves_outcomes_against_the_real_future():
    rng = np.random.default_rng(4)
    series = rng.normal(0, 0.02, 1400)
    obs = walk_forward(
        "X", series, TouchBelow(0.12), horizon_days=60, history_obs=800, config=MC
    )
    assert len(obs) == len(non_overlapping_origins(1400, 800, 60))
    assert all(0 <= o.predicted <= 1 for o in obs)
    assert {type(o.tripped) for o in obs} == {bool}


def test_overlapping_mode_yields_more_observations_and_is_opt_in():
    rng = np.random.default_rng(5)
    series = rng.normal(0, 0.02, 1200)
    sparse = walk_forward("X", series, TouchBelow(0.12), history_obs=800, config=MC)
    dense = walk_forward(
        "X", series, TouchBelow(0.12), history_obs=800, config=MC, overlapping=True
    )
    assert len(dense) > len(sparse)


# -------------------------------------------------------------- reliability


def _obs(pred: float, tripped: bool, n: int) -> list[Observation]:
    return [Observation("X", i, pred, tripped) for i in range(n)]


def test_reliability_bins_partition_the_unit_interval():
    rows = reliability(_obs(0.0, False, 1) + _obs(1.0, True, 1) + _obs(0.4, True, 1), boot=200)
    assert sum(r.n for r in rows) == 3
    # 0.4 lands in [0.4, 0.6), not in [0.2, 0.4).
    by_low = {r.low: r.n for r in rows}
    assert by_low[0.2] == 0 and by_low[0.4] == 1
    # The top edge is inclusive only in the last bin, so 1.0 is not dropped.
    assert by_low[0.8] == 1


def test_reliability_reports_sample_size_for_every_row_including_empty_ones():
    """FR7 wants a sample size beside every figure, and a bin is easy to forget."""
    rows = reliability(_obs(0.5, True, 4), boot=200)
    assert len(rows) == 5
    empty = [r for r in rows if r.n == 0]
    assert empty and all(np.isnan(r.realised) for r in empty)


def test_reliability_recovers_a_known_frequency():
    obs = _obs(0.5, True, 30) + _obs(0.5, False, 70)
    row = next(r for r in reliability(obs, boot=1000) if r.n == 100)
    assert row.realised == pytest.approx(0.30)
    assert row.mean_predicted == pytest.approx(0.5)
    assert row.ci_low < 0.30 < row.ci_high


def test_reliability_interval_narrows_with_sample_size():
    """Bootstrapped over observations, because walk-forward outcomes correlate."""
    small = next(r for r in reliability(_obs(0.5, True, 10) + _obs(0.5, False, 10), boot=2000) if r.n)
    large = next(
        r for r in reliability(_obs(0.5, True, 200) + _obs(0.5, False, 200), boot=2000) if r.n
    )
    assert (large.ci_high - large.ci_low) < (small.ci_high - small.ci_low)


def test_a_perfectly_calibrated_and_a_broken_forecaster_are_distinguishable():
    calibrated = _obs(0.3, True, 30) + _obs(0.3, False, 70)
    broken = _obs(0.3, True, 90) + _obs(0.3, False, 10)
    c = next(r for r in reliability(calibrated, boot=1000) if r.n)
    b = next(r for r in reliability(broken, boot=1000) if r.n)
    assert abs(c.realised - c.mean_predicted) < 0.05
    assert abs(b.realised - b.mean_predicted) > 0.5


def _dated_obs(pred, tripped, n, year, quarter_month):
    from datetime import date as _d

    return [
        Observation("S%d" % i, 0, pred, tripped, origin_date=_d(year, quarter_month, 15))
        for i in range(n)
    ]


def test_clustered_bootstrap_is_wider_than_resampling_observations():
    """The correction that matters on a multi-symbol study.

    Forty-eight symbols in one quarter that all tripped together are one event,
    not forty-eight trials. Resampling observations independently would report an
    interval several times too narrow; resampling quarters keeps the co-movement
    inside the unit.
    """
    # Four quarters, each internally unanimous: within-quarter outcomes are
    # perfectly correlated, which is the case i.i.d. resampling gets wrong.
    dated: list[Observation] = []
    for i, (year, month, tripped) in enumerate(
        [(2020, 1, True), (2020, 4, False), (2020, 7, True), (2020, 10, False)]
    ):
        dated += _dated_obs(0.5, tripped, 25, year, month)

    undated = [Observation(o.symbol, o.origin, o.predicted, o.tripped) for o in dated]

    clustered = next(r for r in reliability(dated, boot=3000) if r.n)
    naive = next(r for r in reliability(undated, boot=3000) if r.n)

    assert clustered.clustered is True
    assert naive.clustered is False
    assert clustered.realised == naive.realised == pytest.approx(0.5)
    assert (clustered.ci_high - clustered.ci_low) > 3 * (naive.ci_high - naive.ci_low)


def test_walk_forward_records_origin_dates_when_given_them():
    from datetime import date as _d

    rng = np.random.default_rng(11)
    series = rng.normal(0, 0.02, 1000)
    dates = [_d(2020, 1, 1) + __import__("datetime").timedelta(days=i) for i in range(1001)]
    obs = walk_forward(
        "X", series, TouchBelow(0.12), horizon_days=60, history_obs=800,
        config=MC, dates=dates,
    )
    assert obs and all(o.origin_date is not None for o in obs)
    # The origin date is the day the history ends, not the day it starts.
    assert obs[0].origin_date == dates[801]
