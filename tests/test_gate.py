"""Tests for the FR9 gates."""

from __future__ import annotations

import numpy as np
import pytest

from co_agent.sim.gate import Band, Verdict, cap_weight, wilson
from co_agent.sim.params import SimInputError


@pytest.mark.parametrize(
    "name,p,low,high,want",
    [
        ("comfortably inside", 0.50, 0.49, 0.51, Verdict.ACCEPT),
        ("inside, touching edges", 0.50, 0.30, 0.70, Verdict.ACCEPT),
        ("clearly too hard", 0.10, 0.09, 0.11, Verdict.REJECT_TOO_HARD),
        ("clearly too easy", 0.90, 0.88, 0.92, Verdict.REJECT_TOO_EASY),
        # The hysteresis cases: the estimate is on one side of the edge but the
        # interval is not, so no verdict is available.
        ("straddling the low edge", 0.31, 0.28, 0.34, Verdict.INDETERMINATE),
        ("straddling the high edge", 0.69, 0.66, 0.72, Verdict.INDETERMINATE),
        ("just outside, straddling", 0.29, 0.26, 0.32, Verdict.INDETERMINATE),
        ("interval spans the band", 0.50, 0.20, 0.80, Verdict.INDETERMINATE),
    ],
)
def test_judge_only_rules_when_the_interval_settles_it(name, p, low, high, want):
    assert Band().judge(p, low, high) is want, name


@pytest.mark.parametrize("band", [Band(0, 0.7), Band(0.3, 1), Band(0.7, 0.3), Band(0.5, 0.5)])
def test_band_validation_rejects_nonsense(band):
    with pytest.raises(SimInputError):
        band.validate()


def test_default_band_is_the_trd_band():
    assert (Band().low, Band().high) == (0.3, 0.7)
    Band().validate()


@pytest.mark.parametrize("k,n", [(0, 1000), (1, 1000), (500, 1000), (999, 1000), (1000, 1000)])
def test_wilson_brackets_the_point_estimate(k, n):
    low, high = wilson(k, n)
    p = k / n
    assert 0 <= low <= high <= 1
    assert low - 1e-9 <= p <= high + 1e-9


def test_wilson_tightens_with_path_count():
    """What escalation buys."""
    narrow = np.subtract(*reversed(wilson(50_000, 100_000)))
    wide = np.subtract(*reversed(wilson(500, 1000)))
    assert narrow < wide


def test_wilson_with_no_trials_is_uninformative():
    assert wilson(0, 0) == (0.0, 1.0)


def test_cap_weight_reduces_to_the_limit():
    # 0.05 weight on a position whose 95th-percentile drawdown is 60% puts 3% at
    # risk against a 2% limit, so the weight must come down to 1/30.
    s = cap_weight(0.60, 0.05, 0.02)
    assert s.reduced
    assert s.weight == pytest.approx(0.02 / 0.60)
    assert s.p95_drawdown_at_weight == pytest.approx(0.02)


def test_cap_weight_leaves_a_compliant_proposal_alone():
    s = cap_weight(0.20, 0.05, 0.02)
    assert not s.reduced
    assert s.weight == 0.05
    assert s.p95_drawdown_at_weight == pytest.approx(0.01)


def test_cap_weight_is_linear_in_weight():
    a = cap_weight(0.3, 0.02, 1)
    b = cap_weight(0.3, 0.04, 1)
    assert b.p95_drawdown_at_weight == pytest.approx(2 * a.p95_drawdown_at_weight)


def test_cap_weight_handles_a_degenerate_drawdown():
    s = cap_weight(0.0, 0.05, 0.02)
    assert not s.reduced
    assert s.weight == 0.05


def test_p95_uses_the_nearest_rank_definition():
    """Pins the percentile convention a stored p95_drawdown was computed with."""
    sample = np.arange(1, 101, dtype=np.float64)
    assert np.quantile(sample, 0.95, method="inverted_cdf") == 95
