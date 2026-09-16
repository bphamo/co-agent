"""Tests for the weekly cycle: snapshot, candidates, scorer, engine."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from co_agent.cycle import (
    Ledger,
    MomentumScreen,
    build_snapshot,
    config_for_horizon,
    resolve,
    suggested_threshold,
)
from co_agent.cycle.candidates import Candidate
from co_agent.paper import PaperBroker
from co_agent.sim import TouchBelow


class FakeSeries:
    def __init__(self, dates, closes):
        self.dates = tuple(dates)
        self.closes = np.asarray(closes, dtype=float)

    @property
    def log_returns(self):
        return np.diff(np.log(self.closes))


def make_series(n=1800, seed=1, mu=0.0003, vol=0.015):
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(mu, vol, n)))
    dates = [date(2018, 1, 1) + timedelta(days=i) for i in range(n)]
    return FakeSeries(dates, closes)


def universe(k=6, n=1800):
    return {f"S{i}.TO": make_series(n=n, seed=i + 1) for i in range(k)}


def baselines(series, value=0.0002):
    days = sorted(set().union(*[set(v.dates) for v in series.values()]))
    return {d: value for d in days}


# ------------------------------------------------------------------- snapshot


def test_snapshot_history_stops_at_its_own_date():
    """No-look-ahead is enforced by slicing, not by convention."""
    s = universe(2)
    b = baselines(s)
    day = s["S0.TO"].dates[1700]
    snap = build_snapshot(s, day, history_obs=1600, baseline_by_date=b)

    px = s["S0.TO"]
    expected = px.log_returns[1700 - 1600 : 1700]
    assert np.array_equal(snap.histories["S0.TO"], expected)
    assert snap.closes["S0.TO"] == pytest.approx(px.closes[1700])
    # Changing a future close must not change the snapshot.
    px.closes[1701] *= 2
    again = build_snapshot(s, day, history_obs=1600, baseline_by_date=b)
    assert again.digest == snap.digest


def test_snapshot_digest_changes_when_inputs_do():
    s = universe(2)
    b = baselines(s)
    day = s["S0.TO"].dates[1700]
    first = build_snapshot(s, day, history_obs=1600, baseline_by_date=b)
    s["S0.TO"].closes[1699] *= 1.01
    second = build_snapshot(s, day, history_obs=1600, baseline_by_date=b)
    assert first.digest != second.digest


def test_names_without_enough_history_are_recorded_not_dropped_silently():
    s = universe(2)
    s["SHORT.TO"] = make_series(n=900, seed=99)
    b = baselines(s)
    day = s["S0.TO"].dates[1700]
    snap = build_snapshot(s, day, history_obs=1600, baseline_by_date=b)
    assert "SHORT.TO" in snap.missing
    assert "SHORT.TO" not in snap.universe


def test_a_snapshot_needs_a_market_baseline():
    s = universe(2)
    with pytest.raises(KeyError):
        build_snapshot(s, s["S0.TO"].dates[1700], history_obs=1600, baseline_by_date={})


# ----------------------------------------------------------------- candidates


def test_suggested_threshold_tracks_the_measured_horizon_map():
    assert suggested_threshold(20) == 0.03
    assert suggested_threshold(60) == 0.06
    assert suggested_threshold(19) == suggested_threshold(20)  # nearest horizon


def test_screen_produces_complete_falsifiable_theses():
    s = universe(8)
    snap = build_snapshot(s, s["S0.TO"].dates[1700], history_obs=1600,
                          baseline_by_date=baselines(s))
    proposed = MomentumScreen(horizon_days=20).propose(snap, 3)
    assert len(proposed) == 3
    for c in proposed:
        assert c.claim and c.mechanism and c.falsifier_text
        assert c.horizon_days == 20
        assert isinstance(c.falsifier, TouchBelow)
        assert c.falsifier.drop == 0.03
        assert c.source.startswith("momentum_screen")


def test_screen_ranks_by_trailing_return():
    s = universe(4)
    # Boost through the snapshot close at index 1700 inclusive. The return
    # ending *on* the snapshot date is known at its close, so it is in scope;
    # only index 1701 onward is the future.
    s["S3.TO"].closes[1640:1701] *= np.linspace(1.0, 1.6, 61)
    snap = build_snapshot(s, s["S0.TO"].dates[1700], history_obs=1600,
                          baseline_by_date=baselines(s))
    assert MomentumScreen(lookback=60).propose(snap, 1)[0].symbol == "S3.TO"


def test_boosting_prices_after_the_snapshot_changes_nothing():
    """The same fixture applied past the snapshot close must not move the ranking.

    Index 1700 is the snapshot date itself, whose close is known; the future
    starts at 1701.
    """
    s = universe(4)
    day = s["S0.TO"].dates[1700]
    before = MomentumScreen(lookback=60).propose(
        build_snapshot(s, day, history_obs=1600, baseline_by_date=baselines(s)), 4
    )
    s["S3.TO"].closes[1701:] *= 2.0
    after = MomentumScreen(lookback=60).propose(
        build_snapshot(s, day, history_obs=1600, baseline_by_date=baselines(s)), 4
    )
    assert [c.symbol for c in before] == [c.symbol for c in after]


def test_a_candidate_without_a_mechanism_is_refused():
    """FR3: a candidate that cannot be stated in the fields is rejected early."""
    with pytest.raises(ValueError, match="mechanism"):
        Candidate("X.TO", "claim", "  ", "falsifier", TouchBelow(0.03), 20, 0.5, "test")


# --------------------------------------------------------------------- scorer


def test_resolution_is_blind_by_signature():
    """There is no parameter through which confidence or a decision could reach it."""
    import inspect

    params = set(inspect.signature(resolve).parameters)
    assert params == {"falsifier", "entry_close", "observed", "horizon_days"}


def test_a_tripped_falsifier_resolves_false_on_the_day_it_trips():
    observed = [(date(2026, 1, d), c) for d, c in zip(range(2, 8), [100, 99, 96, 97, 98, 99])]
    out = resolve(TouchBelow(0.03), 100.0, observed, 5)
    assert out.result == "false"
    assert out.resolved_at == date(2026, 1, 4)
    assert "tripped on day 3" in out.evidence


def test_a_surviving_falsifier_resolves_true_at_the_horizon():
    observed = [(date(2026, 1, d), 100.0) for d in range(2, 9)]
    out = resolve(TouchBelow(0.03), 100.0, observed, 5)
    assert out.result == "true"
    assert out.resolved_at == date(2026, 1, 6)


def test_an_unfinished_window_is_unresolvable_not_true():
    observed = [(date(2026, 1, 2), 100.0)]
    assert resolve(TouchBelow(0.03), 100.0, observed, 5).result == "unresolvable"


# ------------------------------------------------------- horizon-aware config


def test_volatility_conditioning_follows_the_measured_crossover():
    assert config_for_horizon(20).cond_vol is True
    assert config_for_horizon(60).cond_vol is False


# --------------------------------------------------------------------- ledger


def test_the_ledger_is_append_only_by_construction(tmp_path):
    led = Ledger(tmp_path)
    led.append("theses", {"id": "a"})
    led.append("theses", {"id": "b"})
    assert [r["id"] for r in led.read("theses")] == ["a", "b"]
    assert led.count("theses") == 2
    # There is no update or delete to call.
    assert not any(hasattr(led, name) for name in ("update", "delete", "truncate"))


def test_the_ledger_encodes_dates_and_numpy_scalars(tmp_path):
    led = Ledger(tmp_path)
    led.append("x", {"when": date(2026, 1, 2), "v": np.float64(1.5)})
    record = next(iter(led.read("x")))
    assert record == {"when": "2026-01-02", "v": 1.5}


def test_reading_a_kind_that_was_never_written_is_empty(tmp_path):
    assert list(Ledger(tmp_path).read("nothing")) == []
