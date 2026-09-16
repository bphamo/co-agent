"""The side-by-side report: one snapshot, both arms, and a degradable research arm."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np

from co_agent.cycle import ThesisBatch, ThesisDraft, build_snapshot
from co_agent.cycle.compare_arms import (
    benchmark_arm,
    format_arm,
    format_unavailable,
    research_arm,
)


class FakeSeries:
    def __init__(self, dates, closes):
        self.dates, self.closes = tuple(dates), np.asarray(closes, float)

    @property
    def log_returns(self):
        return np.diff(np.log(self.closes))


def make_snapshot(n=1800, k=4, taken_at_index=1700):
    series = {}
    for i in range(k):
        rng = np.random.default_rng(i + 1)
        closes = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
        dates = [date(2018, 1, 1) + timedelta(days=j) for j in range(n)]
        series[f"S{i}.TO"] = FakeSeries(dates, closes)
    day = series["S0.TO"].dates[taken_at_index]
    return build_snapshot(
        series, day, history_obs=1600,
        baseline_by_date={d: 0.0002 for d in series["S0.TO"].dates},
    )


def fake_client(batch: ThesisBatch):
    return SimpleNamespace(
        messages=SimpleNamespace(parse=lambda **kw: SimpleNamespace(parsed_output=batch, usage=None))
    )


def test_the_benchmark_arm_needs_no_client():
    snap = make_snapshot()
    name, candidates = benchmark_arm(snap, 3, horizon_days=60)

    assert name.startswith("momentum_screen/")
    assert len(candidates) == 3
    # FR3: every arm owes the same fields, so the two are comparable at all.
    for c in candidates:
        assert c.claim.strip() and c.mechanism.strip() and c.falsifier_text.strip()
        assert c.horizon_days == 60


def test_both_arms_are_anchored_to_the_same_snapshot():
    """The comparison is only meaningful if neither arm saw a close the other did not."""
    snap = make_snapshot()
    _, screen = benchmark_arm(snap, 2, horizon_days=60)
    _, research = research_arm(
        snap, 2, 60,
        fake_client(ThesisBatch(theses=[
            ThesisDraft(symbol="S0.TO", claim="holds", mechanism="a real cause",
                        drop_threshold=0.06, confidence=0.6),
        ])),
    )

    assert research, "the fake client should have produced a candidate"
    for c in list(screen) + list(research):
        assert str(snap.taken_at) in c.falsifier_text
        assert c.horizon_days == 60


def test_a_missing_key_degrades_the_report_rather_than_failing_it():
    """Half a comparison is a result; losing the snapshot date would make it unrepeatable."""
    snap = make_snapshot()
    text = format_unavailable(snap, RuntimeError("no credentials"))

    assert "RuntimeError" in text and "no credentials" in text
    assert str(snap.taken_at) in text, "the report must say which snapshot to resume from"


def test_an_empty_arm_renders_as_empty_rather_than_blank():
    assert "(none proposed)" in format_arm("research arm -- x", [])
