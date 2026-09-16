"""Tests for the model-driven candidate source.

No network: the client is injected. These cover the parts that decide whether a
bad batch costs a cycle -- validation of what comes back, and the contamination
guard.
"""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from co_agent.cycle import LLMResearch, ThesisBatch, ThesisDraft, build_snapshot


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
    baselines = {d: 0.0002 for d in series["S0.TO"].dates}
    return build_snapshot(series, day, history_obs=1600, baseline_by_date=baselines)


def fake_client(batch: ThesisBatch, usage=None):
    def parse(**kwargs):
        parse.kwargs = kwargs
        return SimpleNamespace(parsed_output=batch, usage=usage)

    return SimpleNamespace(messages=SimpleNamespace(parse=parse)), parse


def draft(symbol="S0.TO", drop=0.06, conf=0.6, claim="holds up", mech="a real cause"):
    return ThesisDraft(
        symbol=symbol, claim=claim, mechanism=mech,
        drop_threshold=drop, confidence=conf,
    )


def test_a_valid_batch_becomes_candidates():
    snap = make_snapshot()
    client, parse = fake_client(ThesisBatch(theses=[draft("S0.TO"), draft("S1.TO", 0.08)]))
    out = LLMResearch(client=client, horizon_days=60).propose(snap, 2)

    assert [c.symbol for c in out] == ["S0.TO", "S1.TO"]
    assert out[1].falsifier.drop == 0.08
    assert out[0].horizon_days == 60
    assert out[0].source == "research/claude-opus-5"
    assert "8.0%" in out[1].falsifier_text and str(snap.taken_at) in out[1].falsifier_text


def test_a_hallucinated_ticker_is_dropped_not_raised():
    """A short batch is recoverable; a KeyError three steps later is not."""
    snap = make_snapshot()
    client, _ = fake_client(ThesisBatch(theses=[draft("NOPE.TO"), draft("S1.TO")]))
    out = LLMResearch(client=client).propose(snap, 2)
    assert [c.symbol for c in out] == ["S1.TO"]


@pytest.mark.parametrize(
    "bad",
    [
        draft(drop=0.001),           # tighter than any gate-passing falsifier
        draft(drop=0.5),             # so wide that surviving it says nothing
        draft(conf=1.7),             # not a probability
        draft(claim="   "),          # FR3: the fields are the record
        draft(mech=""),
    ],
)
def test_malformed_drafts_are_dropped(bad):
    snap = make_snapshot()
    client, _ = fake_client(ThesisBatch(theses=[bad]))
    assert LLMResearch(client=client).propose(snap, 1) == []


def test_duplicate_symbols_are_collapsed():
    snap = make_snapshot()
    client, _ = fake_client(ThesisBatch(theses=[draft("S0.TO"), draft("S0.TO", 0.09)]))
    out = LLMResearch(client=client).propose(snap, 2)
    assert len(out) == 1 and out[0].falsifier.drop == 0.06


def test_symbols_are_matched_case_insensitively():
    snap = make_snapshot()
    client, _ = fake_client(ThesisBatch(theses=[draft("s0.to")]))
    assert [c.symbol for c in LLMResearch(client=client).propose(snap, 1)] == ["S0.TO"]


def test_a_snapshot_inside_the_training_window_is_refused():
    """Replay over data the model may have memorised is recall, not forecast."""
    snap = make_snapshot()
    client, _ = fake_client(ThesisBatch(theses=[draft()]))
    source = LLMResearch(client=client, earliest_date=date(2099, 1, 1))
    with pytest.raises(ValueError, match="recall rather than forecast"):
        source.propose(snap, 1)

    allowed = LLMResearch(
        client=client, earliest_date=date(2099, 1, 1), allow_contaminated=True
    )
    assert allowed.propose(snap, 1), "the opt-out should work for a mechanical test"


def test_the_prompt_carries_only_snapshot_data_and_asks_for_calibration():
    snap = make_snapshot()
    client, parse = fake_client(ThesisBatch(theses=[draft()]))
    LLMResearch(client=client, horizon_days=60).propose(snap, 3)

    prompt = parse.kwargs["messages"][0]["content"]
    assert str(snap.taken_at) in prompt
    assert "exactly 3 theses" in prompt
    assert "Brier" in prompt, "the model should know how its confidence is scored"
    assert "mechanism" in prompt
    for symbol in snap.universe:
        assert symbol in prompt
    assert parse.kwargs["model"] == "claude-opus-5"
    assert parse.kwargs["thinking"] == {"type": "adaptive"}
    assert parse.kwargs["output_format"] is ThesisBatch


def test_token_usage_is_recorded_for_the_cost_ceiling():
    """NFR2 aborts a cycle on spend; that needs the numbers."""
    snap = make_snapshot()
    usage = SimpleNamespace(input_tokens=1234, output_tokens=567)
    client, _ = fake_client(ThesisBatch(theses=[draft()]), usage=usage)
    source = LLMResearch(client=client)
    source.propose(snap, 1)
    assert source._last_usage == {"input_tokens": 1234, "output_tokens": 567}
