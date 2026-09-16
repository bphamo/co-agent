"""Tests for the rolling-window cadence study.

The arithmetic the study reports is only as good as the marking and the window
construction, so those are what is pinned here. The sweep itself is not run --
it needs price data and minutes, and `--report` is separable from it by design.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np

from co_agent.cycle.cadence import (
    FIRST_START,
    STEP_DAYS,
    WINDOW_DAYS,
    NullLedger,
    _final_marks,
    calendar,
    equal_weight_buy_hold,
    index_map,
    report,
    window_starts,
)

class FakeSeries:
    def __init__(self, dates, closes):
        self.dates, self.closes = tuple(dates), np.asarray(closes, float)

    @property
    def log_returns(self):
        return np.diff(np.log(self.closes))

def series_pair():
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(10)]
    full = FakeSeries(days, [100.0] * 10)
    # Delisted after day 4: no price on the final day of the window.
    halted = FakeSeries(days[:5], [50.0, 50.0, 50.0, 50.0, 40.0])
    return {"FULL": full, "HALT": halted}, days

def test_null_ledger_opens_no_path_and_swallows_appends(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ledger = NullLedger()
    ledger.append("theses", {"id": "x"})
    ledger.append("fills", {"id": "y"})
    assert list(tmp_path.iterdir()) == [], "the study must not write a ledger"

def test_a_halted_name_is_marked_at_its_last_observed_close():
    """Not zero, which books a loss that did not happen; not dropped, which
    books no loss at all."""
    series, days = series_pair()
    marks = _final_marks(series, index_map(series), days, ["FULL", "HALT"])
    assert marks["FULL"] == 100.0
    assert marks["HALT"] == 40.0

def test_buy_and_hold_marks_a_halted_name_rather_than_dropping_it():
    series, days = series_pair()
    result = equal_weight_buy_hold(series, index_map(series), days)
    assert result is not None
    # HALT was bought at 50 and is marked at 40, so the arm must show a loss
    # even though FULL never moved.
    assert result.ret < 0
    assert result.fills == 2

def test_windows_overlap_and_start_no_earlier_than_the_study_epoch():
    days = [date(1999, 1, 1) + timedelta(days=i) for i in range(4000)]
    starts = window_starts(days)
    assert starts, "expected at least one window"
    assert all(days[i] >= FIRST_START for i in starts)
    assert starts[1] - starts[0] == STEP_DAYS
    assert STEP_DAYS < WINDOW_DAYS, "windows must overlap for this to be a rolling test"
    assert starts[-1] + WINDOW_DAYS <= len(days), "a window must not run off the end"

def test_calendar_is_the_sorted_union_across_names():
    series, days = series_pair()
    assert calendar(series) == sorted(days)

def test_report_aggregates_shards_and_counts_the_head_to_head(tmp_path, capsys):
    rows = [
        {"window": i, "start": f"20{10 + i:02d}-01-01", "end": f"20{11 + i:02d}-01-01",
         "weekly_hold": {"ret": 0.01 * i, "fills": 200, "fees": 1200.0, "cycles": 51},
         "monthly_hold": {"ret": 0.02 * i, "fills": 60, "fees": 360.0, "cycles": 12},
         "weekly_fexit": {"ret": -0.01 * i, "fills": 290, "fees": 1400.0, "cycles": 51},
         "buy_hold": {"ret": 0.015 * i, "fills": 48, "fees": 240.0}}
        for i in range(1, 6)
    ]
    a = tmp_path / "shard0.jsonl"
    b = tmp_path / "shard1.jsonl"
    a.write_text("\n".join(json.dumps(r) for r in rows[:3]) + "\n")
    b.write_text("\n".join(json.dumps(r) for r in rows[3:]) + "\n")

    assert report([str(tmp_path / "shard*.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "5 windows" in out, "both shards must be read"
    # monthly beats weekly in every row here, by construction.
    assert "monthly_hold vs weekly_hold: wins 5/5 (100%)" in out

def test_report_on_nothing_is_an_error_not_an_empty_table(tmp_path):
    assert report([str(tmp_path / "nothing*.jsonl")]) == 1
