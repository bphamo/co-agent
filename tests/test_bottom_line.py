"""The goal metric, and the hook-safety properties that keep it out of the way."""

from __future__ import annotations

import json

from co_agent.cycle.bottom_line import Reading, format_reading, load_baseline, main


def reading(strategy=0.10, bh=0.15, windows=1, fees=700.0, fills=144):
    return Reading(windows=windows, strategy_ret=strategy, buy_hold_ret=bh,
                   strategy_fees=fees, fills=fills)


def test_the_gap_is_strategy_minus_buy_and_hold():
    assert round(reading(0.10, 0.15).gap, 10) == -0.05
    assert round(reading(0.20, 0.15).gap, 10) == 0.05


def test_a_reading_round_trips_through_its_dict():
    before = reading()
    after = Reading.from_dict(before.to_dict())
    assert after == before


def test_the_verdict_follows_the_sign_of_the_gap():
    assert "BEHIND buy & hold" in format_reading(reading(0.10, 0.15), None)
    assert "BEATS buy & hold" in format_reading(reading(0.20, 0.15), None)


def test_the_report_never_drops_the_paper_caveat_or_the_sample_size():
    """A number this cheap to re-run is a number that invites overfitting."""
    text = format_reading(reading(windows=3), None)
    assert "paper" in text.lower()
    assert "3 windows" in text
    assert "survivor-only" in text


def test_a_baseline_turns_the_reading_into_a_delta():
    text = format_reading(reading(0.12, 0.15), reading(0.10, 0.15))
    assert "closer" in text and "+2.00%" in text

    text = format_reading(reading(0.08, 0.15), reading(0.10, 0.15))
    assert "further" in text


def test_an_unchanged_gap_reads_as_no_change():
    assert "no change" in format_reading(reading(0.10, 0.15), reading(0.10, 0.15))


def test_a_missing_or_corrupt_baseline_is_not_an_error(tmp_path):
    assert load_baseline(tmp_path / "absent.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_baseline(bad) is None
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"windows": 1}))
    assert load_baseline(partial) is None


def test_missing_price_data_exits_zero_so_a_hook_cannot_take_the_session_down(capsys):
    """Price data is gitignored, so a fresh clone has none. That is normal, not a fault."""
    code = main(["--prices", "/nonexistent/prices", "--quiet-if-missing"])
    assert code == 0
    assert "unavailable" in capsys.readouterr().out


def test_missing_price_data_is_still_an_error_when_run_by_hand():
    assert main(["--prices", "/nonexistent/prices"]) == 1
