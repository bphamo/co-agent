"""Tests for the price loader."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from co_agent.data import PriceDataError, load_csv, load_dir, suspicious_returns


def write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


GOOD = "date,close\n2020-01-03,44.88\n2020-01-02,45.13\n2020-01-06,45.50\n"


def test_rows_are_sorted_and_returns_computed(tmp_path):
    series = load_csv(write(tmp_path, "FAKE.csv", GOOD))
    assert series.symbol == "FAKE"
    assert series.dates == (date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6))
    assert series.closes[0] == 45.13
    assert series.log_returns.size == 2
    assert series.log_returns[0] == pytest.approx(np.log(44.88 / 45.13))
    assert series.span_days == 4


def test_blank_closes_are_non_trading_rows_not_errors(tmp_path):
    body = "date,close\n2020-01-02,45.13\n2020-01-03,\n2020-01-06,45.50\n"
    assert load_csv(write(tmp_path, "X.csv", body)).closes.size == 2


@pytest.mark.parametrize(
    "body,fragment",
    [
        ("date,price\n2020-01-02,45\n", "missing column"),
        ("date,close\n2020-01-02,-3\n2020-01-03,4\n", "non-positive"),
        ("date,close\n2020-01-02,45\n2020-01-02,46\n", "duplicate dates"),
        ("date,close\n2020-01-02,45\n", "at least 2 rows"),
        ("date,close\nnot-a-date,45\n2020-01-03,46\n", "not-a-date"),
    ],
)
def test_bad_input_is_rejected_with_the_reason(tmp_path, body, fragment):
    with pytest.raises(PriceDataError, match=fragment):
        load_csv(write(tmp_path, "BAD.csv", body))


def test_load_dir_keys_by_symbol_and_honours_min_obs(tmp_path):
    write(tmp_path, "AAA.csv", GOOD)
    write(tmp_path, "BBB.csv", "date,close\n2020-01-02,10\n2020-01-03,11\n")
    assert set(load_dir(tmp_path)) == {"AAA", "BBB"}
    assert set(load_dir(tmp_path, min_obs=3)) == {"AAA"}
    with pytest.raises(PriceDataError, match="no usable series"):
        load_dir(tmp_path, min_obs=99)


def test_missing_directory_is_reported(tmp_path):
    with pytest.raises(PriceDataError, match="not a directory"):
        load_dir(tmp_path / "nope")


def test_suspicious_returns_flags_an_unadjusted_split(tmp_path):
    # A 2-for-1 split with no adjustment: a -69% "return" the bootstrap would
    # otherwise resample into every synthetic path.
    body = "date,close\n2020-01-02,100\n2020-01-03,101\n2020-01-06,50.5\n"
    series = load_csv(write(tmp_path, "SPLIT.csv", body))
    flagged = suspicious_returns(series)
    assert [d for d, _ in flagged] == [date(2020, 1, 6)]
    assert flagged[0][1] < -0.5
    assert suspicious_returns(series, threshold=1.0) == []
