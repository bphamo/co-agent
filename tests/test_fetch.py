"""Tests for the price fetcher.

No network: the HTTP call is injected, so these cover the parts that actually
break -- retry policy, cache behaviour, and refusing a payload that would
silently produce wrong returns.
"""

from __future__ import annotations

import json
import urllib.error
from datetime import date

import pytest

from co_agent.data.fetch import (
    FetchError,
    fetch_raw,
    fetch_symbol,
    parse_chart,
    write_csv,
)
from co_agent.data.prices import load_csv


def payload(stamps, adj, currency="CAD", close=None):
    return {
        "chart": {
            "result": [
                {
                    "timestamp": stamps,
                    "indicators": {
                        "adjclose": [{"adjclose": adj}],
                        "quote": [{"close": close or adj}],
                    },
                    "meta": {"currency": currency},
                }
            ]
        }
    }


DAY = 86_400
STAMPS = [1_577_923_200, 1_577_923_200 + DAY, 1_577_923_200 + 2 * DAY]


def responder(body, *, fail_times=0, code=429):
    """An injectable HTTP GET that fails a few times before succeeding."""
    state = {"calls": 0}

    def get(url, timeout):
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise urllib.error.HTTPError(url, code, "nope", {}, None)
        return json.dumps(body).encode()

    get.state = state  # type: ignore[attr-defined]
    return get


# ------------------------------------------------------------------- parsing


def test_parse_returns_adjusted_closes_sorted():
    rows, currency = parse_chart("X.TO", payload(STAMPS, [10.0, 11.0, 12.0]))
    assert currency == "CAD"
    assert [r[1] for r in rows] == [10.0, 11.0, 12.0]
    assert rows[0][0] < rows[-1][0]
    assert all(isinstance(r[0], date) for r in rows)


def test_halted_days_are_dropped_not_interpolated():
    """A null close is a day with no trade; inventing one invents a return."""
    rows, _ = parse_chart("X.TO", payload(STAMPS, [10.0, None, 12.0]))
    assert [r[1] for r in rows] == [10.0, 12.0]


def test_a_payload_without_adjusted_closes_is_refused():
    """Unadjusted prices would put every split into the return series."""
    body = payload(STAMPS, [10.0, 11.0, 12.0])
    del body["chart"]["result"][0]["indicators"]["adjclose"]
    with pytest.raises(FetchError, match="no adjusted close"):
        parse_chart("X.TO", body)


@pytest.mark.parametrize(
    "body,fragment",
    [
        ({"chart": {"error": {"code": "Not Found"}}}, "Not Found"),
        ({"chart": {"result": []}}, "no result"),
        (payload(STAMPS[:1], [10.0]), "usable rows"),
        (payload(STAMPS, [None, None, None]), "usable rows"),
    ],
)
def test_unusable_payloads_are_refused_with_the_reason(body, fragment):
    with pytest.raises(FetchError, match=fragment):
        parse_chart("X.TO", body)


# --------------------------------------------------------- retry and caching


def test_rate_limiting_is_retried_with_backoff(tmp_path):
    """The first request from a datacenter address is often refused."""
    get = responder(payload(STAMPS, [1.0, 2.0, 3.0]), fail_times=2)
    slept: list[float] = []
    body, cached = fetch_raw(
        "X.TO", cache_dir=tmp_path, get=get, sleep=slept.append, backoff=2.0
    )
    assert not cached
    assert get.state["calls"] == 3
    assert slept == [2.0, 4.0], "backoff should double"
    assert body["chart"]["result"]


def test_retries_are_bounded_and_the_last_error_is_reported(tmp_path):
    get = responder({}, fail_times=99, code=503)
    with pytest.raises(FetchError, match="503"):
        fetch_raw("X.TO", cache_dir=tmp_path, get=get, sleep=lambda _: None, attempts=3)
    assert get.state["calls"] == 3


def test_a_client_error_is_not_retried(tmp_path):
    """404 means the ticker is wrong; retrying just makes it wrong four times."""
    get = responder({}, fail_times=99, code=404)
    with pytest.raises(FetchError, match="404"):
        fetch_raw("NOPE.TO", cache_dir=tmp_path, get=get, sleep=lambda _: None)
    assert get.state["calls"] == 1


def test_the_cache_is_used_on_the_second_call(tmp_path):
    """A study must re-run offline, or a reproduced figure depends on uptime."""
    get = responder(payload(STAMPS, [1.0, 2.0, 3.0]))
    fetch_raw("X.TO", cache_dir=tmp_path, get=get, sleep=lambda _: None)
    _, cached = fetch_raw("X.TO", cache_dir=tmp_path, get=get, sleep=lambda _: None)
    assert cached
    assert get.state["calls"] == 1, "cache hit should not call out"


def test_refresh_bypasses_the_cache(tmp_path):
    get = responder(payload(STAMPS, [1.0, 2.0, 3.0]))
    fetch_raw("X.TO", cache_dir=tmp_path, get=get, sleep=lambda _: None)
    fetch_raw("X.TO", cache_dir=tmp_path, get=get, sleep=lambda _: None, refresh=True)
    assert get.state["calls"] == 2


# ------------------------------------------------------------------ round trip


def test_written_csv_reloads_through_the_loader(tmp_path):
    """The fetcher's output format and the loader's input format must agree."""
    get = responder(payload(STAMPS, [10.0, 11.0, 12.0]))
    got = fetch_symbol(
        "X.TO",
        cache_dir=tmp_path / "cache",
        out_dir=tmp_path / "csv",
        get=get,
        sleep=lambda _: None,
    )
    assert got.currency == "CAD"
    assert got.span[0] < got.span[1]

    series = load_csv(tmp_path / "csv" / "X.TO.csv")
    assert series.symbol == "X.TO"
    assert series.closes.tolist() == [10.0, 11.0, 12.0]
    assert series.log_returns.size == 2


def test_write_csv_keeps_full_precision(tmp_path):
    rows = [(date(2020, 1, 2), 1.234567), (date(2020, 1, 3), 2.0)]
    path = write_csv("P.TO", rows, tmp_path)
    assert load_csv(path).closes[0] == pytest.approx(1.234567, abs=1e-6)
