"""Fetch daily price history into the CSV layout the loader expects.

One unofficial endpoint is reachable from this environment, so that is what this
uses. It is an undocumented interface that rate-limits aggressively and can
change without notice, which shapes the design:

* every response is cached to disk as raw JSON, so a study re-runs offline and a
  reproduced figure is not at the mercy of the endpoint being up;
* 429 and 5xx retry with exponential backoff, because the first request from a
  datacenter address is frequently refused and the second succeeds;
* the raw JSON is kept alongside the CSV. The CSV carries adjusted closes only,
  but the cache retains open/high/low/volume, so adopting intraday-touch
  falsifiers later needs no re-fetch.

**Adjusted closes.** The CSV's ``close`` column is the split- and
dividend-adjusted close, which is what return modelling wants. The cache's raw
``close`` is unadjusted; do not mix them. If high/low are ever promoted into the
CSV they must be scaled by the same ``adjclose / close`` factor, or a study will
silently combine adjusted and unadjusted prices.

Nothing here commits price data to the repository: the output directory is
gitignored. Redistribution terms differ by source and this data is fetched for
private research.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?period1=0&period2=9999999999&interval=1d&events=div%2Csplit"
)

#: Deliberately not the user's identity: this is an outbound request to a third
#: party and carries no more than it needs to.
USER_AGENT = "co-agent/0.1 (private research)"


class FetchError(RuntimeError):
    """The endpoint could not be read, after retries."""


@dataclass(frozen=True, slots=True)
class Fetched:
    symbol: str
    rows: list[tuple[date, float]]
    currency: str | None
    from_cache: bool

    @property
    def span(self) -> tuple[date, date]:
        return self.rows[0][0], self.rows[-1][0]


def _http_get(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_raw(
    symbol: str,
    *,
    cache_dir: Path,
    refresh: bool = False,
    attempts: int = 4,
    backoff: float = 2.0,
    timeout: float = 45.0,
    get=_http_get,
    sleep=time.sleep,
) -> tuple[dict, bool]:
    """Return the raw chart payload for ``symbol``, and whether it came from cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{symbol}.json"
    if cached.exists() and not refresh:
        return json.loads(cached.read_text(encoding="utf-8")), True

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            body = get(CHART_URL.format(symbol=symbol), timeout)
            payload = json.loads(body)
            cached.write_text(json.dumps(payload), encoding="utf-8")
            return payload, False
        except urllib.error.HTTPError as exc:  # noqa: PERF203 - retry needs the loop
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
        if attempt < attempts - 1:
            sleep(backoff * (2**attempt))
    raise FetchError(f"{symbol}: {last}")


def parse_chart(symbol: str, payload: dict) -> tuple[list[tuple[date, float]], str | None]:
    """Pull (date, adjusted close) rows out of a chart payload.

    Rows with a null adjusted close are dropped: the endpoint emits them for
    halted days, and interpolating would invent a return that never happened.
    """
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise FetchError(f"{symbol}: {chart['error']}")
    results = chart.get("result")
    if not results:
        raise FetchError(f"{symbol}: no result in payload")

    result = results[0]
    stamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    adj = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    if adj is None:
        raise FetchError(f"{symbol}: payload carries no adjusted close")

    rows = [
        (datetime.fromtimestamp(ts, tz=timezone.utc).date(), float(px))
        for ts, px in zip(stamps, adj)
        if px is not None and px > 0
    ]
    if len(rows) < 2:
        raise FetchError(f"{symbol}: {len(rows)} usable rows")
    rows.sort(key=lambda r: r[0])
    return rows, (result.get("meta") or {}).get("currency")


def write_csv(symbol: str, rows: list[tuple[date, float]], out_dir: Path) -> Path:
    """Write the ``date,close`` CSV the loader reads. ``close`` is adjusted."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("date,close\n")
        for when, close in rows:
            handle.write(f"{when.isoformat()},{close:.6f}\n")
    return path


def fetch_symbol(
    symbol: str,
    *,
    cache_dir: Path,
    out_dir: Path | None = None,
    refresh: bool = False,
    **kwargs,
) -> Fetched:
    payload, from_cache = fetch_raw(symbol, cache_dir=cache_dir, refresh=refresh, **kwargs)
    rows, currency = parse_chart(symbol, payload)
    if out_dir is not None:
        write_csv(symbol, rows, out_dir)
    return Fetched(symbol=symbol, rows=rows, currency=currency, from_cache=from_cache)


def main(argv: list[str] | None = None) -> int:
    """Fetch a universe into CSVs.

        python -m co_agent.data.fetch --universe tsx --out data/prices
    """
    import argparse

    from .universes import UNIVERSES

    parser = argparse.ArgumentParser(description="Fetch daily price history.")
    parser.add_argument("--universe", default="tsx", choices=sorted(UNIVERSES))
    parser.add_argument("--symbols", nargs="*", help="override the universe")
    parser.add_argument("--out", default="data/prices")
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache")
    parser.add_argument("--pause", type=float, default=0.8, help="seconds between symbols")
    args = parser.parse_args(argv)

    symbols = tuple(args.symbols) if args.symbols else UNIVERSES[args.universe]
    out_dir, cache_dir = Path(args.out), Path(args.cache)

    ok, failed = 0, []
    for symbol in symbols:
        try:
            got = fetch_symbol(
                symbol, cache_dir=cache_dir, out_dir=out_dir, refresh=args.refresh
            )
        except FetchError as exc:
            print(f"  ! {exc}")
            failed.append(symbol)
            continue
        start, end = got.span
        print(
            f"  {symbol:<12} {len(got.rows):>6} rows  {start} -> {end}  "
            f"{got.currency or '?':<4} {'(cached)' if got.from_cache else ''}"
        )
        ok += 1
        if not got.from_cache:
            time.sleep(args.pause)

    print(f"\n{ok}/{len(symbols)} symbols -> {out_dir}")
    if failed:
        print(f"failed: {', '.join(failed)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
