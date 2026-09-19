"""Loading real price history for the historical walk-forward.

No data ships with this repository. Redistributing vendor price series is a
licensing question rather than a technical one, and the answer differs by source,
so the loader takes a directory you populate yourself.

**Two properties of the data matter more than the loader.**

*Point-in-time universe.* A walk-forward run over a symbol list assembled today
excludes everything that was delisted, acquired or wound up in the window -- which
is precisely where the large drawdowns are. That biases realised trip frequencies
*down* and will make the simulator look better calibrated on the tail than it is.
There is no code fix; either source a point-in-time constituent list or state the
bias beside the result.

*Unadjusted closes.* A split or a large distribution shows up as a one-day move of
the wrong size, which the bootstrap will happily resample into every synthetic
path. Use adjusted closes, and check the largest absolute returns in each series
before trusting a study built on it -- :func:`suspicious_returns` is there for that.

Expected format: one CSV per symbol, named ``<SYMBOL>.csv``, with a date column
and a close column, oldest row first or in any order (rows are sorted on load)::

    date,close
    2020-01-02,45.13
    2020-01-03,44.88
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np


class PriceDataError(ValueError):
    """The series on disk cannot be used as-is."""


@dataclass(frozen=True, slots=True)
class PriceSeries:
    """One symbol's daily closes, oldest first."""

    symbol: str
    dates: tuple[date, ...]
    closes: np.ndarray

    @property
    def log_returns(self) -> np.ndarray:
        """Daily log returns. One shorter than ``closes``."""
        return np.diff(np.log(self.closes))

    @property
    def span_days(self) -> int:
        return (self.dates[-1] - self.dates[0]).days


def load_csv(
    path: str | Path,
    *,
    symbol: str | None = None,
    date_col: str = "date",
    close_col: str = "close",
    date_format: str | None = None,
) -> PriceSeries:
    """Read one symbol's series from a CSV."""
    path = Path(path)
    name = symbol or path.stem

    rows: list[tuple[date, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PriceDataError(f"{path}: no header row")
        missing = {date_col, close_col} - set(reader.fieldnames)
        if missing:
            raise PriceDataError(
                f"{path}: missing column(s) {sorted(missing)}; found {reader.fieldnames}"
            )
        for lineno, row in enumerate(reader, start=2):
            raw_date, raw_close = row[date_col], row[close_col]
            if raw_date is None or raw_close is None or raw_close == "":
                continue  # a blank close is a non-trading row, not an error
            try:
                when = (
                    datetime.strptime(raw_date, date_format).date()
                    if date_format
                    else date.fromisoformat(raw_date)
                )
                close = float(raw_close)
            except ValueError as exc:
                raise PriceDataError(f"{path}:{lineno}: {exc}") from exc
            if close <= 0:
                raise PriceDataError(f"{path}:{lineno}: non-positive close {close}")
            rows.append((when, close))

    if len(rows) < 2:
        raise PriceDataError(f"{path}: need at least 2 rows, got {len(rows)}")

    rows.sort(key=lambda r: r[0])
    dates = [r[0] for r in rows]
    if len(set(dates)) != len(dates):
        raise PriceDataError(f"{path}: duplicate dates")

    return PriceSeries(
        symbol=name,
        dates=tuple(dates),
        closes=np.array([r[1] for r in rows], dtype=np.float64),
    )


def load_dir(
    directory: str | Path,
    *,
    pattern: str = "*.csv",
    min_obs: int = 0,
    **kwargs: object,
) -> dict[str, PriceSeries]:
    """Load every CSV in a directory, keyed by symbol.

    ``min_obs`` drops series too short to be worth loading. Dropping is silent by
    design here -- the caller reports the count, because "how many names had
    enough history" is part of any result built on them.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise PriceDataError(f"{directory} is not a directory")

    out: dict[str, PriceSeries] = {}
    for path in sorted(directory.glob(pattern)):
        series = load_csv(path, **kwargs)  # type: ignore[arg-type]
        if series.closes.size >= min_obs:
            out[series.symbol] = series
    if not out:
        raise PriceDataError(f"{directory}: no usable series matching {pattern!r}")
    return out


def suspicious_returns(series: PriceSeries, threshold: float = 0.25) -> list[tuple[date, float]]:
    """Daily moves large enough to be an unadjusted split rather than a price move.

    Not a validator -- a genuine 30% day happens. It is a list to look at before
    a study is built on the series, because one bad row becomes a fat tail the
    bootstrap resamples into every synthetic path.
    """
    returns = series.log_returns
    return [
        (series.dates[i + 1], float(r))
        for i, r in enumerate(returns)
        if abs(r) >= threshold
    ]
