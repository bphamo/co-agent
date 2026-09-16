"""The frozen input every downstream step reads (TRD §4.3).

Built from price history rather than a broker, because paper trading has no
broker. The load-bearing property is unchanged: a snapshot is taken as of a
date, contains only information available on that date, and everything
downstream reads it instead of going back to the source.

No-look-ahead is enforced by slicing, not by convention -- a history ends at the
snapshot's own index, so a worker cannot reach past it even if it tries.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass(frozen=True, slots=True)
class Snapshot:
    taken_at: date
    universe: tuple[str, ...]
    closes: dict[str, float]
    #: Daily log returns ending on ``taken_at``, per symbol.
    histories: dict[str, np.ndarray]
    #: Equal-weighted trailing market drift as of this date, for Drift.SHRUNK.
    baseline_drift: float
    #: Names dropped for want of history or a price on the day.
    missing: tuple[str, ...]
    history_obs: int

    @property
    def digest(self) -> str:
        """Content hash. Two snapshots with the same digest are the same inputs."""
        h = hashlib.sha256(self.taken_at.isoformat().encode())
        for symbol in self.universe:
            h.update(symbol.encode())
            h.update(np.ascontiguousarray(self.histories[symbol], dtype="<f8").tobytes())
            h.update(f"{self.closes[symbol]:.6f}".encode())
        h.update(f"{self.baseline_drift:.12f}".encode())
        return h.hexdigest()


def build_snapshot(
    series_by_symbol: dict,
    taken_at: date,
    *,
    history_obs: int,
    baseline_by_date: dict[date, float],
) -> Snapshot:
    """Freeze the universe as of ``taken_at``."""
    closes: dict[str, float] = {}
    histories: dict[str, np.ndarray] = {}
    missing: list[str] = []

    for symbol, px in sorted(series_by_symbol.items()):
        index = {d: i for i, d in enumerate(px.dates)}
        i = index.get(taken_at)
        returns = px.log_returns
        # returns[k] ends on dates[k+1], so returns[:i] are exactly those
        # realised on or before taken_at.
        if i is None or i < history_obs or i > returns.size:
            missing.append(symbol)
            continue
        closes[symbol] = float(px.closes[i])
        histories[symbol] = returns[i - history_obs : i]

    if taken_at not in baseline_by_date:
        raise KeyError(f"no market baseline for {taken_at}")

    return Snapshot(
        taken_at=taken_at,
        universe=tuple(sorted(closes)),
        closes=closes,
        histories=histories,
        baseline_drift=baseline_by_date[taken_at],
        missing=tuple(missing),
        history_obs=history_obs,
    )
