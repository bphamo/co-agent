"""Provenance types for the FR9 simulation service.

Two rules from the TRD shape this package:

* It never runs inside an agent loop and its parameters are never chosen by an
  agent.  Everything here is a pure function of ``(history, falsifier,
  config)``; nothing calls a model.
* Synthetic price series must never reach a research agent's context.  The
  package therefore returns summary statistics and provenance only -- no public
  function hands back a simulated path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from enum import StrEnum

import numpy as np

#: Identifies the simulator's behaviour.  Bump it on any change that would move
#: a figure, so a stored ``null_probability`` can be invalidated rather than
#: silently compared against one produced by different code.
VERSION = "sim/3"


class Method(StrEnum):
    """Recorded as ``theses.sim_method``."""

    #: Resamples the symbol's own returns.
    STATIONARY_BOOTSTRAP = "stationary_bootstrap"
    #: Resamples a peer's returns scaled to the symbol's volatility, for names
    #: with too little history of their own (open question 6).  Never selected
    #: implicitly -- see :func:`new_proxy_history`.
    PROXY_BOOTSTRAP = "stationary_bootstrap_proxy"
    #: The event-falsifier class: a null probability that comes from a recorded
    #: base rate, because no return distribution can answer "does the company
    #: guide below $X in Q3".
    EXPLICIT_PRIOR = "explicit_prior"


class Drift(StrEnum):
    """What "by chance" means.

    ``ZERO`` demeans the resampling pool, so the null is a forecaster with no
    information about direction.  ``HISTORICAL`` keeps the symbol's realised
    drift, which makes the null absorb past momentum and generally makes upside
    falsifiers look easier than they are.  The choice materially moves
    ``null_probability``, so it is recorded in ``sim_params`` either way.
    """

    ZERO = "zero"
    HISTORICAL = "historical"
    #: A weighted blend of the symbol's realised drift and a supplied baseline
    #: (a market or sector mean). Zero drift misstates the counterfactual for an
    #: asset that drifts; the symbol's own drift extrapolates its momentum.
    #: Shrinking toward a baseline is the middle, and the weight is measurable.
    SHRUNK = "shrunk"


class SimInputError(ValueError):
    """Any input-validation failure."""


class InsufficientHistoryError(SimInputError):
    """A symbol has too little history for the block bootstrap to mean anything.

    FR9's method needs enough independent blocks that the resampled tail is not
    three realised episodes wearing a Monte Carlo costume.  Below the threshold
    this is a rejection, never a silent fallback to a shorter window or a peer:
    see :func:`new_proxy_history` for the explicit route.
    """

    def __init__(self, have: int, want: int, standardised: bool = False) -> None:
        self.have = have
        self.want = want
        self.standardised = standardised
        what = "usable observations after the EWMA warmup" if standardised else "observations"
        super().__init__(
            f"insufficient history: {have} {what}, need {want} "
            f"(reject the candidate, or supply a peer via new_proxy_history)"
        )


@dataclass(frozen=True, slots=True)
class ProxyInfo:
    """Records that the bootstrap ran on someone else's returns."""

    proxy_symbol: str
    proxy_symbol_id: int
    vol_scale: float
    reason: str


@dataclass(frozen=True, slots=True)
class Prior:
    """The event-falsifier class's null probability."""

    #: Probability the falsifier trips absent the thesis's mechanism.
    p: float
    #: Human-readable provenance, e.g. "8 of 17 comparable quarters, 2019-2025".
    source: str
    #: Observations behind ``p``.  Zero means judgemental, which is permitted
    #: but recorded: a gate verdict on an ungrounded prior is a weaker
    #: statement than one on a counted base rate.
    n: int = 0


@dataclass(frozen=True, slots=True)
class History:
    """A symbol's daily log-return series, oldest first."""

    symbol: str
    symbol_id: int
    log_returns: np.ndarray
    start: date | None = None
    end: date | None = None
    #: Set only by :func:`new_proxy_history`.
    proxy: ProxyInfo | None = None

    def __post_init__(self) -> None:
        arr = np.asarray(self.log_returns, dtype=np.float64)
        if arr.ndim != 1:
            raise SimInputError("log_returns must be one-dimensional")
        object.__setattr__(self, "log_returns", arr)


@dataclass(slots=True)
class Params:
    """Serialised into ``theses.sim_params``.

    It exists so a figure can be reproduced or invalidated, which is the only
    reason FR9 asks for it.
    """

    method: Method
    version: str
    paths: int
    mean_block_len: float
    drift: Drift
    cond_vol: bool
    horizon_days: int
    history_obs: int
    history_sha256: str
    seed: int
    gate_band: tuple[float, float]
    drift_shrinkage: float | None = None
    baseline_drift: float | None = None
    interval_method: str | None = None
    outer_resamples: int | None = None
    ewma_lambda: float | None = None
    ewma_warmup: int | None = None
    sigma_current: float | None = None
    history_from: str | None = None
    history_to: str | None = None
    falsifier: dict[str, object] | None = None
    prior: Prior | None = None
    proxy: ProxyInfo | None = None

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["method"] = str(self.method)
        out["drift"] = str(self.drift)
        out["gate_band"] = list(self.gate_band)
        return {k: v for k, v in out.items() if v is not None}

    def to_json(self) -> str:
        """Render for the ``sim_params`` jsonb column."""
        return json.dumps(self.to_dict(), sort_keys=True)


def new_proxy_history(
    target: History,
    peer: History,
    vol_scale: float,
    reason: str,
) -> History:
    """Build a :class:`History` for a name with too little history of its own.

    This is an explicit constructor rather than a fallback inside :func:`run`
    because a silent proxy is worse than an error: the resulting
    ``null_probability`` looks identical to a real one in the ledger.
    Insufficient history is a rejection reason; using a proxy is a decision
    someone made, and the ledger records who and why.
    """
    if vol_scale <= 0:
        raise SimInputError("vol_scale must be > 0")
    if not reason.strip():
        raise SimInputError("a proxy needs a recorded reason")
    return History(
        symbol=target.symbol,
        symbol_id=target.symbol_id,
        log_returns=peer.log_returns * vol_scale,
        start=peer.start,
        end=peer.end,
        proxy=ProxyInfo(
            proxy_symbol=peer.symbol,
            proxy_symbol_id=peer.symbol_id,
            vol_scale=vol_scale,
            reason=reason,
        ),
    )


def sha256_returns(returns: np.ndarray) -> str:
    """Hash the exact input series.

    Lets a stored figure be shown to belong to the data it was computed from.
    """
    contiguous = np.ascontiguousarray(returns, dtype="<f8")
    return hashlib.sha256(contiguous.tobytes()).hexdigest()
