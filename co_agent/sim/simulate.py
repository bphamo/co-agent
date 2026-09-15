"""The FR9 entry point: null probability, gate verdict, and position sizing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy as np

from .bootstrap import bootstrap_draws, build_pool, draws_to_levels
from .falsifier import Falsifier, max_drawdown
from .gate import Band, Sizing, Verdict, cap_weight, wilson
from .params import (
    VERSION,
    Drift,
    History,
    InsufficientHistoryError,
    Method,
    Params,
    Prior,
    SimInputError,
    sha256_returns,
)


class Interval(StrEnum):
    """How the interval the gate judges is computed.

    ``MC`` is the Wilson interval on the path count: it describes Monte Carlo
    error alone. Measured against known truth it covers 4-44% of the time
    instead of 95%, because Monte Carlo error (~0.018 wide at 10,000 paths) is
    an order of magnitude smaller than the estimator's real error -- so a gate
    judging it is confident about the wrong quantity.

    ``DOUBLE_BOOTSTRAP`` resamples the history itself and takes the spread of
    the resulting estimates. It is 4-7x wider and reaches nominal coverage on
    IID data (0.96 measured). On state-dependent processes it still under-covers
    (0.40-0.64), because resampling a history in blocks loses the state that
    history ended in -- residual specification error no resampling scheme can
    recover. It is the honest interval available, not a correct one.

    Cost: one evaluation is ``outer_resamples`` extra runs at ``inner_paths``
    each, about 2-3 seconds. At FR4 volumes that is nothing; in a test suite it
    is worth passing ``MC``.
    """

    MC = "mc"
    DOUBLE_BOOTSTRAP = "double_bootstrap"


#: Paths simulated per batch.  Bounds peak memory only -- see :func:`_simulate`
#: for why this is not a config knob.
_BATCH_PATHS = 20_000


@dataclass(frozen=True, slots=True)
class Config:
    """Every knob.

    All of it is recorded in ``sim_params``; none of it is ever chosen by an
    agent (FR9).
    """

    #: Initial synthetic path count.  FR9 asks for >= 10,000.
    paths: int = 10_000
    #: Ceiling on escalation when a verdict is indeterminate.  Bounded so a
    #: boundary-hugging falsifier cannot turn into an unbounded compute loop
    #: (NFR2).
    max_paths: int = 160_000
    #: Expected geometric block length, in trading days.
    mean_block_len: float = 10.0
    #: Whether the null carries the symbol's realised drift.
    drift: Drift = Drift.ZERO
    #: Standardise by EWMA volatility and re-inflate at the current forecast,
    #: conditioning the null on today's regime.
    #:
    #: Defaults to False on measured evidence, reversing an earlier assumption.
    #: Conditioning holds today's volatility flat across the whole horizon, which
    #: is right while the horizon is short relative to volatility persistence and
    #: wrong once the process has time to mean-revert. Measured on the synthetic
    #: panel (see co_agent/sim/README.md), it lowers error at horizons of 5-20
    #: days and *raises* it at 60-120 -- on GARCH and on regime-switching data
    #: alike. The TRD's default horizon is 60 days, past the crossover, so the
    #: default here is off. Turn it on for short-horizon theses.
    cond_vol: bool = False
    ewma_lambda: float = 0.94
    ewma_warmup: int = 60
    #: Minimum usable observation count.  ~750 is three years of daily data;
    #: open question 6 asks for this number and this is a defensible starting
    #: answer, not a measured one.
    min_history: int = 750
    #: How the decision interval is computed. See Interval.
    interval: Interval = Interval.DOUBLE_BOOTSTRAP
    #: Resampled histories drawn for the double-bootstrap interval. Raising it
    #: sharpens the *estimate of* the interval, not the interval itself.
    outer_resamples: int = 24
    #: Paths per resampled history. The outer spread dominates, so this is
    #: deliberately smaller than `paths`.
    inner_paths: int = 2_000
    #: Makes a figure reproducible.  Derive it from the thesis id so that two
    #: theses in one batch do not share a path set.
    seed: int = 1

    def validated(self) -> "Config":
        if self.paths <= 0:
            raise SimInputError("paths must be > 0")
        if self.mean_block_len <= 0:
            raise SimInputError("mean_block_len must be > 0")
        if not 0 < self.ewma_lambda < 1:
            raise SimInputError("ewma_lambda must be in (0,1)")
        if self.ewma_warmup < 2:
            raise SimInputError("ewma_warmup must be >= 2")
        if self.min_history <= 0:
            raise SimInputError("min_history must be > 0")
        if self.outer_resamples < 2:
            raise SimInputError("outer_resamples must be >= 2")
        if self.inner_paths <= 0:
            raise SimInputError("inner_paths must be > 0")
        if self.max_paths < self.paths:
            return replace(self, max_paths=self.paths)
        return self


@dataclass(slots=True)
class Request:
    """One FR9 evaluation."""

    history: History
    horizon_days: int

    #: Exactly one of ``falsifier`` or ``prior`` must be set.
    #:
    #: ``falsifier`` is the price_path class: ``null_probability`` is the
    #: fraction of synthetic paths that trip it.
    #:
    #: ``prior`` is the event class.  A bootstrap over returns cannot price
    #: "guides below $X in Q3" or "the 10-Q shows inventory up >20%", and
    #: forcing every thesis into a price-path falsifier just so the gate has
    #: something to compute would narrow the research to pure price bets.  The
    #: event class takes its null probability from a recorded base rate instead,
    #: passes through the same gate, and is still sized by the bootstrap.
    falsifier: Falsifier | None = None
    prior: Prior | None = None

    proposed_weight: float = 0.0
    per_position_drawdown_limit: float = 0.0

    band: Band = field(default_factory=Band)
    config: Config = field(default_factory=Config)


@dataclass(slots=True)
class Result:
    """What the research worker writes to the thesis record.

    It carries no synthetic series, by design: FR9 bars simulated paths from any
    agent's context, and the cheapest way to honour that is for the simulator to
    have no way to emit one.
    """

    method: Method
    null_probability: float
    #: The interval the gate judged. Wide enough to be honest about estimator
    #: variance under DOUBLE_BOOTSTRAP; Monte Carlo error only under MC.
    ci_low: float
    ci_high: float
    #: Monte Carlo error alone, always reported. Useful for telling "I drew too
    #: few paths" apart from "this history cannot pin the number down", which are
    #: different problems with different fixes.
    mc_ci_low: float
    mc_ci_high: float
    interval: Interval
    verdict: Verdict
    paths: int
    escalations: int
    #: 95th-percentile peak-to-trough drawdown of a fully-weighted position over
    #: the horizon.
    p95_drawdown_unit: float
    sizing: Sizing
    params: Params


def run(request: Request) -> Result:
    """Evaluate a candidate: null probability, gate verdict, and sizing.

    A pure function of its inputs.  Two calls with the same request give the
    same result, which is what lets a stored figure be checked against the code
    that produced it.
    """
    cfg = request.config.validated()
    band = request.band
    band.validate()

    if request.horizon_days <= 0:
        raise SimInputError("horizon_days must be > 0")
    if (request.falsifier is None) == (request.prior is None):
        raise SimInputError(
            "set exactly one of falsifier (price_path) or prior (event)"
        )
    if request.falsifier is not None:
        request.falsifier.validate()
    if request.prior is not None:
        if not 0 <= request.prior.p <= 1:
            raise SimInputError("prior.p must be in [0,1]")
        if not request.prior.source.strip():
            raise SimInputError(
                "prior.source is required: an unsourced base rate is not evidence"
            )
        if request.prior.n < 0:
            raise SimInputError("prior.n must be >= 0")
    if request.proposed_weight < 0:
        raise SimInputError("proposed_weight must be >= 0")

    returns = request.history.log_returns
    if returns.size < cfg.min_history:
        raise InsufficientHistoryError(returns.size, cfg.min_history)

    pool = build_pool(
        returns,
        cond_vol=cfg.cond_vol,
        drift_zero=cfg.drift is Drift.ZERO,
        ewma_lambda=cfg.ewma_lambda,
        ewma_warmup=cfg.ewma_warmup,
        min_history=cfg.min_history,
    )

    method = Method.STATIONARY_BOOTSTRAP
    if request.history.proxy is not None:
        method = Method.PROXY_BOOTSTRAP
    if request.prior is not None:
        method = Method.EXPLICIT_PRIOR

    paths = cfg.paths
    escalations = 0
    while True:
        trips, drawdowns = _simulate(pool, request.falsifier, request.horizon_days, paths, cfg)

        if request.prior is not None:
            # Sizing comes from the bootstrap; the null probability does not.
            p = request.prior.p
            if request.prior.n > 0:
                ci_low, ci_high = wilson(round(p * request.prior.n), request.prior.n)
            else:
                # A judgemental prior has no sampling interval.  Recording the
                # point estimate as its own interval keeps the gate usable while
                # leaving prior.n = 0 in sim_params as the flag that this
                # verdict rests on judgement.
                ci_low = ci_high = p
            mc_low, mc_high = ci_low, ci_high
            verdict = band.judge(p, ci_low, ci_high)
            break

        p = trips / paths
        mc_low, mc_high = wilson(trips, paths)

        if cfg.interval is Interval.DOUBLE_BOOTSTRAP:
            ci_low, ci_high = _double_bootstrap_interval(
                returns, request.falsifier, request.horizon_days, cfg
            )
        else:
            ci_low, ci_high = mc_low, mc_high

        verdict = band.judge(p, ci_low, ci_high)

        # Escalating the path count narrows Monte Carlo error, which is not what
        # a double-bootstrap interval is made of: its width is a property of the
        # history, so more paths cannot resolve an indeterminate verdict. Under
        # MC the escalation is meaningful, so it is kept there and only there.
        if (
            cfg.interval is not Interval.MC
            or verdict is not Verdict.INDETERMINATE
            or paths >= cfg.max_paths
        ):
            break
        paths = min(paths * 4, cfg.max_paths)
        escalations += 1

    # Nearest-rank percentile: the 95th percentile is the smallest observed
    # drawdown that at least 95% of paths do not exceed.
    p95 = float(np.quantile(drawdowns, 0.95, method="inverted_cdf"))
    sizing = cap_weight(p95, request.proposed_weight, request.per_position_drawdown_limit)

    params = Params(
        method=method,
        version=VERSION,
        paths=paths,
        mean_block_len=cfg.mean_block_len,
        drift=cfg.drift,
        cond_vol=cfg.cond_vol,
        horizon_days=request.horizon_days,
        history_obs=int(returns.size),
        history_sha256=sha256_returns(returns),
        seed=cfg.seed,
        gate_band=(band.low, band.high),
        interval_method=str(cfg.interval),
        outer_resamples=(
            cfg.outer_resamples if cfg.interval is Interval.DOUBLE_BOOTSTRAP else None
        ),
        ewma_lambda=cfg.ewma_lambda if cfg.cond_vol else None,
        ewma_warmup=cfg.ewma_warmup if cfg.cond_vol else None,
        sigma_current=pool.sigma_current,
        history_from=request.history.start.isoformat() if request.history.start else None,
        history_to=request.history.end.isoformat() if request.history.end else None,
        falsifier=request.falsifier.spec() if request.falsifier is not None else None,
        prior=request.prior,
        proxy=request.history.proxy,
    )

    return Result(
        method=method,
        null_probability=p,
        ci_low=ci_low,
        ci_high=ci_high,
        mc_ci_low=mc_low,
        mc_ci_high=mc_high,
        interval=cfg.interval,
        verdict=verdict,
        paths=paths,
        escalations=escalations,
        p95_drawdown_unit=p95,
        sizing=sizing,
        params=params,
    )


def _simulate(
    pool,
    falsifier: Falsifier | None,
    horizon: int,
    paths: int,
    cfg: Config,
) -> tuple[int, np.ndarray]:
    """Draw ``paths`` synthetic paths in batches.

    Returns how many trip the falsifier (zero when there is none) and each
    path's maximum drawdown.  Batching bounds peak memory: a full 160,000-path
    escalation held as one array would be ~78 MB.

    ``_BATCH_PATHS`` is a module constant rather than a config knob on purpose.
    Changing it changes which draws a given seed produces, so exposing it would
    put a second, undeclared input into every recorded figure. Reproducibility
    is a property of ``(seed, paths)`` alone.
    """
    rng = np.random.default_rng(cfg.seed)
    jump_prob = min(1.0, 1.0 / cfg.mean_block_len)
    drawdowns = np.empty(paths, dtype=np.float64)
    trips = 0
    done = 0
    while done < paths:
        n = min(_BATCH_PATHS, paths - done)
        levels = draws_to_levels(
            bootstrap_draws(rng, pool.returns, pool.scale, jump_prob, horizon, n)
        )
        if falsifier is not None:
            trips += int(falsifier.trips(levels).sum())
        drawdowns[done : done + n] = max_drawdown(levels)
        done += n
    return trips, drawdowns


def _double_bootstrap_interval(
    returns: np.ndarray,
    falsifier: Falsifier,
    horizon: int,
    cfg: Config,
) -> tuple[float, float]:
    """Spread of the estimate across resampled histories.

    The outer resample is itself a stationary bootstrap of the raw return series,
    so dependence survives into each replicate; the pool is then rebuilt from
    scratch per replicate, which means any volatility conditioning is re-estimated
    too and its sampling error is included rather than assumed away.

    What this captures: "how far would this number move had I seen a different
    sample of the same process". What it cannot capture: the state the real
    history ended in, which block resampling scrambles -- so on state-dependent
    data the interval is still too narrow. Measured coverage is 0.96 on IID data
    and 0.40-0.64 on GARCH and regime-switching data.
    """
    rng = np.random.default_rng([cfg.seed, 0xD00D])
    jump_prob = min(1.0, 1.0 / cfg.mean_block_len)
    estimates = np.empty(cfg.outer_resamples)

    for j in range(cfg.outer_resamples):
        replicate = bootstrap_draws(
            rng, returns, 1.0, jump_prob, returns.size, 1
        )[0]
        try:
            pool = build_pool(
                replicate,
                cond_vol=cfg.cond_vol,
                drift_zero=cfg.drift is Drift.ZERO,
                ewma_lambda=cfg.ewma_lambda,
                ewma_warmup=cfg.ewma_warmup,
                min_history=cfg.min_history,
            )
        except (SimInputError, InsufficientHistoryError):
            estimates[j] = np.nan
            continue
        inner = replace(cfg, paths=cfg.inner_paths, seed=cfg.seed + j)
        trips, _ = _simulate(pool, falsifier, horizon, cfg.inner_paths, inner)
        estimates[j] = trips / cfg.inner_paths

    usable = estimates[np.isfinite(estimates)]
    if usable.size < 2:
        # Not enough replicates to say anything about spread. Returning the
        # widest possible interval makes the verdict indeterminate rather than
        # confidently wrong.
        return 0.0, 1.0
    return float(np.quantile(usable, 0.025)), float(np.quantile(usable, 0.975))
