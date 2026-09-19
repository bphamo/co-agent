"""The stationary bootstrap (Politis & Romano, 1994) and EWMA volatility.

The TRD asks for contiguous blocks of 5-20 days to preserve volatility
clustering and autocorrelation, and explicitly rejects geometric Brownian motion
for understating tails.  Fixed-length blocks do preserve dependence, but the
resampled series is not stationary: observations near a block boundary are
systematically under-represented, and the artefact shows up in exactly the tail
the simulation exists to measure.  Drawing block lengths from a geometric
distribution instead makes the resampled series strictly stationary at the same
computational cost -- the block length becomes a single parameter (its mean)
rather than a range, so ``mean_block_len = 10`` covers the TRD's 5-20.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import InsufficientHistoryError, SimInputError


@dataclass(frozen=True, slots=True)
class Pool:
    """What the bootstrap draws from."""

    returns: np.ndarray
    #: Multiplier applied to each draw (1, or the current sigma forecast).
    scale: float
    #: The current sigma forecast when conditioning on volatility, else None.
    sigma_current: float | None


def bootstrap_draws(
    rng: np.random.Generator,
    pool: np.ndarray,
    scale: float,
    jump_prob: float,
    horizon: int,
    paths: int,
) -> np.ndarray:
    """Draw ``paths`` synthetic return series of length ``horizon``.

    Walks the pool forward from a random start and, at each step, jumps to a new
    random index with probability ``jump_prob``, wrapping at the end.  All paths
    advance together so the walk vectorises across the batch.
    """
    n = pool.size
    idx = rng.integers(0, n, size=paths)
    out = np.empty((paths, horizon), dtype=np.float64)
    for t in range(horizon):
        out[:, t] = pool[idx]
        jump = rng.random(paths) < jump_prob
        fresh = rng.integers(0, n, size=paths)
        idx = np.where(jump, fresh, (idx + 1) % n)
    if scale != 1.0:
        out *= scale
    return out


def draws_to_levels(draws: np.ndarray) -> np.ndarray:
    """Turn log-return draws into price paths starting at 1."""
    paths, horizon = draws.shape
    levels = np.empty((paths, horizon + 1), dtype=np.float64)
    levels[:, 0] = 1.0
    np.cumsum(draws, axis=1, out=levels[:, 1:])
    np.exp(levels[:, 1:], out=levels[:, 1:])
    return levels


def ewma_sigma(
    returns: np.ndarray,
    lam: float,
    warmup: int,
) -> tuple[np.ndarray, float]:
    """One-step-ahead volatility forecast for every index, plus the next step.

    ``sigma[t]`` is computed from returns strictly before ``t``, so standardising
    ``returns[t]`` by ``sigma[t]`` introduces no look-ahead.  Indices below
    ``warmup`` are NaN: the recursion is seeded from the first ``warmup``
    observations, and those observations therefore have no clean sigma of their
    own.
    """
    n = returns.size
    sigma = np.full(n, np.nan, dtype=np.float64)
    if n <= warmup or warmup < 2:
        return sigma, float("nan")

    seed = returns[:warmup]
    var = float(seed.var())
    if var <= 0:
        var = 1e-12
    sigma[warmup] = np.sqrt(var)

    for t in range(warmup, n - 1):
        var = lam * var + (1 - lam) * returns[t] * returns[t]
        sigma[t + 1] = np.sqrt(var)

    var = lam * var + (1 - lam) * returns[n - 1] * returns[n - 1]
    return sigma, float(np.sqrt(var))


def build_pool(
    returns: np.ndarray,
    *,
    cond_vol: bool,
    drift_target: float | None,
    ewma_lambda: float,
    ewma_warmup: int,
    min_history: int,
) -> Pool:
    """Prepare the resampling pool.

    ``drift_target`` recentres the pool to a given mean daily return; ``None``
    leaves the symbol's realised drift untouched.

    With ``cond_vol``, returns are standardised by their own one-step-ahead EWMA
    volatility and re-inflated by the current forecast, so the null is
    conditioned on today's regime rather than on the unconditional average of
    the whole history.  This matters: the unconditional probability of "down 15%
    in 60 days" is wrong in both directions depending on where volatility
    currently sits, and it is the figure the gate keys off.

    The re-inflation holds volatility flat across the horizon.  A vol path that
    mean-reverts (GARCH-style) would be more faithful; it is a deliberate
    simplification, recorded via ``cond_vol`` in ``sim_params`` so a later
    version can invalidate these figures rather than quietly replace them.
    """
    if cond_vol:
        sigma, sigma_next = ewma_sigma(returns, ewma_lambda, ewma_warmup)
        if not np.isfinite(sigma_next) or sigma_next <= 0:
            raise SimInputError(
                "volatility conditioning needs more history than the warmup window"
            )
        usable = np.arange(ewma_warmup, returns.size)
        s = sigma[usable]
        keep = np.isfinite(s) & (s > 0)
        standardised = returns[usable][keep] / s[keep]
        if standardised.size < min_history:
            raise InsufficientHistoryError(
                standardised.size, min_history, standardised=True
            )
        pool, scale, current = standardised, sigma_next, sigma_next
    else:
        pool, scale, current = returns.astype(np.float64, copy=True), 1.0, None

    if drift_target is not None:
        # Recentre so that a drawn value (pool entry x scale) has mean
        # `drift_target` per day. With volatility conditioning the pool holds
        # standardised returns, so the target is divided by the same scale it
        # will be multiplied by.
        pool = pool - pool.mean() + drift_target / scale

    return Pool(returns=pool, scale=scale, sigma_current=current)
