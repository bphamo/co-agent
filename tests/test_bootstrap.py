"""Tests for the stationary bootstrap and the EWMA volatility estimate."""

from __future__ import annotations

import numpy as np
import pytest

from co_agent.sim.bootstrap import (
    bootstrap_draws,
    build_pool,
    draws_to_levels,
    ewma_sigma,
)
from co_agent.sim.falsifier import max_drawdown
from co_agent.sim.params import InsufficientHistoryError, SimInputError


def lag1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation."""
    d = x - x.mean()
    denom = float((d * d).sum())
    if denom == 0:
        return 0.0
    return float((d[1:] * d[:-1]).sum() / denom)


def test_blocks_preserve_autocorrelation():
    """The whole reason for blocks rather than IID draws.

    At mean block length 1 the stationary bootstrap degenerates to IID and the
    dependence is destroyed -- the behaviour IID resampling and GBM share, and
    that FR9 rejects.
    """
    rng = np.random.default_rng(1)
    phi = 0.6
    src = np.zeros(4000)
    for i in range(1, src.size):
        src[i] = phi * src[i - 1] + rng.normal(0, 0.01)
    assert lag1(src) > 0.4, "fixture is not autocorrelated enough"

    def measure(mean_block_len: float) -> float:
        draws = bootstrap_draws(
            np.random.default_rng(9), src, 1.0, 1 / mean_block_len, 20_000, 1
        )
        return lag1(draws[0])

    assert measure(10) > 0.4, "blocked resample lost the dependence"
    assert abs(measure(1)) < 0.05, "mean block length 1 should be IID"


def test_bootstrap_covers_the_pool_uniformly():
    """Fixed-length blocks under-represent observations near block boundaries."""
    pool = np.arange(50, dtype=np.float64)
    draws = bootstrap_draws(np.random.default_rng(3), pool, 1.0, 0.1, 500_000, 1)
    counts = np.bincount(draws[0].astype(int), minlength=pool.size)
    want = draws.size / pool.size
    assert np.all(np.abs(counts - want) / want < 0.05)


def test_draws_to_levels_starts_at_one_and_compounds():
    draws = np.array([[0.1, -0.1], [0.0, 0.0]])
    levels = draws_to_levels(draws)
    assert levels.shape == (2, 3)
    assert np.all(levels[:, 0] == 1.0)
    assert levels[0, 1] == pytest.approx(np.exp(0.1))
    assert levels[0, 2] == pytest.approx(1.0)
    assert np.all(levels[1] == 1.0)


def test_ewma_has_no_look_ahead():
    """Standardising by an estimate that already saw the return would leak."""
    r = np.random.default_rng(21).normal(0, 0.02, 400)
    sigma_a, _ = ewma_sigma(r, 0.94, 60)

    perturbed = r.copy()
    perturbed[-1] = 0.35  # a huge final move
    sigma_b, _ = ewma_sigma(perturbed, 0.94, 60)

    both_nan = np.isnan(sigma_a) & np.isnan(sigma_b)
    assert np.all(both_nan | (sigma_a == sigma_b)), (
        "sigma changed after altering only the last return: it is looking ahead"
    )


def test_ewma_sigma_tracks_volatility():
    quiet = np.random.default_rng(31).normal(0, 0.005, 600)
    loud = np.random.default_rng(32).normal(0, 0.04, 600)
    _, next_quiet = ewma_sigma(quiet, 0.94, 60)
    _, next_loud = ewma_sigma(loud, 0.94, 60)
    assert next_loud > 3 * next_quiet


def test_ewma_sigma_returns_nan_without_enough_history():
    sigma, nxt = ewma_sigma(np.zeros(10), 0.94, 60)
    assert np.isnan(nxt)
    assert np.all(np.isnan(sigma))


def test_build_pool_recentres_only_when_given_a_target():
    r = np.random.default_rng(5).normal(0.01, 0.02, 1600)
    zero = build_pool(
        r, cond_vol=False, drift_target=0.0, ewma_lambda=0.94, ewma_warmup=60, min_history=750
    )
    historical = build_pool(
        r, cond_vol=False, drift_target=None, ewma_lambda=0.94, ewma_warmup=60, min_history=750
    )
    shifted = build_pool(
        r, cond_vol=False, drift_target=0.001, ewma_lambda=0.94, ewma_warmup=60, min_history=750
    )
    assert zero.returns.mean() == pytest.approx(0.0, abs=1e-15)
    assert historical.returns.mean() == pytest.approx(r.mean())
    assert shifted.returns.mean() == pytest.approx(0.001)


def test_build_pool_rejects_history_that_cannot_be_standardised():
    with pytest.raises(SimInputError):
        build_pool(
            np.zeros(30),
            cond_vol=True,
            drift_target=0.0,
            ewma_lambda=0.94,
            ewma_warmup=60,
            min_history=10,
        )
    with pytest.raises(InsufficientHistoryError):
        build_pool(
            np.random.default_rng(1).normal(0, 0.02, 200),
            cond_vol=True,
            drift_target=0.0,
            ewma_lambda=0.94,
            ewma_warmup=60,
            min_history=750,
        )


def test_max_drawdown():
    levels = np.array(
        [
            [1.0, 1.1, 1.2],
            [1.0, 0.9, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 0.5, 2.0, 1.0],
        ][:3]
        + [[1.0, 0.5, 2.0]]
    )
    got = max_drawdown(levels)
    assert got == pytest.approx([0.0, 0.1, 0.5, 0.5])


def test_recentring_survives_volatility_conditioning():
    """The target is in return space, but the pool holds standardised returns.

    Recentring without dividing by the scale it will later be multiplied by
    would put the drift in the wrong units -- silently, and by a factor of the
    current volatility.
    """
    r = np.random.default_rng(9).normal(0.0, 0.02, 1600)
    pool = build_pool(
        r, cond_vol=True, drift_target=0.0005, ewma_lambda=0.94, ewma_warmup=60,
        min_history=750,
    )
    drawn_mean = pool.returns.mean() * pool.scale
    assert drawn_mean == pytest.approx(0.0005, rel=1e-6)
