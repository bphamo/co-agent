"""Tests for the FR9 simulation service.

Mostly property tests: there is no closed form to check a block bootstrap
against, so the suite pins the properties the method is chosen for, plus one
analytic anchor.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date

import numpy as np
import pytest

import co_agent.sim as sim
from co_agent.sim.simulate import Interval


def synth_returns(n: int, vol: float, seed: int) -> np.ndarray:
    """A deterministic daily log-return series."""
    return np.random.default_rng(seed).normal(0.0, vol, n)


def history(n: int = 1600, vol: float = 0.02, seed: int = 42) -> sim.History:
    return sim.History(
        symbol="TEST.TO",
        symbol_id=1,
        log_returns=synth_returns(n, vol, seed),
        start=date(2020, 1, 2),
        end=date(2026, 1, 2),
    )


def base_request(falsifier: sim.Falsifier | None = None, **kwargs) -> sim.Request:
    params = dict(
        history=history(),
        horizon_days=60,
        falsifier=falsifier,
        proposed_weight=0.05,
        per_position_drawdown_limit=0.02,
        # MC unless a test is about interval width: the double bootstrap runs
        # `outer_resamples` extra simulations, which is right in production and
        # 25x too slow for a unit test.
        config=sim.Config(interval=Interval.MC),
    )
    params.update(kwargs)
    return sim.Request(**params)


def test_run_is_deterministic():
    req = base_request(sim.TouchBelow(0.12))
    a = sim.run(req)
    b = sim.run(req)
    assert a.null_probability == b.null_probability
    assert a.p95_drawdown_unit == b.p95_drawdown_unit
    assert a.params.history_sha256
    assert a.params.version == sim.VERSION

    # A different seed must move the figure, or the seed is not being used.
    c = sim.run(base_request(sim.TouchBelow(0.12), config=sim.Config(seed=99, interval=Interval.MC)))
    assert c.null_probability != a.null_probability


def test_demeaned_pool_ends_above_half_the_time():
    """The one analytic anchor available without a closed form for the rest."""
    res = sim.run(base_request(sim.TerminalAbove(0.0), config=sim.Config(paths=20_000, interval=Interval.MC)))
    assert abs(res.null_probability - 0.5) < 0.03


def test_null_probability_falls_as_falsifier_gets_harder():
    previous = 1.1
    for drop in (0.05, 0.10, 0.15, 0.25, 0.40):
        res = sim.run(base_request(sim.TouchBelow(drop)))
        assert res.null_probability < previous, f"drop={drop}"
        previous = res.null_probability
    assert previous < 0.05, "a 40% drop in 60 days should be rare"


def test_touch_is_easier_than_terminal():
    """'Trades below' and 'closes below' are different falsifiers."""
    touch = sim.run(base_request(sim.TouchBelow(0.12)))
    terminal = sim.run(base_request(sim.TerminalBelow(0.12)))
    assert touch.null_probability > terminal.null_probability


def test_insufficient_history_is_an_error():
    req = base_request(sim.TouchBelow(0.12), history=history(n=200))
    with pytest.raises(sim.InsufficientHistoryError) as excinfo:
        sim.run(req)
    assert excinfo.value.have == 200
    assert excinfo.value.want == sim.Config().min_history


def test_conditioning_needs_history_beyond_the_warmup():
    """Standardised observations, not raw ones, must clear min_history."""
    req = base_request(
        sim.TouchBelow(0.12),
        history=history(n=800),
        config=sim.Config(min_history=760, ewma_warmup=60, cond_vol=True),
    )
    with pytest.raises(sim.InsufficientHistoryError) as excinfo:
        sim.run(req)
    assert excinfo.value.standardised
    assert excinfo.value.have == 740  # 800 observations less the 60-day warmup


def test_proxy_is_explicit_and_recorded():
    """A proxy is a decision someone recorded, never a silent fallback."""
    target = sim.History("NEWCO.V", 2, synth_returns(90, 0.02, 3))
    peer = sim.History("PEER.TO", 3, synth_returns(1600, 0.015, 11))

    with pytest.raises(sim.SimInputError):
        sim.new_proxy_history(target, peer, 1.4, "")
    with pytest.raises(sim.SimInputError):
        sim.new_proxy_history(target, peer, 0, "thin history")

    proxied = sim.new_proxy_history(target, peer, 1.4, "IPO 2026-03, 90 obs")
    res = sim.run(base_request(sim.TouchBelow(0.12), history=proxied))
    assert res.method is sim.Method.PROXY_BOOTSTRAP
    assert res.params.proxy is not None
    assert res.params.proxy.proxy_symbol == "PEER.TO"
    assert res.params.proxy.vol_scale == 1.4


def test_event_class_uses_prior_but_is_still_sized():
    """A base rate the bootstrap cannot compute, gated the same way."""
    req = base_request(
        prior=sim.Prior(p=0.48, source="8 of 17 comparable quarters, 2019-2025", n=17)
    )
    res = sim.run(req)
    assert res.method is sim.Method.EXPLICIT_PRIOR
    assert res.null_probability == 0.48
    assert res.p95_drawdown_unit > 0, "event theses still hold a position"
    # 17 observations is a wide interval, so the gate cannot settle it.
    assert res.verdict is sim.Verdict.INDETERMINATE

    res = sim.run(base_request(prior=sim.Prior(p=0.92, source="22 of 24 quarters", n=24)))
    assert res.verdict is sim.Verdict.REJECT_TOO_EASY


def test_judgemental_prior_is_allowed_but_flagged():
    res = sim.run(base_request(prior=sim.Prior(p=0.5, source="analyst judgement", n=0)))
    assert res.verdict is sim.Verdict.ACCEPT
    assert res.ci_low == res.ci_high == 0.5
    assert res.params.to_dict()["prior"]["n"] == 0


def test_prior_needs_a_source():
    with pytest.raises(sim.SimInputError):
        sim.run(base_request(prior=sim.Prior(p=0.5, source="")))


def test_exactly_one_falsifier_class():
    with pytest.raises(sim.SimInputError):
        sim.run(base_request(sim.TouchBelow(0.12), prior=sim.Prior(p=0.5, source="x")))
    with pytest.raises(sim.SimInputError):
        sim.run(base_request(None))


def test_falsifier_inputs_are_validated():
    for bad in (sim.TouchBelow(0.0), sim.TouchBelow(1.0), sim.DrawdownExceeds(1.5)):
        with pytest.raises(sim.SimInputError):
            sim.run(base_request(bad))
    with pytest.raises(sim.SimInputError):
        sim.run(base_request(sim.TouchAbove(-0.1)))


def test_horizon_must_be_positive():
    with pytest.raises(sim.SimInputError):
        sim.run(base_request(sim.TouchBelow(0.12), horizon_days=0))


def test_indeterminate_escalates_the_path_count():
    """A boundary falsifier buys precision before anyone gives up on it."""
    first = sim.run(base_request(sim.TouchBelow(0.12)))
    assert first.escalations == 0

    # Put a band edge on the measured value: the interval straddles it by
    # construction at this path count.
    req = base_request(
        sim.TouchBelow(0.12),
        band=sim.Band(low=first.null_probability, high=first.null_probability + 0.25),
        config=sim.Config(max_paths=160_000, interval=Interval.MC),
    )
    res = sim.run(req)
    assert res.escalations > 0
    assert res.paths > first.paths


def test_boundary_holds_indeterminate_rather_than_guessing():
    first = sim.run(base_request(sim.TouchBelow(0.12)))
    p = first.null_probability

    # A band narrower than the Monte Carlo interval, centred on the measured
    # value, with no room to escalate: the simulator cannot tell which side of
    # the gate this falls on, and says so instead of picking one.
    req = base_request(
        sim.TouchBelow(0.12),
        band=sim.Band(low=p - 0.0005, high=p + 0.0005),
        config=sim.Config(paths=10_000, max_paths=10_000, interval=Interval.MC),
    )
    res = sim.run(req)
    assert res.verdict is sim.Verdict.INDETERMINATE
    assert res.escalations == 0
    assert res.paths == 10_000
    # The point of the hysteresis: a second run does not flip the answer.
    assert sim.run(req).verdict is res.verdict


def test_conditioning_on_current_vol_moves_the_null():
    """Otherwise the gate keys off an unconditional average."""
    calm_last = np.concatenate([synth_returns(800, 0.035, 5), synth_returns(800, 0.008, 6)])
    storm_last = np.concatenate([synth_returns(800, 0.008, 6), synth_returns(800, 0.035, 5)])

    def null(returns: np.ndarray, cond_vol: bool) -> float:
        req = base_request(
            sim.TouchBelow(0.15),
            history=sim.History("T", 1, returns),
            config=sim.Config(cond_vol=cond_vol, interval=Interval.MC),
        )
        return sim.run(req).null_probability

    calm = null(calm_last, True)
    storm = null(storm_last, True)
    assert storm > calm + 0.10, f"conditioning barely moved: {calm=} {storm=}"

    # Unconditionally the two are near-identical, which is the failure mode the
    # conditioning exists to avoid.
    assert abs(null(calm_last, False) - null(storm_last, False)) < 0.10


def test_batching_handles_a_path_count_above_one_batch():
    """A 40,000-path escalation spans batches; the estimate must stay sane."""
    small = sim.run(base_request(sim.TouchBelow(0.12)))
    large = sim.run(base_request(sim.TouchBelow(0.12), config=sim.Config(paths=40_000, interval=Interval.MC)))
    assert large.paths == 40_000
    assert abs(large.null_probability - small.null_probability) < 0.03
    # Same (seed, paths) is reproducible even across batch boundaries.
    again = sim.run(base_request(sim.TouchBelow(0.12), config=sim.Config(paths=40_000, interval=Interval.MC)))
    assert again.null_probability == large.null_probability


def test_weight_is_reduced_when_drawdown_breaches_the_limit():
    res = sim.run(
        base_request(
            sim.TouchBelow(0.12), proposed_weight=0.5, per_position_drawdown_limit=0.02
        )
    )
    assert res.sizing.reduced
    assert res.sizing.weight < 0.5
    assert res.sizing.p95_drawdown_at_weight == pytest.approx(0.02)


def test_params_serialise_for_sim_params_column():
    res = sim.run(base_request(sim.TouchBelow(0.12)))
    back = json.loads(res.params.to_json())
    for key in (
        "method",
        "version",
        "paths",
        "mean_block_len",
        "drift",
        "cond_vol",
        "horizon_days",
        "history_obs",
        "history_sha256",
        "seed",
        "gate_band",
        "falsifier",
    ):
        assert key in back, f"sim_params is missing {key!r}"
    assert back["method"] == sim.Method.STATIONARY_BOOTSTRAP
    assert back["falsifier"] == {"kind": "touch_below", "drop": 0.12}
    assert back["history_from"] == "2020-01-02"


def test_result_carries_no_synthetic_paths():
    """FR9 bars simulated series from an agent's context."""
    res = sim.run(base_request(sim.TouchBelow(0.12)))
    for f in dataclasses.fields(res):
        assert not isinstance(getattr(res, f.name), np.ndarray)


# --------------------------------------------------------- interval and defaults


def test_cond_vol_is_off_by_default():
    """Measured on the synthetic panel: conditioning raises error at 60 days."""
    assert sim.Config().cond_vol is False


def test_double_bootstrap_is_the_default_interval():
    assert sim.Config().interval is Interval.DOUBLE_BOOTSTRAP


def test_double_bootstrap_is_much_wider_than_monte_carlo_error():
    """The two intervals answer different questions, and the gap is the point.

    Monte Carlo error says how much the estimate would move on a different seed.
    The double bootstrap says how much it would move on a different sample of the
    same process -- which is 3-7x larger, and is what the gate should be judging.
    """
    req = base_request(sim.TouchBelow(0.12), config=sim.Config(outer_resamples=12))
    res = sim.run(req)
    assert res.interval is Interval.DOUBLE_BOOTSTRAP
    mc_width = res.mc_ci_high - res.mc_ci_low
    width = res.ci_high - res.ci_low
    assert width > 2 * mc_width, f"{width=} {mc_width=}"
    # The Monte Carlo interval is still reported, so "too few paths" stays
    # distinguishable from "this history cannot pin the number down".
    assert 0 < mc_width < 0.05


def test_monte_carlo_interval_is_reported_under_both_methods():
    for interval in (Interval.MC, Interval.DOUBLE_BOOTSTRAP):
        res = sim.run(
            base_request(
                sim.TouchBelow(0.12),
                config=sim.Config(interval=interval, outer_resamples=8),
            )
        )
        assert res.mc_ci_low < res.null_probability < res.mc_ci_high


def test_escalation_only_applies_to_the_monte_carlo_interval():
    """More paths narrow Monte Carlo error and nothing else.

    A double-bootstrap width is a property of the history, so escalating the
    path count cannot resolve an indeterminate verdict -- and a loop that tried
    would burn four times the compute to return the same answer.
    """
    first = sim.run(base_request(sim.TouchBelow(0.12)))
    boundary = sim.Band(low=first.null_probability, high=first.null_probability + 0.25)

    mc = sim.run(
        base_request(
            sim.TouchBelow(0.12),
            band=boundary,
            config=sim.Config(interval=Interval.MC, max_paths=160_000),
        )
    )
    assert mc.escalations > 0

    db = sim.run(
        base_request(
            sim.TouchBelow(0.12),
            band=boundary,
            config=sim.Config(
                interval=Interval.DOUBLE_BOOTSTRAP, outer_resamples=8, max_paths=160_000
            ),
        )
    )
    assert db.escalations == 0
    assert db.paths == sim.Config().paths


def test_interval_method_is_recorded_for_reproducibility():
    """A stored verdict depends on which interval produced it."""
    res = sim.run(
        base_request(sim.TouchBelow(0.12), config=sim.Config(outer_resamples=8))
    )
    params = res.params.to_dict()
    assert params["interval_method"] == "double_bootstrap"
    assert params["outer_resamples"] == 8
    assert params["version"] == "sim/2"

    mc = sim.run(base_request(sim.TouchBelow(0.12)))
    assert mc.params.to_dict()["interval_method"] == "mc"
    assert "outer_resamples" not in mc.params.to_dict()


def test_outer_resamples_is_validated():
    with pytest.raises(sim.SimInputError):
        sim.run(base_request(sim.TouchBelow(0.12), config=sim.Config(outer_resamples=1)))
    with pytest.raises(sim.SimInputError):
        sim.run(base_request(sim.TouchBelow(0.12), config=sim.Config(inner_paths=0)))
