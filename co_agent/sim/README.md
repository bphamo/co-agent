# `co_agent.sim` — the FR9 simulation service

Computes `null_probability` and `p95_drawdown` for a candidate thesis, and
applies the two gates FR9 puts in code: falsifiers that trip by chance too
often or too rarely are returned for restatement, and proposed weights are cut
until the 95th-percentile drawdown fits the per-position limit.

Nothing here calls a model. `run` is a pure function of `(history, falsifier,
config)`, so a stored figure can be recomputed and checked against the code
that produced it.

```python
import co_agent.sim as sim

res = sim.run(sim.Request(
    history=sim.History("FAKE.TO", 9876, log_returns),  # daily, oldest first
    horizon_days=60,
    falsifier=sim.TouchBelow(drop=0.12),                # "trades 12% below entry"
    proposed_weight=0.05,
    per_position_drawdown_limit=0.02,
))

res.null_probability       # -> theses.null_probability
res.p95_drawdown_unit      # -> theses.p95_drawdown
res.sizing.weight          # -> theses.proposed_weight (reduced if it breached the limit)
res.verdict                # accept | reject_too_easy | reject_too_hard | indeterminate
res.params.to_json()       # -> theses.sim_params
```

Falsifiers: `TouchBelow`, `TouchAbove`, `TerminalBelow`, `TerminalAbove`,
`DrawdownExceeds`. Pass a `Prior` instead for the event class.

## Where this departs from the TRD, and why

**Geometric block lengths instead of fixed 5–20 day blocks.** Fixed-length
blocks preserve dependence, but the resampled series is not stationary:
observations near a block boundary are systematically under-represented, and the
artefact lands in the tail the simulation exists to measure. Drawing lengths from
a geometric distribution (the stationary bootstrap, Politis & Romano 1994) is
strictly stationary at the same cost, and collapses the 5–20 range into one
recorded parameter, `mean_block_len`. `test_blocks_preserve_autocorrelation`
pins the property that motivates blocks in the first place — and shows that at
`mean_block_len = 1` the dependence is gone, which is the IID/GBM behaviour FR9
rejects.

**The null is conditioned on the current volatility regime.** Returns are
standardised by their own one-step-ahead EWMA volatility and re-inflated at the
latest forecast. An unconditional null for "down 15% in 60 days" is wrong in both
directions depending on where volatility sits today, and it is the figure the
0.3–0.7 gate keys off. `test_conditioning_on_current_vol_moves_the_null` holds
two histories with identical pooled observations in opposite order and shows the
conditional estimates differ by more than 10 points while the unconditional ones
barely move. The re-inflation holds volatility flat across the horizon — a
mean-reverting vol path would be more faithful, and `cond_vol` in `sim_params` is
what lets these figures be invalidated rather than quietly replaced.

**The gate has a third answer.** FR9 says a `null_probability` outside 0.3–0.7
returns the falsifier for restatement. Taken literally, a falsifier whose true
null probability sits at 0.30 is accepted or bounced on Monte Carlo noise, and
the researcher reads a coin flip as signal. `run` therefore judges the Wilson
interval, not the point estimate: a verdict is only returned when the interval
settles the question. When it straddles a band edge, `run` buys precision
(escalating the path count up to `max_paths`) and, if that still does not settle
it, returns `indeterminate`. **Callers must not treat `indeterminate` as a
rejection** — the value is genuinely near the boundary and the call belongs to a
human.

**Event falsifiers are a first-class class.** A bootstrap over returns cannot
price "guides below $X in Q3" or "the 10-Q shows inventory up >20%". Requiring
every thesis to carry a price-path falsifier just so the gate has something to
compute would quietly narrow the research to pure price bets. So `Request` takes
either a `falsifier` (price path) or a `prior` (a recorded base rate, with its
observation count). Both go through the same gate; both are still sized by the
bootstrap, because an event thesis still holds a position. A prior with `n = 0`
is permitted and recorded as such — a gate verdict resting on judgement is a
weaker statement than one resting on a counted base rate, and the ledger should
be able to tell them apart later.

**Insufficient history is a rejection, never a fallback.** Below `min_history`
(default 750 observations, ≈3 years) `run` raises `InsufficientHistoryError`.
Fewer than that leaves too few independent blocks for the tail to mean anything.
A peer proxy is available but only through `new_proxy_history`, which demands a
volatility scale and a written reason and stamps `sim_method =
stationary_bootstrap_proxy`. A silent proxy is worse than an error, because the
resulting figure is indistinguishable from a real one in the ledger.
`min_history` is a defensible starting answer to open question 6, not a measured
one.

**Zero drift by default.** `Drift.ZERO` demeans the pool, so "by chance" means a
forecaster with no information about direction. `Drift.HISTORICAL` keeps the
symbol's realised drift, which makes the null absorb past momentum and generally
makes upside falsifiers look easier than they are. The choice moves
`null_probability` materially, so it is recorded either way.

## Implementation notes

**numpy, and paths are simulated in batches.** The bootstrap walk vectorises
across paths: all paths advance one day together, so a 60-day horizon is 60
vector operations rather than 600,000 scalar ones. Paths are drawn in batches of
`_BATCH_PATHS` (20,000) to bound peak memory — a full 160,000-path escalation
held as one array is about 78 MB.

`_BATCH_PATHS` is a module constant, not a `Config` field, and that is
deliberate: changing it changes which draws a given seed produces, so exposing it
would put a second, undeclared input into every recorded figure. **Reproducibility
is a property of `(seed, paths)` alone.** Derive `seed` from the thesis id so two
theses in one batch do not share a path set.

**`p95_drawdown` uses the nearest-rank percentile** (`numpy`'s `inverted_cdf`
method): the smallest observed drawdown that at least 95% of paths do not exceed.
Pinned by a test, because a stored figure is only comparable to another one
computed the same way.

## What this does not do

`p95_drawdown` is a **per-position** figure, linear in weight. It does not bound
portfolio drawdown: six correlated names each inside the per-position limit can
breach it together. FR5's sector and currency caps are what stand between this
number and that case, and Appendix A excludes the correlation analysis that would
close the gap properly.

No public function returns a synthetic path, and `test_result_carries_no_synthetic_paths`
keeps it that way. FR9 bars simulated series from any agent's context, and the
cheapest way to honour that is for the simulator to have no way to emit one.
Stress scenarios for the red team should be enumerated labels, not numbers from
here.

## Tests

```
pytest tests/test_sim.py tests/test_bootstrap.py tests/test_gate.py
```

Mostly property tests, since there is no closed form to check against:
monotonicity in threshold, touch-vs-terminal ordering, uniform pool coverage,
autocorrelation preservation, EWMA look-ahead freedom, reproducibility under a
fixed seed, and one analytic anchor — a demeaned pool ends above its start half
the time.
