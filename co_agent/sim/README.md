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
    falsifier=sim.TouchBelow(drop=0.12),                # closes 12% down on any day
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

**All of them are evaluated on daily closes**, because synthetic paths are
close-to-close. `TouchBelow` means "closes 12% down on at least one day", not
"trades 12% down" — despite the name, which is worth changing. A live thesis
resolved against intraday lows would trip more often than this null predicts, so
the falsifier wording and the resolver have to share one definition or
`null_probability` is biased low and nothing downstream notices. Deciding to
resolve on intraday lows means feeding the simulator OHLC bars and modelling the
daily range; resolving on closes needs only a close series.

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

**Volatility conditioning is off by default, on measured evidence.** An earlier
version of this package conditioned the null on the current volatility regime by
default, on the argument that an unconditional estimate of "down 15% in 60 days"
is wrong in both directions depending on where volatility sits today. The
argument is right; the implementation trades one error for a larger one, because
re-inflating at the latest forecast holds volatility *flat across the whole
horizon*. Measured against known truth on the synthetic panel (mean absolute
error, lower is better):

| process | h=5 | h=10 | h=20 | h=60 | h=120 |
|---|---|---|---|---|---|
| GARCH(1,1), conditioning off | .068 | .070 | .070 | **.059** | **.048** |
| GARCH(1,1), conditioning on | **.025** | **.034** | **.041** | .062 | .080 |
| regime-switching, off | .168 | .164 | .140 | **.081** | **.056** |
| regime-switching, on | **.115** | **.110** | **.113** | .148 | .174 |

The crossover sits between 20 and 60 days on both processes, near the
volatility half-life (~34 days for this GARCH parameterisation). The TRD's
default horizon is 60 days, past the crossover, so `cond_vol` defaults to
`False`. Turn it on for short-horizon theses; `sim_params` records which was
used either way. Reproduce with
`python -m co_agent.sim.validate --horizon 20 --cond-vol`.

**The gate judges an interval, and the interval is not Monte Carlo error.** FR9
says a `null_probability` outside 0.3–0.7 returns the falsifier for restatement.
Taken literally, a falsifier whose true null probability sits at 0.30 is accepted
or bounced on simulation noise, so `run` judges an interval rather than the point
estimate and returns `indeterminate` when the interval straddles a band edge.
**Callers must not treat `indeterminate` as a rejection** — the value is near the
boundary and the call belongs to a human.

Which interval turns out to matter more than the hysteresis. The Wilson interval
on the path count describes Monte Carlo error alone, and measured against known
truth it covers 3–42% of the time instead of 95%: at 10,000 paths it is ~0.019
wide while the estimator's real error is 0.015–0.081. A gate judging it is
confident about the wrong quantity, and escalating the path count narrows an
error term that was never dominant.

`Interval.DOUBLE_BOOTSTRAP` (the default) resamples the history itself and takes
the spread of the resulting estimates:

| process | true p | bias | MAE | coverage (MC) | coverage (double) | width (double) |
|---|---|---|---|---|---|---|
| IID normal | 0.369 | +0.000 | 0.015 | 0.42 | **1.00** | 0.066 |
| Student-t(4) | 0.351 | +0.004 | 0.020 | 0.33 | **0.95** | 0.080 |
| GARCH(1,1) | 0.325 | +0.021 | 0.059 | 0.05 | 0.40 | 0.088 |
| regime-switching | 0.269 | +0.024 | 0.081 | 0.03 | 0.35 | 0.119 |

It reaches nominal coverage where the process has no state, and improves the
state-dependent cases 5–10× without fixing them: block-resampling a history
scrambles the state that history *ended in*, which is the part the estimator
cannot represent anyway. This is the honest interval available, not a correct
one. Because its width is a property of the history rather than of compute, path
escalation cannot resolve an indeterminate verdict, and is therefore confined to
`Interval.MC`.

**With that interval the gate discriminates correctly.** Sweeping the falsifier
threshold so truth spans the band (verdicts as accept/reject/indeterminate over
12 trials, regime-switching process):

| threshold | true p | verdicts |
|---|---|---|
| 4% | 0.657 (inside) | 4 / 0 / 8 |
| 8% | 0.441 (inside) | **12 / 0 / 0** |
| 12% | 0.291 (just outside) | 1 / 1 / 10 |
| 20% | 0.117 (outside) | **0 / 12 / 0** |
| 30% | 0.030 (outside) | **0 / 12 / 0** |

Mid-band accepts, clearly-outside rejects, and `indeterminate` concentrated in a
collar of roughly ±0.03 around the edges — which is what it is for. Reported at
a single threshold the error rates look alarming (57% "false reject" on this
process with the MC interval); the sweep shows that headline is a boundary
artifact, and that the fix is the interval, not a wider band. On this evidence
TRD open question 7 — should the band widen for high-conviction theses — does not
need a yes; the cost of the 0.3–0.7 band is a 30–75% indeterminate rate near its
edges, not misclassification.

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

## Validating the number

`null_probability` is a reference the rest of the system leans on: the 0.3–0.7
gate keys off it, and FR7 reports calibration net of it. If it is biased, the
gate rejects sound falsifiers and every Brier skill score is wrong by an unknown
amount. So it has its own validation, in two parts.

**Bias study** — `co_agent/sim/dgp.py` plus `bias_study` in `validate.py`. Real
data has no known answer: you observe one realisation and cannot ask it what the
probability *was*. A synthetic process can be asked, by simulating forward from a
known state, which is the only way to measure bias rather than self-consistency.
The panel is an IID baseline, Student-t for tails without clustering, GARCH(1,1)
for clustering, and a regime-switching process built to defeat an unconditional
estimator. Every table above came from it:

```
python -m co_agent.sim.validate                                  # fast, MC interval
python -m co_agent.sim.validate --interval double_bootstrap      # the honest width
```

**Historical walk-forward** — `walk_forward` and `reliability` in `validate.py`.
At each origin, predict using only prior data (enforced by slicing, not
convention), then observe whether the falsifier actually tripped. Truth is
unknowable per observation, but predicted probability can be compared against
realised frequency in bins. No price data ships here; see
`co_agent/data/prices.py` for the CSV format and for the two data properties that
matter more than the loader — a point-in-time universe, and adjusted closes.

```
python -m co_agent.sim.validate --prices ./path/to/csvs
```

Windows are non-overlapping by default, because sampling daily gives thousands of
observations of nearly the same event; effective sample size is roughly span ÷
horizon. Intervals on realised frequency are bootstrapped over observations
rather than binomial, since symbols move together. Every row carries its `n`.

Neither part substitutes for the other. A process that flatters the estimator and
history that does not means the panel is too kind; passing on the panel and
failing on history localises the problem to a property the panel omits.

### First run on real data (TSX, 48 names, 4,119 observations)

48 liquid TSX names, histories from 1986–2015 starts to 2026-09, non-overlapping
60-day windows, `TouchBelow(0.12)`, 1,600-observation history per origin.
Predicted probability against what actually happened:

| predicted | n | mean pred | realised | 95% CI | |
|---|---|---|---|---|---|
| **zero drift** (current default) | | | | | |
| [0.0, 0.2) | 1622 | 0.126 | 0.103 | [0.069, 0.144] | ok |
| [0.2, 0.4) | 1637 | 0.277 | 0.164 | [0.133, 0.197] | **overstates** |
| [0.4, 0.6) | 817 | 0.490 | 0.400 | [0.352, 0.447] | **overstates** |
| [0.6, 0.8) | 43 | 0.630 | 0.535 | [0.390, 0.691] | ok (thin) |
| **historical drift** (w = 1) | | | | | |
| [0.0, 0.2) | 2315 | 0.108 | 0.113 | [0.084, 0.145] | ok |
| [0.2, 0.4) | 1120 | 0.281 | 0.210 | [0.166, 0.255] | overstates |
| [0.4, 0.6) | 607 | 0.488 | 0.422 | [0.368, 0.477] | ok |
| [0.6, 0.8) | 77 | 0.624 | 0.429 | [0.316, 0.570] | overstates (thin) |

**Drift is the largest single error, and neither extreme is the answer.**
Sweeping the drift mode over the same 4,119 windows (`shrunk` blends the
symbol's own drift with an equal-weighted market baseline; *w* is the weight on
the symbol's own):

| setting | weighted \|pred − realised\| | bins outside CI | worst gap |
|---|---|---|---|
| zero | 0.0726 | 2 of 4 | 0.073 |
| historical | 0.0352 | 3 of 4 | 0.078 |
| **shrunk, w = 0.00** | **0.0268** | **1 of 4** | 0.025 |
| shrunk, w = 0.25 | 0.0278 | 1 of 4 | 0.026 |
| shrunk, w = 0.50 | 0.0276 | 2 of 4 | 0.023 |

The known-truth panel says the same thing and explains why, because there the
error decomposes. On drifting processes (`mu` ≈ 8%/year):

| process | drift | w | bias | MAE |
|---|---|---|---|---|
| IID + drift | zero | – | **+0.0383** | 0.0383 |
| IID + drift | historical | – | +0.0010 | **0.0527** |
| IID + drift | shrunk | 0.00 | +0.0012 | **0.0144** |
| IID + drift | shrunk | 0.50 | +0.0000 | 0.0305 |
| GARCH + drift | zero | – | **+0.0593** | 0.0774 |
| GARCH + drift | historical | – | +0.0218 | 0.0722 |
| GARCH + drift | shrunk | 0.00 | +0.0219 | **0.0602** |

`ZERO` carries the bias and `HISTORICAL` carries the variance. Stripping drift
from an asset that drifts upward does not produce a forecaster with no
information about direction -- it produces a counterfactual in which the stock
falls more often than it really does, worth +0.038 to +0.059 of spurious null
probability. Using the symbol's own drift removes that bias and replaces it with
estimation noise: **a single name's drift over 1,600 days has a standard error
near 12.6%/year against a signal of maybe 8%**, so the estimate's sign is not
reliable, and on the IID process `HISTORICAL` ends up with a *worse* total error
than `ZERO`. Averaging across N names cuts that error by sqrt(N), which is the
whole argument for shrinking toward a baseline.

The GARCH rows also separate the two effects cleanly: its zero-drift bias of
+0.059 is roughly the IID drift effect (+0.038) plus the clustering bias
measured on the driftless GARCH (+0.021). Drift and clustering are close to
additive, and drift is the larger of the two.

`Drift.SHRUNK` at `w = 0.25` is the default. On real data 0.00, 0.25 and 0.50
score 0.0268, 0.0278 and 0.0276 -- indistinguishable, a flat region. The
synthetic panel prefers 0.00 outright, but its baseline was each process's exact
`mu`; a real cross-sectional baseline is itself estimated, so that preference is
optimistic. 0.25 sits inside the flat region without discarding the symbol's own
history on the strength of an idealisation. `SHRUNK` requires
`Request.baseline_drift` and will not invent one -- an assumed market drift
would be an undeclared input to every figure -- so a one-off with no universe to
hand should pass `Drift.HISTORICAL` explicitly.

**What this result cannot settle.** The universe was assembled today, so
delisted names are missing and realised trip frequencies are biased *down* --
the same direction as the estimator's apparent overstatement. The two are
confounded, and no amount of extra symbols fixes it; only a point-in-time
constituent list would. The *comparison* between drift settings is unaffected,
since both share the bias.

### More history does not help, on real data

The synthetic panel says error falls 12-31% going from ~1,600 observations to
6,400. It does not replicate. Controlled test on real TSX data -- **the same 32
symbols and the same 722 origins at every setting**, so only the history handed
to the estimator varies:

| history | Brier | reliability | weighted \|pred − realised\| | bins outside CI |
|---|---|---|---|---|
| 400 | 0.1381 | 0.0043 | 0.0585 | 1 of 4 |
| 800 | **0.1304** | 0.0012 | 0.0179 | 1 of 4 |
| **1600** (default) | 0.1347 | **0.0008** | **0.0177** | **0 of 3** |
| 3200 | 0.1367 | 0.0022 | 0.0303 | 1 of 3 |
| 6400 | 0.1354 | 0.0037 | 0.0422 | 1 of 3 |

The curve is U-shaped with its floor at 800-1600. Twenty-five years of history
*quadruples* the calibration error rather than reducing it.

The panel was wrong here for a reason worth keeping in mind whenever it is used
again: every process in it is stationary by construction, so more data is
unconditionally more information. Real markets have structural breaks, and the
bootstrap treats a 2003 trading day as an equally valid draw for 2026. It is
not. **Leave `min_history` at 1600**; it was right by accident and is now right
by measurement. Adding a regime-shifting process to the panel would close the
gap that let it mislead.

Varying `min_history` alone measures nothing, incidentally -- it is a floor
check, not the amount of history the estimator uses. The sweep has to vary
`history_obs` at fixed symbols and origins, or it compares different universes
and different periods instead.

**Still to run:** the falsifier-threshold sweep above covers one process and one
falsifier family. The same sweep across `TerminalBelow` and `DrawdownExceeds`, and
across horizon buckets, would say whether some falsifier *shapes* are
intrinsically easier to estimate than others — which would be worth knowing before
FR3's horizon policy is revisited against the ledger.

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
