# `co_agent.cycle` — the weekly cycle

```
snapshot -> candidates -> FR9 gate -> decisions -> paper fills
                                                       |
                            daily: resolve falsifiers, exit positions
```

```python
engine = Engine(
    series=load_dir("data/prices"),
    baseline_by_date=market_drift_by_date(series, 1600),
    source=MomentumScreen(lookback=60, horizon_days=60),   # or LLMResearch(client=...)
    broker=PaperBroker(cash=10_000.0),
    ledger=Ledger("ledger/"),
)
for day in trading_days:
    engine.settle(day)          # resolve and exit
    if is_cycle_day(day):
        engine.run_cycle(day)   # snapshot, propose, gate, decide, fill
```

## The falsifier is not the exit

This was the original design and it was wrong. A falsifier is *tuned to trip*:
the FR9 gate wants it inside 0.30–0.70, so roughly half fire on noise alone, and
that is exactly what makes resolving one informative. A stop wants the opposite —
rare enough to fire only on information.

Using one as the other guarantees being stopped out of about half of all
positions on noise, paying spread and commission each time. Measured over 43
sessions: **18 of 21 falsifiers tripped**, and the forced exits were the single
largest contributor to a −7.6% result in a market that returned +1.6%. Over a
full year the same rule cost 14 points against holding to the horizon.

`settle()` now records the outcome the moment the falsifier decides, and closes
the position on its own rule: horizon by default, an optional `stop_drop` sized
to be rare, or the old behaviour behind `exit_on_falsifier=True`.

## Turnover is the largest controllable cost

On a $10,000 account, measured across an up year and a down year:

| | UP year | DOWN year | annual costs | fills |
|---|---|---|---|---|
| weekly, falsifier exit | +6.8% | −3.7% | $1,782 | 288 |
| weekly, hold to horizon | +20.5% | −4.6% | $1,256 | 200 |
| monthly, hold to horizon | +10.7% | +9.5% | **$367** | 62 |
| *equal-weight buy & hold* | *+26.3%* | *+5.0%* | *~$240* | *48* |

Weekly trading costs **18% of capital per year**. Monthly costs 4%. That part is
arithmetic and not in doubt. Whether the monthly *returns* hold up is two data
points, and a rolling test across every overlapping 12-month window since 2002 is
what settles it — until it reports, the cadence is not a default.

Note that buy-and-hold beat every active variant in the up year. Nothing in this
package has demonstrated directional edge; the negative control showed the
simulator's score contains none by construction. What is demonstrated is cost
control and volatility-aware sizing.

## Falsifier thresholds that the gate will actually accept

Realised trip rate for "closes below on any day", 48 TSX names, non-overlapping
windows. The FR9 band is 0.30–0.70:

| threshold | h=10 | h=20 | h=30 | h=60 |
|---|---|---|---|---|
| 2% | **0.445** | **0.572** | **0.626** | 0.706 |
| 3% | **0.331** | **0.465** | **0.528** | **0.620** |
| 4% | 0.243 | **0.369** | **0.439** | **0.541** |
| 6% | 0.137 | 0.239 | **0.306** | **0.415** |
| 8% | 0.078 | 0.154 | 0.209 | **0.321** |
| 12% | 0.031 | 0.072 | 0.106 | 0.191 |

A 12% falsifier at 60 days — the obvious thing to write — trips 19% of the time
and **the gate rejects it as too hard**. Only 21–27% of scores landed inside the
band before these thresholds were used; the research prompt carries this guidance
for the same reason.

Terminal-form falsifiers ("closes below *on* day N") never reach the band at any
sensible threshold at 60 days, peaking at 0.239.

## Two caps, two jobs

`batch_cap` bounds how many rows a human reviews in one sitting (FR5). It does
**not** bound exposure: with a 20-day horizon and a weekly cycle, entries outpace
exits and open positions reached 10 against a cap of 6. `max_open` is the
exposure bound.

## The research arm dies at the gate on threshold choice, not on reasoning

`LLMResearch` has now executed against a live snapshot (2026-09-15, 45 names).
The first batch was **accepted 1 of 6**, and all five rejections were
`reject_too_hard` — the gate never reached the mechanisms.

The cause was not the reasoning. It was the threshold, and the prompt caused it.
The old guidance named the universe figure (6% at 60 days) and then said "choose
per name based on how volatile it is". The model scaled *monotonically with
volatility* — the ordering was right — but it read 6% as a **floor** and only
ever adjusted upward:

| name | ann.vol | chose | mid-band | null_p | verdict |
|---|---|---|---|---|---|
| FTS | 13.1% | 6% | 2.8% | 0.218 | too hard |
| RY | 17.1% | 7% | 2.8% | 0.188 | too hard |
| SLF | 16.7% | 7% | 3.4% | 0.234 | too hard |
| L | 20.2% | 8% | 3.6% | 0.170 | too hard |
| CNQ | 28.9% | 11% | 7.3% | 0.334 | accept |
| FNV | 36.2% | 12% | 6.6% | 0.258 | too hard |

Uniformly 1.5–2.5× too wide. A defensive name with a 6% floor asks whether
something unlikely happens, which is exactly the falsifier the gate rejects.

**The rule.** Bisecting each name to `null_probability = 0.50` at 60 days gives a
threshold that scales linearly with volatility:

    threshold ≈ 0.21 × annualised vol      (sd 0.041 over 0.162–0.284, n=10)

This *generalises* the flat map rather than replacing it: the universe's mean
volatility at this snapshot is 28.1%, and 0.21 × 28.1% = 6.0%, which is the map's
own 60-day figure. The coefficient sits below the driftless random-walk value
(~0.33) because the null carries the shrunk baseline drift, +3.8% over 60 days,
which pushes paths up and away from the falsifier.

It also explains why the momentum screen's flat 6% works: momentum ranking
selects high-volatility names (24–50% here), and 6% is close to mid-band for
them. The screen is not choosing thresholds well — it is choosing names whose
volatility happens to suit one fixed threshold.

The prompt now carries the scale, the direction, and the failure mode. Re-run on
the same snapshot: **accepted 6 of 6**, thresholds 2.8%–10.8%, every `null_p`
between 0.385 and 0.562.

## Research and screen pick different portfolios

Same snapshot, same horizon, same gate, six theses each. Overlap was 2 of 6
(CNQ, FNV): the screen took the momentum leaders (CVE +33.6%, SU +28.7%, WPM,
IMO — energy and gold), the research arm took defensives on stated mechanisms
(FTS regulated utility, RY capital position, SLF fee-based earnings, L discount
banners). Whether either carries edge is an outcomes question the ledger will
answer at the horizon; what is established today is only that they are not the
same bet. Cost was ~3.5k input / ~3.4k output tokens per cycle.

## What the auto-approve arm is not

The paper engine records `policy="auto_approve_paper"` and a null
`human_confidence`. **G4 — whether the human gate adds or destroys value — can
never be computed from these batches**, because no human saw them. That
comparison needs a human stating a probability before deciding, which is what the
`human_confidence` column exists for.

## Known gaps

- **`LLMResearch` has executed twice, on one snapshot.** Both runs are above.
  One snapshot is not a sample: the threshold rule is measured on 10 names at a
  single date and a single horizon, and the 6-of-6 acceptance that followed the
  prompt fix is one batch. Neither has an outcome yet — no thesis from either
  arm has reached its horizon, so nothing here speaks to edge.
- **The ledger here is a JSONL stand-in** for `db/`. No foreign keys, no CHECK
  constraints, no privilege separation — the three things that make the Postgres
  schema trustworthy. Move before anything depends on the history being unedited.
- **Red team does not exist.** The gate runs; adversarial review does not.
- **Bank of Canada and StatCan are unreachable** from this environment. SEC
  EDGAR is reachable.
