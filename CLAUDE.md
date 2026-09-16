# co-agent

A personal market research system: a hand-written Claude agent loop, the FR9
simulation service, a paper broker, and the ledger that records what was
claimed and what happened.

## The goal

**Beat equal-weight buy and hold of the same universe, net of fees and
slippage.** Not "make money" — a rising market does that for free, and an arm
that returns +15% while the universe returns +19% has destroyed value with extra
steps. The one number that matters is the *gap*:

```bash
python -m co_agent.cycle.bottom_line          # gap on the most recent window
python -m co_agent.cycle.bottom_line -h       # --windows 4 for a steadier read
```

**The goal is currently not met, and it is not close.** Over 95 overlapping
12-month windows from 2002, buy and hold returned roughly 14 points a year more
than either active arm, and was profitable in 86% of windows against roughly
51% (`studies/cadence-2026-09-16.jsonl`, reported in `co_agent/cycle/README.md`).
A change that does not move that gap has not advanced the goal, however good the
diff looks.

Recording that honestly is the point of the ledger. This package has
demonstrated cost control and volatility-aware sizing, and nothing yet that
beats owning the universe.

## Instruments: long equity only, bought and sold with settled cash

**Buy and sell. No options, no shorts, no margin, no leverage, no derivatives
of any kind.** This is a fixed constraint on the system, not a feature that has
not been built yet. Adding an instrument is a decision for the owner of the
account, never an implementation detail of a change that is chasing the gap.

The broker enforces it structurally rather than by convention, and
`tests/test_paper.py` pins it: `sell` only ever sells what is held and opens
nothing when the position is absent, so a position cannot go short; `buy` is
bounded by settled cash and raises `InsufficientCash` rather than borrowing.

The reason is that everything downstream assumes a bounded loss. `p95_drawdown`
is a per-position number computed from price paths, the weight reduction sizes
against it, and both are meaningless for a payoff that can pass -100%. A short
or a written option turns the worst case from "the position goes to zero" into
"the position is unbounded", and nothing in the sizing path would notice.

That matters most precisely because the goal above is stated in money. Leverage
is the fastest way to move a return number without any edge behind it, and it
would move the reported gap while making the system strictly worse. If an arm
only beats buy and hold with leverage, it has not beaten buy and hold.

## Simulation is the instrument, for now

We simulate to start. That is the deliberate phase, not an apology: paper
removes the broker dependency entirely, so the cycle, the gate, the sizing and
the ledger can all be exercised before any account exists. The question that
matters is therefore not "is the paper number right" -- it is not, exactly --
but **does the paper number move the way the real one would?**

**Why the gap and not the return.** Both arms take the same modelled fills at
the same closes under the same slippage and commission, so friction error
largely cancels between them. It does not cancel cleanly: over a year the
strategy pays about 144 commissions and the benchmark about 43, so any error in
the cost model hits the strategy roughly three times harder. That residual is
the whole reason the gap needs a sensitivity check rather than trust.

**What that check says today** (`co_agent/cycle/frictions.py`, output in
`studies/frictions-2026-09-16.txt`, 4 windows spanning 2002-2026):

```
                    strategy   buy & hold      gap
  frictionless         9.12%       14.04%   -4.91%
  default (10bps)      0.04%       11.86%  -11.82%
  50bps, x1           -6.82%       11.53%  -18.35%
  10bps, x4          -20.60%        5.03%  -25.64%
```

The gap is negative at every setting, **including a frictionless account with no
slippage and no commission on either arm**. The best case that could possibly
exist for the strategy still trails by 4.91%. So being behind is not an artifact
of the execution model, and that conclusion is insensitive to whatever real
fills turn out to cost. Costs make it worse; they did not create it.

Run `python -m co_agent.cycle.frictions` before quoting any gap as a finding.
Both arms must always move together -- a gap measured with one arm's frictions
changed is not a gap.

## Graduating from paper

The correlation that actually matters -- paper fills against real fills -- cannot
be measured until real fills exist, so it is the last step and not the first.
Before any real capital:

1. **The gap is positive at the default frictions, and keeps its sign across the
   whole grid.** A gap that only appears below 5bps is an execution fantasy.
2. **Measured over the full cadence study, not one window.** 95 overlapping
   windows, with the paired interval excluding zero. `bottom_line` is a
   development signal; it is not the thing that clears this bar.
3. **Against a point-in-time universe that includes delisted names.** Today's
   benchmark is survivor-only and therefore flattered, so the measured gap is
   *harsher* than the truth -- which is the safe direction for a bar to be wrong
   in, and the wrong direction for a green light.
4. **A resolved-outcome record in the ledger**, with calibration reported net of
   `null_probability` (FR7), over enough resolved theses to mean something.
5. **Then a deliberately small allocation**, with real fills recorded against the
   paper model, so the friction assumptions get checked against reality rather
   than against themselves. That measurement is what closes this section.

None of these is met. The system is in step 0.

## What the goal does not license

A money goal plus a fast feedback number is how a research system turns into an
overfitted one. These hold regardless of what the gap says:

- **A paper *level* is not evidence; a paper *gap* is weak evidence.** Every fill
  is modelled at a daily close with 10bps slippage and Questrade-like commission,
  and `paper/broker.py` says a frictionless paper broker is the single most
  reliable way to make a strategy look good. The gap is a paper number too --
  both sides of it are -- but it is the more robust one, for the reason in
  *Simulation is the instrument* below. Present the gap, with its sample size and
  its friction sensitivity; never present a paper return on its own.
- **The benchmark is flattered too.** The universe is survivor-only
  (`data/universes.py`), which biases buy and hold *up* most of all, since it
  holds every name for the whole window. The real gap is narrower than the
  measured one by an unknown amount. That is an argument for better data, never
  for quietly dropping the benchmark.
- **One window is one draw.** `bottom_line` runs the most recent window or two so
  it is fast. The cadence study's spread is the context: window-level differences
  of ±9% were routine while the underlying effect was zero. Do not tune on it,
  and do not read a single window as a result.
- **Never change the benchmark, the universe or the frictions to close the gap.**
  Those are the measuring instrument. Improving the score by moving them is the
  one move that cannot be detected downstream.
- **Sample size beside every figure** (FR7), and `indeterminate` from the FR9
  gate is not a rejection — that call belongs to a human.

## Layout

```
co_agent/            the agent loop and its tools
co_agent/sim/        FR9 simulation service (null_probability, p95_drawdown, the gate)
co_agent/cycle/      snapshot -> candidates -> gate -> decisions -> paper fills
co_agent/cycle/cadence.py      the rolling-window study (slow, authoritative)
co_agent/cycle/bottom_line.py  the fast check against the goal
co_agent/cycle/frictions.py    how far the gap depends on the cost model
co_agent/cycle/compare_arms.py both candidate sources against one live snapshot
co_agent/data/       price loading and fetching
db/                  the Postgres ledger schema
studies/             raw study rows, so tables can be checked without a re-run
```

`cadence.py` is authoritative; `bottom_line.py` is the fast check. They share
`run_arm` and `equal_weight_buy_hold` deliberately, so the two cannot drift.

## Working here

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest            # no network, no API key
./db/test/run_migrations.sh           # schema assertions, throwaway cluster
```

Price data is **gitignored** — vendor series are not redistributed here, so a
fresh clone has none and anything reading `data/prices` must degrade rather than
fail. Rebuild it with:

```bash
.venv/bin/python -m co_agent.data.fetch --universe tsx --out data/prices --cache data/cache
```

An `ANTHROPIC_API_KEY` is needed only for the research arm (`LLMResearch`). The
tests, the simulator, the cadence study and `bottom_line` all run without one.

Measurement code belongs in the repository. The first cadence harness was a
scratch script and did not survive the session that wrote it; that is why
`cadence.py` is a module.
