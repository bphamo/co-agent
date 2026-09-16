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

## What the goal does not license

A money goal plus a fast feedback number is how a research system turns into an
overfitted one. These hold regardless of what the gap says:

- **Paper is not evidence.** Every fill is modelled at a daily close with 10bps
  slippage and Questrade-like commission. `paper/broker.py` says a frictionless
  paper broker is the single most reliable way to make a strategy look good, and
  FR9 bans simulated equity curves for exactly that reason. Never present a paper
  return as a result; present the gap, with its sample size.
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
