# co-agent

A hand-written Claude agent loop, plus the simulation service and ledger schema
for the personal market research system described in the TRD.

Python 3.11+.

```
co_agent/           the agent loop and its tools
co_agent/sim/       the FR9 simulation service (null_probability, p95_drawdown)
co_agent/sim/dgp.py   synthetic processes with a computable truth
co_agent/sim/validate.py  bias study + historical walk-forward
co_agent/data/      price loading for validation studies
db/                 the Postgres ledger schema
tests/              pytest suite
```

## Running the agent

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env   # or run: ant auth login
python -m co_agent
```

A `.env` file is convenient but not required — the SDK also resolves
`ANTHROPIC_AUTH_TOKEN` and an `ant auth login` profile, so credentials are
checked when the first request is made rather than at startup.

The loop is deliberately explicit: send the conversation, print what came back,
run any tools Claude asked for, send the results, repeat until Claude stops
calling tools. The SDK also ships a tool runner
(`client.beta.messages.tool_runner`) that owns this loop for you; it is the
better default for new code, and this module exists because owning the loop is
the point.

**`read_file` is not sandboxed.** It reads whatever path it is given, relative
to the process's working directory. That is a property of a local single-user
developer tool, not an oversight — but it means the agent can read anything you
can, so don't point it at an untrusted conversation.

## Tests

```bash
pytest                        # 197 tests, no network, no API key needed
./db/test/run_migrations.sh   # schema assertions against a throwaway cluster
```

The agent-loop tests run against a fake client, so the suite never spends
tokens.

## Validating the simulator

`null_probability` is a reference the gate and the calibration report both lean
on, so it has its own validation. The bias study needs no market data:

```bash
python -m co_agent.sim.validate                              # fast, Monte Carlo interval
python -m co_agent.sim.validate --interval double_bootstrap  # the honest interval
python -m co_agent.sim.validate --prices ./csvs              # historical walk-forward
```

What it found, and what changed as a result, is in
[`co_agent/sim/README.md`](co_agent/sim/README.md#validating-the-number).

## Sub-package documentation

- [`co_agent/sim/README.md`](co_agent/sim/README.md) — the simulation method,
  the falsifier gate, and where it departs from the TRD.
- [`db/README.md`](db/README.md) — the ledger schema, how immutability is
  enforced, and the departures from §5.1.
