"""Both candidate sources against one frozen snapshot, side by side.

FR7 measures the model-driven arm against the benchmark arm, and the only
honest way to do that is to hand both the *same* frozen snapshot: `build_snapshot`
is what freezes it, so neither arm can see a close the other did not. That is
the whole reason `MomentumScreen` and `LLMResearch` sit behind one interface.

This is the qualitative half of the comparison -- what each arm proposes, before
the gate and before any position exists. `cadence.py` is the quantitative half.
Neither replaces the ledger: an edge is a claim about resolved outcomes, and
reading two lists of theses side by side cannot settle it.

The benchmark arm is deterministic and needs no credentials. The research arm
needs the Messages API, so a missing key degrades the report to naming what it
could not reach, and prints the snapshot date -- the snapshot is reproducible,
so the same comparison can be completed later against the same universe and the
same closes rather than against whatever the market has done since.
"""

from __future__ import annotations

import argparse
from datetime import date

from .candidates import Candidate, MomentumScreen
from .snapshot import Snapshot, build_snapshot

HISTORY_OBS = 1600


def benchmark_arm(snapshot: Snapshot, n: int, horizon_days: int) -> tuple[str, list[Candidate]]:
    screen = MomentumScreen(lookback=60, horizon_days=horizon_days)
    return screen.name, screen.propose(snapshot, n)


def research_arm(
    snapshot: Snapshot, n: int, horizon_days: int, client: object
) -> tuple[str, list[Candidate]]:
    from .research import LLMResearch

    arm = LLMResearch(client=client, horizon_days=horizon_days)
    return arm.name, arm.propose(snapshot, n)


def format_arm(title: str, candidates: list[Candidate]) -> str:
    out = [f"\n{title}", "-" * len(title)]
    if not candidates:
        out.append("  (none proposed)")
    for i, c in enumerate(candidates, 1):
        out += [
            f"\n  {i}. {c.symbol}   horizon {c.horizon_days}d   confidence {c.confidence:.2f}",
            f"     claim:     {c.claim}",
            f"     mechanism: {c.mechanism}",
            f"     falsifier: {c.falsifier_text}",
        ]
    return "\n".join(out)


def format_header(snapshot: Snapshot) -> str:
    return (
        f"snapshot {snapshot.taken_at}  digest {snapshot.digest[:16]}\n"
        f"  universe {len(snapshot.universe)}   missing {len(snapshot.missing)}"
        f"   baseline drift {snapshot.baseline_drift:+.6f}/day"
    )


def format_unavailable(snapshot: Snapshot, exc: BaseException) -> str:
    """What the research arm could not reach, and how to finish the comparison."""
    return (
        f"\nresearch arm\n------------\n"
        f"  UNAVAILABLE: {type(exc).__name__}\n  {exc}\n\n"
        f"  The snapshot above is frozen, so this is resumable rather than lost:\n"
        f"  rerun with --date {snapshot.taken_at} once credentials resolve and the\n"
        f"  research arm is scored against the same universe and the same closes."
    )


def main(argv=None) -> int:
    from ..data import load_dir
    from ..sim.validate import market_drift_by_date

    p = argparse.ArgumentParser(description="Both candidate sources, one snapshot.")
    p.add_argument("--prices", default="data/prices")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--date", default=None, help="YYYY-MM-DD; default is the latest close")
    args = p.parse_args(argv)

    series = load_dir(args.prices, min_obs=HISTORY_OBS + 1)
    baseline = market_drift_by_date(series, HISTORY_OBS)
    when = date.fromisoformat(args.date) if args.date else max(baseline)
    snapshot = build_snapshot(series, when, history_obs=HISTORY_OBS, baseline_by_date=baseline)

    print(format_header(snapshot))
    name, candidates = benchmark_arm(snapshot, args.n, args.horizon)
    print(format_arm(f"benchmark arm -- {name}", candidates))

    try:
        import anthropic

        name, candidates = research_arm(snapshot, args.n, args.horizon, anthropic.Anthropic())
    except Exception as exc:  # noqa: BLE001 -- the benchmark half is still a result
        print(format_unavailable(snapshot, exc))
        return 2
    print(format_arm(f"research arm -- {name}", candidates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
