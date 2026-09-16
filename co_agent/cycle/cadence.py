"""Rolling-window cadence study: does the monthly cadence's *return* hold up?

The turnover finding is arithmetic and not in doubt -- weekly trading costs
about 18% of a $10,000 account per year against monthly's 4%. Whether monthly
also *returns* more was two data points, one up year and one down year, and
`cycle/README.md` said so: "a rolling test across every overlapping 12-month
window since 2002 is what settles it -- until it reports, the cadence is not a
default."

This is that test. It exists as a module rather than a scratch script because
the first version of it was a scratch script, and it did not survive the session
that wrote it.

Design, and what is held fixed so that only cadence varies:

* The cycle calendar is the only difference between the two hold-to-horizon
  arms. Same screen, same horizon, same gate, same broker, same frictions.
* The horizon is held at the screen's own default of 20 days across both arms,
  rather than matched to each cadence. Matching it would confound cadence with
  horizon, and horizon is a separate question with its own answer. At 20 days a
  monthly cycle closes a position at about the moment the next cycle opens one,
  which is what makes monthly a coherent cadence rather than an arbitrary one.
* Windows are 252 trading days and start every 63 (quarterly), so they overlap.
  Overlapping windows are not independent observations -- the head-to-head count
  below is a description of the sample, not a significance test, and the paired
  bootstrap interval inherits the same dependence. Treat both as descriptive.

Run::

    python -m co_agent.cycle.cadence --prices data/prices            # all windows
    python -m co_agent.cycle.cadence --limit 4 --offset 0            # a shard
    python -m co_agent.cycle.cadence --report results/*.jsonl        # aggregate
"""

from __future__ import annotations

import argparse
import glob as globlib
import json
import random
import statistics as st
import sys
import time
from dataclasses import dataclass
from datetime import date

from ..data.prices import load_dir
from ..paper import InsufficientCash, PaperBroker
from ..sim.validate import market_drift_by_date
from .candidates import MomentumScreen
from .ledger import Ledger
from .run import Engine

WINDOW_DAYS = 252
STEP_DAYS = 63
HISTORY_OBS = 1600
START_CASH = 10_000.0
FIRST_START = date(2002, 1, 1)

#: (label, trading days between cycles, whether a tripped falsifier also exits)
ARMS = (
    ("weekly_hold", 5, False),
    ("monthly_hold", 21, False),
    ("weekly_fexit", 5, True),
)


class NullLedger(Ledger):
    """Writes nothing. The study reads equity, not records.

    A real Ledger would write several million JSONL rows across 95 windows and
    four arms, for a result that never reads them back.
    """

    def __init__(self) -> None:  # noqa: D107 - deliberately does not open a path
        pass

    def append(self, kind: str, record: object) -> None:
        pass


@dataclass(frozen=True, slots=True)
class ArmResult:
    ret: float
    fills: int
    fees: float
    cycles: int
    #: True when the benchmark ran out of cash before buying every name that
    #: was *available at the window's start* -- not every name in the universe,
    #: since an early window predates half of it. Only reachable at frictions
    #: well above the default, and a caller comparing against a truncated
    #: benchmark is comparing against something other than owning the universe.
    truncated: bool = False


def index_map(series: dict) -> dict:
    return {sym: {d: i for i, d in enumerate(px.dates)} for sym, px in series.items()}


def closes_for(series: dict, idx: dict, day: date) -> dict[str, float]:
    out: dict[str, float] = {}
    for sym, px in series.items():
        i = idx[sym].get(day)
        if i is not None:
            out[sym] = float(px.closes[i])
    return out


def calendar(series: dict) -> list[date]:
    return sorted({d for px in series.values() for d in px.dates})


def _final_marks(series: dict, idx: dict, days: list[date], symbols) -> dict[str, float]:
    """Last close each name actually had inside the window.

    A name delisted or halted mid-window has no price on the final day. Marking
    it at zero would book a total loss that did not happen; dropping it would
    book no loss at all. Its last observed close is the honest mark.
    """
    marks = closes_for(series, idx, days[-1])
    for symbol in symbols:
        if symbol in marks:
            continue
        for day in reversed(days):
            i = idx[symbol].get(day)
            if i is not None:
                marks[symbol] = float(series[symbol].closes[i])
                break
    return marks


def equal_weight_buy_hold(
    series: dict,
    idx: dict,
    days: list[date],
    *,
    slippage_bps: float = 10.0,
    commission_scale: float = 1.0,
) -> ArmResult | None:
    """The reference arm: buy the universe on day one, hold, same frictions.

    The frictions are arguments so a sensitivity study can move them, and they
    must move on *both* arms together -- a gap measured with one arm's frictions
    changed is not a gap, it is a thumb on the scale.
    """
    broker = PaperBroker(
        cash=START_CASH, slippage_bps=slippage_bps, commission_scale=commission_scale
    )
    opening = closes_for(series, idx, days[0])
    if not opening:
        return None
    budget = START_CASH / len(opening)
    truncated = False
    for symbol in sorted(opening):
        try:
            broker.buy(symbol, days[0], opening[symbol], budget)
        except InsufficientCash:
            truncated = True
            # Only reachable at frictions far above the default, where a single
            # commission exceeds what is left. Stopping leaves the remainder in
            # cash, which is what an equal-weight buyer would actually hold;
            # `names` records the shortfall rather than hiding it.
            break
    marks = _final_marks(series, idx, days, list(broker.positions))
    return ArmResult(
        ret=broker.equity(marks) / START_CASH - 1.0,
        fills=len(broker.fills),
        fees=sum(f.fees for f in broker.fills),
        cycles=1,
        truncated=truncated,
    )


def run_arm(
    series: dict,
    idx: dict,
    baseline: dict[date, float],
    days: list[date],
    *,
    every: int,
    exit_on_falsifier: bool,
    screen: MomentumScreen,
    slippage_bps: float = 10.0,
    commission_scale: float = 1.0,
) -> ArmResult:
    broker = PaperBroker(
        cash=START_CASH, slippage_bps=slippage_bps, commission_scale=commission_scale
    )
    engine = Engine(
        series=series,
        baseline_by_date=baseline,
        source=screen,
        broker=broker,
        ledger=NullLedger(),
        history_obs=HISTORY_OBS,
        exit_on_falsifier=exit_on_falsifier,
    )
    cycles = 0
    for i, day in enumerate(days):
        engine.settle(day)
        if i % every == 0:
            try:
                engine.run_cycle(day)
            except KeyError:
                continue  # no market baseline for this day
            cycles += 1
    marks = _final_marks(series, idx, days, list(broker.positions))
    return ArmResult(
        ret=broker.equity(marks) / START_CASH - 1.0,
        fills=len(broker.fills),
        fees=sum(f.fees for f in broker.fills),
        cycles=cycles,
    )


def window_starts(days: list[date]) -> list[int]:
    return [
        i
        for i in range(0, len(days) - WINDOW_DAYS, STEP_DAYS)
        if days[i] >= FIRST_START
    ]


def sweep(prices: str, limit: int | None, offset: int, horizon: int, out) -> int:
    series = load_dir(prices)
    baseline = market_drift_by_date(series, HISTORY_OBS)
    idx = index_map(series)
    days = [d for d in calendar(series) if d in baseline]
    starts = window_starts(days)
    screen = MomentumScreen(lookback=60, horizon_days=horizon)

    chosen = starts[offset : offset + limit] if limit else starts[offset:]
    print(
        f"# {len(series)} series, {len(days)} trading days, "
        f"{len(starts)} windows of {WINDOW_DAYS}d stepping {STEP_DAYS}d; "
        f"running {len(chosen)} from offset {offset}; screen {screen.name} h={horizon}",
        file=sys.stderr, flush=True,
    )
    for k, si in enumerate(chosen):
        window = days[si : si + WINDOW_DAYS]
        started = time.time()
        row: dict = {
            "window": offset + k,
            "start": window[0].isoformat(),
            "end": window[-1].isoformat(),
            "horizon": horizon,
        }
        for label, every, fexit in ARMS:
            r = run_arm(series, idx, baseline, window, every=every,
                        exit_on_falsifier=fexit, screen=screen)
            row[label] = {"ret": r.ret, "fills": r.fills, "fees": r.fees,
                          "cycles": r.cycles}
        bh = equal_weight_buy_hold(series, idx, window)
        if bh is not None:
            row["buy_hold"] = {"ret": bh.ret, "fills": bh.fills, "fees": bh.fees}
        row["secs"] = round(time.time() - started, 1)
        print(json.dumps(row), file=out, flush=True)
    return 0


# ----------------------------------------------------------------------- report


def _paired(rows: list[dict], a: str, b: str, seed: int = 7) -> str:
    pairs = [(x[a]["ret"], x[b]["ret"]) for x in rows if a in x and b in x]
    diffs = [u - v for u, v in pairs]
    wins = sum(1 for d in diffs if d > 0)
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(10_000)
    )
    lo, hi = means[250], means[9_750]
    return (
        f"{a} vs {b}: wins {wins}/{len(pairs)} ({wins / len(pairs):.0%}), "
        f"mean diff {st.mean(diffs):+.2%} [{lo:+.2%}, {hi:+.2%}]"
    )


def report(patterns: list[str]) -> int:
    rows: list[dict] = []
    for pattern in patterns:
        for path in sorted(globlib.glob(pattern)):
            with open(path, encoding="utf-8") as handle:
                rows.extend(json.loads(ln) for ln in handle if ln.strip().startswith("{"))
    if not rows:
        print("no rows", file=sys.stderr)
        return 1
    rows.sort(key=lambda r: r["start"])
    print(f"{len(rows)} windows, {rows[0]['start']} -> {rows[-1]['end']}\n")

    arms = [a for a, _, _ in ARMS] + ["buy_hold"]
    head = (f"{'arm':<14}{'median':>9}{'mean':>9}{'p10':>9}{'p90':>9}"
            f"{'>0':>6}{'fills':>7}{'fees$':>8}")
    print(head)
    print("-" * len(head))
    for a in arms:
        r = sorted(x[a]["ret"] for x in rows if a in x)
        if not r:
            continue
        fl = [x[a]["fills"] for x in rows if a in x]
        fe = [x[a]["fees"] for x in rows if a in x]
        q = lambda p: r[int(p * (len(r) - 1))]  # noqa: E731
        print(f"{a:<14}{st.median(r):>8.1%}{st.mean(r):>9.1%}{q(0.10):>9.1%}"
              f"{q(0.90):>9.1%}{sum(1 for v in r if v > 0) / len(r):>6.0%}"
              f"{st.mean(fl):>7.0f}{st.mean(fe):>8.0f}")

    print("\nHead to head (overlapping windows: descriptive, not a significance test)")
    for a, b in (("monthly_hold", "weekly_hold"), ("monthly_hold", "buy_hold"),
                 ("weekly_hold", "buy_hold"), ("weekly_hold", "weekly_fexit")):
        print("  " + _paired(rows, a, b))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prices", default="data/prices")
    parser.add_argument("--limit", type=int, default=None, help="windows to run")
    parser.add_argument("--offset", type=int, default=0, help="first window index")
    parser.add_argument("--horizon", type=int, default=20,
                        help="held equal across arms; the screen's own default")
    parser.add_argument("--report", nargs="+", metavar="GLOB",
                        help="aggregate JSONL from a previous sweep instead of running")
    args = parser.parse_args(argv)
    if args.report:
        return report(args.report)
    return sweep(args.prices, args.limit, args.offset, args.horizon, sys.stdout)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # `--report ... | head` closes the pipe early; that is not a failure.
        # Python flushes stdout at exit and would re-raise, so retarget it.
        import os

        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise SystemExit(0)
