"""The goal, as a number: does the strategy beat owning the universe, net of cost?

`cadence.py` measured this over 95 windows and the answer was no -- buy and hold
returned about 14 points a year more than either active arm. That result set the
goal: **beat equal-weight buy and hold, net of fees and slippage.** This module
is the fast check against that goal, meant to run after a change rather than as
a study.

It is deliberately the same code as the study. `run_arm` and
`equal_weight_buy_hold` are imported from `cadence.py` rather than reimplemented,
so a number reported here and a number reported there cannot drift apart. What
differs is only how much is run: the most recent window or two, not all 95.

**This is paper.** Virtual cash, modelled fills, 10bps slippage, Questrade-like
commission. `broker.py` is explicit that a paper broker is the easiest way to
make a strategy look good, and FR9 bans simulated equity curves for that reason.
A fast feedback loop pointed at a simulated P&L is a machine for overfitting to
one window, so what this prints is the *gap against the benchmark* rather than a
return to admire, on a sample small enough that the sample size is printed
beside it. One window is one draw; the cadence study's spread is the context.

Two more biases stay true here and are not fixable downstream. The universe is
survivor-only, which flatters buy and hold most of all, since it holds every
name for the whole window. And a single recent window is the one most likely to
have fed whatever change is being tested.

Run::

    python -m co_agent.cycle.bottom_line                     # gap on the last window
    python -m co_agent.cycle.bottom_line --save-baseline     # record it to compare against
    python -m co_agent.cycle.bottom_line --windows 4         # steadier, slower
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ..data.prices import load_dir
from ..sim.validate import market_drift_by_date
from .cadence import (
    HISTORY_OBS,
    START_CASH,
    WINDOW_DAYS,
    calendar,
    equal_weight_buy_hold,
    index_map,
    run_arm,
)
from .candidates import MomentumScreen

#: The committed default: monthly cadence, hold to horizon. `cadence.py` made it
#: the default on cost, the returns being a dead heat.
STRATEGY_EVERY = 21
BASELINE_PATH = Path("studies/bottom-line-baseline.json")


@dataclass(frozen=True, slots=True)
class Reading:
    windows: int
    strategy_ret: float
    buy_hold_ret: float
    strategy_fees: float
    fills: int

    @property
    def gap(self) -> float:
        """The goal metric: strategy minus buy and hold, net of cost. Wants > 0."""
        return self.strategy_ret - self.buy_hold_ret

    def to_dict(self) -> dict:
        return {
            "windows": self.windows,
            "strategy_ret": self.strategy_ret,
            "buy_hold_ret": self.buy_hold_ret,
            "strategy_fees": self.strategy_fees,
            "fills": self.fills,
            "gap": self.gap,
        }

    @staticmethod
    def from_dict(d: dict) -> "Reading":
        return Reading(
            windows=d["windows"], strategy_ret=d["strategy_ret"],
            buy_hold_ret=d["buy_hold_ret"], strategy_fees=d["strategy_fees"],
            fills=d["fills"],
        )


def measure(prices: str, windows: int, horizon: int) -> Reading:
    series = load_dir(prices)
    baseline = market_drift_by_date(series, HISTORY_OBS)
    idx = index_map(series)
    days = [d for d in calendar(series) if d in baseline]
    if len(days) < WINDOW_DAYS:
        raise ValueError(f"need {WINDOW_DAYS} sessions with a baseline, have {len(days)}")

    screen = MomentumScreen(lookback=60, horizon_days=horizon)
    strat_rets, bh_rets, fees, fills, used = [], [], 0.0, 0, 0
    for w in range(windows):
        hi = len(days) - w * WINDOW_DAYS
        lo = hi - WINDOW_DAYS
        if lo < 0:
            break
        window = days[lo:hi]
        arm = run_arm(series, idx, baseline, window, every=STRATEGY_EVERY,
                      exit_on_falsifier=False, screen=screen)
        bh = equal_weight_buy_hold(series, idx, window)
        if bh is None:
            continue
        strat_rets.append(arm.ret)
        bh_rets.append(bh.ret)
        fees += arm.fees
        fills += arm.fills
        used += 1

    if not used:
        raise ValueError("no window produced a result")
    return Reading(
        windows=used,
        strategy_ret=sum(strat_rets) / used,
        buy_hold_ret=sum(bh_rets) / used,
        strategy_fees=fees / used,
        fills=fills // used,
    )


def format_reading(now: Reading, before: Reading | None) -> str:
    verdict = "BEATS buy & hold" if now.gap > 0 else "BEHIND buy & hold"
    lines = [
        f"bottom line (paper, {now.windows} window{'s' if now.windows != 1 else ''} "
        f"of {WINDOW_DAYS}d, ${START_CASH:,.0f}):",
        f"  strategy   {now.strategy_ret:+7.2%}   "
        f"cost ${now.strategy_fees:,.0f}   {now.fills} fills",
        f"  buy & hold {now.buy_hold_ret:+7.2%}",
        f"  gap        {now.gap:+7.2%}   {verdict}",
    ]
    if before is not None:
        moved = now.gap - before.gap
        arrow = "no change" if abs(moved) < 5e-5 else ("closer" if moved > 0 else "further")
        lines.append(f"  vs baseline {moved:+7.2%} ({arrow}; baseline gap {before.gap:+.2%})")
    lines.append(
        "  Paper fills, survivor-only universe, and a small recent sample that a "
        "change can be fit to.\n  The gap is the goal; the level is not evidence."
    )
    return "\n".join(lines)


def load_baseline(path: Path) -> Reading | None:
    try:
        return Reading.from_dict(json.loads(path.read_text()))
    except (OSError, ValueError, KeyError):
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Strategy vs buy and hold, net of cost.")
    p.add_argument("--prices", default="data/prices")
    p.add_argument("--windows", type=int, default=1)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--baseline", default=str(BASELINE_PATH))
    p.add_argument("--save-baseline", action="store_true")
    p.add_argument("--quiet-if-missing", action="store_true",
                   help="exit 0 with a short note when prices are absent (for hooks)")
    args = p.parse_args(argv)

    try:
        now = measure(args.prices, args.windows, args.horizon)
    except Exception as exc:  # noqa: BLE001
        # A hook must never take the session down, and price data is gitignored,
        # so its absence is the normal state of a fresh clone rather than a fault.
        note = f"bottom line: unavailable ({type(exc).__name__}: {exc})"
        if args.quiet_if_missing:
            print(note)
            return 0
        print(note, file=sys.stderr)
        return 1

    path = Path(args.baseline)
    print(format_reading(now, load_baseline(path)))
    if args.save_baseline:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(now.to_dict(), indent=2) + "\n")
        print(f"  baseline written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
