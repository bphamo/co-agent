"""How much of the gap is the friction model, and how much is the strategy?

The goal is stated against a benchmark measured in simulation, so the honest
question is not "is the paper number right" -- it is not, exactly -- but "does
the paper number move the same way the real one would?" Without real fills that
cannot be measured directly. What can be measured is how far the answer depends
on the part that is modelled, which is the frictions.

The asymmetry is the point. Over a 252-day window the strategy pays about 144
commissions and the benchmark about 43, so raising frictions hurts the strategy
roughly three times as hard, and lowering them flatters it. If the gap keeps its
sign all the way down to a **frictionless** account -- no slippage, no
commission, the most generous execution that could possibly exist -- then no
error in the execution model explains it, and the conclusion survives whatever
real fills turn out to cost. If instead the gap closes somewhere inside the
plausible range, the finding is an artifact of the cost assumption and the
strategy is not actually behind.

Both arms always move together. A gap measured with one arm's frictions changed
is not a gap.

**This is not a knob for tuning.** Every reported result uses the defaults
(10bps, the Questrade-like schedule). The grid exists to bound a conclusion, and
the honest way to read it is the whole row, never the friendliest cell.

Run::

    python -m co_agent.cycle.frictions                  # slippage + commission sweeps
    python -m co_agent.cycle.frictions --windows 6      # steadier, slower
"""

from __future__ import annotations

import argparse
import statistics as st
from dataclasses import dataclass

from ..data.prices import load_dir
from ..sim.validate import market_drift_by_date
from .cadence import (
    HISTORY_OBS,
    WINDOW_DAYS,
    calendar,
    equal_weight_buy_hold,
    index_map,
    run_arm,
    window_starts,
)
from .candidates import MomentumScreen

#: Defaults in bold below. 0 is the frictionless corner, 50bps is punitive for
#: liquid large caps -- together they bracket anything real execution could cost.
SLIPPAGE_GRID = (0.0, 5.0, 10.0, 20.0, 30.0, 50.0)
COMMISSION_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
DEFAULT_SLIPPAGE, DEFAULT_COMMISSION = 10.0, 1.0
STRATEGY_EVERY = 21


@dataclass(frozen=True, slots=True)
class Row:
    label: str
    slippage: float
    commission: float
    strategy: float
    buy_hold: float
    windows: int
    is_default: bool
    #: True if the benchmark ran out of cash in any window at this setting.
    truncated: bool = False

    @property
    def gap(self) -> float:
        return self.strategy - self.buy_hold


def measure(
    series, idx, baseline, windows, screen, *, slippage: float, commission: float
) -> tuple[float, float, int, bool]:
    """Mean strategy return, mean benchmark return, and whether it stayed whole."""
    strat, bh, cut = [], [], False
    for window in windows:
        arm = run_arm(
            series, idx, baseline, window,
            every=STRATEGY_EVERY, exit_on_falsifier=False, screen=screen,
            slippage_bps=slippage, commission_scale=commission,
        )
        ref = equal_weight_buy_hold(
            series, idx, window, slippage_bps=slippage, commission_scale=commission
        )
        if ref is None:
            continue
        strat.append(arm.ret)
        bh.append(ref.ret)
        cut = cut or ref.truncated
    if not strat:
        raise ValueError("no window produced a result")
    return st.mean(strat), st.mean(bh), len(strat), cut


def format_rows(rows: list[Row]) -> str:
    out = [
        f"{'':<26}{'strategy':>10}{'buy & hold':>12}{'gap':>10}",
        "-" * 58,
    ]
    for r in rows:
        mark = "  <- default" if r.is_default else ""
        if r.truncated:
            mark += "  (! benchmark ran out of cash; not the whole universe)"
        out.append(
            f"{r.label:<26}{r.strategy:>9.2%}{r.buy_hold:>12.2%}{r.gap:>10.2%}{mark}"
        )
    return "\n".join(out)


def verdict(rows: list[Row]) -> str:
    """Does the conclusion survive the whole grid, or only part of it?"""
    gaps = [r.gap for r in rows]
    best = max(rows, key=lambda r: r.gap)
    if all(g < 0 for g in gaps):
        return (
            "The gap is negative at every setting, including a frictionless account.\n"
            f"  Best case for the strategy ({best.label}) still trails by {abs(best.gap):.2%}.\n"
            "  No error in the execution model explains being behind, so the finding\n"
            "  does not depend on what real fills turn out to cost."
        )
    if all(g > 0 for g in gaps):
        return (
            "The gap is positive at every setting, including the most punitive.\n"
            "  The conclusion does not rest on the friction assumption."
        )
    flips = [r for r in rows if (r.gap > 0) != (rows[0].gap > 0)]
    return (
        "The gap CHANGES SIGN inside the grid -- first at "
        f"{flips[0].label if flips else '?'}.\n"
        "  The conclusion is an artifact of the cost assumption, not a property of\n"
        "  the strategy. It cannot be reported without the friction setting beside it."
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Friction sensitivity of the gap.")
    p.add_argument("--prices", default="data/prices")
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--no-memo", action="store_true")
    args = p.parse_args(argv)

    if not args.no_memo:
        _install_memo()

    series = load_dir(args.prices)
    baseline = market_drift_by_date(series, HISTORY_OBS)
    idx = index_map(series)
    days = [d for d in calendar(series) if d in baseline]
    starts = window_starts(days)
    if not starts:
        p.error("no usable windows")

    # Evenly spaced rather than most-recent: a robustness claim made on one
    # regime is not a robustness claim.
    picked = [starts[round(i * (len(starts) - 1) / max(1, args.windows - 1))]
              for i in range(args.windows)]
    windows = [days[s : s + WINDOW_DAYS] for s in sorted(set(picked))]
    screen = MomentumScreen(lookback=60, horizon_days=args.horizon)

    print(
        f"friction sensitivity: {len(series)} symbols, {len(windows)} windows of "
        f"{WINDOW_DAYS}d\n  {windows[0][0]} -> {windows[-1][-1]}, "
        f"monthly cadence, hold to horizon\n"
    )

    rows: list[Row] = []
    for slip in SLIPPAGE_GRID:
        s_ret, b_ret, n, held = measure(
            series, idx, baseline, windows, screen,
            slippage=slip, commission=DEFAULT_COMMISSION,
        )
        rows.append(Row(
            f"slippage {slip:>4.0f}bps", slip, DEFAULT_COMMISSION, s_ret, b_ret, n,
            slip == DEFAULT_SLIPPAGE, held,
        ))
    print("commission at the default schedule, slippage varied")
    print(format_rows(rows), "\n")

    crows: list[Row] = []
    for scale in COMMISSION_GRID:
        s_ret, b_ret, n, held = measure(
            series, idx, baseline, windows, screen,
            slippage=DEFAULT_SLIPPAGE, commission=scale,
        )
        label = "commission free" if scale == 0 else f"commission x{scale:g}"
        crows.append(Row(label, DEFAULT_SLIPPAGE, scale, s_ret, b_ret, n,
                         scale == DEFAULT_COMMISSION, held))
    print("slippage at 10bps, commission schedule scaled")
    print(format_rows(crows), "\n")

    s_ret, b_ret, n, held = measure(
        series, idx, baseline, windows, screen, slippage=0.0, commission=0.0
    )
    free = Row("frictionless", 0.0, 0.0, s_ret, b_ret, n, False, held)
    print("the generous corner: no slippage, no commission, both arms")
    print(format_rows([free]), "\n")

    print(verdict(rows + crows + [free]))
    print(f"\n  {n} windows per row. Every reported result elsewhere uses the default\n"
          "  row; this grid bounds a conclusion and is not a setting to choose from.")
    return 0


def _install_memo() -> None:
    """Memoise the FR9 gate across friction settings.

    The gate is a pure function of price history, horizon, falsifier, baseline
    and config -- the broker is not an input to it. So every friction setting
    recomputes exactly the same verdicts, and reusing them is exact rather than
    approximate. Without this the grid is a dozen full backtests.
    """
    from . import run as run_mod

    real, memo = run_mod.run, {}

    def memoised(request):
        key = (
            request.history.symbol,
            hash(request.history.log_returns.tobytes()),
            request.horizon_days,
            repr(request.falsifier),
            round(float(request.baseline_drift), 12),
            round(float(request.proposed_weight), 12),
            round(float(request.per_position_drawdown_limit), 12),
            repr(request.config),
        )
        if key not in memo:
            memo[key] = real(request)
        return memo[key]

    run_mod.run = memoised


if __name__ == "__main__":
    raise SystemExit(main())
