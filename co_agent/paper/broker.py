"""A paper broker: virtual cash, real prices, honest frictions.

There is no broker API here and that is the point. Paper trading removes the
Questrade dependency entirely -- no token rotation, no account calls, no
reconciliation loop -- which is why it can run before any of that exists.

**What paper fills cannot tell you.** Every fill here happens at a daily close
at a modelled cost. Real fills happen at a price someone else was willing to
take, in a size the book could absorb, at a moment the market had moved. So:

* `slippage_bps` defaults to 10, not 0. A paper broker with frictionless fills
  is the single most reliable way to make a strategy look good, and FR9 bans
  simulated equity curves precisely because they flatter.
* Commission defaults to Questrade-like flat pricing. On a $10,000 account this
  dominates: a measured 251-day daily-rebalance run paid $1,168 in commissions,
  12% of capital, and turned +7.9% gross into -3.7% net.
* Partial fills, liquidity limits and gaps are not modelled at all. Nothing here
  should be read as evidence about execution; the `execution` attribution bucket
  stays empty until real fills exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

#: Questrade-like: one cent per share, floored and capped.
MIN_COMMISSION, MAX_COMMISSION, PER_SHARE = 4.95, 9.95, 0.01


class InsufficientCash(RuntimeError):
    """The order costs more than the account holds."""


@dataclass(frozen=True, slots=True)
class Fill:
    symbol: str
    side: str  # "buy" | "sell"
    when: date
    qty: float
    price: float  # the price actually paid, after slippage
    close: float  # the day's close, before slippage
    fees: float

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass(slots=True)
class Position:
    symbol: str
    qty: float
    cost_basis: float  # total paid, including fees

    @property
    def average_price(self) -> float:
        return self.cost_basis / self.qty if self.qty else 0.0


def commission(qty: float) -> float:
    return min(MAX_COMMISSION, max(MIN_COMMISSION, qty * PER_SHARE))


@dataclass(slots=True)
class PaperBroker:
    """Virtual cash and positions, marked against real closes."""

    cash: float = 10_000.0
    slippage_bps: float = 10.0
    #: Multiplier on the commission schedule. 1.0 is the Questrade-like default
    #: and is what every reported result uses. `cycle/frictions.py` varies it to
    #: ask how far a conclusion depends on the schedule -- which is the opposite
    #: of picking a kinder one, and the only reason this knob exists.
    commission_scale: float = 1.0
    positions: dict[str, Position] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    starting_cash: float = field(init=False)

    def __post_init__(self) -> None:
        self.starting_cash = self.cash

    def _fee(self, qty: float) -> float:
        return commission(qty) * self.commission_scale

    @property
    def _fee_ceiling(self) -> float:
        """The most one order can be charged, used to reserve cash before trimming.

        Scaled like the fee itself: reserving the unscaled ceiling while charging
        a scaled fee lets a buy overdraw the account, which is the one thing the
        broker must never do.
        """
        return MAX_COMMISSION * self.commission_scale

    # ------------------------------------------------------------------ orders

    def buy(self, symbol: str, when: date, close: float, notional: float) -> Fill | None:
        """Spend up to ``notional`` on whole shares of ``symbol``."""
        if close <= 0:
            raise ValueError(f"{symbol}: non-positive close {close}")
        price = close * (1 + self.slippage_bps / 10_000)
        qty = float(int(notional // price))
        if qty <= 0:
            return None
        fees = self._fee(qty)
        total = qty * price + fees
        if total > self.cash:
            qty = float(int((self.cash - self._fee_ceiling) // price))
            if qty <= 0:
                raise InsufficientCash(
                    f"{symbol}: ${self.cash:,.2f} cash cannot buy one share at ${price:,.2f}"
                )
            fees = self._fee(qty)
            total = qty * price + fees

        self.cash -= total
        held = self.positions.get(symbol)
        if held is None:
            self.positions[symbol] = Position(symbol, qty, total)
        else:
            held.qty += qty
            held.cost_basis += total
        fill = Fill(symbol, "buy", when, qty, price, close, fees)
        self.fills.append(fill)
        return fill

    def sell(self, symbol: str, when: date, close: float, qty: float | None = None) -> Fill | None:
        """Sell ``qty`` shares, or the whole position when ``qty`` is None."""
        held = self.positions.get(symbol)
        if held is None or held.qty <= 0:
            return None
        qty = held.qty if qty is None else min(qty, held.qty)
        price = close * (1 - self.slippage_bps / 10_000)
        fees = self._fee(qty)
        self.cash += qty * price - fees

        held.cost_basis *= 1 - qty / held.qty
        held.qty -= qty
        if held.qty <= 0:
            del self.positions[symbol]
        fill = Fill(symbol, "sell", when, qty, price, close, fees)
        self.fills.append(fill)
        return fill

    # ------------------------------------------------------------------ marking

    def equity(self, closes: dict[str, float]) -> float:
        """Cash plus positions marked at the given closes.

        A position with no close available is marked at its average cost rather
        than dropped, so a halted or missing name cannot quietly vanish from the
        account's value.
        """
        held = sum(
            p.qty * closes.get(s, p.average_price) for s, p in self.positions.items()
        )
        return self.cash + held

    def exposure(self, closes: dict[str, float]) -> dict[str, float]:
        """Each position's share of equity."""
        total = self.equity(closes)
        if total <= 0:
            return {}
        return {
            s: p.qty * closes.get(s, p.average_price) / total
            for s, p in self.positions.items()
        }

    @property
    def total_fees(self) -> float:
        return sum(f.fees for f in self.fills)

    @property
    def slippage_cost(self) -> float:
        """What the modelled spread cost, separately from commission."""
        return sum(abs(f.price - f.close) * f.qty for f in self.fills)
