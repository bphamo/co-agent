"""Tests for the paper broker.

The frictions are what these check. A paper broker that fills at the close for
free is the most reliable way to make a strategy look good, so the tests pin
that it does not.
"""

from __future__ import annotations

from datetime import date

import pytest

from co_agent.paper import InsufficientCash, PaperBroker, commission

DAY = date(2026, 1, 5)


def test_buys_whole_shares_and_pays_the_spread():
    b = PaperBroker(cash=10_000, slippage_bps=10)
    fill = b.buy("X.TO", DAY, 150.0, 2_000)
    assert fill is not None
    assert fill.qty == 13  # 2000 / 150.15, floored
    assert fill.price > fill.close, "a buy should not fill at the close"
    assert fill.price == pytest.approx(150.0 * 1.001)
    assert b.cash == pytest.approx(10_000 - 13 * fill.price - fill.fees)


def test_sells_into_the_spread_too():
    b = PaperBroker(cash=10_000, slippage_bps=10)
    b.buy("X.TO", DAY, 150.0, 2_000)
    fill = b.sell("X.TO", DAY, 150.0)
    assert fill is not None and fill.price < fill.close
    assert "X.TO" not in b.positions
    # Round-tripping at an unchanged price must lose money.
    assert b.cash < 10_000


def test_slippage_defaults_to_nonzero():
    """Zero-friction fills are the lie FR9 is guarding against."""
    assert PaperBroker().slippage_bps > 0


@pytest.mark.parametrize(
    "qty,expected", [(1, 4.95), (100, 4.95), (700, 7.0), (5_000, 9.95)]
)
def test_commission_is_floored_and_capped(qty, expected):
    assert commission(qty) == pytest.approx(expected)


def test_an_unaffordable_order_is_refused_not_silently_shrunk_to_zero():
    b = PaperBroker(cash=50)
    with pytest.raises(InsufficientCash):
        b.buy("X.TO", DAY, 150.0, 1_000)


def test_a_large_order_is_trimmed_to_available_cash():
    b = PaperBroker(cash=1_000, slippage_bps=0)
    fill = b.buy("X.TO", DAY, 100.0, 5_000)
    assert fill is not None and fill.qty <= 10
    assert b.cash >= 0


def test_selling_nothing_is_a_no_op():
    assert PaperBroker().sell("X.TO", DAY, 100.0) is None


def test_partial_sale_leaves_a_proportional_cost_basis():
    b = PaperBroker(cash=10_000, slippage_bps=0)
    b.buy("X.TO", DAY, 100.0, 5_000)
    basis_before = b.positions["X.TO"].cost_basis
    b.sell("X.TO", DAY, 100.0, qty=25)
    held = b.positions["X.TO"]
    assert held.qty == 25
    assert held.cost_basis == pytest.approx(basis_before / 2)


def test_equity_marks_a_missing_price_at_cost_rather_than_dropping_it():
    """A halted name must not quietly vanish from the account's value."""
    b = PaperBroker(cash=10_000, slippage_bps=0)
    b.buy("X.TO", DAY, 100.0, 1_000)
    with_price, without = b.equity({"X.TO": 100.0}), b.equity({})
    # The fallback is average cost, which carries the entry fee, so the two are
    # close but not equal. What matters is that the position is still counted.
    assert without > b.cash
    assert without == pytest.approx(with_price, abs=10.0)


def test_exposure_sums_to_the_invested_share():
    b = PaperBroker(cash=10_000, slippage_bps=0)
    b.buy("A.TO", DAY, 100.0, 2_000)
    b.buy("B.TO", DAY, 50.0, 2_000)
    closes = {"A.TO": 100.0, "B.TO": 50.0}
    exposure = b.exposure(closes)
    assert 0.3 < sum(exposure.values()) < 0.45
    assert set(exposure) == {"A.TO", "B.TO"}


def test_fees_and_slippage_are_reported_separately():
    """They are different costs with different fixes; a single number hides that."""
    b = PaperBroker(cash=10_000, slippage_bps=25)
    b.buy("X.TO", DAY, 100.0, 5_000)
    b.sell("X.TO", DAY, 100.0)
    assert b.total_fees == pytest.approx(9.9)
    assert b.slippage_cost > 0
