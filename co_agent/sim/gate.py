"""The FR9 gates: which falsifiers carry information, and how big a position."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .params import SimInputError


class Verdict(StrEnum):
    """The gate's answer, recorded in ``falsifier_gate_events``."""

    #: Neither a near-certainty nor a near-impossibility, so resolving it
    #: carries information.
    ACCEPT = "accept"
    #: Trips by chance too often.  This is the base-rate drift the TRD's risk
    #: table names: easy falsifiers inflate Brier with no research behind them.
    REJECT_TOO_EASY = "reject_too_easy"
    #: Almost never trips by chance, so a "false" outcome says nothing about
    #: the claim.
    REJECT_TOO_HARD = "reject_too_hard"
    #: The estimate's confidence interval straddles a band edge, so the
    #: simulator cannot tell which side of the gate this falsifier is on.
    #:
    #: This is the hysteresis.  Without it, a falsifier whose true null
    #: probability sits at 0.30 is accepted or bounced depending on Monte Carlo
    #: noise, and restating it is a coin flip the researcher will read as
    #: signal.
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class Band:
    """The acceptable null-probability range.

    FR9 sets it at 0.3-0.7.  Open question 7 asks whether it should widen for
    high-conviction theses, so it is a parameter rather than a constant, and it
    is recorded in ``sim_params``.
    """

    low: float = 0.3
    high: float = 0.7

    def validate(self) -> None:
        if not (0 < self.low < self.high < 1):
            raise SimInputError("band must satisfy 0 < low < high < 1")

    def judge(self, p: float, ci_low: float, ci_high: float) -> Verdict:
        """Apply the band to a point estimate and its confidence interval.

        A verdict is only returned when the interval settles the question::

            ci entirely below low   -> too hard
            ci entirely above high  -> too easy
            ci entirely inside band -> accept
            otherwise               -> indeterminate

        Callers must not treat ``INDETERMINATE`` as a rejection.  :func:`run`
        escalates the path count first; if the interval still straddles an edge,
        the true value is genuinely near the boundary and the decision belongs
        to a human, not to the next Monte Carlo seed.
        """
        if ci_high < self.low:
            return Verdict.REJECT_TOO_HARD
        if ci_low > self.high:
            return Verdict.REJECT_TOO_EASY
        if ci_low >= self.low and ci_high <= self.high:
            return Verdict.ACCEPT
        return Verdict.INDETERMINATE


#: 97.5th percentile of the standard normal.
_Z = 1.959963984540054


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for ``k`` successes in ``n`` trials.

    Preferred over the normal approximation because the band edges sit where
    the normal interval is least well behaved at small path counts.
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    z2 = _Z * _Z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = _Z / denom * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True, slots=True)
class Sizing:
    """FR9's second gate: sizing is never proposed by an agent."""

    proposed_weight: float
    weight: float
    reduced: bool
    p95_drawdown_at_weight: float
    per_position_limit: float


def cap_weight(
    p95_drawdown_unit: float,
    proposed_weight: float,
    per_position_limit: float,
) -> Sizing:
    """Reduce a proposed weight until its 95th-percentile drawdown fits the limit.

    Drawdown at weight *w* is linear in *w* for a single position, so the
    reduction is exact.  It does **not** bound *portfolio* drawdown: six
    correlated names each inside the per-position limit can breach it together.
    FR5's sector and currency caps are what stand between this figure and that
    case, and Appendix A excludes the correlation analysis that would close the
    gap properly.
    """
    if p95_drawdown_unit <= 0 or per_position_limit <= 0:
        return Sizing(
            proposed_weight=proposed_weight,
            weight=proposed_weight,
            reduced=False,
            p95_drawdown_at_weight=proposed_weight * max(0.0, p95_drawdown_unit),
            per_position_limit=per_position_limit,
        )

    weight, reduced = proposed_weight, False
    if proposed_weight * p95_drawdown_unit > per_position_limit:
        weight = per_position_limit / p95_drawdown_unit
        reduced = True
    return Sizing(
        proposed_weight=proposed_weight,
        weight=weight,
        reduced=reduced,
        p95_drawdown_at_weight=weight * p95_drawdown_unit,
        per_position_limit=per_position_limit,
    )
