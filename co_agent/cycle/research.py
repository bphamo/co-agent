"""The model-driven candidate source.

The one component that could plausibly carry directional edge. Everything else
in this package measures or sizes; nothing else predicts. The momentum screen it
sits alongside is the benchmark it has to beat -- if this does not outperform a
rule anyone could write in an afternoon, the ledger should say so.

**Contamination.** Run forward, this is clean: the model cannot have seen a date
that has not happened. Run over historical snapshots it is not, and no prompt
instruction fixes that, because the leakage is in the weights rather than in the
tool layer. `propose` therefore refuses a snapshot older than `earliest_date`
unless the caller explicitly opts in, so a replay cannot quietly become evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
from pydantic import BaseModel, Field

from ..sim.falsifier import TouchBelow
from .candidates import Candidate, suggested_threshold
from .snapshot import Snapshot

DEFAULT_MODEL = "claude-opus-5"

#: Drop thresholds outside this band are almost never inside the FR9 gate at any
#: horizon we trade, so a thesis written outside it is rejected before it costs a
#: simulation.
MIN_DROP, MAX_DROP = 0.01, 0.20


class ThesisDraft(BaseModel):
    """One thesis, in the FR3 fields and nothing else."""

    symbol: str = Field(description="Ticker, exactly as given in the candidate table.")
    claim: str = Field(description="What you assert will hold, in one sentence.")
    mechanism: str = Field(
        description="Why it should hold. Name the cause, not the observation."
    )
    drop_threshold: float = Field(
        description=(
            "Falsifier threshold as a decimal fraction: 0.06 means the thesis is "
            "false if the name closes 6% or more below the snapshot close on any "
            "day within the horizon."
        )
    )
    confidence: float = Field(
        description="Probability the claim survives the horizon, 0 to 1."
    )


class ThesisBatch(BaseModel):
    theses: list[ThesisDraft]


@dataclass(slots=True)
class LLMResearch:
    """Proposes theses from a frozen snapshot, via the Messages API."""

    client: object
    model: str = DEFAULT_MODEL
    horizon_days: int = 60
    max_tokens: int = 16_000
    #: Snapshots before this are refused: the model's training data may cover
    #: them, so its "forecast" would be recall.
    earliest_date: date | None = None
    allow_contaminated: bool = False
    _last_usage: dict = field(default_factory=dict, init=False)

    @property
    def name(self) -> str:
        return f"research/{self.model}"

    # ------------------------------------------------------------------ prompt

    def _features(self, snapshot: Snapshot) -> str:
        rows = []
        for symbol in snapshot.universe:
            r = snapshot.histories[symbol]
            def ret(n: int) -> float:
                return float(np.exp(r[-n:].sum()) - 1)
            vol = float(r[-60:].std() * np.sqrt(252))
            rows.append(
                f"{symbol:<10} close {snapshot.closes[symbol]:>9.2f}  "
                f"1m {ret(21):>+7.1%}  3m {ret(63):>+7.1%}  12m {ret(252):>+7.1%}  "
                f"ann.vol {vol:>5.1%}"
            )
        return "\n".join(rows)

    def _prompt(self, snapshot: Snapshot, n: int) -> str:
        drop = suggested_threshold(self.horizon_days)
        return f"""You are proposing investment theses for a single private investor.

Snapshot date: {snapshot.taken_at}. Every figure below is computed only from
data available on that date. You have no information after it, and you must not
use any.

Universe as of the snapshot:

{self._features(snapshot)}

Propose exactly {n} theses, each on a different name from the table.

Each thesis must be falsifiable in these fields and nothing else:

- claim: what holds over the next {self.horizon_days} trading days.
- mechanism: why. Name a cause. "It has gone up" is an observation, not a
  mechanism, and a thesis whose only support is recent price is worth less than
  the momentum screen this is measured against.
- drop_threshold: the fraction below the snapshot close that, if closed through
  on any day in the horizon, makes the thesis false.
- confidence: your probability the claim survives, calibrated. You are scored on
  a Brier score against realised outcomes, so systematically high confidence
  costs you; so does hedging everything to 0.5, which scores as no information.

On the threshold: measured on this universe, a falsifier near {drop:.0%} at a
{self.horizon_days}-day horizon trips roughly half the time by chance, which is
where a falsifier carries the most information. Much tighter and it trips on
noise; much wider and surviving it tells us nothing. Choose per name based on
how volatile it is -- the table gives annualised volatility -- not by copying
one number across all {n}.

Pick names where you can state a real mechanism. Fewer good theses beat {n}
padded ones, but return exactly {n} so the batch can be compared like for like."""

    # ----------------------------------------------------------------- propose

    def propose(self, snapshot: Snapshot, n: int) -> list[Candidate]:
        if (
            self.earliest_date is not None
            and snapshot.taken_at < self.earliest_date
            and not self.allow_contaminated
        ):
            raise ValueError(
                f"snapshot {snapshot.taken_at} predates {self.earliest_date}: the model "
                "may have been trained on what happened next, so its output would be "
                "recall rather than forecast. Pass allow_contaminated=True only for a "
                "mechanical test whose result is not treated as evidence."
            )

        response = self.client.messages.parse(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": self._prompt(snapshot, n)}],
            output_format=ThesisBatch,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._last_usage = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
            }
        return self._to_candidates(response.parsed_output, snapshot)

    def _to_candidates(self, batch: ThesisBatch, snapshot: Snapshot) -> list[Candidate]:
        """Validate drafts into candidates, dropping what cannot be traded.

        A hallucinated ticker or an out-of-band threshold is dropped here rather
        than raising downstream: the batch is still usable, and the loss shows up
        as a short batch rather than a failed cycle.
        """
        out: list[Candidate] = []
        seen: set[str] = set()
        for draft in batch.theses:
            symbol = draft.symbol.strip().upper()
            if symbol not in snapshot.closes or symbol in seen:
                continue
            if not MIN_DROP <= draft.drop_threshold <= MAX_DROP:
                continue
            if not 0 <= draft.confidence <= 1:
                continue
            if not (draft.claim.strip() and draft.mechanism.strip()):
                continue
            seen.add(symbol)
            out.append(
                Candidate(
                    symbol=symbol,
                    claim=draft.claim.strip(),
                    mechanism=draft.mechanism.strip(),
                    falsifier_text=(
                        f"{symbol} closes {draft.drop_threshold:.1%} or more below its "
                        f"{snapshot.taken_at} close of {snapshot.closes[symbol]:.2f} on "
                        f"any of the next {self.horizon_days} trading days."
                    ),
                    falsifier=TouchBelow(draft.drop_threshold),
                    horizon_days=self.horizon_days,
                    confidence=draft.confidence,
                    source=self.name,
                )
            )
        return out
