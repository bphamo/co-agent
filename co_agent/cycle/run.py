"""One weekly cycle, and the daily loop that carries positions between them.

snapshot -> candidates -> FR9 gate -> decisions -> paper fills, then a daily pass
that resolves falsifiers and closes what they settle.

The falsifier is the exit. A thesis says "this does not fall X% within N days";
when it does, the position is sold and the thesis is recorded false. That keeps
the falsifier operationally real instead of decorative, and it means the ledger
and the account cannot disagree about what happened.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date

from ..paper import PaperBroker
from ..paper.broker import MAX_COMMISSION
from ..sim import Config, History, Interval, Request, Verdict, run
from ..sim.params import Drift
from .candidates import Candidate, CandidateSource
from .ledger import Ledger
from .scorer import resolve
from .snapshot import Snapshot, build_snapshot

#: Volatility conditioning helps at horizons of 5-20 days and hurts at 60-120,
#: measured on the synthetic panel. The policy lives here so a horizon change
#: cannot silently pick the wrong side of that crossover.
CONDITION_VOL_UP_TO_DAYS = 20


def config_for_horizon(horizon_days: int, **overrides) -> Config:
    return Config(
        cond_vol=horizon_days <= CONDITION_VOL_UP_TO_DAYS,
        interval=Interval.MC,
        drift=Drift.SHRUNK,
        **overrides,
    )


@dataclass(slots=True)
class OpenThesis:
    thesis_id: str
    symbol: str
    entry_date: date
    entry_close: float
    candidate: Candidate
    null_probability: float
    weight: float
    observed: list[tuple[date, float]] = field(default_factory=list)


@dataclass(slots=True)
class CycleReport:
    taken_at: date
    snapshot_digest: str
    proposed: int
    accepted: int
    rejected_too_easy: int
    rejected_too_hard: int
    indeterminate: int
    presented: int
    opened: int
    equity: float
    open_after: int = 0


@dataclass(slots=True)
class Engine:
    series: dict
    baseline_by_date: dict[date, float]
    source: CandidateSource
    broker: PaperBroker
    ledger: Ledger
    history_obs: int = 1600
    batch_cap: int = 6
    #: Cap on concurrent open theses. FR5's batch cap bounds how many rows a
    #: human sees in one sitting; it does not bound exposure. With a 20-day
    #: horizon and a weekly cycle, entries outpace exits and three batches'
    #: worth of positions accumulate -- measured at 10 open against a batch cap
    #: of 6 before this existed.
    max_open: int = 8
    per_position_drawdown_limit: float = 0.025
    open_theses: dict[str, OpenThesis] = field(default_factory=dict)
    _index: dict[str, dict[date, int]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        # PriceSeries is frozen, so the date index lives here rather than being
        # attached to it.
        self._index = {s: {d: i for i, d in enumerate(px.dates)} for s, px in self.series.items()}

    # ------------------------------------------------------------------ daily

    def closes_on(self, day: date) -> dict[str, float]:
        out = {}
        for symbol, px in self.series.items():
            i = self._index[symbol].get(day)
            if i is not None:
                out[symbol] = float(px.closes[i])
        return out

    def settle(self, day: date) -> list[str]:
        """Resolve every open thesis against today, closing the ones that settle."""
        closes = self.closes_on(day)
        settled: list[str] = []
        for thesis_id, open_thesis in list(self.open_theses.items()):
            close = closes.get(open_thesis.symbol)
            if close is None:
                continue
            open_thesis.observed.append((day, close))
            outcome = resolve(
                open_thesis.candidate.falsifier,
                open_thesis.entry_close,
                open_thesis.observed,
                open_thesis.candidate.horizon_days,
            )
            if outcome.result == "unresolvable":
                continue

            self.broker.sell(open_thesis.symbol, day, close)
            self.ledger.append(
                "outcomes",
                {
                    "thesis_id": thesis_id,
                    "resolved_at": day,
                    "result": outcome.result,
                    "evidence": outcome.evidence,
                    "blinded": outcome.blinded,
                    "resolver": "code",
                    "sessions_observed": len(open_thesis.observed),
                },
            )
            del self.open_theses[thesis_id]
            settled.append(thesis_id)
        return settled

    # ------------------------------------------------------------------ weekly

    def run_cycle(self, day: date) -> CycleReport:
        snapshot = build_snapshot(
            self.series,
            day,
            history_obs=self.history_obs,
            baseline_by_date=self.baseline_by_date,
        )
        batch_id = str(uuid.uuid4())
        self.ledger.append(
            "snapshots",
            {
                "taken_at": day,
                "digest": snapshot.digest,
                "universe_size": len(snapshot.universe),
                "missing": list(snapshot.missing),
                "baseline_drift": snapshot.baseline_drift,
            },
        )

        # Over-generate: the gate rejects most falsifiers written outside the
        # band, so proposing exactly batch_cap would present a half-empty batch.
        proposed = self.source.propose(snapshot, self.batch_cap * 3)
        accepted: list[tuple[Candidate, object]] = []
        counts = {"reject_too_easy": 0, "reject_too_hard": 0, "indeterminate": 0}

        for candidate in proposed:
            if candidate.symbol in self.open_theses_by_symbol():
                continue
            result = run(
                Request(
                    history=History(candidate.symbol, 0, snapshot.histories[candidate.symbol]),
                    horizon_days=candidate.horizon_days,
                    falsifier=candidate.falsifier,
                    baseline_drift=snapshot.baseline_drift,
                    config=config_for_horizon(candidate.horizon_days),
                    proposed_weight=1.0 / self.batch_cap,
                    per_position_drawdown_limit=self.per_position_drawdown_limit,
                )
            )
            self.ledger.append(
                "falsifier_gate_events",
                {
                    "batch_id": batch_id,
                    "symbol": candidate.symbol,
                    "falsifier": candidate.falsifier_text,
                    "null_probability": result.null_probability,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "verdict": str(result.verdict),
                    "sim_method": str(result.method),
                    "sim_params": result.params.to_dict(),
                },
            )
            if result.verdict is Verdict.ACCEPT:
                accepted.append((candidate, result))
            else:
                counts[str(result.verdict)] = counts.get(str(result.verdict), 0) + 1

        room = max(0, self.max_open - len(self.open_theses))
        slots = min(self.batch_cap, room)
        presented, not_presented = accepted[:slots], accepted[slots:]
        opened = 0
        equity = self.broker.equity(snapshot.closes)

        for candidate, result in presented:
            thesis_id = str(uuid.uuid4())
            self.ledger.append(
                "theses",
                {
                    "id": thesis_id,
                    "batch_id": batch_id,
                    "snapshot_digest": snapshot.digest,
                    "symbol": candidate.symbol,
                    "claim": candidate.claim,
                    "mechanism": candidate.mechanism,
                    "falsifier": candidate.falsifier_text,
                    "falsifier_class": "price_path",
                    "snapshot_taken_at": day,
                    "horizon_days": candidate.horizon_days,
                    "confidence": candidate.confidence,
                    "null_probability": result.null_probability,
                    "p95_drawdown": result.p95_drawdown_unit,
                    "proposed_weight": result.sizing.weight,
                    "weight_reduced": result.sizing.reduced,
                    "sim_method": str(result.method),
                    "sim_params": result.params.to_dict(),
                    "source": candidate.source,
                },
            )
            # No human in this arm. Recorded as a policy, not as a person, so
            # the approved-vs-passed comparison (G4) is never computed from it.
            self.ledger.append(
                "decisions",
                {
                    "thesis_id": thesis_id,
                    "batch_id": batch_id,
                    "decision": "approved",
                    "policy": "auto_approve_paper",
                    "human_confidence": None,
                    "decided_at": day,
                },
            )
            # Size against equity but spend only cash. Earlier cycles' positions
            # hold most of the account, so a target computed from equity alone
            # tries to spend money that is already invested.
            close = snapshot.closes[candidate.symbol]
            target = equity * min(result.sizing.weight, 1.0 / self.max_open)
            budget = min(target, max(0.0, self.broker.cash - MAX_COMMISSION))
            if budget < close:
                self.ledger.append(
                    "decisions",
                    {"thesis_id": thesis_id, "batch_id": batch_id,
                     "decision": "not_filled", "reason": "insufficient cash",
                     "cash": self.broker.cash, "close": close, "decided_at": day},
                )
                continue
            fill = self.broker.buy(candidate.symbol, day, close, budget)
            if fill is None:
                continue
            self.ledger.append(
                "fills",
                {
                    "thesis_id": thesis_id, "symbol": candidate.symbol, "side": "buy",
                    "filled_at": day, "qty": fill.qty, "price": fill.price,
                    "close": fill.close, "fees": fill.fees,
                },
            )
            self.open_theses[thesis_id] = OpenThesis(
                thesis_id=thesis_id, symbol=candidate.symbol, entry_date=day,
                entry_close=fill.price, candidate=candidate,
                null_probability=result.null_probability, weight=result.sizing.weight,
            )
            opened += 1

        for candidate, _ in not_presented:
            self.ledger.append(
                "decisions",
                {"thesis_id": None, "batch_id": batch_id, "symbol": candidate.symbol,
                 "decision": "not_presented", "human_confidence": None, "decided_at": day},
            )

        return CycleReport(
            taken_at=day, snapshot_digest=snapshot.digest, proposed=len(proposed),
            accepted=len(accepted), rejected_too_easy=counts.get("reject_too_easy", 0),
            rejected_too_hard=counts.get("reject_too_hard", 0),
            indeterminate=counts.get("indeterminate", 0), presented=len(presented),
            opened=opened, equity=equity,
        )

    def open_theses_by_symbol(self) -> set[str]:
        return {t.symbol for t in self.open_theses.values()}
