"""Validating ``null_probability``: is the number the simulator reports correct?

Two harnesses, because "correct" has two available meanings.

**Bias study** (:func:`bias_study`).  On a process with a computable truth, run
the estimator against that truth many times.  Reports bias, error, whether the
Wilson interval covers truth at its nominal rate, and -- the figure that
actually matters -- how often the FR9 gate reaches the wrong verdict.  Needs no
market data, so it runs today.

**Historical walk-forward** (:func:`walk_forward`).  On a real series, compute
``null_probability`` at date *t* using only data up to *t*, then observe whether
the falsifier actually tripped over the following horizon.  Truth is unknowable
per observation, but predicted probability can be compared against realised
frequency in bins.  This is the instrument that says whether the estimator
works on the data it will actually see.

Why this exists at all: the 0.3-0.7 gate and FR7's "calibration net of
``null_probability``" both treat this number as a reference.  If it is biased,
the gate rejects sound falsifiers and every Brier skill score is wrong by an
unknown amount -- so the reference needs its own validation before it can carry
that weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .bootstrap import draws_to_levels
from .dgp import DGP
from .falsifier import Falsifier
from .gate import Band, Verdict
from .params import History, InsufficientHistoryError, SimInputError
from .simulate import Config, Request, run

# ---------------------------------------------------------------- known truth


def true_probability(
    dgp: DGP,
    state: object,
    falsifier: Falsifier,
    horizon: int,
    rng: np.random.Generator,
    paths: int = 40_000,
) -> float:
    """The falsifier's true trip probability under ``dgp``, given ``state``.

    Computed by simulating forward from the process itself, so it is a Monte
    Carlo estimate of truth rather than truth exactly.  ``paths`` is larger than
    the estimator's default for that reason: the reference should be tighter
    than the thing being measured.
    """
    levels = draws_to_levels(dgp.forward_paths(rng, state, horizon, paths))
    return float(falsifier.trips(levels).mean())


@dataclass(slots=True)
class Trial:
    """One estimate against its truth."""

    truth: float
    estimate: float
    ci_low: float
    ci_high: float
    verdict: Verdict


@dataclass(slots=True)
class BiasReport:
    """What a bias study found for one (process, falsifier, config) combination."""

    dgp: str
    dgp_params: dict[str, object]
    falsifier: dict[str, object]
    horizon_days: int
    history_obs: int
    cond_vol: bool
    trials: list[Trial] = field(default_factory=list)
    skipped: int = 0

    @property
    def n(self) -> int:
        return len(self.trials)

    @property
    def mean_truth(self) -> float:
        return float(np.mean([t.truth for t in self.trials])) if self.trials else float("nan")

    @property
    def bias(self) -> float:
        """Mean signed error.  Positive means the estimator overstates."""
        if not self.trials:
            return float("nan")
        return float(np.mean([t.estimate - t.truth for t in self.trials]))

    @property
    def mae(self) -> float:
        if not self.trials:
            return float("nan")
        return float(np.mean([abs(t.estimate - t.truth) for t in self.trials]))

    @property
    def coverage(self) -> float:
        """Fraction of trials whose 95% interval contains truth.

        Well below 0.95 means the interval is describing Monte Carlo noise while
        the real error is bias -- which makes the gate's hysteresis confident
        about the wrong thing.
        """
        if not self.trials:
            return float("nan")
        hits = [1 for t in self.trials if t.ci_low <= t.truth <= t.ci_high]
        return len(hits) / self.n

    def gate_errors(self, band: Band) -> dict[str, float]:
        """How often the gate reaches a verdict truth does not support.

        ``false_reject`` -- truth is inside the band and the gate rejected it: a
        sound falsifier sent back for restatement.
        ``false_accept`` -- truth is outside the band and the gate accepted it:
        an easy or impossible falsifier admitted to the ledger, which is the
        base-rate drift the TRD's risk table names.
        """
        inside = [t for t in self.trials if band.low <= t.truth <= band.high]
        outside = [t for t in self.trials if not band.low <= t.truth <= band.high]
        rejects = (Verdict.REJECT_TOO_EASY, Verdict.REJECT_TOO_HARD)
        return {
            "n_truth_inside": len(inside),
            "n_truth_outside": len(outside),
            "false_reject": (
                sum(t.verdict in rejects for t in inside) / len(inside) if inside else float("nan")
            ),
            "false_accept": (
                sum(t.verdict is Verdict.ACCEPT for t in outside) / len(outside)
                if outside
                else float("nan")
            ),
            "indeterminate": sum(t.verdict is Verdict.INDETERMINATE for t in self.trials) / self.n
            if self.trials
            else float("nan"),
        }


def bias_study(
    dgp: DGP,
    falsifier: Falsifier,
    *,
    horizon_days: int = 60,
    history_obs: int = 1600,
    trials: int = 100,
    config: Config | None = None,
    band: Band | None = None,
    truth_paths: int = 40_000,
    seed: int = 12345,
) -> BiasReport:
    """Measure the estimator against a process whose truth can be computed.

    Each trial draws a fresh history from ``dgp``, computes the falsifier's true
    probability conditional on the state that history ended in, then runs the
    estimator on the history alone and records both.

    The per-trial estimator seed varies with the trial, because a fixed seed
    across trials would make the Monte Carlo error correlated and understate the
    spread.
    """
    cfg = (config or Config()).validated()
    band = band or Band()
    report = BiasReport(
        dgp=dgp.name,
        dgp_params=dgp.params(),
        falsifier=falsifier.spec(),
        horizon_days=horizon_days,
        history_obs=history_obs,
        cond_vol=cfg.cond_vol,
    )

    for i in range(trials):
        rng = np.random.default_rng([seed, i])
        returns, state = dgp.simulate(rng, history_obs)
        truth = true_probability(dgp, state, falsifier, horizon_days, rng, truth_paths)

        request = Request(
            history=History(symbol=dgp.name, symbol_id=0, log_returns=returns),
            horizon_days=horizon_days,
            falsifier=falsifier,
            band=band,
            # Vary the estimator's seed per trial; everything else is held.
            config=replace(cfg, seed=cfg.seed + i),
        )
        try:
            result = run(request)
        except (InsufficientHistoryError, SimInputError):
            # A process can produce a history the estimator refuses (too few
            # usable observations after the warmup).  Count it rather than
            # dropping it silently: a high skip rate is itself a finding.
            report.skipped += 1
            continue

        report.trials.append(
            Trial(
                truth=truth,
                estimate=result.null_probability,
                ci_low=result.ci_low,
                ci_high=result.ci_high,
                verdict=result.verdict,
            )
        )
    return report


# ------------------------------------------------------- historical walk-forward


@dataclass(slots=True)
class Observation:
    """One walk-forward prediction and what actually happened."""

    symbol: str
    origin: int
    predicted: float
    tripped: bool


def non_overlapping_origins(n_obs: int, history_obs: int, horizon: int) -> list[int]:
    """Origin indices spaced so their outcome windows do not overlap.

    Sampling every day would give thousands of observations whose outcomes are
    almost the same event.  Effective sample size is roughly span / horizon, and
    pretending otherwise is the independence error that makes a reliability table
    look far more precise than it is.
    """
    if horizon <= 0:
        raise SimInputError("horizon must be > 0")
    return list(range(history_obs, n_obs - horizon + 1, horizon))


def walk_forward(
    symbol: str,
    log_returns: np.ndarray,
    falsifier: Falsifier,
    *,
    horizon_days: int = 60,
    history_obs: int = 1600,
    config: Config | None = None,
    overlapping: bool = False,
) -> list[Observation]:
    """Predict at each origin using only prior data, then observe the outcome.

    The estimator never sees data at or beyond its origin -- that is the whole
    point, and it is enforced here by slicing the series rather than by
    convention.
    """
    cfg = (config or Config()).validated()
    returns = np.asarray(log_returns, dtype=np.float64)
    step = 1 if overlapping else horizon_days
    origins = (
        list(range(history_obs, returns.size - horizon_days + 1, step))
        if overlapping
        else non_overlapping_origins(returns.size, history_obs, horizon_days)
    )

    out: list[Observation] = []
    for origin in origins:
        past = returns[origin - history_obs : origin]
        future = returns[origin : origin + horizon_days]

        try:
            result = run(
                Request(
                    history=History(symbol=symbol, symbol_id=0, log_returns=past),
                    horizon_days=horizon_days,
                    falsifier=falsifier,
                    config=cfg,
                )
            )
        except (InsufficientHistoryError, SimInputError):
            continue

        realised = draws_to_levels(future.reshape(1, -1))
        out.append(
            Observation(
                symbol=symbol,
                origin=origin,
                predicted=result.null_probability,
                tripped=bool(falsifier.trips(realised)[0]),
            )
        )
    return out


@dataclass(slots=True)
class Bin:
    """One row of a reliability table."""

    low: float
    high: float
    n: int
    mean_predicted: float
    realised: float
    ci_low: float
    ci_high: float


def reliability(
    observations: list[Observation],
    edges: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    rng_seed: int = 7,
    boot: int = 2000,
) -> list[Bin]:
    """Predicted probability against realised frequency, by bin.

    The interval on the realised frequency comes from a bootstrap over
    observations rather than a binomial formula, because walk-forward outcomes
    across symbols are cross-correlated -- a market-wide drawdown trips many at
    once -- and a binomial interval would claim precision the sample does not
    have.  With one symbol it reduces to roughly the binomial answer.

    Every row carries ``n``: FR7 requires a sample size beside every figure, and
    a reliability table is the easiest place to forget it.
    """
    rng = np.random.default_rng(rng_seed)
    rows: list[Bin] = []
    for low, high in zip(edges[:-1], edges[1:]):
        # Upper edge inclusive only in the last bin, so bins partition [0, 1].
        last = high == edges[-1]

        def in_bin(p: float, low: float = low, high: float = high, last: bool = last) -> bool:
            return low <= p <= high if last else low <= p < high

        members = [o for o in observations if in_bin(o.predicted)]
        if not members:
            rows.append(Bin(low, high, 0, float("nan"), float("nan"), float("nan"), float("nan")))
            continue

        outcomes = np.array([o.tripped for o in members], dtype=float)
        draws = rng.integers(0, outcomes.size, size=(boot, outcomes.size))
        means = outcomes[draws].mean(axis=1)
        rows.append(
            Bin(
                low=low,
                high=high,
                n=outcomes.size,
                mean_predicted=float(np.mean([o.predicted for o in members])),
                realised=float(outcomes.mean()),
                ci_low=float(np.quantile(means, 0.025)),
                ci_high=float(np.quantile(means, 0.975)),
            )
        )
    return rows


# ------------------------------------------------------------------ reporting


def format_bias_table(reports: list[BiasReport], band: Band) -> str:
    """One row per process, plus the gate-error columns that decide anything."""
    head = (
        f"{'process':<15}{'n':>4}{'truth':>8}{'bias':>9}{'MAE':>8}"
        f"{'coverage':>10}{'width':>8}"
        f"{'in/out':>9}{'false rej':>11}{'false acc':>11}{'indet':>8}"
    )
    lines = [head, "-" * len(head)]
    for r in reports:
        if not r.trials:
            lines.append(f"{r.dgp:<15}{0:>4}  (every history refused: {r.skipped} skipped)")
            continue
        e = r.gate_errors(band)
        width = sum(t.ci_high - t.ci_low for t in r.trials) / r.n
        lines.append(
            f"{r.dgp:<15}{r.n:>4}{r.mean_truth:>8.3f}{r.bias:>+9.4f}{r.mae:>8.4f}"
            f"{r.coverage:>10.2f}{width:>8.4f}"
            # Rates over a handful of trials are not evidence; the denominators
            # belong next to them (FR7 asks the same of every reported figure).
            f"{str(e['n_truth_inside']) + '/' + str(e['n_truth_outside']):>9}"
            f"{_pct(e['false_reject']):>11}{_pct(e['false_accept']):>11}"
            f"{_pct(e['indeterminate']):>8}"
        )
    return "\n".join(lines)


def format_reliability(rows: list[Bin]) -> str:
    head = f"{'predicted':>18}{'n':>6}{'mean pred':>11}{'realised':>10}{'95% CI':>18}"
    lines = [head, "-" * len(head)]
    for r in rows:
        label = f"[{r.low:.1f}, {r.high:.1f})"
        if r.n == 0:
            lines.append(f"{label:>18}{0:>6}{'-':>11}{'-':>10}{'-':>18}")
            continue
        ci = f"[{r.ci_low:.3f}, {r.ci_high:.3f}]"
        lines.append(
            f"{label:>18}{r.n:>6}{r.mean_predicted:>11.3f}{r.realised:>10.3f}{ci:>18}"
        )
    return "\n".join(lines)


def _pct(value: float) -> str:
    return "-" if value != value else f"{value * 100:.0f}%"


def main(argv: list[str] | None = None) -> int:
    """Run a validation study and print it.

        python -m co_agent.sim.validate                      # synthetic bias study
        python -m co_agent.sim.validate --prices ./data/px   # historical walk-forward

    The bias study needs no data. The walk-forward needs a directory of CSVs; see
    co_agent/data/prices.py for the format and for the two data properties that
    matter more than the loader.
    """
    import argparse

    from .dgp import DEFAULT_PANEL
    from .simulate import Interval

    parser = argparse.ArgumentParser(description="Validate null_probability.")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--drop", type=float, default=0.12, help="TouchBelow threshold")
    parser.add_argument("--trials", type=int, default=40, help="bias study trials per process")
    parser.add_argument("--paths", type=int, default=10_000)
    parser.add_argument("--history", type=int, default=1600, help="observations per history")
    parser.add_argument(
        "--interval",
        choices=[str(Interval.MC), str(Interval.DOUBLE_BOOTSTRAP)],
        default=str(Interval.MC),
        help="MC is fast; double_bootstrap is the honest width and ~25x slower",
    )
    parser.add_argument("--cond-vol", action="store_true", help="condition on current volatility")
    parser.add_argument("--prices", type=str, default=None, help="run the walk-forward instead")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args(argv)

    falsifier = __import__("co_agent.sim", fromlist=["TouchBelow"]).TouchBelow(args.drop)
    cfg = Config(
        paths=args.paths,
        interval=Interval(args.interval),
        cond_vol=args.cond_vol,
        seed=args.seed,
    )
    band = Band()

    if args.prices:
        from ..data import load_dir, suspicious_returns

        series = load_dir(args.prices, min_obs=args.history + args.horizon + 1)
        print(
            f"walk-forward: {len(series)} symbols, horizon {args.horizon}d, "
            f"non-overlapping windows, interval={args.interval}"
        )
        observations: list[Observation] = []
        for symbol, px in sorted(series.items()):
            odd = suspicious_returns(px)
            if odd:
                print(f"  ! {symbol}: {len(odd)} daily move(s) >= 25% -- check for splits")
            observations += walk_forward(
                symbol,
                px.log_returns,
                falsifier,
                horizon_days=args.horizon,
                history_obs=args.history,
                config=cfg,
            )
        print(f"  {len(observations)} observations\n")
        print(format_reliability(reliability(observations)))
        print(
            "\nNote: windows do not overlap, but symbols do move together, so the "
            "\nintervals above are bootstrapped over observations rather than binomial."
            "\nA universe assembled today also excludes what was delisted, which biases"
            "\nrealised tail frequencies down."
        )
        return 0

    print(
        f"bias study: horizon {args.horizon}d, {args.trials} trials/process, "
        f"{args.paths} paths, history {args.history}, interval={args.interval}, "
        f"cond_vol={args.cond_vol}"
    )
    reports = [
        bias_study(
            dgp,
            falsifier,
            horizon_days=args.horizon,
            history_obs=args.history,
            trials=args.trials,
            config=cfg,
            band=band,
            seed=args.seed,
        )
        for dgp in DEFAULT_PANEL
    ]
    print()
    print(format_bias_table(reports, band))
    print(
        "\ncoverage is the fraction of trials whose interval contained truth; it should"
        "\nbe 0.95. 'in/out' is how many trials had truth inside vs outside the band --"
        "\nthe denominators for the two error rates. 'false rej' is truth inside the band"
        "\nbut the gate rejected; 'false acc' is truth outside but the gate accepted."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
