package sim

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand/v2"
	"testing"
	"time"
)

// synthReturns builds a deterministic daily log-return series.
func synthReturns(n int, vol float64, seed uint64) []float64 {
	rng := rand.New(rand.NewPCG(seed, 7))
	out := make([]float64, n)
	for i := range out {
		out[i] = rng.NormFloat64() * vol
	}
	return out
}

func testHistory(n int, vol float64, seed uint64) History {
	return History{
		Symbol:     "TEST.TO",
		SymbolID:   1,
		LogReturns: synthReturns(n, vol, seed),
		From:       time.Date(2020, 1, 2, 0, 0, 0, 0, time.UTC),
		To:         time.Date(2026, 1, 2, 0, 0, 0, 0, time.UTC),
	}
}

func baseRequest(f Falsifier) Request {
	return Request{
		History:                  testHistory(1600, 0.02, 42),
		HorizonDays:              60,
		Falsifier:                f,
		ProposedWeight:           0.05,
		PerPositionDrawdownLimit: 0.02,
		Config:                   DefaultConfig(),
	}
}

func TestRunIsDeterministic(t *testing.T) {
	req := baseRequest(TouchBelow{Drop: 0.12})
	a, err := Run(req)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Run(req)
	if err != nil {
		t.Fatal(err)
	}
	if a.NullProbability != b.NullProbability || a.P95DrawdownUnit != b.P95DrawdownUnit {
		t.Fatalf("not reproducible: %v/%v vs %v/%v",
			a.NullProbability, a.P95DrawdownUnit, b.NullProbability, b.P95DrawdownUnit)
	}
	if a.Params.HistorySHA256 == "" || a.Params.Version != Version {
		t.Fatalf("provenance not recorded: %+v", a.Params)
	}

	// A different seed must move the figure, or the seed is not being used.
	req.Config.Seed = 99
	c, err := Run(req)
	if err != nil {
		t.Fatal(err)
	}
	if c.NullProbability == a.NullProbability {
		t.Fatal("seed had no effect on the estimate")
	}
}

// With a demeaned pool, ending above the start is a coin flip. This is the one
// analytic anchor available without a closed form for the rest.
func TestDemeanedPoolEndsAboveHalfTheTime(t *testing.T) {
	req := baseRequest(TerminalAbove{Rise: 0})
	req.Config.Paths = 20_000
	res, err := Run(req)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(res.NullProbability-0.5) > 0.03 {
		t.Fatalf("P(terminal >= start) = %.4f, want ~0.5", res.NullProbability)
	}
}

func TestNullProbabilityFallsAsFalsifierGetsHarder(t *testing.T) {
	var prev float64 = 1.1
	for _, drop := range []float64{0.05, 0.10, 0.15, 0.25, 0.40} {
		res, err := Run(baseRequest(TouchBelow{Drop: drop}))
		if err != nil {
			t.Fatal(err)
		}
		if res.NullProbability >= prev {
			t.Fatalf("drop=%.2f gave p=%.4f, not below %.4f", drop, res.NullProbability, prev)
		}
		prev = res.NullProbability
	}
	if prev > 0.05 {
		t.Fatalf("a 40%% drop in 60 days should be rare, got p=%.4f", prev)
	}
}

// "Trades below" and "closes below" are different falsifiers, and the gap is
// exactly what the gate exists to surface.
func TestTouchIsEasierThanTerminal(t *testing.T) {
	touch, err := Run(baseRequest(TouchBelow{Drop: 0.12}))
	if err != nil {
		t.Fatal(err)
	}
	terminal, err := Run(baseRequest(TerminalBelow{Drop: 0.12}))
	if err != nil {
		t.Fatal(err)
	}
	if !(touch.NullProbability > terminal.NullProbability) {
		t.Fatalf("touch %.4f should exceed terminal %.4f",
			touch.NullProbability, terminal.NullProbability)
	}
}

func TestInsufficientHistoryIsAnError(t *testing.T) {
	req := baseRequest(TouchBelow{Drop: 0.12})
	req.History.LogReturns = synthReturns(200, 0.02, 1)
	_, err := Run(req)
	var ihe *InsufficientHistoryError
	if !errors.As(err, &ihe) {
		t.Fatalf("want InsufficientHistoryError, got %v", err)
	}
	if ihe.Have != 200 || ihe.Want != DefaultConfig().MinHistory {
		t.Fatalf("unexpected bounds: %+v", ihe)
	}
}

// A proxy must be a decision someone recorded, never a fallback the simulator
// reached for on its own.
func TestProxyIsExplicitAndRecorded(t *testing.T) {
	target := History{Symbol: "NEWCO.V", SymbolID: 2,
		LogReturns: synthReturns(90, 0.02, 3)}
	peer := testHistory(1600, 0.015, 11)
	peer.Symbol = "PEER.TO"
	peer.SymbolID = 3

	if _, err := NewProxyHistory(target, peer, 1.4, ""); err == nil {
		t.Fatal("a proxy without a reason should be refused")
	}
	if _, err := NewProxyHistory(target, peer, 0, "thin history"); err == nil {
		t.Fatal("a non-positive vol scale should be refused")
	}

	h, err := NewProxyHistory(target, peer, 1.4, "IPO 2026-03, 90 obs")
	if err != nil {
		t.Fatal(err)
	}
	req := baseRequest(TouchBelow{Drop: 0.12})
	req.History = h
	res, err := Run(req)
	if err != nil {
		t.Fatal(err)
	}
	if res.Method != MethodProxyBootstrap {
		t.Fatalf("method = %q, want %q", res.Method, MethodProxyBootstrap)
	}
	if res.Params.Proxy == nil || res.Params.Proxy.Symbol != "PEER.TO" ||
		res.Params.Proxy.VolScale != 1.4 {
		t.Fatalf("proxy provenance missing: %+v", res.Params.Proxy)
	}
}

// The event class: a base rate the bootstrap cannot compute, gated the same way
// and still sized by the bootstrap.
func TestEventClassUsesPriorButIsStillSized(t *testing.T) {
	req := baseRequest(nil)
	req.Prior = &Prior{P: 0.48, Source: "8 of 17 comparable quarters, 2019-2025", N: 17}
	res, err := Run(req)
	if err != nil {
		t.Fatal(err)
	}
	if res.Method != MethodExplicitPrior {
		t.Fatalf("method = %q", res.Method)
	}
	if res.NullProbability != 0.48 {
		t.Fatalf("null probability = %v, want the prior", res.NullProbability)
	}
	if res.P95DrawdownUnit <= 0 {
		t.Fatal("event-class theses still need a drawdown figure for sizing")
	}
	// n=17 is a wide interval, so the gate cannot settle it either way.
	if res.Verdict != VerdictIndeterminate {
		t.Fatalf("verdict = %q, want indeterminate on a 17-observation prior", res.Verdict)
	}

	req.Prior = &Prior{P: 0.92, Source: "22 of 24 quarters", N: 24}
	res, err = Run(req)
	if err != nil {
		t.Fatal(err)
	}
	if res.Verdict != VerdictRejectTooEasy {
		t.Fatalf("verdict = %q, want reject_too_easy", res.Verdict)
	}
}

func TestPriorNeedsASource(t *testing.T) {
	req := baseRequest(nil)
	req.Prior = &Prior{P: 0.5}
	if _, err := Run(req); !errors.Is(err, ErrInvalid) {
		t.Fatalf("want ErrInvalid, got %v", err)
	}
}

func TestExactlyOneFalsifierClass(t *testing.T) {
	req := baseRequest(TouchBelow{Drop: 0.12})
	req.Prior = &Prior{P: 0.5, Source: "x"}
	if _, err := Run(req); !errors.Is(err, ErrInvalid) {
		t.Fatalf("both classes set should be refused, got %v", err)
	}
	req2 := baseRequest(nil)
	if _, err := Run(req2); !errors.Is(err, ErrInvalid) {
		t.Fatalf("neither class set should be refused, got %v", err)
	}
}

// A falsifier whose true null probability sits on a band edge must not be
// bounced by Monte Carlo noise. Two behaviours together make that true: Run
// buys precision first, and when precision cannot settle it, the answer is
// "indeterminate" rather than whichever side the seed happened to land on.
func TestIndeterminateEscalatesThePathCount(t *testing.T) {
	req := baseRequest(TouchBelow{Drop: 0.12})
	first, err := Run(req)
	if err != nil {
		t.Fatal(err)
	}
	if first.Escalations != 0 {
		t.Fatalf("a mid-band value should not have escalated: %+v", first.Verdict)
	}

	// Put a band edge on the measured value: at this path count the interval
	// straddles it by construction.
	req.Band = Band{Low: first.NullProbability, High: first.NullProbability + 0.25}
	req.Config.MaxPaths = 160_000
	res, err := Run(req)
	if err != nil {
		t.Fatal(err)
	}
	if res.Escalations == 0 || res.Paths <= first.Paths {
		t.Fatalf("expected escalation, got %d escalations at %d paths",
			res.Escalations, res.Paths)
	}
}

func TestBoundaryHoldsIndeterminateRatherThanGuessing(t *testing.T) {
	req := baseRequest(TouchBelow{Drop: 0.12})
	first, err := Run(req)
	if err != nil {
		t.Fatal(err)
	}

	// A band narrower than the Monte Carlo interval, centred on the measured
	// value, with no room to escalate: the simulator cannot tell which side of
	// the gate this falsifier falls on, and says so instead of picking one.
	p := first.NullProbability
	req.Band = Band{Low: p - 0.0005, High: p + 0.0005}
	req.Config.MaxPaths = req.Config.Paths

	res, err := Run(req)
	if err != nil {
		t.Fatal(err)
	}
	if res.Verdict != VerdictIndeterminate {
		t.Fatalf("verdict = %q, want indeterminate", res.Verdict)
	}
	if res.Escalations != 0 || res.Paths != req.Config.Paths {
		t.Fatalf("escalated past MaxPaths: %d escalations at %d paths",
			res.Escalations, res.Paths)
	}
	// The point of the hysteresis: a second run does not flip the answer.
	again, err := Run(req)
	if err != nil {
		t.Fatal(err)
	}
	if again.Verdict != res.Verdict {
		t.Fatalf("verdict flipped between identical runs: %q then %q",
			res.Verdict, again.Verdict)
	}
}

// Volatility conditioning has to be doing something, or the gate is keyed off an
// unconditional average that is wrong in both directions depending on where
// volatility currently sits.
func TestConditioningOnCurrentVolMovesTheNull(t *testing.T) {
	// Same pooled observations, opposite regimes: calm now vs stormy now.
	calmFirst := append(synthReturns(800, 0.035, 5), synthReturns(800, 0.008, 6)...)
	stormFirst := append(synthReturns(800, 0.008, 6), synthReturns(800, 0.035, 5)...)

	run := func(r []float64, condVol bool) float64 {
		req := baseRequest(TouchBelow{Drop: 0.15})
		req.History.LogReturns = r
		req.Config.CondVol = condVol
		res, err := Run(req)
		if err != nil {
			t.Fatal(err)
		}
		return res.NullProbability
	}

	calm := run(calmFirst, true)   // history ends quiet
	storm := run(stormFirst, true) // history ends volatile
	if !(storm > calm+0.10) {
		t.Fatalf("conditioning barely moved the null: calm=%.4f storm=%.4f", calm, storm)
	}

	// Unconditionally the two are near-identical, which is the failure mode the
	// conditioning exists to avoid.
	if d := math.Abs(run(calmFirst, false) - run(stormFirst, false)); d > 0.10 {
		t.Fatalf("unconditional estimates differ by %.4f; the fixture is not "+
			"isolating the regime effect", d)
	}
}

func TestParamsSerialiseForSimParamsColumn(t *testing.T) {
	res, err := Run(baseRequest(TouchBelow{Drop: 0.12}))
	if err != nil {
		t.Fatal(err)
	}
	blob, err := res.Params.JSON()
	if err != nil {
		t.Fatal(err)
	}
	var back map[string]any
	if err := json.Unmarshal(blob, &back); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"method", "version", "paths", "mean_block_len",
		"drift", "cond_vol", "horizon_days", "history_obs", "history_sha256",
		"seed", "gate_band", "falsifier"} {
		if _, ok := back[k]; !ok {
			t.Errorf("sim_params is missing %q; the figure would not be reproducible", k)
		}
	}
	if back["method"] != string(MethodStationaryBootstrap) {
		t.Errorf("method = %v", back["method"])
	}
}
