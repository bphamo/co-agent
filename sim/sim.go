package sim

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
)

// ErrInvalid wraps every input-validation failure.
var ErrInvalid = errors.New("sim: invalid input")

func errInvalid(msg string) error { return fmt.Errorf("%w: %s", ErrInvalid, msg) }

// InsufficientHistoryError is returned when a symbol has too little history for
// the block bootstrap to mean anything.
//
// FR9's method needs enough independent blocks that the resampled tail is not
// three realised episodes wearing a Monte Carlo costume. Below the threshold
// this is a rejection, never a silent fallback to a shorter window or a peer:
// see NewProxyHistory for the explicit route.
type InsufficientHistoryError struct {
	Have, Want   int
	Standardised bool
}

func (e *InsufficientHistoryError) Error() string {
	what := "observations"
	if e.Standardised {
		what = "usable observations after the EWMA warmup"
	}
	return fmt.Sprintf("sim: insufficient history: %d %s, need %d "+
		"(reject the candidate, or supply a peer via NewProxyHistory)",
		e.Have, what, e.Want)
}

// Config holds every knob. All of it is recorded in sim_params; none of it is
// ever chosen by an agent (FR9).
type Config struct {
	// Paths is the initial synthetic path count. FR9 asks for ≥10,000.
	Paths int
	// MaxPaths caps escalation when a verdict is indeterminate. Bounded so a
	// boundary-hugging falsifier cannot turn into an unbounded compute loop
	// (NFR2).
	MaxPaths int
	// MeanBlockLen is the expected geometric block length in trading days.
	MeanBlockLen float64
	// Drift decides whether the null carries the symbol's realised drift.
	Drift DriftMode
	// CondVol standardises by EWMA volatility and re-inflates at the current
	// forecast, conditioning the null on today's regime.
	CondVol    bool
	EWMALambda float64
	EWMAWarmup int
	// MinHistory is the minimum usable observation count. ~750 is three years
	// of daily data; open question 6 asks for this number and this is a
	// defensible starting answer, not a measured one.
	MinHistory int
	// Seed makes a figure reproducible. Derive it from the thesis id so that
	// two theses in one batch do not share a path set.
	Seed uint64
}

// DefaultConfig is the FR9 baseline.
func DefaultConfig() Config {
	return Config{
		Paths:        10_000,
		MaxPaths:     160_000,
		MeanBlockLen: 10,
		Drift:        DriftZero,
		CondVol:      true,
		EWMALambda:   0.94,
		EWMAWarmup:   60,
		MinHistory:   750,
		Seed:         1,
	}
}

func (c *Config) applyDefaults() {
	d := DefaultConfig()
	if c.Paths <= 0 {
		c.Paths = d.Paths
	}
	if c.MaxPaths < c.Paths {
		c.MaxPaths = max(c.Paths, d.MaxPaths)
	}
	if c.MeanBlockLen <= 0 {
		c.MeanBlockLen = d.MeanBlockLen
	}
	if c.Drift == "" {
		c.Drift = d.Drift
	}
	if c.EWMALambda <= 0 || c.EWMALambda >= 1 {
		c.EWMALambda = d.EWMALambda
	}
	if c.EWMAWarmup <= 0 {
		c.EWMAWarmup = d.EWMAWarmup
	}
	if c.MinHistory <= 0 {
		c.MinHistory = d.MinHistory
	}
	if c.Seed == 0 {
		c.Seed = d.Seed
	}
}

// Request is one FR9 evaluation.
type Request struct {
	History     History
	HorizonDays int

	// Exactly one of Falsifier or Prior must be set.
	//
	// Falsifier is the price_path class: null_probability is the fraction of
	// synthetic paths that trip it.
	//
	// Prior is the event class. A bootstrap over returns cannot price "guides
	// below $X in Q3" or "the 10-Q shows inventory up >20%", and forcing every
	// thesis into a price-path falsifier just so the gate has something to
	// compute would narrow the research to pure price bets. The event class
	// takes its null probability from a recorded base rate instead, passes
	// through the same gate, and is still sized by the bootstrap below.
	Falsifier Falsifier
	Prior     *Prior

	ProposedWeight           float64
	PerPositionDrawdownLimit float64

	Band   Band
	Config Config
}

// Result is what the research worker writes to the thesis record.
//
// It carries no synthetic series, by design: FR9 bars simulated paths from any
// agent's context, and the cheapest way to honour that is for the simulator to
// have no way to emit one.
type Result struct {
	Method          Method
	NullProbability float64
	CILow, CIHigh   float64
	Verdict         Verdict
	Paths           int
	Escalations     int

	// P95DrawdownUnit is the 95th-percentile peak-to-trough drawdown of a
	// fully-weighted position over the horizon.
	P95DrawdownUnit float64
	Sizing          Sizing

	Params Params
}

// Run evaluates a candidate: null probability, gate verdict, and the drawdown
// figure that caps its weight.
//
// It is a pure function of its inputs. Two calls with the same Request give the
// same Result, which is what lets a stored figure be checked against the code
// that produced it.
func Run(req Request) (*Result, error) {
	cfg := req.Config
	cfg.applyDefaults()

	band := req.Band
	if band == (Band{}) {
		band = DefaultBand()
	}
	if err := band.validate(); err != nil {
		return nil, err
	}
	if req.HorizonDays <= 0 {
		return nil, errInvalid("horizon_days must be > 0")
	}
	if (req.Falsifier == nil) == (req.Prior == nil) {
		return nil, errInvalid("set exactly one of Falsifier (price_path) or Prior (event)")
	}
	if req.Falsifier != nil {
		if err := req.Falsifier.Validate(); err != nil {
			return nil, err
		}
	}
	if req.Prior != nil {
		if req.Prior.P < 0 || req.Prior.P > 1 {
			return nil, errInvalid("prior.p must be in [0,1]")
		}
		if req.Prior.Source == "" {
			return nil, errInvalid("prior.source is required: an unsourced base rate is not evidence")
		}
	}
	if req.ProposedWeight < 0 {
		return nil, errInvalid("proposed_weight must be >= 0")
	}
	if n := len(req.History.LogReturns); n < cfg.MinHistory {
		return nil, &InsufficientHistoryError{Have: n, Want: cfg.MinHistory}
	}

	pool, err := buildPool(req.History, cfg)
	if err != nil {
		return nil, err
	}

	method := MethodStationaryBootstrap
	if req.History.proxy != nil {
		method = MethodProxyBootstrap
	}
	if req.Prior != nil {
		method = MethodExplicitPrior
	}

	res := &Result{Method: method}
	paths := cfg.Paths
	var drawdowns []float64

	for {
		trips, dds := simulate(pool, req.Falsifier, req.HorizonDays, paths, cfg)
		drawdowns = dds
		res.Paths = paths

		if req.Prior != nil {
			// Sizing comes from the bootstrap; the null probability does not.
			res.NullProbability = req.Prior.P
			if req.Prior.N > 0 {
				k := int(math.Round(req.Prior.P * float64(req.Prior.N)))
				res.CILow, res.CIHigh = wilson(k, req.Prior.N)
			} else {
				// A judgemental prior has no sampling interval. Recording the
				// point estimate as its own interval keeps the gate usable
				// while leaving prior.n = 0 in sim_params as the flag that
				// this verdict rests on judgement.
				res.CILow, res.CIHigh = req.Prior.P, req.Prior.P
			}
			res.Verdict = band.Judge(res.NullProbability, res.CILow, res.CIHigh)
			break
		}

		res.NullProbability = float64(trips) / float64(paths)
		res.CILow, res.CIHigh = wilson(trips, paths)
		res.Verdict = band.Judge(res.NullProbability, res.CILow, res.CIHigh)

		if res.Verdict != VerdictIndeterminate || paths >= cfg.MaxPaths {
			break
		}
		paths = min(paths*4, cfg.MaxPaths)
		res.Escalations++
	}

	slices.Sort(drawdowns)
	res.P95DrawdownUnit = percentile(drawdowns, 0.95)
	res.Sizing = CapWeight(res.P95DrawdownUnit, req.ProposedWeight, req.PerPositionDrawdownLimit)

	res.Params = Params{
		Method:        method,
		Version:       Version,
		Paths:         res.Paths,
		MeanBlockLen:  cfg.MeanBlockLen,
		Drift:         cfg.Drift,
		CondVol:       cfg.CondVol,
		HorizonDays:   req.HorizonDays,
		HistoryObs:    len(req.History.LogReturns),
		HistorySHA256: sha256Returns(req.History.LogReturns),
		Seed:          cfg.Seed,
		GateBand:      [2]float64{band.Low, band.High},
		Prior:         req.Prior,
		Proxy:         req.History.proxy,
	}
	if cfg.CondVol {
		res.Params.EWMALambda = cfg.EWMALambda
		res.Params.EWMAWarmup = cfg.EWMAWarmup
		res.Params.SigmaCurrent = pool.sigma
	}
	if !req.History.From.IsZero() {
		res.Params.HistoryFrom = req.History.From.UTC().Format("2006-01-02")
	}
	if !req.History.To.IsZero() {
		res.Params.HistoryTo = req.History.To.UTC().Format("2006-01-02")
	}
	if req.Falsifier != nil {
		res.Params.Falsifier = req.Falsifier.Spec()
	}
	return res, nil
}

// simulate draws `paths` synthetic paths and returns how many trip the
// falsifier (zero when there is none) along with each path's max drawdown.
func simulate(pool resamplePool, f Falsifier, horizon, paths int, cfg Config) (trips int, drawdowns []float64) {
	rng := rand.New(rand.NewPCG(cfg.Seed, 0x9E3779B97F4A7C15))
	jumpProb := 1 / cfg.MeanBlockLen
	if jumpProb > 1 {
		jumpProb = 1
	}

	draw := make([]float64, horizon)
	path := make([]float64, horizon+1)
	drawdowns = make([]float64, paths)

	for i := 0; i < paths; i++ {
		bootstrapPath(rng, pool.returns, pool.scale, jumpProb, draw)
		levels(draw, path)
		if f != nil && f.Trips(path) {
			trips++
		}
		drawdowns[i] = maxDrawdown(path)
	}
	return trips, drawdowns
}

// percentile takes the q-quantile of an already-sorted slice.
func percentile(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(q*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}
