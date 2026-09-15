package sim

import (
	"math"
	"math/rand/v2"
)

// The stationary bootstrap (Politis & Romano, 1994).
//
// The TRD asks for contiguous blocks of 5–20 days to preserve volatility
// clustering and autocorrelation, and explicitly rejects GBM for understating
// tails. Fixed-length blocks do preserve dependence, but the resampled series
// is not stationary: observations near a block boundary are systematically
// under-represented, and the artefact shows up in exactly the tail the
// simulation exists to measure. Drawing block lengths from a geometric
// distribution instead makes the resampled series strictly stationary at the
// same computational cost — the block length becomes a single parameter (its
// mean) rather than a range, so `mean_block_len: 10` covers the TRD's 5–20.
//
// Implementation: walk the pool forward from a random start, and at each step
// jump to a new random index with probability 1/meanBlockLen, wrapping at the
// end.
func bootstrapPath(rng *rand.Rand, poolReturns []float64, scale, jumpProb float64, dst []float64) {
	n := len(poolReturns)
	i := rng.IntN(n)
	for t := range dst {
		dst[t] = poolReturns[i] * scale
		if rng.Float64() < jumpProb {
			i = rng.IntN(n)
		} else {
			i++
			if i == n {
				i = 0
			}
		}
	}
}

// levels turns a log-return draw into a price path starting at 1.
func levels(logReturns []float64, dst []float64) {
	lvl := 1.0
	dst[0] = 1
	for t, r := range logReturns {
		lvl *= math.Exp(r)
		dst[t+1] = lvl
	}
}

// ewmaSigma returns the one-step-ahead volatility forecast for every index,
// plus the forecast for the step after the last observation.
//
// sigma[t] is computed from returns strictly before t, so standardising r[t] by
// sigma[t] introduces no look-ahead. Indices below warmup are NaN: the
// recursion is seeded from the first `warmup` observations, and those
// observations therefore have no clean sigma of their own.
func ewmaSigma(r []float64, lambda float64, warmup int) (sigma []float64, sigmaNext float64) {
	sigma = make([]float64, len(r))
	for i := range sigma {
		sigma[i] = math.NaN()
	}
	if len(r) <= warmup || warmup < 2 {
		return sigma, math.NaN()
	}
	// Seed from the warmup window (uses r[0:warmup] only).
	var sum, sumsq float64
	for _, v := range r[:warmup] {
		sum += v
		sumsq += v * v
	}
	mean := sum / float64(warmup)
	v := sumsq/float64(warmup) - mean*mean
	if v <= 0 {
		v = 1e-12
	}
	sigma[warmup] = math.Sqrt(v)

	var2 := v
	for t := warmup; t < len(r)-1; t++ {
		var2 = lambda*var2 + (1-lambda)*r[t]*r[t]
		sigma[t+1] = math.Sqrt(var2)
	}
	var2 = lambda*var2 + (1-lambda)*r[len(r)-1]*r[len(r)-1]
	return sigma, math.Sqrt(var2)
}

// resamplePool is what the bootstrap draws from.
type resamplePool struct {
	returns []float64 // log returns, possibly standardised
	scale   float64   // multiplier applied to each draw (1, or sigmaNext)
	sigma   float64   // sigmaNext when conditioning on current vol, else 0
}

// buildPool prepares the resampling pool.
//
// With CondVol, returns are standardised by their own one-step-ahead EWMA
// volatility and re-inflated by the current forecast, so the null is
// conditioned on today's regime rather than on the unconditional average of the
// whole history. This matters: the unconditional probability of "down 15% in 60
// days" is badly wrong in both directions depending on where volatility
// currently sits, and it is the figure the gate keys off.
//
// The re-inflation holds volatility flat across the horizon. A vol path that
// mean-reverts (GARCH-style) would be more faithful; it is a deliberate
// simplification, recorded via cond_vol in sim_params so a later version can
// invalidate these figures rather than quietly replace them.
func buildPool(h History, cfg Config) (resamplePool, error) {
	r := h.LogReturns
	var pool resamplePool

	if cfg.CondVol {
		sigma, sigmaNext := ewmaSigma(r, cfg.EWMALambda, cfg.EWMAWarmup)
		if math.IsNaN(sigmaNext) || sigmaNext <= 0 {
			return pool, errInvalid("volatility conditioning needs more history than the warmup window")
		}
		z := make([]float64, 0, len(r))
		for t := cfg.EWMAWarmup; t < len(r); t++ {
			if s := sigma[t]; !math.IsNaN(s) && s > 0 {
				z = append(z, r[t]/s)
			}
		}
		if len(z) < cfg.MinHistory {
			return pool, &InsufficientHistoryError{Have: len(z), Want: cfg.MinHistory, Standardised: true}
		}
		pool = resamplePool{returns: z, scale: sigmaNext, sigma: sigmaNext}
	} else {
		cp := make([]float64, len(r))
		copy(cp, r)
		pool = resamplePool{returns: cp, scale: 1}
	}

	if cfg.Drift == DriftZero {
		var sum float64
		for _, v := range pool.returns {
			sum += v
		}
		mean := sum / float64(len(pool.returns))
		for i := range pool.returns {
			pool.returns[i] -= mean
		}
	}
	return pool, nil
}
