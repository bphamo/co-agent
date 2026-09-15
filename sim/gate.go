package sim

import "math"

// Verdict is the FR9 gate's answer, recorded in falsifier_gate_events.
type Verdict string

const (
	// VerdictAccept — the falsifier is neither a near-certainty nor a
	// near-impossibility, so resolving it carries information.
	VerdictAccept Verdict = "accept"
	// VerdictRejectTooEasy — it trips by chance too often. This is the
	// base-rate drift the TRD's risk table names: easy falsifiers inflate
	// Brier without any research behind them.
	VerdictRejectTooEasy Verdict = "reject_too_easy"
	// VerdictRejectTooHard — it almost never trips by chance, so a "false"
	// outcome says nothing about the claim.
	VerdictRejectTooHard Verdict = "reject_too_hard"
	// VerdictIndeterminate — the estimate's confidence interval straddles a
	// band edge, so the simulator cannot tell which side of the gate this
	// falsifier is on.
	//
	// This is the hysteresis. Without it, a falsifier whose true null
	// probability sits at 0.30 is accepted or bounced depending on Monte Carlo
	// noise, and restating it is a coin flip the researcher will read as
	// signal.
	VerdictIndeterminate Verdict = "indeterminate"
)

// Band is the acceptable null-probability range. FR9 sets it at 0.3–0.7; open
// question 7 asks whether it should widen for high-conviction theses, so it is
// a parameter rather than a constant, and it is recorded in sim_params.
type Band struct{ Low, High float64 }

// DefaultBand is FR9's stated band.
func DefaultBand() Band { return Band{Low: 0.3, High: 0.7} }

func (b Band) validate() error {
	if !(b.Low > 0 && b.Low < b.High && b.High < 1) {
		return errInvalid("band must satisfy 0 < low < high < 1")
	}
	return nil
}

// Judge applies the band to a point estimate and its confidence interval.
//
// A verdict is only returned when the interval settles the question:
//
//	ci entirely below Low   -> too hard
//	ci entirely above High  -> too easy
//	ci entirely inside band -> accept
//	otherwise               -> indeterminate
//
// Callers must not treat indeterminate as a rejection. Run escalates the path
// count first; if the interval still straddles an edge, the true value is
// genuinely near the boundary and the decision belongs to a human, not to the
// next Monte Carlo seed.
func (b Band) Judge(p, ciLow, ciHigh float64) Verdict {
	switch {
	case ciHigh < b.Low:
		return VerdictRejectTooHard
	case ciLow > b.High:
		return VerdictRejectTooEasy
	case ciLow >= b.Low && ciHigh <= b.High:
		return VerdictAccept
	default:
		return VerdictIndeterminate
	}
}

// wilson returns a 95% Wilson score interval for k successes in n trials.
// Preferred over the normal approximation because the band edges sit where the
// normal interval is least well behaved for small path counts.
func wilson(k, n int) (low, high float64) {
	if n == 0 {
		return 0, 1
	}
	const z = 1.959963984540054 // 97.5th percentile of the standard normal
	nf := float64(n)
	p := float64(k) / nf
	z2 := z * z
	denom := 1 + z2/nf
	centre := (p + z2/(2*nf)) / denom
	half := z / denom * math.Sqrt(p*(1-p)/nf+z2/(4*nf*nf))
	return math.Max(0, centre-half), math.Min(1, centre+half)
}

// Sizing is FR9's second gate: sizing is never proposed by an agent.
type Sizing struct {
	ProposedWeight      float64 `json:"proposed_weight"`
	Weight              float64 `json:"weight"`
	Reduced             bool    `json:"reduced"`
	P95DrawdownAtWeight float64 `json:"p95_drawdown_at_weight"`
	PerPositionLimit    float64 `json:"per_position_limit"`
}

// CapWeight reduces the proposed weight until the position's 95th-percentile
// drawdown fits the per-position limit.
//
// Drawdown at weight w is linear in w for a single position, so the reduction
// is exact. It does not bound *portfolio* drawdown: six correlated names each
// inside the per-position limit can breach it together. FR5's sector and
// currency caps are what stand between this figure and that case, and
// Appendix A excludes the correlation analysis that would close the gap
// properly.
func CapWeight(p95DrawdownUnit, proposedWeight, perPositionLimit float64) Sizing {
	s := Sizing{
		ProposedWeight:   proposedWeight,
		Weight:           proposedWeight,
		PerPositionLimit: perPositionLimit,
	}
	if p95DrawdownUnit <= 0 || perPositionLimit <= 0 {
		s.P95DrawdownAtWeight = proposedWeight * math.Max(0, p95DrawdownUnit)
		return s
	}
	if at := proposedWeight * p95DrawdownUnit; at > perPositionLimit {
		s.Weight = perPositionLimit / p95DrawdownUnit
		s.Reduced = true
	}
	s.P95DrawdownAtWeight = s.Weight * p95DrawdownUnit
	return s
}
