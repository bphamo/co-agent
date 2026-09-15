package sim

import (
	"math"
	"testing"
)

func TestJudgeOnlyRulesWhenTheIntervalSettlesIt(t *testing.T) {
	b := DefaultBand()
	cases := []struct {
		name         string
		p, low, high float64
		want         Verdict
	}{
		{"comfortably inside", 0.50, 0.49, 0.51, VerdictAccept},
		{"inside, touching edges", 0.50, 0.30, 0.70, VerdictAccept},
		{"clearly too hard", 0.10, 0.09, 0.11, VerdictRejectTooHard},
		{"clearly too easy", 0.90, 0.88, 0.92, VerdictRejectTooEasy},
		// The hysteresis cases: the estimate is on one side of the edge but the
		// interval is not, so no verdict is available.
		{"straddling the low edge", 0.31, 0.28, 0.34, VerdictIndeterminate},
		{"straddling the high edge", 0.69, 0.66, 0.72, VerdictIndeterminate},
		{"just outside, interval straddles", 0.29, 0.26, 0.32, VerdictIndeterminate},
		{"interval spans the whole band", 0.50, 0.20, 0.80, VerdictIndeterminate},
	}
	for _, c := range cases {
		if got := b.Judge(c.p, c.low, c.high); got != c.want {
			t.Errorf("%s: Judge(%.2f, %.2f, %.2f) = %q, want %q",
				c.name, c.p, c.low, c.high, got, c.want)
		}
	}
}

func TestBandValidation(t *testing.T) {
	for _, b := range []Band{{0, 0.7}, {0.3, 1}, {0.7, 0.3}, {0.5, 0.5}} {
		if err := b.validate(); err == nil {
			t.Errorf("Band%v should be refused", b)
		}
	}
	if err := (Band{0.3, 0.7}).validate(); err != nil {
		t.Error(err)
	}
}

func TestWilsonBracketsThePointEstimate(t *testing.T) {
	for _, c := range []struct{ k, n int }{{0, 1000}, {1, 1000}, {500, 1000}, {999, 1000}, {1000, 1000}} {
		low, high := wilson(c.k, c.n)
		p := float64(c.k) / float64(c.n)
		if low < 0 || high > 1 || low > high {
			t.Errorf("wilson(%d,%d) = [%v,%v] out of bounds", c.k, c.n, low, high)
		}
		if p < low-1e-9 || p > high+1e-9 {
			t.Errorf("wilson(%d,%d) = [%v,%v] excludes p=%v", c.k, c.n, low, high, p)
		}
	}
	// More paths, tighter interval — that is what escalation buys.
	l1, h1 := wilson(500, 1000)
	l2, h2 := wilson(50_000, 100_000)
	if (h2 - l2) >= (h1 - l1) {
		t.Fatalf("interval did not tighten with path count: %.4f vs %.4f", h2-l2, h1-l1)
	}
}

func TestCapWeightReducesToTheLimit(t *testing.T) {
	// 0.05 weight on a position whose 95th-percentile drawdown is 60% puts 3%
	// at risk against a 2% limit, so the weight must come down to 1/30.
	s := CapWeight(0.60, 0.05, 0.02)
	if !s.Reduced {
		t.Fatal("weight should have been reduced")
	}
	if math.Abs(s.Weight-0.02/0.60) > 1e-12 {
		t.Fatalf("weight = %v, want %v", s.Weight, 0.02/0.60)
	}
	if math.Abs(s.P95DrawdownAtWeight-0.02) > 1e-12 {
		t.Fatalf("drawdown at weight = %v, want the limit", s.P95DrawdownAtWeight)
	}

	// Inside the limit, the proposal stands untouched.
	s = CapWeight(0.20, 0.05, 0.02)
	if s.Reduced || s.Weight != 0.05 {
		t.Fatalf("unnecessary reduction: %+v", s)
	}
	if math.Abs(s.P95DrawdownAtWeight-0.01) > 1e-12 {
		t.Fatalf("drawdown at weight = %v, want 0.01", s.P95DrawdownAtWeight)
	}
}

func TestCapWeightIsLinearInWeight(t *testing.T) {
	a := CapWeight(0.3, 0.02, 1)
	b := CapWeight(0.3, 0.04, 1)
	if math.Abs(b.P95DrawdownAtWeight-2*a.P95DrawdownAtWeight) > 1e-12 {
		t.Fatalf("not linear: %v vs %v", a.P95DrawdownAtWeight, b.P95DrawdownAtWeight)
	}
}

func TestPercentile(t *testing.T) {
	sorted := make([]float64, 100)
	for i := range sorted {
		sorted[i] = float64(i + 1)
	}
	if got := percentile(sorted, 0.95); got != 95 {
		t.Errorf("p95 = %v, want 95", got)
	}
	if got := percentile(nil, 0.95); got != 0 {
		t.Errorf("p95 of empty = %v", got)
	}
}
