package sim

import (
	"math"
	"math/rand/v2"
	"testing"
)

// lag1 autocorrelation of a series.
func lag1(x []float64) float64 {
	n := len(x)
	var mean float64
	for _, v := range x {
		mean += v
	}
	mean /= float64(n)
	var num, den float64
	for i := 0; i < n; i++ {
		d := x[i] - mean
		den += d * d
		if i > 0 {
			num += d * (x[i-1] - mean)
		}
	}
	if den == 0 {
		return 0
	}
	return num / den
}

// The whole reason for blocks rather than IID draws: dependence survives the
// resample. With mean block length 1 the stationary bootstrap degenerates to IID
// and the autocorrelation is destroyed, which is the behaviour GBM and IID
// resampling share and that FR9 rejects.
func TestBlocksPreserveAutocorrelation(t *testing.T) {
	const phi = 0.6
	rng := rand.New(rand.NewPCG(1, 2))
	src := make([]float64, 4000)
	for i := 1; i < len(src); i++ {
		src[i] = phi*src[i-1] + rng.NormFloat64()*0.01
	}
	srcAC := lag1(src)
	if srcAC < 0.4 {
		t.Fatalf("fixture is not autocorrelated enough: %.3f", srcAC)
	}

	measure := func(meanBlockLen float64) float64 {
		out := make([]float64, 20_000)
		bootstrapPath(rand.New(rand.NewPCG(9, 9)), src, 1, 1/meanBlockLen, out)
		return lag1(out)
	}

	blocked := measure(10)
	iid := measure(1)

	if blocked < 0.4 {
		t.Errorf("blocked resample lost the dependence: %.3f (source %.3f)", blocked, srcAC)
	}
	if math.Abs(iid) > 0.05 {
		t.Errorf("mean block length 1 should be IID, got autocorrelation %.3f", iid)
	}
}

// Fixed-length blocks under-represent observations near block boundaries;
// geometric lengths do not. The check here is the weaker but sufficient one:
// every observation in the pool is reachable and none is favoured.
func TestBootstrapCoversThePoolUniformly(t *testing.T) {
	pool := make([]float64, 50)
	for i := range pool {
		pool[i] = float64(i)
	}
	out := make([]float64, 500_000)
	bootstrapPath(rand.New(rand.NewPCG(3, 4)), pool, 1, 1.0/10.0, out)

	counts := make([]int, len(pool))
	for _, v := range out {
		counts[int(v)]++
	}
	want := float64(len(out)) / float64(len(pool))
	for i, c := range counts {
		if rel := math.Abs(float64(c)-want) / want; rel > 0.05 {
			t.Errorf("pool index %d drawn %d times, want ~%.0f (%.1f%% off)",
				i, c, want, rel*100)
		}
	}
}

// Standardising by a volatility estimate that already saw the return it is
// standardising would leak future information into the null.
func TestEWMAHasNoLookAhead(t *testing.T) {
	r := synthReturns(400, 0.02, 21)
	sigmaA, _ := ewmaSigma(r, 0.94, 60)

	perturbed := make([]float64, len(r))
	copy(perturbed, r)
	perturbed[len(perturbed)-1] = 0.35 // a huge final move
	sigmaB, _ := ewmaSigma(perturbed, 0.94, 60)

	for i := 0; i < len(r); i++ {
		a, b := sigmaA[i], sigmaB[i]
		if math.IsNaN(a) && math.IsNaN(b) {
			continue
		}
		if a != b {
			t.Fatalf("sigma[%d] changed (%v -> %v) after altering only the last "+
				"return: the estimate is looking ahead", i, a, b)
		}
	}
}

func TestEWMASigmaTracksVolatility(t *testing.T) {
	quiet := synthReturns(600, 0.005, 31)
	loud := synthReturns(600, 0.04, 32)
	_, nextQuiet := ewmaSigma(quiet, 0.94, 60)
	_, nextLoud := ewmaSigma(loud, 0.94, 60)
	if !(nextLoud > 3*nextQuiet) {
		t.Fatalf("sigma did not track the regime: quiet=%.5f loud=%.5f", nextQuiet, nextLoud)
	}
}

func TestMaxDrawdown(t *testing.T) {
	cases := []struct {
		path []float64
		want float64
	}{
		{[]float64{1, 1.1, 1.2}, 0},
		{[]float64{1, 0.9, 1.0}, 0.1},
		{[]float64{1, 2, 1}, 0.5},
		{[]float64{1, 0.5, 2, 1}, 0.5},
	}
	for _, c := range cases {
		if got := maxDrawdown(c.path); math.Abs(got-c.want) > 1e-12 {
			t.Errorf("maxDrawdown(%v) = %v, want %v", c.path, got, c.want)
		}
	}
}
