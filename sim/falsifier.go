package sim

import "fmt"

// Falsifier is a price-path predicate evaluated against one synthetic path.
//
// A path is a level series normalised to path[0] == 1, length horizon+1. Only
// the predicate sees it; no path leaves this package.
type Falsifier interface {
	Trips(path []float64) bool
	Kind() string
	Spec() map[string]any
	Validate() error
}

// TouchBelow trips if the level falls Drop below the start at any point.
// Drop is expressed as a positive fraction: 0.12 means "down 12%".
type TouchBelow struct{ Drop float64 }

func (f TouchBelow) Kind() string         { return "touch_below" }
func (f TouchBelow) Spec() map[string]any { return map[string]any{"kind": f.Kind(), "drop": f.Drop} }
func (f TouchBelow) Validate() error      { return checkFrac("drop", f.Drop) }
func (f TouchBelow) Trips(path []float64) bool {
	threshold := 1 - f.Drop
	for _, lvl := range path {
		if lvl <= threshold {
			return true
		}
	}
	return false
}

// TouchAbove trips if the level rises Rise above the start at any point.
type TouchAbove struct{ Rise float64 }

func (f TouchAbove) Kind() string         { return "touch_above" }
func (f TouchAbove) Spec() map[string]any { return map[string]any{"kind": f.Kind(), "rise": f.Rise} }
func (f TouchAbove) Validate() error {
	if f.Rise < 0 {
		return errInvalid("rise must be >= 0")
	}
	return nil
}
func (f TouchAbove) Trips(path []float64) bool {
	threshold := 1 + f.Rise
	for _, lvl := range path {
		if lvl >= threshold {
			return true
		}
	}
	return false
}

// TerminalBelow trips on the horizon date only. "Closes below" rather than
// "trades below" — a materially different and usually much harder falsifier,
// which is exactly the kind of difference null_probability is there to expose.
type TerminalBelow struct{ Drop float64 }

func (f TerminalBelow) Kind() string         { return "terminal_below" }
func (f TerminalBelow) Spec() map[string]any { return map[string]any{"kind": f.Kind(), "drop": f.Drop} }
func (f TerminalBelow) Validate() error      { return checkFrac("drop", f.Drop) }
func (f TerminalBelow) Trips(path []float64) bool {
	return path[len(path)-1] <= 1-f.Drop
}

// TerminalAbove trips on the horizon date only.
type TerminalAbove struct{ Rise float64 }

func (f TerminalAbove) Kind() string         { return "terminal_above" }
func (f TerminalAbove) Spec() map[string]any { return map[string]any{"kind": f.Kind(), "rise": f.Rise} }
func (f TerminalAbove) Validate() error {
	if f.Rise < 0 {
		return errInvalid("rise must be >= 0")
	}
	return nil
}
func (f TerminalAbove) Trips(path []float64) bool {
	return path[len(path)-1] >= 1+f.Rise
}

// DrawdownExceeds trips if peak-to-trough drawdown within the horizon exceeds
// Limit.
type DrawdownExceeds struct{ Limit float64 }

func (f DrawdownExceeds) Kind() string { return "drawdown_exceeds" }
func (f DrawdownExceeds) Spec() map[string]any {
	return map[string]any{"kind": f.Kind(), "limit": f.Limit}
}
func (f DrawdownExceeds) Validate() error { return checkFrac("limit", f.Limit) }
func (f DrawdownExceeds) Trips(path []float64) bool {
	return maxDrawdown(path) >= f.Limit
}

func checkFrac(name string, v float64) error {
	if v <= 0 || v >= 1 {
		return errInvalid(fmt.Sprintf("%s must be in (0,1), got %g", name, v))
	}
	return nil
}

// maxDrawdown is the largest peak-to-trough decline on the path, as a fraction.
func maxDrawdown(path []float64) float64 {
	peak := path[0]
	worst := 0.0
	for _, lvl := range path {
		if lvl > peak {
			peak = lvl
		}
		if dd := (peak - lvl) / peak; dd > worst {
			worst = dd
		}
	}
	return worst
}
