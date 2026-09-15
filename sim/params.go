// Package sim implements the FR9 simulation service: the code-only module that
// computes null_probability and p95_drawdown for a candidate thesis.
//
// Two rules from the TRD shape the whole package:
//
//   - It never runs inside an agent loop and its parameters are never chosen by
//     an agent. Everything here is a pure function of (history, falsifier,
//     config); nothing calls a model.
//   - Synthetic price series must never reach a research agent's context. This
//     package therefore returns summary statistics and provenance only. There
//     is no exported type that carries a simulated path out of it.
package sim

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"time"
)

// Version identifies the simulator's behaviour. Bump it on any change that
// would move a figure, so a stored null_probability can be invalidated rather
// than silently compared against one produced by different code.
const Version = "sim/1"

// Method is recorded as theses.sim_method.
type Method string

const (
	// MethodStationaryBootstrap resamples the symbol's own returns.
	MethodStationaryBootstrap Method = "stationary_bootstrap"
	// MethodProxyBootstrap resamples a peer's returns scaled to the symbol's
	// volatility, for names with too little history of their own (open
	// question 6). Never selected implicitly — see NewProxyHistory.
	MethodProxyBootstrap Method = "stationary_bootstrap_proxy"
	// MethodExplicitPrior is the event-falsifier class: a null probability that
	// comes from a recorded base rate, because no return distribution can
	// answer "does the company guide below $X in Q3".
	MethodExplicitPrior Method = "explicit_prior"
)

// DriftMode controls what "by chance" means.
//
// DriftZero demeans the resampling pool, so the null is a forecaster with no
// information about direction. DriftHistorical keeps the symbol's realised
// drift, which makes the null absorb past momentum and generally makes upside
// falsifiers look easier than they are. The choice materially moves
// null_probability, so it is recorded in sim_params either way.
type DriftMode string

const (
	DriftZero       DriftMode = "zero"
	DriftHistorical DriftMode = "historical"
)

// ProxyInfo records that the bootstrap ran on someone else's returns.
type ProxyInfo struct {
	Symbol   string  `json:"proxy_symbol"`
	SymbolID int64   `json:"proxy_symbol_id"`
	VolScale float64 `json:"vol_scale"`
	Reason   string  `json:"reason"`
}

// Prior is the event-falsifier class's null probability.
type Prior struct {
	// P is the probability the falsifier trips absent the thesis's mechanism.
	P float64 `json:"p"`
	// Source is the human-readable provenance, e.g.
	// "8 of 17 comparable quarters, 2019-2025".
	Source string `json:"source"`
	// N is the number of observations behind P. Zero means judgemental, which
	// is permitted but recorded: a gate verdict on an ungrounded prior is a
	// weaker statement than one on a counted base rate.
	N int `json:"n"`
}

// Params is serialised into theses.sim_params. It exists so a figure can be
// reproduced or invalidated, which is the only reason FR9 asks for it.
type Params struct {
	Method        Method         `json:"method"`
	Version       string         `json:"version"`
	Paths         int            `json:"paths"`
	MeanBlockLen  float64        `json:"mean_block_len"`
	Drift         DriftMode      `json:"drift"`
	CondVol       bool           `json:"cond_vol"`
	EWMALambda    float64        `json:"ewma_lambda,omitempty"`
	EWMAWarmup    int            `json:"ewma_warmup,omitempty"`
	SigmaCurrent  float64        `json:"sigma_current,omitempty"`
	HorizonDays   int            `json:"horizon_days"`
	HistoryObs    int            `json:"history_obs"`
	HistoryFrom   string         `json:"history_from,omitempty"`
	HistoryTo     string         `json:"history_to,omitempty"`
	HistorySHA256 string         `json:"history_sha256"`
	Seed          uint64         `json:"seed"`
	GateBand      [2]float64     `json:"gate_band"`
	Falsifier     map[string]any `json:"falsifier,omitempty"`
	Prior         *Prior         `json:"prior,omitempty"`
	Proxy         *ProxyInfo     `json:"proxy,omitempty"`
}

// JSON renders Params for the sim_params jsonb column.
func (p Params) JSON() ([]byte, error) { return json.Marshal(p) }

// History is a symbol's daily log-return series, oldest first.
type History struct {
	Symbol     string
	SymbolID   int64
	LogReturns []float64
	From, To   time.Time

	// proxy is set only by NewProxyHistory.
	proxy *ProxyInfo
}

// NewProxyHistory builds a History for a name with too little history of its
// own, from a peer's returns scaled by volScale.
//
// This exists as an explicit constructor rather than a fallback inside Run
// because a silent proxy is worse than an error: the resulting
// null_probability looks identical to a real one in the ledger. Insufficient
// history is a rejection reason; using a proxy is a decision someone made and
// the ledger records who and why.
func NewProxyHistory(target History, peer History, volScale float64, reason string) (History, error) {
	if volScale <= 0 {
		return History{}, errInvalid("vol_scale must be > 0")
	}
	if reason == "" {
		return History{}, errInvalid("a proxy needs a recorded reason")
	}
	scaled := make([]float64, len(peer.LogReturns))
	for i, r := range peer.LogReturns {
		scaled[i] = r * volScale
	}
	return History{
		Symbol:     target.Symbol,
		SymbolID:   target.SymbolID,
		LogReturns: scaled,
		From:       peer.From,
		To:         peer.To,
		proxy: &ProxyInfo{
			Symbol:   peer.Symbol,
			SymbolID: peer.SymbolID,
			VolScale: volScale,
			Reason:   reason,
		},
	}, nil
}

// sha256Returns hashes the exact input series, so a stored figure can be shown
// to belong to the data it was computed from.
func sha256Returns(r []float64) string {
	h := sha256.New()
	var buf [8]byte
	for _, v := range r {
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(v))
		_, _ = h.Write(buf[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}
