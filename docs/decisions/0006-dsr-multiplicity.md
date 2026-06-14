# ADR-0006: DSR `n_trials` counts the full configuration grid

- **Status:** Accepted
- **Date:** 2026-06-14
- **Deciders:** hrp-portfolio maintainers
- **Related:** [ADR-0001](0001-single-linkage-default.md), [ADR-0002](0002-shared-covariance-fairness.md), [ADR-0003](0003-shrunk-mu.md)

## Context

The Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) corrects an observed
Sharpe for **selection bias**: if you try many strategy configurations and report
the best, the maximum Sharpe is inflated purely by chance, and the inflation grows
with the number of trials. The DSR deflates the observed Sharpe toward the Sharpe
you would expect from the *best of N independent random trials* given the return
distribution's skew and kurtosis.

The integrity of this correction hinges entirely on **what counts as a trial**. A
tempting under-count is to set `n_trials = 1` ("we only ran HRP once") or to count
only the headline allocators. But every knob we *could* have turned and reported
the best of is a trial — including the covariance estimator, the linkage, the RMT
toggle, the rebalance frequency, the cost level, the lookback window, and the `mu`
estimator. Under-counting `n_trials` is exactly how a lucky configuration gets
laundered into a "significant" edge.

## Decision

The DSR `n_effective_trials` counts the **full explored configuration grid**:

```
n_trials = #allocators
         × #linkages
         × #covariance-estimators
         × #rmt(on/off)
         × #rebalance-freqs
         × #cost-levels
         × #lookback-windows
         × #mu-estimators
```

The DSR uses the **full `(k + 2) / 4` kurtosis term** (not a Gaussian
simplification), implemented in `evaluation/dsr.py`. A property test asserts the
DSR is **non-increasing in `n_trials`**, and the DSR vs the Bailey–LdP reference
table is parity-tested to 1e-4.

This count is reported on the API and surfaced in the UI alongside the verdict, so
the multiplicity is visible to the reader, not buried.

## Consequences

- **Positive.** The "winning" Sharpe found by exploring the grid is correctly
  deflated. With the full grid counted, the HRP-vs-1/N DSR lands near zero — the
  honest finding — rather than a spuriously significant one.
- **Positive.** Every other ADR that *adds* an axis (a linkage, an estimator, a
  `mu` choice) does so knowing it raises `n_trials`. The cost of exploration is
  paid in the right place.
- **Positive.** Because `headline_verdict` is a pure function of the JKM p-value,
  the DSR, and the bootstrap-CI sign, a deflated-to-zero DSR mechanically blocks
  an over-claim.
- **Cost.** The verdict is conservative: a genuine small edge could be deflated
  below significance. We accept this — for a benchmark whose purpose is honesty,
  a false "no edge" is far cheaper than a false "HRP beats 1/N."
- **Risk addressed.** "Multiplicity / data-snooping inflating the Sharpe" is
  countered by counting the full grid and testing DSR monotonicity.
