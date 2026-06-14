# ADR-0002: All allocators share one covariance estimator per window

- **Status:** Accepted
- **Date:** 2026-06-14
- **Deciders:** hrp-portfolio maintainers
- **Related:** [ADR-0003](0003-shrunk-mu.md) (the matching fairness choice for `mu`)

## Context

The whole point of the horse race is to isolate the **allocation rule** as the
treatment. HRP, IVP, 1/N, and Markowitz (min-var + max-Sharpe) all consume a
covariance matrix. If each allocator were handed a *different* covariance
estimator — say, HRP a shrunk one and Markowitz the raw sample covariance — then
any difference in out-of-sample performance would be confounded: we could not say
whether HRP "won" because of its allocation logic or because it happened to get a
better-conditioned input.

This is one of the most common ways HRP benchmarks are unintentionally rigged.
The raw sample covariance is also near-singular when the lookback window is short
relative to the number of assets (`T` close to `N`), which advantages methods
that never invert it and penalizes those that do — again confounding the
estimator with the rule.

## Decision

On every walk-forward window, **all four allocators are fed the identical
covariance estimator**: **Ledoit–Wolf shrinkage by default**, with an optional
Marchenko–Pastur RMT eigenvalue clip. The covariance is estimated once, in
`estimators/covariance.py`, and passed unchanged to each allocator. The covariance
estimator and the RMT toggle are explicit axes of the configuration grid, applied
uniformly.

The shrinkage intensity and the RMT cutoff are deterministic functions of the
in-sample window only and are asserted to be invariant to future data
(no-lookahead test).

## Consequences

- **Positive.** The allocation rule is the only treatment; a Sharpe difference is
  attributable to the rule, not the input. The comparison is honest.
- **Positive.** Sharing a well-conditioned (shrunk) covariance gives Markowitz a
  *fair* shot — it is not sabotaged by a singular sample covariance — so when
  Markowitz still shows higher turnover and the singular-cov robustness gap, that
  finding is credible rather than an artifact.
- **Positive.** The HRP-vs-1/N headline depends only on covariance, making it
  immune to the separate `mu`-estimation problem ([ADR-0003](0003-shrunk-mu.md)).
- **Cost.** Ledoit–Wolf is a modeling choice; we mitigate by exposing
  `sample` / `oas` and the RMT clip as ablations and parity-testing Ledoit–Wolf
  against scikit-learn to 1e-10.
- **Risk addressed.** "Rigged comparison via mismatched estimators" is removed by
  construction and enforced by the shared code path.
