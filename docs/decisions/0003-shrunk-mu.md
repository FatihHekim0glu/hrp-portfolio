# ADR-0003: Max-Sharpe uses a James–Stein shrunk mean, with naive mu as an ablation

- **Status:** Accepted
- **Date:** 2026-06-14
- **Deciders:** hrp-portfolio maintainers
- **Related:** [ADR-0002](0002-shared-covariance-fairness.md) (the matching fairness choice for covariance)

## Context

Three of the four allocators (HRP, IVP, 1/N) need only a covariance matrix. The
max-Sharpe Markowitz portfolio additionally needs **expected returns**, `mu`. The
sample mean return is a notoriously noisy estimator: its estimation error
dominates portfolio optimization and is the main reason "optimal" mean-variance
portfolios underperform 1/N out of sample (DeMiguel, Garlappi & Uppal 2009).

If we fed max-Sharpe the **naive sample mean**, it would post a dismal OOS Sharpe
— and a careless reader could conclude "the max-Sharpe *allocator* is bad." That
conclusion would be wrong: the failure is `mu`-*estimation* noise, not the
allocation rule. Reporting only naive `mu` would itself be a subtle rigging of the
comparison, just in the opposite direction from [ADR-0002](0002-shared-covariance-fairness.md).

## Decision

The max-Sharpe allocator's `mu` is an **explicit James–Stein / grand-mean-shrunk
estimator** (`estimators/mu.py`), shrinking each asset's sample mean toward the
cross-sectional grand mean. The **naive sample mean is reported as an ablation**,
so the reader can see directly that switching `mu` estimators — not switching
allocators — is what moves max-Sharpe.

Crucially, the **headline comparison is HRP vs 1/N**, which is covariance-only and
therefore **`mu`-immune**. The `mu` choice never touches the headline verdict; it
only affects the max-Sharpe context line.

## Consequences

- **Positive.** Max-Sharpe gets a fair, estimation-error-aware input, so its
  results reflect the allocation rule rather than naive-mean noise.
- **Positive.** The naive-`mu` ablation makes the `mu`-estimation effect visible
  and quantified instead of asserted.
- **Positive.** Keeping the headline on the `mu`-immune HRP-vs-1/N axis means the
  central claim of the repository does not depend on this modeling choice at all.
- **Cost.** `mu_estimator` (`james_stein` vs `sample`) is one more axis in the
  configuration grid and the DSR `n_trials` ([ADR-0006](0006-dsr-multiplicity.md)).
- **Risk addressed.** "Naive `mu` sinking max-Sharpe and being misread as an
  allocator failure" is prevented; the ADR and the ablation document the choice.
