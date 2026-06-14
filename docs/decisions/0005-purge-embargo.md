# ADR-0005: Purge and embargo re-derived for the portfolio setting

- **Status:** Accepted
- **Date:** 2026-06-14
- **Deciders:** hrp-portfolio maintainers
- **Related:** [ADR-0002](0002-shared-covariance-fairness.md) (estimator-level no-lookahead)

## Context

De Prado's purge-and-embargo machinery (AFML Ch. 7) prevents leakage in
cross-validation when **labels overlap in time** — e.g. a triple-barrier label
whose outcome window spans several days, so a training observation can "see" into
a test observation's future. Pairs-trading and other event-labeled backtests in
the sibling repositories configure purge/embargo for *that* overlapping-label
regime, often with multi-day embargoes tuned to a label horizon.

It would be cargo-culting to copy those values here. This is a **portfolio
allocation** backtest with a fundamentally simpler temporal structure:

- Returns are **non-overlapping daily** observations (return horizon = 1 day).
- Weights are estimated on an in-sample window and applied to the **subsequent**
  OOS window via `signal.shift(1)` at the rebalance boundary — there is no
  multi-day label whose outcome leaks backward.

The only leakage surface is the **single shared boundary observation** between the
in-sample and OOS windows.

## Decision

Purge and embargo are **re-derived from first principles for this setting**, not
inherited from the pairs config:

- **Embargo = return horizon = 1 day** (daily returns). With `shift(1)`
  application the embargo collapses to the one-day gap already enforced by the
  shift.
- **Purge removes the single shared boundary observation** between in-sample and
  OOS, so no datum is used both to fit and to evaluate.

Combined with `signal.shift(1)`, this guarantees weights formed at the boundary
are applied strictly forward. The derivation is documented here and the
no-lookahead property is enforced by a future-perturbation-invariance test
extended down to the shrinkage intensity and RMT cutoff
([ADR-0002](0002-shared-covariance-fairness.md)), not just the final weights.

## Consequences

- **Positive.** The leakage guard matches the actual data-generating structure
  instead of importing an over-conservative or mis-sized embargo.
- **Positive.** The minimal-but-correct purge/embargo maximizes usable OOS data
  without sacrificing rigor.
- **Cost.** The derivation must be re-checked if the return horizon ever changes
  (e.g. weekly returns or overlapping windows would re-introduce a multi-period
  embargo). This dependency is stated explicitly so it is not forgotten.
- **Risk addressed.** "Cargo-culting the pairs purge/embargo config" is rejected;
  the chosen values are justified by the portfolio setting and tested.
