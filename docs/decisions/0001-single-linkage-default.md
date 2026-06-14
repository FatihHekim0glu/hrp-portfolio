# ADR-0001: Single linkage is the default; ward/complete/average are ablations

- **Status:** Accepted
- **Date:** 2026-06-14
- **Deciders:** hrp-portfolio maintainers
- **Related:** [ADR-0004](0004-distance-formula.md) (distance metric the linkage consumes)

## Context

The first stage of HRP clusters assets with SciPy agglomerative clustering. The
linkage criterion (how the distance between two *clusters* is defined from the
distances between their members) is a real degree of freedom: single, complete,
average, and ward all produce different dendrograms, different leaf orders after
quasi-diagonalization, and therefore different weights.

A recurring footgun in public implementations is silently shipping `ward` or
`average` — often because it is a library default or "looks smoother" — while
citing de Prado. That drift is invisible at the API surface but moves the
allocation, so a benchmark built on the wrong default is not reproducing the
paper.

De Prado (2016) and AFML Ch. 16 use **single linkage**. Single linkage defines
inter-cluster distance as the *nearest* pair across clusters, which produces the
"friend of a friend" chaining that the recursive-bisection step is designed
around: it keeps the dendrogram order meaningful for placing large covariances on
the diagonal.

## Decision

`single` is the **validated default** linkage. `ward`, `complete`, and `average`
are exposed as **configurable ablations only** — selectable via the `linkage`
parameter for sensitivity analysis, never silently substituted.

The default is asserted, not assumed: the parity oracle reconciles our
single-linkage leaf order against the reference implementations, and a regression
test catches any drift in the shipped default.

## Consequences

- **Positive.** The headline reproduces the paper. Linkage becomes a transparent,
  countable axis of the configuration grid rather than a hidden assumption.
- **Positive.** Because the alternatives are first-class ablations, a reader can
  quantify how much the verdict depends on the criterion.
- **Cost.** Every linkage we expose multiplies the DSR `n_trials`
  ([ADR-0006](0006-dsr-multiplicity.md)). This is correct accounting — exploring
  more criteria *should* deflate any "best" Sharpe — but it means more linkages
  is not free.
- **Risk addressed.** "Linkage drift" (silent ward/average default) is eliminated
  by the asserted default plus the parity test.
