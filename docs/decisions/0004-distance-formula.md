# ADR-0004: Correlation distance is `sqrt(0.5(1 - rho))`, not `1 - rho`

- **Status:** Accepted
- **Date:** 2026-06-14
- **Deciders:** hrp-portfolio maintainers
- **Related:** [ADR-0001](0001-single-linkage-default.md) (the linkage consuming this distance)

## Context

HRP clusters assets on a distance derived from their correlation. The choice of
formula matters because clustering only behaves sensibly on a **proper metric**
— one that is non-negative, zero iff the points coincide, symmetric, and obeys
the triangle inequality.

A widespread shortcut is to use `d_ij = 1 - rho_ij`. This is *not* a metric: it
violates the triangle inequality, and it conflates anti-correlation with
distance in a way that distorts the dendrogram. It is the single most common
correctness bug in HRP code, and because it still "runs" and produces plausible
weights, it goes unnoticed.

De Prado's `correlDist` (2016; AFML Ch. 16) uses

```
d_ij = sqrt( 0.5 * (1 - rho_ij) )
```

which maps `rho = +1 -> d = 0`, `rho = 0 -> d = 1/sqrt(2)`, `rho = -1 -> d = 1`,
is bounded in `[0, 1]`, and **is** a true Euclidean-embeddable metric. HRP then
takes a **second-order** Euclidean co-distance over the columns of `d`:

```
D_ij = sqrt( sum_k (d_ik - d_jk)^2 )
```

so two assets are close when they relate to the *rest of the universe* the same
way — the quantity the linkage and quasi-diagonalization steps actually rely on.

## Decision

The first-order distance is **`d_ij = sqrt(0.5(1 - rho_ij))`**, implemented in
`cluster/distance.py`, followed by the second-order Euclidean co-distance `D`
before linkage. The `1 - rho` form is **rejected** and is never used anywhere in
the codebase.

The line carries an inline `HONESTY-REQUIREMENT` comment naming the footgun, and
the formula is pinned by both a unit test (exact values at `rho in {-1, 0, 1}`)
and the parity oracle against the reference implementations.

## Consequences

- **Positive.** Clustering operates on a genuine metric, so dendrograms,
  leaf order, and weights match the paper and the reference oracles to 1e-7.
- **Positive.** The most common silent HRP bug is caught by an exact unit test
  and a visible comment rather than slipping into the benchmark.
- **Cost.** Two distance passes (first-order metric, then second-order
  co-distance) are slightly more work than a single `1 - rho`, which is
  negligible at the universe sizes here.
- **Risk addressed.** The "`1 - rho` vs `sqrt(0.5(1 - rho))` footgun" is closed by
  a golden test and an inline honesty comment.
