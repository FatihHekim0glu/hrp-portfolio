# hrp-portfolio

> From-scratch Hierarchical Risk Parity (Lopez de Prado, 2016), benchmarked
> honestly out-of-sample against Markowitz, IVP, and naive 1/N.

[![CI](https://img.shields.io/badge/CI-pending-lightgrey)](https://github.com/FatihHekim0glu/hrp-portfolio)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Honest headline

HRP does what it advertises on the dimension it actually targets — materially
**lower out-of-sample portfolio variance** than Markowitz CLA, and it **never
blows up on a near-singular covariance matrix** (no inversion required). But it
does **not** reliably beat naive 1/N on out-of-sample Sharpe after transaction
costs. On a point-in-time S&P 500 universe with walk-forward rebalancing and
~10 bps/side costs, the HRP-vs-1/N Sharpe gap is economically small and
**statistically insignificant** under the Jobson–Korkie–Memmel test, with the
Deflated Sharpe Ratio near zero once the *full* configuration grid is counted as
the number of trials.

This is the literature-consistent outcome. De Prado (2016) claims lower OOS
variance — not higher Sharpe — and DeMiguel, Garlappi & Uppal (2009) make 1/N a
brutal benchmark that most "optimal" allocators fail to beat once estimation
error is paid for. HRP is a **better-behaved risk allocator, not a return
generator**. Its **lower turnover** versus Markowitz is its most defensible
practical edge.

The repository encodes this honesty as a regression: the `headline_verdict` is a
pure function of `(jkm_pvalue, deflated_sharpe, bootstrap-CI sign)` and *cannot*
emit "HRP beats 1/N" while the bootstrap confidence interval straddles zero.

## Why this library

Most public HRP implementations optimize for adoption convenience and quietly
take shortcuts that flatter the method:

- They use the **wrong correlation distance** (`1 - rho` instead of the metric
  `sqrt(0.5 * (1 - rho))`).
- They use **equal intra-cluster weights** inside `getClusterVar` instead of the
  inverse-variance weights de Prado specifies.
- They benchmark HRP against a **rigged comparison** — feeding each allocator a
  *different* covariance estimator, or sinking max-Sharpe with a naive sample
  mean — so the allocation rule is confounded with the estimator.
- They report a single in-sample point estimate and **skip multiplicity
  correction** entirely, so a "winning" Sharpe is indistinguishable from luck.

This library exists to get those four things right and to answer **one question
correctly**: does HRP deliver what the paper claims, and does that translate into
a *statistically significant* risk-adjusted edge over naive diversification once
realistic costs and estimation error are paid for?

It is built for practitioners and researchers who want a **pure, typed, side-
effect-free compute core** (`mypy --strict`, `py.typed`, zero import-time side
effects) they can audit line by line, parity-tested to 1e-7 against independent
reference implementations, with every footgun marked by an inline
`HONESTY-REQUIREMENT` comment and every contested design choice justified in a
numbered ADR.

## Method

All four allocators are fed the **identical covariance estimator** on each
walk-forward window (Ledoit–Wolf shrinkage by default), so the **allocation rule
is the only treatment** ([ADR-0002](docs/decisions/0002-shared-covariance-fairness.md)).
HRP itself is three hand-rolled, parity-tested stages.

### 1. Tree clustering

Convert the correlation matrix to a proper distance metric, then take a
second-order Euclidean distance over the distance columns, then cluster with
SciPy single linkage.

The correlation distance is the **true metric**, not `1 - rho`
([ADR-0004](docs/decisions/0004-distance-formula.md)):

```
d_ij = sqrt(0.5 * (1 - rho_ij))
```

This maps `rho = +1 -> d = 0`, `rho = 0 -> d = 1/sqrt(2)`, `rho = -1 -> d = 1`
and satisfies the triangle inequality. The second-order co-distance is the
Euclidean distance between the *columns* of `d`:

```
D_ij = sqrt( sum_k (d_ik - d_jk)^2 )
```

so two assets are "close" when they relate to the rest of the universe in the
same way. Single linkage is the paper default; `ward` / `complete` / `average`
are exposed only as configurable ablations
([ADR-0001](docs/decisions/0001-single-linkage-default.md)).

### 2. Quasi-diagonalization

`getQuasiDiag` walks the SciPy linkage matrix to recover the dendrogram leaf
order, then reorders the covariance matrix so that **large covariances sit along
the diagonal** and similar assets are adjacent. This is a pure permutation — a
bijection on the asset index — and the reordered covariance stays symmetric.

### 3. Recursive bisection

`getRecBipart` splits the ordered list top-down. At each split it allocates
between the two sub-clusters **inversely to their cluster variance**, where each
cluster's variance is computed with **inverse-variance intra-cluster weights**
via `getClusterVar` — *not* equal weights:

```
w_cluster = diag(V)^-1 / sum( diag(V)^-1 )      # intra-cluster IVP weights
var_cluster = w_cluster' V w_cluster            # cluster variance
alpha_left  = 1 - var_left / (var_left + var_right)
```

Weights cascade multiplicatively down the tree and the final vector lands on the
simplex (sums to 1, all non-negative). No matrix is ever inverted, which is why
HRP survives the singular covariance that breaks Markowitz CLA.

### Estimating the inputs fairly

- **Covariance:** Ledoit–Wolf shrinkage by default, shared by all four
  allocators; optional Marchenko–Pastur RMT eigenvalue clip
  ([ADR-0002](docs/decisions/0002-shared-covariance-fairness.md)).
- **Expected returns (max-Sharpe only):** an explicit James–Stein /
  grand-mean-shrunk estimator, with the naive sample mean reported as an
  ablation so the reader sees it is *`mu`-estimation noise*, not the allocator,
  that sinks max-Sharpe ([ADR-0003](docs/decisions/0003-shrunk-mu.md)). The
  **headline HRP-vs-1/N comparison is covariance-only and `mu`-immune.**
- **No-lookahead:** covariance, shrinkage intensity, the RMT cutoff, the
  dendrogram, and all weights are estimated on the in-sample window only, then
  applied to the *subsequent* OOS window via `signal.shift(1)`. Purge and embargo
  are re-derived for the portfolio setting in
  [ADR-0005](docs/decisions/0005-purge-embargo.md), not cargo-culted from a pairs
  config.
- **Significance:** Jobson–Korkie–Memmel Sharpe-difference test +
  Politis–Romano stationary block-bootstrap CIs on the Sharpe gap + Deflated
  Sharpe Ratio whose `n_trials` is the **full configuration grid**
  ([ADR-0006](docs/decisions/0006-dsr-multiplicity.md)).

## Validation table

Every claim the library makes is pinned to an external reference, a numeric
tolerance, and a test suite.

| Claim | Reference | Tolerance | Test |
|-------|-----------|-----------|------|
| HRP weights match PyPortfolioOpt (+ second reference) on de Prado's worked example | de Prado (2016), AFML Ch. 16 | 1e-7 | `tests/parity/` |
| Correlation distance is `sqrt(0.5(1-rho))`, not `1-rho` | de Prado (2016) | exact | `tests/unit/`, `tests/parity/` |
| `getQuasiDiag` is a valid bijection; reordered covariance stays symmetric | de Prado (2016) | exact | `tests/property/` |
| `getClusterVar` uses inverse-variance (not equal) intra-cluster weights | de Prado (2016) | 1e-7 | `tests/parity/`, `tests/unit/` |
| Ledoit–Wolf shrinkage matches scikit-learn | Ledoit & Wolf (2004) | 1e-10 | `tests/parity/` |
| JKM statistic matches Memmel closed form | Memmel (2003) | 1e-8 | `tests/parity/` |
| DSR matches Bailey–LdP reference table (full `(k+2)/4` kurtosis term) | Bailey & LdP (2014) | 1e-4 | `tests/parity/` |
| Weights live on the simplex (sum-to-1, all >= 0) | — | 1e-12 | `tests/property/` |
| Scale- and permutation-invariance of allocations | — | 1e-10 | `tests/property/` |
| No-lookahead: shrinkage intensity + RMT cutoff invariant to future data | — | exact | `tests/property/` |
| DSR is non-increasing in `n_trials` | Bailey & LdP (2014) | — | `tests/property/` |
| **HRP yields valid weights on a singular / block-perfectly-correlated covariance that breaks Markowitz CLA** | de Prado (2016) | — | `tests/regression/` |
| **HRP OOS variance < Markowitz min-var on the fixture, while HRP-vs-1/N Sharpe-gap CI straddles zero** | this repo's honest null | — | `tests/regression/` |
| `headline_verdict` truth table cannot claim a win while the CI straddles zero | — | exact | `tests/unit/`, `tests/property/` |
| Determinism: same `RunManifest` seed -> byte-identical weights/gap/CI | — | exact | `tests/regression/` |

## Limitations

These are disclosed, not hidden. Each has a corresponding caveat in the data
strategy and an ADR or inline `HONESTY-REQUIREMENT` comment.

- **Survivorship is mitigated, not eliminated.** A static current-S&P-500
  universe would inflate every allocator's return and could spuriously favor
  concentration, so the universe is rebuilt point-in-time at each rebalance.
  **Residual blind spot:** Polygon Starter has no true historical
  *index-membership* snapshots, so the PIT universe is a **liquidity/price-traded
  approximation**, and delisting-return completeness is imperfect.
- **New-constituent (estimable-universe) caveat.** A ticker entering the universe
  at rebalance `t` with fewer than `lookback_window` observations known strictly
  before `t` is **excluded** from that window's allocation (never padded), and the
  exclusion is logged. This is a longer-listing bias, distinct from survivorship,
  disclosed rather than hidden.
- **1/N is a brutal benchmark.** Per DeMiguel, Garlappi & Uppal (2009), HRP is
  *expected* to tie or lose on Sharpe after costs. The honest finding is the
  insignificant gap — the repository resists cherry-picking by deriving the
  verdict purely from the JKM p-value, the DSR, and the bootstrap-CI sign.
- **The Sharpe edge is insignificant by design of the test, not by tuning.**
  DSR `n_trials` counts the full explored grid (#allocators × #linkages ×
  #covariance-estimators × #rmt × #rebalance-freqs × #cost-levels ×
  #lookback-windows), so a "winning" configuration found by search is correctly
  deflated toward zero.

## Reproduce

```bash
# install with the dev + data extras
uv sync --extra dev --extra data

# run the full test suite (unit, property, parity, regression, integration)
uv run pytest

# run only the parity oracles and the honest-null regressions
uv run pytest -m parity
uv run pytest -m regression

# type-check (strict) and lint
uv run mypy
uv run ruff check .

# run the walk-forward horse race
uv run hrp run --help
```

A fixed `RunManifest` seed makes the horse race deterministic: the same seed
yields byte-identical weights, Sharpe gap, and bootstrap CI.

## References

- Lopez de Prado (2016), *Building Diversified Portfolios that Outperform
  Out-of-Sample*, JPM 42(4). SSRN
  [2708678](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678).
- Lopez de Prado (2018), *Advances in Financial Machine Learning*, Ch. 16
  (`getQuasiDiag`, `getRecBipart`, `getClusterVar`, `correlDist`).
- DeMiguel, Garlappi & Uppal (2009), *Optimal Versus Naive Diversification*,
  RFS 22(5).
- Bailey & Lopez de Prado (2014), *The Deflated Sharpe Ratio*, JPM
  (`n_trials` and the `(k+2)/4` kurtosis term).
- Memmel (2003) / Jobson & Korkie (1981), Sharpe-difference test.
- Politis & Romano (1994), the stationary bootstrap.
- Ledoit & Wolf (2004), honest covariance-matrix shrinkage.
- Marchenko & Pastur (1967) / Plerou et al. (1999), RMT eigenvalue clipping.
- James & Stein (1961), shrinkage estimation of the mean.
- Andrews (1991), HAC / Newey–West bandwidth selection.
- PyPortfolioOpt / Riskfolio-Lib / mlfinlab — HRP parity oracles (dev-only).
- *Estimation Windows in HRP Methods*, Springer (2024) — OOS sensitivity.

## Live link

https://fatihhekimoglu.com/tools/hrp-portfolio
