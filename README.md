# hrp-portfolio

> From-scratch Hierarchical Risk Parity (Lopez de Prado, 2016), benchmarked
> honestly out-of-sample against IVP and naive 1/N on real Polygon EOD data.

[![CI](https://img.shields.io/badge/CI-pending-lightgrey)](https://github.com/FatihHekim0glu/hrp-portfolio)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Honest headline

On a real Polygon end-of-day backtest — a 20-name large-cap basket, **893
trading days (2021-06-15 .. 2024-12-31)**, monthly rebalance, 10 bps/side costs,
252-day lookback — **HRP delivers the lowest out-of-sample volatility and never
blows up, but it does *not* beat naive 1/N on after-cost Sharpe.**

- **HRP:** OOS Sharpe **0.817**, OOS vol **0.1199** (lowest of the three), avg turnover 0.096
- **1/N:** OOS Sharpe **1.041**, OOS vol 0.1330, avg turnover 0.016
- **IVP:** OOS Sharpe 0.784, OOS vol 0.1202, avg turnover 0.036

The HRP-vs-1/N Sharpe gap is **-0.224**. The Jobson–Korkie–Memmel test returns
**p = 0.189 (not significant)**, and a stationary block-bootstrap puts the 95%
confidence interval on the gap at **[-0.592, 0.123]**, which **straddles zero**.
The verdict the repository emits is therefore **`NO_SIGNIFICANT_DIFFERENCE`**.

This is the literature-consistent result. De Prado (2016) claims **lower OOS
variance**, not higher Sharpe; DeMiguel, Garlappi & Uppal (2009) make 1/N a
benchmark most "optimal" allocators fail to beat once estimation error is paid
for. **HRP's real edge here is lower out-of-sample variance and robustness to a
singular covariance matrix — not a higher Sharpe.** It is a better-behaved risk
allocator, not a return generator.

> **Survivorship caveat.** The basket is a *fixed current* large-cap set
> (AAPL, MSFT, GOOGL, AMZN, JPM, BAC, XOM, CVX, JNJ, PFE, PG, KO, WMT, HD, CAT,
> BA, NEE, DUK, VZ, T) and is therefore survivor-biased. A true point-in-time
> universe would include names that were later delisted, which would pull all
> three allocators' returns down. The numbers above should be read as an
> *internal, like-for-like* comparison, not a clean live-trading estimate.

## Why this library

Most public HRP implementations optimize for adoption convenience and quietly
take shortcuts that flatter the method:

- They use the **wrong correlation distance** (`1 - rho` instead of the metric
  `sqrt(0.5 * (1 - rho))`).
- They use **equal intra-cluster weights** inside `getClusterVar` instead of the
  inverse-variance weights de Prado specifies.
- They benchmark HRP against a **rigged comparison** — feeding each allocator a
  *different* covariance estimator — so the allocation rule is confounded with
  the estimator.
- They report a single in-sample point estimate and **skip significance testing**
  entirely, so a "winning" Sharpe is indistinguishable from luck.

This library gets those four things right and answers **one question correctly**:
does HRP deliver what the paper claims, and does that translate into a
*statistically significant* risk-adjusted edge over naive diversification once
realistic costs and estimation error are paid for? (On the real data above, the
answer is: yes to lower variance, no to a significant Sharpe edge.)

It is built for practitioners and researchers who want a **pure, typed,
side-effect-free compute core** they can audit line by line, parity-tested
against independent reference implementations.

## Method

All allocators are fed the **identical covariance estimator** on each
walk-forward window (Ledoit–Wolf shrinkage by default), so the **allocation rule
is the only treatment**. HRP itself is three hand-rolled, parity-tested stages.

### 1. Tree clustering — correlation distance

Convert the correlation matrix to a proper distance metric. The correlation
distance is the **true metric**, not `1 - rho`:

```
d_ij = sqrt(0.5 * (1 - rho_ij))
```

This maps `rho = +1 -> d = 0`, `rho = 0 -> d = 1/sqrt(2)`, `rho = -1 -> d = 1`
and satisfies the triangle inequality. A second-order co-distance is then the
Euclidean distance between the *columns* of `d`:

```
D_ij = sqrt( sum_k (d_ik - d_jk)^2 )
```

so two assets are "close" when they relate to the rest of the universe in the
same way. SciPy single linkage is the paper default.

### 2. Quasi-diagonalization

`getQuasiDiag` walks the SciPy linkage matrix to recover the dendrogram leaf
order, then reorders the covariance matrix so that **large covariances sit along
the diagonal** and similar assets are adjacent. This is a pure permutation — a
bijection on the asset index — and the reordered covariance stays symmetric.

### 3. Recursive bisection — inverse-variance cluster weights

`getRecBipart` splits the ordered list top-down. At each split it allocates
between the two sub-clusters **inversely to their cluster variance**, where each
cluster's variance is computed with **inverse-variance intra-cluster weights**
via `getClusterVar` — *not* equal weights:

```
w_cluster   = diag(V)^-1 / sum( diag(V)^-1 )     # intra-cluster IVP weights
var_cluster = w_cluster' V w_cluster             # cluster variance
alpha_left  = 1 - var_left / (var_left + var_right)
```

Weights cascade multiplicatively down the tree and the final vector lands on the
simplex (sums to 1, all non-negative). **No matrix is ever inverted**, which is
why HRP survives the singular covariance that breaks Markowitz CLA.

### Shared-estimator fairness

Every allocator on a given window receives the **same Ledoit–Wolf covariance**,
so any performance difference is attributable to the *allocation rule alone* and
not to a covariance-estimator advantage. Covariance, shrinkage intensity, the
dendrogram, and all weights are estimated on the in-sample window only and
applied to the *subsequent* OOS window — no lookahead.

## Results

Real Polygon EOD backtest. 20-name large-cap basket, 893 trading days
(2021-06-15 .. 2024-12-31), monthly rebalance, 10 bps/side, 252-day lookback.

| Allocator | OOS Sharpe | OOS vol      | Avg turnover |
|-----------|-----------:|-------------:|-------------:|
| **HRP**   |      0.817 | **0.1199** (lowest) |        0.096 |
| 1/N       |  **1.041** |       0.1330 |        0.016 |
| IVP       |      0.784 |       0.1202 |        0.036 |

**HRP vs 1/N:** Sharpe gap **-0.224**; Jobson–Korkie–Memmel **p = 0.189** (not
significant); block-bootstrap 95% CI on the gap **[-0.592, 0.123]** (straddles
zero). **Verdict: `NO_SIGNIFICANT_DIFFERENCE`.**

HRP wins on the dimension it actually targets — **lowest out-of-sample
variance** — and ties (statistically) on Sharpe. That is exactly what the
literature predicts.

## Validation

Every claim is pinned to an external reference, a numeric tolerance, and a test
file you can run.

| Claim | Reference | Tolerance | Test |
|-------|-----------|-----------|------|
| HRP weights match PyPortfolioOpt on de Prado's worked example | de Prado (2016) | ~1e-7 | [`tests/parity/test_parity_hrp_oracle.py`](tests/parity/test_parity_hrp_oracle.py) |
| Ledoit–Wolf shrinkage matches `sklearn.covariance.LedoitWolf` | Ledoit & Wolf (2004) | 1e-10 | [`tests/parity/test_parity_hrp_oracle.py`](tests/parity/test_parity_hrp_oracle.py) |
| Correlation distance is `sqrt(0.5(1-rho))`, not `1-rho` | de Prado (2016) | exact | [`tests/unit/`](tests/unit/), [`tests/parity/`](tests/parity/) |
| `getQuasiDiag` is a valid bijection; reordered covariance stays symmetric | de Prado (2016) | exact | [`tests/property/test_invariants.py`](tests/property/test_invariants.py) |
| Weights live on the simplex (sum-to-1, all >= 0) | — | 1e-12 | [`tests/property/test_invariants.py`](tests/property/test_invariants.py) |
| HRP yields valid weights on a singular covariance that breaks Markowitz | de Prado (2016) | — | [`tests/regression/test_regression_hrp_horse_race.py`](tests/regression/test_regression_hrp_horse_race.py) |
| `headline_verdict` cannot claim a win while the CI straddles zero | — | exact | [`tests/regression/`](tests/regression/), [`tests/unit/`](tests/unit/) |

The PyPortfolioOpt oracle is dev-only and `importorskip`-guarded, so the parity
suite skips cleanly in lean environments rather than hard-failing on import.

## Data

- **Real runs:** Polygon end-of-day OHLC. The headline numbers above come from a
  Polygon EOD pull of the 20-name basket.
- **Dev / CI:** a `yfinance -> Stooq -> synthetic` fallback chain. If `yfinance`
  is unavailable or rate-limited, the loader falls back to Stooq, and finally to
  a deterministic synthetic generator so the test suite stays hermetic and
  offline-reproducible.

## Limitations

These are disclosed, not hidden.

- **Survivorship: the basket is fixed and survivor-biased.** The 20 names are a
  *current* large-cap set, so every allocator's return is inflated relative to a
  true point-in-time universe that would include later-delisted names. Read the
  results as an internal like-for-like comparison.
- **Index-membership PIT is approximated.** Polygon Starter has no true
  historical index-membership snapshots, so any point-in-time universe is a
  liquidity/price-traded approximation, and delisting-return completeness is
  imperfect.
- **Short window.** 893 trading days (~3.5 years, 2021-06-15 .. 2024-12-31) is a
  single regime-limited sample. The insignificant Sharpe gap is consistent with
  the literature but should not be over-extrapolated to other periods or
  universes.
- **1/N is a brutal benchmark.** Per DeMiguel, Garlappi & Uppal (2009), HRP is
  *expected* to tie or lose on Sharpe after costs. The honest finding is the
  insignificant gap — the verdict is derived purely from the JKM p-value and the
  bootstrap-CI sign, not from cherry-picking.

## Reproduce

```bash
# create the env and install with the dev extras
uv venv
uv pip install -e '.[dev]'

# run the full test suite (220 tests, ~91% coverage)
pytest

# real Polygon backtest: 20-name basket, 893 days, monthly rebalance,
# 10 bps/side, 252-day lookback (tickers are a positional, space-separated list)
hrp run \
  AAPL MSFT GOOGL AMZN JPM BAC XOM CVX JNJ PFE PG KO WMT HD CAT BA NEE DUK VZ T \
  --start 2021-06-15 --end 2024-12-31 \
  --rebalance monthly --cost-bps 10 --lookback-window 252 \
  --data-source polygon
```

`--data-source polygon` requires a `POLYGON_API_KEY` in your environment (or a
`.env` file). On any failure — or with the default `--data-source auto` — the
loader falls back to the `yfinance -> Stooq -> synthetic` dev chain.

## Install

```bash
uv venv
uv pip install -e '.[dev]'     # dev + parity-oracle + lint/type/test stack
# extras: '.[data]' for the data path, '.[viz]' for Plotly figures,
#         '.[all]' for every runtime extra
```

The console entry point `hrp` is installed alongside the package.

## License

[MIT](LICENSE).

## Live link

https://fatihhekimoglu.com/tools/hrp-portfolio
