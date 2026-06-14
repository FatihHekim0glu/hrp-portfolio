# Design

This document explains how `hrp-portfolio` is put together: the layering, the
data flow through a single walk-forward window, the invariants the compute core
guarantees, and the testing strategy that keeps the honest headline honest. For
*why* individual contested choices were made, see the numbered ADRs in
[`docs/decisions/`](decisions/).

## Goals and non-goals

**Goals**

- A pure, typed (`mypy --strict`, `py.typed`), side-effect-free compute core that
  can be audited line by line and vendored into a backend without dragging UI or
  network dependencies along.
- A faithful from-scratch HRP (clustering, quasi-diagonalization, recursive
  bisection) parity-tested to 1e-7 against independent references.
- A **fair** horse race in which the allocation rule is the only treatment, so
  any Sharpe difference cannot be blamed on a mismatched estimator.
- A statistically defensible verdict that survives multiplicity correction and is
  *mechanically* prevented from over-claiming.

**Non-goals**

- Beating 1/N. The honest finding is that HRP does not, after costs, by a
  significant margin.
- A live trading system. This is a research/benchmark library.
- A generic clustering toolkit. The clustering exists to serve HRP.

## Layered architecture

The package is strictly layered; each layer imports only from the ones below it.
`src/` has **zero import-time side effects**, guarded by a subprocess import-purity
test.

```
                 cli.py (Typer)        plots.py (Plotly)        app/ (Streamlit)
                      |                      |                       |
   ┌──────────────────┴──────────────────────┴───────────────────────┘
   │                          evaluation/
   │           dsr.py · comparison.py · verdict.py
   │   (Deflated Sharpe, ComparisonResult, pure headline_verdict deriver)
   ├──────────────────────────────────────────────────────────────────
   │                          backtest/
   │            walk_forward.py · costs.py · stats.py
   │   (no-lookahead engine · per-side bps · JKM, block bootstrap, HAC)
   ├──────────────────────────────────────────────────────────────────
   │            allocate/                         estimators/
   │   hrp.py · ivp.py · naive.py        covariance.py · rmt.py · mu.py
   │   markowitz_adapter.py              (Ledoit-Wolf · MP clip · shrunk mu)
   ├──────────────────────────────────────────────────────────────────
   │                          cluster/
   │       distance.py · linkage.py · quasidiag.py · ClusterTree
   ├──────────────────────────────────────────────────────────────────
   │   data.py · data_providers/        foundation (no internal deps)
   │   (yfinance->stooq, PIT universe,  _validation · _constants · _typing
   │    risk-free)                       _exceptions · _manifest · _rng
   └──────────────────────────────────────────────────────────────────
```

### Foundation (`_*.py`)

- `_constants.py` — `PERIODS_PER_YEAR = 252` and friends; one source of truth.
- `_validation.py` — input guards (shape, finiteness, sufficient observations).
- `_typing.py` / `_exceptions.py` — shared aliases and the exception taxonomy.
- `_manifest.py` / `_rng.py` — `RunManifest` plus seeded PCG64 substreams. The
  manifest makes a whole run reproducible; the same seed yields byte-identical
  weights, Sharpe gap, and bootstrap CI.

### `cluster/`

`distance.py` builds `d_ij = sqrt(0.5(1 - rho_ij))` and the second-order
co-distance ([ADR-0004](decisions/0004-distance-formula.md)). `linkage.py` wraps
SciPy agglomerative clustering with single linkage as the default and ward /
complete / average as ablations ([ADR-0001](decisions/0001-single-linkage-default.md)).
`quasidiag.py` recovers the dendrogram leaf order. A frozen `ClusterTree`
dataclass carries the linkage matrix and the ordered leaves between stages.

### `estimators/`

The fairness layer. `covariance.py` produces the one estimator shared by every
allocator ([ADR-0002](decisions/0002-shared-covariance-fairness.md)); `rmt.py`
applies the optional Marchenko–Pastur clip; `mu.py` is the James–Stein
grand-mean-shrunk expected-returns estimator used only by max-Sharpe
([ADR-0003](decisions/0003-shrunk-mu.md)).

### `allocate/`

Four allocators behind a common interface: `hrp.py` (the only genuinely new code,
returning a frozen `HRPResult`), `ivp.py`, `naive.py` (1/N, first-class), and
`markowitz_adapter.py` (min-var + max-Sharpe on the shared covariance, using a
Cholesky solve — it never inverts `Sigma`). cvxpy is lazily imported and only for
max-Sharpe.

### `backtest/`

`walk_forward.py` is the no-lookahead engine: anchored/expanding windows, purge +
embargo ([ADR-0005](decisions/0005-purge-embargo.md)), and `signal.shift(1)` at
the rebalance boundary. `costs.py` applies the per-side bps grid against realized
turnover. `stats.py` holds JKM, the Politis–Romano stationary block bootstrap, and
HAC (Andrews 1991) inference.

### `evaluation/`

`dsr.py` computes the Deflated/Probabilistic Sharpe with the full-grid `n_trials`
([ADR-0006](decisions/0006-dsr-multiplicity.md)). `comparison.py` assembles a
frozen `ComparisonResult`. `verdict.py` is a **pure function** mapping
`(jkm_pvalue, deflated_sharpe, bootstrap-CI sign)` to a fixed verdict enum.

## Data flow through one walk-forward window

```
in-sample window  ──►  covariance estimator (Ledoit-Wolf [+ RMT clip])
   (prices)              │              │
                         │              ├─► HRP:  distance ─► linkage ─► quasi-diag
                         │              │        ─► recursive bisection ─► w_HRP
                         │              ├─► IVP:  diag(Sigma)^-1 normalized ─► w_IVP
                         │              ├─► 1/N:  equal weights ─► w_1N
                         │              └─► Markowitz: min-var / max-Sharpe
                         │                     (max-Sharpe also consumes shrunk mu)
                         ▼
              all weights frozen on in-sample data ONLY
                         │
                         ▼  signal.shift(1) at the rebalance boundary
   OOS window  ──►  realized returns · turnover · per-side bps cost ─► net OOS returns
                         │
                         ▼ (aggregate across all windows)
        JKM Sharpe-diff test · stationary block-bootstrap CI · Deflated Sharpe
                         │
                         ▼
            verdict.py  ──►  headline_verdict (pure-derived enum)
```

The headline comparison is **HRP vs 1/N**, which depends only on covariance and
is therefore immune to `mu`-estimation noise.

## Key invariants

The compute core guarantees, and tests enforce:

1. **Simplex.** Every allocator's weights sum to 1 (1e-12) and are non-negative.
2. **No-lookahead.** Shrinkage intensity and the RMT cutoff are deterministic
   functions of the in-sample window — perturbing future data leaves them
   unchanged. Weights apply only to the subsequent OOS window.
3. **Bijection.** `getQuasiDiag` is a permutation of the asset index; the
   reordered covariance stays symmetric.
4. **Scale & permutation invariance.** Rescaling returns or relabeling assets
   does not change the realized allocation.
5. **Robustness.** A block-perfectly-correlated (singular) covariance that breaks
   Markowitz CLA still yields valid HRP weights.
6. **DSR monotonicity.** The Deflated Sharpe is non-increasing in `n_trials`
   (full `(k+2)/4` kurtosis term).
7. **Verdict safety.** `headline_verdict` cannot emit "HRP beats 1/N" while the
   bootstrap CI straddles zero (truth-table unit-tested).
8. **Determinism.** Same `RunManifest` seed -> byte-identical outputs.
9. **Import purity.** Importing any `src/hrp` module triggers no I/O, no network,
   no RNG draw (subprocess-tested).

## Testing strategy

Tests are partitioned by intent under `tests/` (markers in `pyproject.toml`):

- **`unit/`** — isolated kernels: distance formula, `getClusterVar` weighting, the
  verdict truth table.
- **`property/`** (Hypothesis) — the invariants above: simplex, no-lookahead,
  quasi-diag bijection, scale/permutation invariance, DSR monotonicity.
- **`parity/`** — golden checks against independent references: HRP vs
  PyPortfolioOpt (+ a second oracle) at 1e-7, Ledoit–Wolf vs scikit-learn at
  1e-10, JKM vs Memmel's closed form at 1e-8, DSR vs the Bailey–LdP table at 1e-4.
- **`regression/`** — the honest null, locked: HRP OOS variance < Markowitz
  min-var while the HRP-vs-1/N Sharpe-gap CI straddles zero; the singular-cov
  robustness case; the new-constituent exclusion fixture; determinism; cost-grid
  monotonicity; the import-purity subprocess test.
- **`integration/`** — end-to-end pipeline runs (may touch data/network).

Seeded fixtures in `conftest.py` (`one_factor`, `block_correlation`, `pure_noise`,
`singular_cov`) give every layer deterministic, adversarial inputs.

## Backend & frontend boundary

The compute core is decoupled from delivery. The backend vendors
`hrp-portfolio[data]` (not `[all]`) under `api/lib/hrp/` and exposes
`POST /tools/hrp-portfolio/run`, returning summary scalars plus Plotly
`{data, layout}` figures. cvxpy is lazy, the mlfinlab oracle is never importable
in the API, and a compute-budget gate returns 422 / async for grids whose
block-bootstrap resample budget would blow the 512 MB container. The frontend
renders the figures and surfaces the pure-derived `headline_verdict` as the first
thing a visitor reads.
