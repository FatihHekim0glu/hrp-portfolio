"""Regression tests: locked behaviour and pinned horse-race snapshots.

Three locked behaviours:

* SINGULAR ROBUSTNESS - the ``singular_cov`` fixture (block-perfectly-correlated,
  rank-deficient) still yields *valid simplex* HRP weights with no exception.
  This is de Prado's headline robustness claim (Markowitz CLA cannot invert this
  matrix; HRP never inverts the full covariance).
* DETERMINISM - identical inputs and the same seed produce byte-identical
  weights and bootstrap CIs.
* PINNED HORSE-RACE SNAPSHOT - on a deterministic synthetic panel, HRP's
  out-of-sample variance is ``<=`` minimum-variance's OOS variance, while the
  HRP-vs-1/N Sharpe-gap bootstrap CI straddles zero (the honest null is locked,
  not papered over).

The synthetic-panel builder and allocator configuration are fixed in-module so
the snapshot is exactly reproducible; any change to the engine or estimators
that moves these numbers is surfaced loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hrp._rng import make_rng
from hrp.allocate.hrp import hrp_allocate
from hrp.allocate.markowitz_adapter import min_var_weights
from hrp.allocate.naive import naive_weights
from hrp.backtest.stats import annualized_vol
from hrp.backtest.walk_forward import walk_forward_backtest
from hrp.estimators.covariance import sample_cov
from hrp.evaluation.comparison import block_bootstrap_sharpe_gap

pytestmark = pytest.mark.regression


# --------------------------------------------------------------------------- #
# Locked horse-race configuration (pin the snapshot)                           #
# --------------------------------------------------------------------------- #
_PANEL_SEED = 20260614
_BOOTSTRAP_SEED = 7
_LOOKBACK = 20
_N_BOOTSTRAP = 2000


def _synthetic_block_panel(
    *,
    seed: int = _PANEL_SEED,
    n_obs: int = 600,
    n_blocks: int = 3,
    per_block: int = 4,
) -> pd.DataFrame:
    """Deterministic block-factor returns panel for the pinned horse race.

    Twelve assets in three correlated blocks share a global factor plus a
    per-block factor and idiosyncratic noise. With a SHORT lookback and the
    deliberately fragile SAMPLE covariance, minimum-variance over-fits estimation
    error (its OOS variance balloons) while HRP - which never inverts the full
    covariance - stays well-conditioned. This is the regime in which de Prado's
    OOS-variance advantage is real, so the snapshot is honest rather than rigged.
    """
    gen = make_rng(seed)
    n_assets = n_blocks * per_block
    glob = gen.standard_normal(n_obs) * 0.006
    data = np.zeros((n_obs, n_assets), dtype=np.float64)
    for b in range(n_blocks):
        block_factor = gen.standard_normal(n_obs) * 0.008
        for k in range(per_block):
            j = b * per_block + k
            beta = gen.uniform(0.6, 1.4)
            data[:, j] = beta * glob + 0.9 * block_factor + gen.standard_normal(n_obs) * 0.01
    labels = [f"A{i:02d}" for i in range(n_assets)]
    index = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    return pd.DataFrame(data, index=index, columns=labels)


def _hrp_allocator(window: pd.DataFrame) -> pd.Series:
    """HRP on the shared sample covariance of the in-sample window."""
    return hrp_allocate(window, cov=sample_cov(window)).weights


def _min_var_allocator(window: pd.DataFrame) -> pd.Series:
    """Unconstrained minimum-variance on the same sample covariance.

    ``long_only=False`` exposes the closed-form solution, which over-fits the
    noisy sample covariance - exactly the failure HRP is meant to avoid.
    """
    return min_var_weights(sample_cov(window), long_only=False)


def _naive_allocator(window: pd.DataFrame) -> pd.Series:
    """1/N equal-weight - the estimation-error-free yardstick."""
    return naive_weights(list(window.columns))


@pytest.fixture(scope="module")
def horse_race_returns() -> dict[str, pd.Series]:
    """Run the three allocators once through the walk-forward engine.

    Module-scoped so the (mildly expensive) backtests run a single time and feed
    every assertion in this file from the identical OOS return series.
    """
    panel = _synthetic_block_panel()
    common = {"lookback_window": _LOOKBACK, "rebalance": "monthly", "cost_bps": 10.0}
    hrp_bt = walk_forward_backtest(panel, _hrp_allocator, **common)
    mv_bt = walk_forward_backtest(panel, _min_var_allocator, **common)
    nv_bt = walk_forward_backtest(panel, _naive_allocator, **common)
    return {
        "hrp": hrp_bt.oos_returns,
        "min_var": mv_bt.oos_returns,
        "naive": nv_bt.oos_returns,
    }


# --------------------------------------------------------------------------- #
# SINGULAR ROBUSTNESS - HRP must survive a non-invertible covariance           #
# --------------------------------------------------------------------------- #
def test_hrp_survives_singular_covariance(singular_cov: pd.DataFrame) -> None:
    """HRP yields valid simplex weights on a rank-deficient covariance.

    The ``singular_cov`` fixture is block-perfectly-correlated (rank 2 < 6), so
    Markowitz CLA cannot invert it. HRP must return a clean allocation - no
    exception, weights non-negative and summing to one over exactly the input
    assets - which is the paper's robustness headline.
    """
    # Sanity: the fixture really is rank-deficient (otherwise the test is vacuous).
    assert np.linalg.matrix_rank(singular_cov.to_numpy(dtype=np.float64)) < singular_cov.shape[0]

    # A label-only returns frame so the entry point can recover asset names.
    returns = pd.DataFrame(np.zeros((4, singular_cov.shape[0])), columns=list(singular_cov.columns))

    result = hrp_allocate(returns, cov=singular_cov)
    weights = result.weights

    # Valid simplex: same universe, non-negative, sums to one.
    assert list(weights.index) == list(singular_cov.columns)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert float(weights.min()) >= -1e-12
    assert np.all(np.isfinite(weights.to_numpy(dtype=np.float64)))
    # The leaf order is a valid permutation of the asset positions.
    assert sorted(result.quasidiag_order) == list(range(singular_cov.shape[0]))


def test_min_var_fails_where_hrp_survives(singular_cov: pd.DataFrame) -> None:
    """The contrast: Markowitz min-variance cannot factor the singular covariance.

    Pins the asymmetry that motivates HRP - same input, one allocator raises,
    the other allocates. Guards against a future change that silently "repairs"
    the singular matrix and erases the comparison.
    """
    from hrp._exceptions import SingularCovarianceError

    with pytest.raises(SingularCovarianceError):
        min_var_weights(singular_cov, long_only=False)


# --------------------------------------------------------------------------- #
# DETERMINISM - same seed / same inputs -> identical outputs                   #
# --------------------------------------------------------------------------- #
def test_hrp_weights_are_deterministic(one_factor_returns: pd.DataFrame) -> None:
    """Two HRP runs on identical inputs produce byte-identical weights."""
    cov = sample_cov(one_factor_returns)
    w1 = hrp_allocate(one_factor_returns, cov=cov).weights
    w2 = hrp_allocate(one_factor_returns, cov=cov).weights
    # Bit-for-bit equality (no tolerance): the pipeline is fully deterministic.
    assert list(w1.index) == list(w2.index)
    assert np.array_equal(w1.to_numpy(dtype=np.float64), w2.to_numpy(dtype=np.float64))


def test_bootstrap_ci_is_deterministic(horse_race_returns: dict[str, pd.Series]) -> None:
    """The block bootstrap is reproducible for a fixed seed (RNG, not wall clock)."""
    a, b = horse_race_returns["hrp"], horse_race_returns["naive"]
    c1 = block_bootstrap_sharpe_gap(a, b, n_bootstrap=500, confidence=0.95, seed=_BOOTSTRAP_SEED)
    c2 = block_bootstrap_sharpe_gap(a, b, n_bootstrap=500, confidence=0.95, seed=_BOOTSTRAP_SEED)
    assert c1.ci_low == c2.ci_low
    assert c1.ci_high == c2.ci_high
    assert c1.sharpe_gap == c2.sharpe_gap


def test_bootstrap_ci_seed_changes_result(horse_race_returns: dict[str, pd.Series]) -> None:
    """A different seed perturbs the CI bounds (the seed actually drives the RNG)."""
    a, b = horse_race_returns["hrp"], horse_race_returns["naive"]
    c_seed7 = block_bootstrap_sharpe_gap(a, b, n_bootstrap=500, confidence=0.95, seed=7)
    c_seed8 = block_bootstrap_sharpe_gap(a, b, n_bootstrap=500, confidence=0.95, seed=8)
    # The point estimate is seed-free; only the resampled CI bounds move.
    assert c_seed7.sharpe_gap == c_seed8.sharpe_gap
    assert (c_seed7.ci_low, c_seed7.ci_high) != (c_seed8.ci_low, c_seed8.ci_high)


# --------------------------------------------------------------------------- #
# PINNED HORSE-RACE SNAPSHOT - HRP OOS variance <= min-var; null CI straddles 0 #
# --------------------------------------------------------------------------- #
def test_hrp_oos_variance_beats_min_variance(horse_race_returns: dict[str, pd.Series]) -> None:
    """HRP's OOS variance is at most minimum-variance's (de Prado's claim).

    With the fragile sample covariance and a short lookback, unconstrained
    minimum-variance over-fits estimation noise and its realized OOS variance
    blows up; HRP stays controlled. The inequality is the pinned behaviour.
    """
    hrp_var = float(horse_race_returns["hrp"].var(ddof=1))
    mv_var = float(horse_race_returns["min_var"].var(ddof=1))

    assert hrp_var <= mv_var
    # Snapshot the magnitudes so a silent drift in either engine is caught.
    assert hrp_var == pytest.approx(6.298141426394225e-05, rel=1e-6)
    assert mv_var == pytest.approx(0.00015962386605742148, rel=1e-6)
    # And the annualized-vol view (consumed by the evaluation layer) agrees.
    assert annualized_vol(horse_race_returns["hrp"]) <= annualized_vol(
        horse_race_returns["min_var"]
    )


def test_hrp_vs_naive_sharpe_gap_ci_straddles_zero(
    horse_race_returns: dict[str, pd.Series],
) -> None:
    """The HRP-vs-1/N Sharpe-gap 95% CI straddles zero - honest null, locked.

    HRP need not beat the brutal 1/N benchmark on Sharpe; pretending otherwise
    would be the dishonest result this suite exists to prevent. The pinned
    snapshot asserts the bootstrap CI contains zero and pins its exact bounds.
    """
    cmp = block_bootstrap_sharpe_gap(
        horse_race_returns["hrp"],
        horse_race_returns["naive"],
        n_bootstrap=_N_BOOTSTRAP,
        confidence=0.95,
        seed=_BOOTSTRAP_SEED,
    )

    # The honest null: the confidence interval brackets zero.
    assert cmp.ci_low < 0.0 < cmp.ci_high, (
        f"Sharpe-gap CI [{cmp.ci_low:.4f}, {cmp.ci_high:.4f}] must straddle zero "
        "(HRP is not claimed to beat 1/N on Sharpe)."
    )

    # Pinned snapshot of the exact figures for this locked configuration.
    assert cmp.sharpe_gap == pytest.approx(-0.10234753691209464, rel=1e-6)
    assert cmp.ci_low == pytest.approx(-0.4006874895719483, rel=1e-6)
    assert cmp.ci_high == pytest.approx(0.18435484541767222, rel=1e-6)
    assert cmp.n_bootstrap == _N_BOOTSTRAP
    # A non-significant gap is consistent with a CI that straddles zero.
    assert cmp.jkm_pvalue > 0.05
