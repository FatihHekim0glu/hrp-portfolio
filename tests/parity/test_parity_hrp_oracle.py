"""Parity tests against independent reference implementations.

Two independent oracles pin the HRP library to existing, trusted code:

* :func:`hrp.allocate.hrp.hrp_allocate` is validated against
  PyPortfolioOpt's :class:`pypfopt.hierarchical_portfolio.HRPOpt` on an
  identical covariance and the paper-default **single** linkage, to ``~1e-7``.
* :func:`hrp.estimators.covariance.ledoit_wolf_cov` is validated against
  :class:`sklearn.covariance.LedoitWolf` to ``1e-10``.

The PyPortfolioOpt oracle is optional: ``pytest.importorskip('pypfopt')``
makes the whole module SKIP cleanly when the dev-only reference is absent, so
the parity gate never turns into a hard import failure in lean environments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hrp.allocate.hrp import hrp_allocate
from hrp.estimators.covariance import ledoit_wolf_cov

pytestmark = pytest.mark.parity


# --------------------------------------------------------------------------- #
# Ledoit-Wolf covariance vs sklearn.covariance.LedoitWolf  (1e-10)             #
# --------------------------------------------------------------------------- #
def test_ledoit_wolf_matches_sklearn_to_1e_10(one_factor_returns: pd.DataFrame) -> None:
    """``ledoit_wolf_cov`` must equal ``sklearn.covariance.LedoitWolf`` to 1e-10.

    The library's estimator is a thin, label-preserving wrapper around sklearn's
    closed-form shrinkage; this pins it to the reference numerically (not merely
    "close"), guarding against any future drift in centering/normalization.
    """
    from sklearn.covariance import LedoitWolf

    x = one_factor_returns.to_numpy(dtype=np.float64)
    reference = LedoitWolf(assume_centered=False).fit(x).covariance_

    ours = ledoit_wolf_cov(one_factor_returns)

    # Labels preserved from the returns columns.
    assert list(ours.index) == list(one_factor_returns.columns)
    assert list(ours.columns) == list(one_factor_returns.columns)

    np.testing.assert_allclose(
        ours.to_numpy(dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        rtol=0.0,
        atol=1e-10,
    )


def test_ledoit_wolf_matches_sklearn_pure_noise(pure_noise_returns: pd.DataFrame) -> None:
    """Parity holds on the no-structure null panel too (different shrinkage regime)."""
    from sklearn.covariance import LedoitWolf

    x = pure_noise_returns.to_numpy(dtype=np.float64)
    reference = LedoitWolf(assume_centered=False).fit(x).covariance_

    ours = ledoit_wolf_cov(pure_noise_returns).to_numpy(dtype=np.float64)
    np.testing.assert_allclose(ours, np.asarray(reference, dtype=np.float64), rtol=0.0, atol=1e-10)


# --------------------------------------------------------------------------- #
# hrp_allocate vs PyPortfolioOpt HRPOpt  (single linkage, identical cov, 1e-7) #
# --------------------------------------------------------------------------- #
def _pypfopt_hrp_weights(cov: pd.DataFrame) -> pd.Series:
    """Run PyPortfolioOpt HRP on a *given* covariance and single linkage.

    PyPortfolioOpt's :class:`HRPOpt` accepts a pre-computed covariance via
    ``cov_matrix=...``; passing the same matrix both allocators see removes the
    covariance estimator as a confound, so any difference is purely the
    clustering / recursive-bisection algebra.
    """
    # Import the EXACT submodule used (not just the package shell), so a
    # partial/absent PyPortfolioOpt install is caught by ``importorskip`` and the
    # parity test skips cleanly instead of erroring inside this helper.
    hp = pytest.importorskip("pypfopt.hierarchical_portfolio")
    HRPOpt = hp.HRPOpt

    opt = HRPOpt(cov_matrix=cov)
    opt.optimize(linkage_method="single")
    weights = pd.Series(opt.clean_weights(rounding=None), dtype="float64")
    # Align to the covariance's asset order for an apples-to-apples comparison.
    return weights.reindex(cov.columns).astype("float64")


def _shared_cov(one_factor_returns: pd.DataFrame) -> pd.DataFrame:
    """A shared, well-conditioned shrunk covariance fed to BOTH allocators."""
    return ledoit_wolf_cov(one_factor_returns)


def test_hrp_weights_match_pypfopt_single_linkage(one_factor_returns: pd.DataFrame) -> None:
    """``hrp_allocate`` weights must match PyPortfolioOpt HRP to ~1e-7.

    Both allocators receive the IDENTICAL shrunk covariance and single linkage,
    so the only thing under test is the hand-rolled
    ``correl_dist -> euclidean_codistance -> linkage -> quasi-diag -> recursive
    bisection`` pipeline against an independent reference of the same algorithm.
    """
    pytest.importorskip("pypfopt.hierarchical_portfolio")

    cov = _shared_cov(one_factor_returns)

    ours = hrp_allocate(one_factor_returns, cov=cov, linkage_method="single").weights
    theirs = _pypfopt_hrp_weights(cov)

    # Same asset universe / order before numeric comparison.
    assert list(ours.index) == list(theirs.index)

    np.testing.assert_allclose(
        ours.to_numpy(dtype=np.float64),
        theirs.to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-7,
    )
    # Sanity: both are valid simplex weights.
    assert ours.sum() == pytest.approx(1.0, abs=1e-12)
    assert float(ours.min()) >= -1e-12


def test_hrp_weights_match_pypfopt_block_structure(
    one_factor_returns: pd.DataFrame,
    block_correlation_cov: pd.DataFrame,
) -> None:
    """Parity on a block-structured covariance (the case HRP clustering targets).

    The ``block_correlation_cov`` fixture is a unit-variance correlation matrix,
    so it doubles as a covariance. Both allocators run on it directly with single
    linkage; a returns panel is supplied only so the public entry point can
    recover labels — the injected ``cov`` is what is actually clustered.
    """
    pytest.importorskip("pypfopt.hierarchical_portfolio")

    cov = block_correlation_cov
    # A label-only returns frame matching the covariance's nine assets.
    returns = pd.DataFrame(np.zeros((4, cov.shape[0])), columns=list(cov.columns))

    ours = hrp_allocate(returns, cov=cov, linkage_method="single").weights
    theirs = _pypfopt_hrp_weights(cov)

    assert list(ours.index) == list(theirs.index)
    np.testing.assert_allclose(
        ours.to_numpy(dtype=np.float64),
        theirs.to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-7,
    )
