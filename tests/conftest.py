"""Shared, seeded test fixtures.

Every fixture is deterministic (driven by :func:`hrp._rng.make_rng`) and returns
pandas objects, so tests across the suite share identical synthetic data with
known structure:

- ``one_factor_returns`` - a single common factor plus idiosyncratic noise
  (positive average correlation; a clean, well-conditioned panel).
- ``block_correlation_cov`` - a block-structured covariance with high
  within-block and low cross-block correlation (the case HRP clustering targets).
- ``pure_noise_returns`` - independent assets with no common structure (the null:
  clustering should find nothing meaningful).
- ``singular_cov`` - a block-perfectly-correlated, non-invertible covariance on
  which Markowitz CLA fails but HRP must still produce valid weights.
- ``de_prado_example`` - a small, known correlation matrix from de Prado's worked
  example, used by the parity oracle.

Importing this module has no side effects beyond fixture registration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hrp._rng import make_rng

_SEED = 20260614


def _asset_labels(n: int) -> list[str]:
    """Return ``n`` deterministic asset labels ``A00, A01, ...``."""
    return [f"A{i:02d}" for i in range(n)]


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded PCG64 generator shared by tests that need raw randomness."""
    return make_rng(_SEED)


@pytest.fixture
def one_factor_returns() -> pd.DataFrame:
    """One-factor returns panel: common factor + idiosyncratic noise.

    Shape ``(750, 8)``. Each asset loads on a single common factor with a random
    positive beta, producing a positive average pairwise correlation and a
    well-conditioned covariance.
    """
    gen = make_rng(_SEED)
    n_obs, n_assets = 750, 8
    factor = gen.standard_normal(n_obs) * 0.01
    betas = gen.uniform(0.5, 1.5, size=n_assets)
    idio = gen.standard_normal((n_obs, n_assets)) * 0.01
    data = np.outer(factor, betas) + idio
    index = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    return pd.DataFrame(data, index=index, columns=_asset_labels(n_assets))


@pytest.fixture
def block_correlation_cov() -> pd.DataFrame:
    """Block-structured covariance: high within-block, low cross-block correlation.

    Shape ``(9, 9)``, three blocks of three assets. Within a block the
    correlation is ``0.8``; across blocks it is ``0.1``. Unit variances, so the
    covariance equals the correlation. This is the structure HRP clustering is
    designed to discover.
    """
    n_assets = 9
    block_size = 3
    corr = np.full((n_assets, n_assets), 0.1)
    for b in range(0, n_assets, block_size):
        corr[b : b + block_size, b : b + block_size] = 0.8
    np.fill_diagonal(corr, 1.0)
    labels = _asset_labels(n_assets)
    return pd.DataFrame(corr, index=labels, columns=labels)


@pytest.fixture
def pure_noise_returns() -> pd.DataFrame:
    """Independent-asset returns panel: the no-structure null.

    Shape ``(750, 8)``. Every column is i.i.d. Gaussian noise with no common
    factor, so the population correlation is the identity and clustering should
    find no meaningful structure.
    """
    gen = make_rng(_SEED + 1)
    n_obs, n_assets = 750, 8
    data = gen.standard_normal((n_obs, n_assets)) * 0.01
    index = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    return pd.DataFrame(data, index=index, columns=_asset_labels(n_assets))


@pytest.fixture
def singular_cov() -> pd.DataFrame:
    """Block-perfectly-correlated, singular (non-invertible) covariance.

    Shape ``(6, 6)``, two blocks of three perfectly-correlated assets (within-block
    correlation ``1.0``). The matrix is rank-deficient, so Markowitz CLA cannot
    invert it; HRP, which never inverts the full covariance, must still return
    valid simplex weights (the paper's headline robustness claim, encoded as a
    regression test).
    """
    n_assets = 6
    block_size = 3
    corr = np.zeros((n_assets, n_assets))
    for b in range(0, n_assets, block_size):
        corr[b : b + block_size, b : b + block_size] = 1.0
    # Off-block correlation kept mild but the perfect within-block blocks already
    # make the matrix rank-deficient (rank 2 < 6).
    np.fill_diagonal(corr, 1.0)
    labels = _asset_labels(n_assets)
    return pd.DataFrame(corr, index=labels, columns=labels)


@pytest.fixture
def de_prado_example() -> pd.DataFrame:
    """A small, known correlation matrix for the parity oracle.

    A ``4 x 4`` correlation matrix with two clearly-correlated pairs, used to
    pin the hand-rolled ``getQuasiDiag`` / ``getRecBipart`` against PyPortfolioOpt
    (and a second reference) on a hand-checkable example.
    """
    corr = np.array(
        [
            [1.00, 0.70, 0.20, 0.10],
            [0.70, 1.00, 0.15, 0.05],
            [0.20, 0.15, 1.00, 0.80],
            [0.10, 0.05, 0.80, 1.00],
        ]
    )
    labels = _asset_labels(4)
    return pd.DataFrame(corr, index=labels, columns=labels)
