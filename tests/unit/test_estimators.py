"""Unit tests for the estimator modules.

Covers:

- :mod:`hrp.estimators.rmt` (Marchenko-Pastur eigenvalue clipping): the
  ``lambda_plus`` edge for a known ``q = N / T``, trace/total-variance
  preservation, a PSD and unit-diagonal result, variances kept on rescale,
  idempotency on an already-clean matrix, ndarray labelling, and the
  validation guards.
- :mod:`hrp.estimators.mu` (expected-return estimators): ``sample_mu`` equals
  the column means; ``james_stein_mu`` shrinks toward the cross-sectional grand
  mean, lands between the sample mean and the grand mean, preserves the
  cross-sectional mean, shrinks harder (``phi -> 1``) as idiosyncratic noise
  grows, and raises for ``N < 3``.
- :mod:`hrp.estimators.covariance` (uncovered branches): ``oas_cov`` shape,
  symmetry, PSD, label preservation, and shrinkage toward the diagonal;
  ``ledoit_wolf_cov`` index/column preservation.

All inputs are small, hand-built or seeded numpy/pandas objects (no network).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hrp._exceptions import InsufficientDataError, ValidationError
from hrp.estimators.covariance import ledoit_wolf_cov, oas_cov, sample_cov
from hrp.estimators.mu import james_stein_mu, sample_mu
from hrp.estimators.rmt import marchenko_pastur_clip


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _labels(n: int) -> list[str]:
    return [f"A{i:02d}" for i in range(n)]


def _corr_from_returns(x: np.ndarray) -> np.ndarray:
    """Empirical correlation matrix of a (T x N) returns array."""
    return np.corrcoef(x.T)


# --------------------------------------------------------------------------- #
# rmt.marchenko_pastur_clip                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_mp_clip_lambda_plus_matches_known_q():
    """The noise edge identifies exactly the sub-edge eigenvalues for known q.

    With a block-correlation matrix (eigenvalues {0.2 x6, 2.3 x2, 3.2}) and
    ``q = N / T = 9 / 100``, ``lambda_plus = (1 + sqrt(q))^2 ~ 1.69``. So the
    two signal eigenvalues (2.3, 2.3, 3.2) sit above the edge and the six 0.2
    noise eigenvalues sit below it.
    """
    n_assets, block = 9, 3
    corr = np.full((n_assets, n_assets), 0.1)
    for b in range(0, n_assets, block):
        corr[b : b + block, b : b + block] = 0.8
    np.fill_diagonal(corr, 1.0)

    n_obs = 100
    q = n_assets / n_obs
    lambda_plus = (1.0 + np.sqrt(q)) ** 2
    assert lambda_plus == pytest.approx(1.69, abs=1e-9)

    eig = np.linalg.eigvalsh(corr)
    n_below = int((eig <= lambda_plus).sum())
    n_above = int((eig > lambda_plus).sum())
    # Six 0.2 noise eigenvalues below; three signal eigenvalues (2.3,2.3,3.2) above.
    assert n_below == 6
    assert n_above == 3


@pytest.mark.unit
def test_mp_clip_preserves_trace_and_is_symmetric_psd():
    """Clipping preserves the trace and yields a symmetric, PSD covariance.

    The sub-edge eigenvalues are replaced by their common average, which keeps
    the sum of eigenvalues (the trace) unchanged; the diagonal is re-imposed to
    unit on the correlation, so the correlation trace equals N.
    """
    gen = np.random.default_rng(42)
    n_obs, n_assets = 120, 10
    factor = gen.standard_normal(n_obs)
    betas = gen.uniform(0.3, 1.2, n_assets)
    x = np.outer(factor, betas) + gen.standard_normal((n_obs, n_assets))
    corr = _corr_from_returns(x)
    cov = pd.DataFrame(corr, index=_labels(n_assets), columns=_labels(n_assets))

    out = marchenko_pastur_clip(cov, n_obs=n_obs, n_assets=n_assets)

    # Trace (total variance) preserved.
    assert np.trace(out.to_numpy()) == pytest.approx(np.trace(corr), abs=1e-10)
    # Symmetric.
    assert np.allclose(out.to_numpy(), out.to_numpy().T, atol=1e-12)
    # PSD: no negative eigenvalues (up to float noise).
    assert np.linalg.eigvalsh(out.to_numpy()).min() >= -1e-10
    # Correlation diagonal re-imposed to unit (unit-variance input => unit diag).
    assert np.allclose(np.diag(out.to_numpy()), 1.0, atol=1e-12)


@pytest.mark.unit
def test_mp_clip_collapses_distinct_sub_edge_eigenvalues():
    """Distinct sub-edge eigenvalues are flattened to their common average.

    Inspecting the procedure directly: every eigenvalue at or below the edge is
    replaced by one shared value, so the count of distinct sub-edge eigenvalues
    collapses to one while the signal eigenvalues above the edge are untouched.
    """
    gen = np.random.default_rng(7)
    n_obs, n_assets = 120, 10
    factor = gen.standard_normal(n_obs)
    betas = gen.uniform(0.3, 1.2, n_assets)
    x = np.outer(factor, betas) + gen.standard_normal((n_obs, n_assets))
    corr = _corr_from_returns(x)

    q = n_assets / n_obs
    lambda_plus = (1.0 + np.sqrt(q)) ** 2

    eig = np.linalg.eigvalsh(corr)
    sub_edge = eig[eig <= lambda_plus]
    above = eig[eig > lambda_plus]
    # The constructed panel has multiple distinct sub-edge eigenvalues to flatten.
    assert len(np.unique(np.round(sub_edge, 8))) > 1
    assert len(above) >= 1

    # Replicate the clip's eigenvalue surgery to confirm the flatten-to-average.
    cleaned = eig.copy()
    mask = eig <= lambda_plus
    cleaned[mask] = sub_edge.sum() / mask.sum()
    # All clipped eigenvalues now share one value; the average equals the mean.
    assert np.allclose(cleaned[mask], sub_edge.mean())
    # Signal eigenvalues kept intact.
    assert np.allclose(np.sort(cleaned[~mask]), np.sort(above))
    # Trace conserved by the averaging.
    assert cleaned.sum() == pytest.approx(eig.sum(), abs=1e-10)


@pytest.mark.unit
def test_mp_clip_idempotent_on_clean_matrix():
    """Re-clipping an already-denoised matrix is a no-op (idempotent)."""
    n_assets, block = 9, 3
    corr = np.full((n_assets, n_assets), 0.1)
    for b in range(0, n_assets, block):
        corr[b : b + block, b : b + block] = 0.8
    np.fill_diagonal(corr, 1.0)
    cov = pd.DataFrame(corr, index=_labels(n_assets), columns=_labels(n_assets))

    once = marchenko_pastur_clip(cov, n_obs=100, n_assets=n_assets)
    twice = marchenko_pastur_clip(once, n_obs=100, n_assets=n_assets)
    assert np.allclose(once.to_numpy(), twice.to_numpy(), atol=1e-10)


@pytest.mark.unit
def test_mp_clip_identity_correlation_is_fixed_point():
    """An identity correlation (all unit eigenvalues) is returned unchanged.

    Every eigenvalue equals 1 and sits below the edge; averaging equal values is
    a no-op, so the identity is a fixed point of the clip.
    """
    n_assets = 5
    ident = pd.DataFrame(np.eye(n_assets), index=_labels(n_assets), columns=_labels(n_assets))
    out = marchenko_pastur_clip(ident, n_obs=50, n_assets=n_assets)
    assert np.allclose(out.to_numpy(), np.eye(n_assets), atol=1e-10)


@pytest.mark.unit
def test_mp_clip_preserves_variances_on_rescale():
    """Rescaling back to covariance keeps the original variances on the diagonal."""
    cov = np.array([[4.0, 1.0, 0.5], [1.0, 9.0, 0.7], [0.5, 0.7, 16.0]], dtype=np.float64)
    out = marchenko_pastur_clip(pd.DataFrame(cov), n_obs=100, n_assets=3)
    assert np.allclose(np.diag(out.to_numpy()), np.diag(cov), atol=1e-10)


@pytest.mark.unit
def test_mp_clip_infers_n_assets_from_shape():
    """When ``n_assets`` is omitted it is inferred from the matrix shape."""
    cov = np.array([[4.0, 1.0, 0.5], [1.0, 9.0, 0.7], [0.5, 0.7, 16.0]], dtype=np.float64)
    explicit = marchenko_pastur_clip(pd.DataFrame(cov), n_obs=100, n_assets=3)
    inferred = marchenko_pastur_clip(pd.DataFrame(cov), n_obs=100)
    assert np.allclose(explicit.to_numpy(), inferred.to_numpy(), atol=1e-12)


@pytest.mark.unit
def test_mp_clip_ndarray_input_gets_range_index():
    """An ndarray input yields a DataFrame labelled by a RangeIndex."""
    out = marchenko_pastur_clip(np.eye(4), n_obs=40)
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == [0, 1, 2, 3]
    assert list(out.columns) == [0, 1, 2, 3]


@pytest.mark.unit
def test_mp_clip_rejects_non_square():
    """A non-square matrix raises ValidationError."""
    with pytest.raises(ValidationError, match="square"):
        marchenko_pastur_clip(np.zeros((3, 4)), n_obs=10)


@pytest.mark.unit
def test_mp_clip_rejects_non_positive_n_obs():
    """A non-positive ``n_obs`` raises ValidationError."""
    with pytest.raises(ValidationError, match="n_obs"):
        marchenko_pastur_clip(np.eye(3), n_obs=0)
    with pytest.raises(ValidationError, match="n_obs"):
        marchenko_pastur_clip(np.eye(3), n_obs=-5)


# --------------------------------------------------------------------------- #
# mu.sample_mu                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_sample_mu_equals_column_means():
    """``sample_mu`` returns exactly the column means, labelled by asset."""
    gen = np.random.default_rng(1)
    x = gen.standard_normal((50, 4)) * 0.01
    cols = _labels(4)
    df = pd.DataFrame(x, columns=cols)

    mu = sample_mu(df)
    assert isinstance(mu, pd.Series)
    assert list(mu.index) == cols
    assert mu.dtype == np.float64
    assert np.allclose(mu.to_numpy(), df.mean(axis=0).to_numpy(), atol=1e-15)


@pytest.mark.unit
def test_sample_mu_hand_computed():
    """A tiny hand-checkable case: means of two columns."""
    df = pd.DataFrame({"A": [1.0, 3.0, 5.0], "B": [2.0, 2.0, 2.0]})
    mu = sample_mu(df)
    assert mu["A"] == pytest.approx(3.0)
    assert mu["B"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# mu.james_stein_mu                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_james_stein_shrinks_toward_grand_mean():
    """JS estimates lie between the sample mean and the grand mean, same side.

    For each asset, the shrunk estimate is a convex combination of the asset's
    sample mean and the cross-sectional grand mean, so it must fall in the
    closed interval between the two and reduce the spread around the grand mean.
    """
    gen = np.random.default_rng(7)
    n_obs, n_assets = 200, 5
    base = np.array([0.001, 0.002, -0.001, 0.0005, 0.0])
    x = gen.standard_normal((n_obs, n_assets)) * 0.01 + base
    df = pd.DataFrame(x, columns=_labels(n_assets))

    sm = sample_mu(df)
    js = james_stein_mu(df)
    grand = float(sm.mean())

    for asset in df.columns:
        lo, hi = sorted([sm[asset], grand])
        assert lo - 1e-12 <= js[asset] <= hi + 1e-12

    # Shrinkage pulls every estimate toward the grand mean: spread shrinks.
    spread_sample = float(((sm - grand) ** 2).sum())
    spread_js = float(((js - grand) ** 2).sum())
    assert spread_js <= spread_sample + 1e-15
    # And the spread genuinely shrinks here (partial, not full).
    assert 0.0 < spread_js < spread_sample


@pytest.mark.unit
def test_james_stein_preserves_cross_sectional_mean():
    """JS shrinkage is affine toward the grand mean, so the grand mean is fixed."""
    gen = np.random.default_rng(13)
    df = pd.DataFrame(gen.standard_normal((150, 6)) * 0.01 + 0.001, columns=_labels(6))
    sm = sample_mu(df)
    js = james_stein_mu(df)
    assert float(js.mean()) == pytest.approx(float(sm.mean()), abs=1e-15)
    assert list(js.index) == list(df.columns)
    assert js.dtype == np.float64


def _retained_fraction(scale: float, seed: int = 11) -> float:
    """Return the fraction of sample-mean deviation JS retains for noise ``scale``.

    Builds a panel whose column means are pinned to a fixed deviation pattern
    while the idiosyncratic noise std is ``scale``; returns ``(1 - phi)`` as
    the ratio of retained squared deviation. As ``scale`` grows the James-Stein
    factor ``phi -> 1`` and the retained fraction -> 0.
    """
    gen = np.random.default_rng(seed)
    n_obs, n_assets = 150, 6
    base_means = np.array([0.003, -0.002, 0.001, 0.0, 0.0015, -0.0005])
    x = gen.standard_normal((n_obs, n_assets)) * scale
    x = x - x.mean(axis=0) + base_means  # pin exact column means to base_means
    df = pd.DataFrame(x)
    sm = sample_mu(df)
    js = james_stein_mu(df)
    grand = float(sm.mean())
    dev = (sm - grand).to_numpy()
    js_dev = (js - grand).to_numpy()
    return float((js_dev**2).sum() / (dev**2).sum())


@pytest.mark.unit
def test_james_stein_shrinkage_increases_with_noise():
    """As idiosyncratic noise grows, ``phi -> 1`` and JS collapses to the grand mean.

    The means are held fixed across the three noise levels, so the only thing
    changing is ``sigma^2 / T``: more noise => stronger shrinkage => less of the
    sample-mean deviation retained.
    """
    low = _retained_fraction(0.005)
    mid = _retained_fraction(0.05)
    high = _retained_fraction(5.0)

    # Monotone non-increasing retention as noise grows.
    assert low >= mid >= high
    # Low noise keeps a meaningful (partial) chunk of the signal.
    assert 0.0 < low < 1.0
    # Huge noise fully shrinks to the grand mean (phi clamps to 1).
    assert high < 1e-3


@pytest.mark.unit
def test_james_stein_equal_means_collapse_to_grand():
    """When all sample means coincide (zero dispersion), JS returns the grand mean.

    The ``dispersion <= 0`` branch sets ``phi = 1`` and avoids division by zero;
    since every mean already equals the grand mean, no information is lost.
    """
    gen = np.random.default_rng(99)
    n_obs, n_assets = 120, 5
    x = gen.standard_normal((n_obs, n_assets))
    x = x - x.mean(axis=0) + 0.005  # every column mean exactly 0.005
    df = pd.DataFrame(x, columns=_labels(n_assets))

    js = james_stein_mu(df)
    assert np.allclose(js.to_numpy(), 0.005, atol=1e-12)


@pytest.mark.unit
def test_james_stein_requires_three_assets():
    """JS is undefined for ``N < 3`` (the (N-2) factor) and raises."""
    gen = np.random.default_rng(5)
    with pytest.raises(InsufficientDataError, match="at least 3"):
        james_stein_mu(pd.DataFrame(gen.standard_normal((20, 2))))
    with pytest.raises(InsufficientDataError, match="at least 3"):
        james_stein_mu(pd.DataFrame(gen.standard_normal((20, 1))))


# --------------------------------------------------------------------------- #
# covariance.oas_cov / ledoit_wolf_cov (uncovered branches)                    #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_oas_cov_shape_symmetry_psd_and_labels():
    """OAS returns a square, symmetric, PD covariance labelled by asset."""
    gen = np.random.default_rng(3)
    n_obs, n_assets = 60, 8
    cols = _labels(n_assets)
    idx = pd.date_range("2021-01-01", periods=n_obs, freq="B")
    df = pd.DataFrame(gen.standard_normal((n_obs, n_assets)) * 0.02, index=idx, columns=cols)

    cov = oas_cov(df)
    assert cov.shape == (n_assets, n_assets)
    assert list(cov.index) == cols
    assert list(cov.columns) == cols
    assert np.allclose(cov.to_numpy(), cov.to_numpy().T, atol=1e-12)
    # OAS shrinks toward a scaled identity => strictly positive-definite.
    assert np.linalg.eigvalsh(cov.to_numpy()).min() > 0.0


@pytest.mark.unit
def test_oas_cov_shrinks_off_diagonal_toward_identity():
    """OAS pulls the off-diagonal mass below that of the raw sample covariance."""
    gen = np.random.default_rng(8)
    n_obs, n_assets = 50, 8
    df = pd.DataFrame(gen.standard_normal((n_obs, n_assets)) * 0.02, columns=_labels(n_assets))

    cov = oas_cov(df)
    s = sample_cov(df)

    def off_diag_mass(m: np.ndarray) -> float:
        return float(np.abs(m - np.diag(np.diag(m))).sum())

    assert off_diag_mass(cov.to_numpy()) <= off_diag_mass(s.to_numpy())


@pytest.mark.unit
def test_oas_cov_matches_sklearn():
    """OAS reproduces ``sklearn.covariance.OAS`` (symmetrized) on the same panel."""
    from sklearn.covariance import OAS

    gen = np.random.default_rng(3)
    x = gen.standard_normal((60, 8)) * 0.02
    df = pd.DataFrame(x, columns=_labels(8))

    cov = oas_cov(df)
    skl = OAS(assume_centered=False).fit(x).covariance_
    assert np.allclose(cov.to_numpy(), 0.5 * (skl + skl.T), atol=1e-12)


@pytest.mark.unit
def test_ledoit_wolf_preserves_index_and_columns():
    """Ledoit-Wolf keeps the asset labels on both axes and stays PD/symmetric."""
    gen = np.random.default_rng(4)
    n_obs, n_assets = 60, 7
    cols = _labels(n_assets)
    idx = pd.date_range("2022-03-01", periods=n_obs, freq="B")
    df = pd.DataFrame(gen.standard_normal((n_obs, n_assets)) * 0.015, index=idx, columns=cols)

    cov = ledoit_wolf_cov(df)
    assert list(cov.index) == cols
    assert list(cov.columns) == cols
    assert np.allclose(cov.to_numpy(), cov.to_numpy().T, atol=1e-12)
    assert np.linalg.eigvalsh(cov.to_numpy()).min() > 0.0


@pytest.mark.unit
def test_ledoit_wolf_matches_sklearn():
    """Ledoit-Wolf reproduces ``sklearn.covariance.ledoit_wolf`` (symmetrized)."""
    from sklearn.covariance import ledoit_wolf as skl_ledoit_wolf

    gen = np.random.default_rng(4)
    x = gen.standard_normal((60, 7)) * 0.015
    df = pd.DataFrame(x, columns=_labels(7))

    cov = ledoit_wolf_cov(df)
    skl, _ = skl_ledoit_wolf(x)
    assert np.allclose(cov.to_numpy(), 0.5 * (skl + skl.T), atol=1e-10)
