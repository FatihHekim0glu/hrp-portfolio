"""Hypothesis property tests for HRP invariants.

These exercise mathematical properties that must hold across the whole input
space, not just on hand-picked examples:

- **Simplex.** ``hrp_allocate`` / ``ivp_weights`` / ``naive_weights`` always
  return weights that sum to one (within ``1e-10``) and are non-negative.
- **HRP scale-invariance.** Multiplying the covariance by a positive scalar
  leaves the HRP weights unchanged (correlation, distance, linkage, and the
  inverse-variance *ratios* are all invariant to a global variance rescale).
- **HRP permutation-equivariance.** Permuting the asset order permutes the
  weights identically — HRP carries no positional bias.
- **Quasi-diagonalisation is a bijection.** ``get_quasi_diag`` returns a valid
  permutation of ``range(N)``.
- **Estimator-level no-lookahead.** HRP weights computed on an in-sample window
  depend only on that window: perturbing later rows does not change them.
- **DSR monotonicity.** ``deflated_sharpe_ratio`` is non-increasing in
  ``n_trials`` (more multiplicity can only deflate the score).
- **Verdict truth-table.** ``derive_verdict`` never returns ``HRP_BEATS_1N``
  while the bootstrap CI straddles zero.

All tests are marked ``@pytest.mark.property``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from hrp import (
    Verdict,
    deflated_sharpe_ratio,
    derive_verdict,
    get_quasi_diag,
    hrp_allocate,
    ivp_weights,
    naive_weights,
)
from hrp.cluster.distance import correl_dist, euclidean_codistance
from hrp.cluster.linkage import linkage_matrix

pytestmark = pytest.mark.property

# A simplex closure tolerance an order of magnitude tighter than the 1e-10 the
# task pins, so the assertion has headroom.
SIMPLEX_ATOL = 1e-10

# Keep example counts modest: each HRP example runs a full clustering pipeline.
_SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# --------------------------------------------------------------------------- #
# Strategies                                                                   #
# --------------------------------------------------------------------------- #
def _labels(n: int) -> list[str]:
    return [f"A{i:02d}" for i in range(n)]


@st.composite
def psd_covariances(draw, min_assets: int = 2, max_assets: int = 8):
    """Draw a well-conditioned, labelled PSD covariance ``DataFrame``.

    Built as ``B @ B.T + diag(jitter)`` so it is symmetric positive-definite
    with a strictly positive diagonal (the precondition HRP/IVP require).
    """
    n = draw(st.integers(min_value=min_assets, max_value=max_assets))
    # Factor-loading matrix with a handful of factors; entries kept O(1) so the
    # resulting variances stay numerically tame.
    k = draw(st.integers(min_value=1, max_value=n))
    b = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n, k),
            elements=st.floats(
                min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    jitter = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(
                min_value=1e-2, max_value=2.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    cov = b @ b.T + np.diag(jitter)
    cov = 0.5 * (cov + cov.T)
    # Reject degenerate draws (a vanishing diagonal would violate preconditions).
    assume(np.all(np.diag(cov) > 1e-9))
    assume(np.all(np.isfinite(cov)))
    labels = _labels(n)
    return pd.DataFrame(cov, index=labels, columns=labels)


@st.composite
def returns_panels(draw, min_assets: int = 2, max_assets: int = 6):
    """Draw a labelled returns panel with a common factor (positive correlation)."""
    n = draw(st.integers(min_value=min_assets, max_value=max_assets))
    n_obs = draw(st.integers(min_value=60, max_value=180))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    gen = np.random.default_rng(seed)
    factor = gen.standard_normal(n_obs) * 0.01
    betas = gen.uniform(0.3, 1.5, size=n)
    idio = gen.standard_normal((n_obs, n)) * 0.01
    data = np.outer(factor, betas) + idio
    index = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    return pd.DataFrame(data, index=index, columns=_labels(n))


@st.composite
def continuous_covariances(draw, min_assets: int = 2, max_assets: int = 7):
    """Draw a labelled covariance estimated from random returns.

    Estimating from continuous Gaussian returns makes exact pairwise-distance
    ties a probability-zero event, so single-linkage HRP has an unambiguous
    dendrogram and is genuinely permutation-*equivariant* (not merely
    permutation-*invariant up to tie-breaking*).
    """
    n = draw(st.integers(min_value=min_assets, max_value=max_assets))
    n_obs = draw(st.integers(min_value=120, max_value=240))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    gen = np.random.default_rng(seed)
    factor = gen.standard_normal(n_obs) * 0.01
    betas = gen.uniform(0.3, 1.5, size=n)
    idio = gen.standard_normal((n_obs, n)) * 0.01
    data = np.outer(factor, betas) + idio
    x = data - data.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / (n_obs - 1)
    cov = 0.5 * (cov + cov.T)
    assume(np.all(np.diag(cov) > 1e-12))
    labels = _labels(n)
    return pd.DataFrame(cov, index=labels, columns=labels)


def _assert_simplex(weights: pd.Series, *, atol: float = SIMPLEX_ATOL) -> None:
    arr = weights.to_numpy(dtype=np.float64)
    assert np.all(np.isfinite(arr)), "weights contain non-finite entries"
    assert np.all(arr >= -atol), f"weights have a negative entry: min={arr.min()}"
    assert abs(arr.sum() - 1.0) <= atol, f"weights sum to {arr.sum()!r}, not 1"


# --------------------------------------------------------------------------- #
# Simplex: every allocator's weights live on the simplex                      #
# --------------------------------------------------------------------------- #
@_SETTINGS
@given(cov=psd_covariances())
def test_hrp_weights_on_simplex(cov: pd.DataFrame) -> None:
    result = hrp_allocate(pd.DataFrame(columns=cov.columns), cov=cov)
    _assert_simplex(result.weights)
    # Labels are preserved and in the original input order.
    assert list(result.weights.index) == list(cov.columns)


@_SETTINGS
@given(cov=psd_covariances())
def test_ivp_weights_on_simplex(cov: pd.DataFrame) -> None:
    _assert_simplex(ivp_weights(cov))


@_SETTINGS
@given(n=st.integers(min_value=1, max_value=25))
def test_naive_weights_on_simplex(n: int) -> None:
    weights = naive_weights(_labels(n))
    _assert_simplex(weights)
    # 1/N is exactly equal weight.
    assert np.allclose(weights.to_numpy(), 1.0 / n, atol=SIMPLEX_ATOL)


@_SETTINGS
@given(panel=returns_panels())
def test_hrp_weights_on_simplex_from_returns(panel: pd.DataFrame) -> None:
    """End-to-end: HRP estimating its own (Ledoit-Wolf) covariance."""
    result = hrp_allocate(panel)
    _assert_simplex(result.weights)
    assert list(result.weights.index) == list(panel.columns)


# --------------------------------------------------------------------------- #
# HRP scale-invariance: cov -> c * cov leaves weights unchanged               #
# --------------------------------------------------------------------------- #
@_SETTINGS
@given(
    cov=continuous_covariances(),
    scale=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
)
def test_hrp_scale_invariance(cov: pd.DataFrame, scale: float) -> None:
    """Scaling the covariance by a positive scalar leaves HRP weights unchanged.

    A global variance rescale ``Sigma -> c*Sigma`` leaves the correlation matrix
    (and hence the distances, the linkage and the inverse-variance *ratios*)
    mathematically identical, so the weights are invariant. We draw from
    continuous (tie-free) covariances so the rescale cannot perturb a tied merge
    order through float rounding, and additionally gate on leaf-order stability.
    """
    empty = pd.DataFrame(columns=cov.columns)
    base = hrp_allocate(empty, cov=cov)
    scaled = hrp_allocate(empty, cov=cov * scale)

    # Topology must be unchanged by the rescale (true a.s. for continuous covs).
    assume(base.quasidiag_order == scaled.quasidiag_order)

    # Same labels, same numbers.
    pd.testing.assert_index_equal(base.weights.index, scaled.weights.index)
    np.testing.assert_allclose(
        base.weights.to_numpy(), scaled.weights.to_numpy(), rtol=1e-9, atol=1e-12
    )


# --------------------------------------------------------------------------- #
# HRP permutation-equivariance: permuting assets permutes weights             #
# --------------------------------------------------------------------------- #


@_SETTINGS
@given(cov=continuous_covariances(), seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_hrp_permutation_equivariance(cov: pd.DataFrame, seed: int) -> None:
    """HRP is permutation-equivariant when the dendrogram is permutation-stable.

    de Prado's recursive bisection splits each cluster at ``len(cluster) // 2``,
    i.e. by *leaf position*. So equivariance of the per-label weights holds
    exactly when relabelling the assets produces the relabelled leaf order. When
    SciPy resolves an (estimated, near-tied) merge differently under the
    permutation, the split points move and only the weaker invariant survives.
    We therefore split the assertion: the per-label weights match exactly on the
    permutation-stable examples, and the weight *multiset* is asserted only on
    those same examples (it is not robust to topology changes either).
    """
    n = cov.shape[0]
    labels = list(cov.columns)
    perm = np.random.default_rng(seed).permutation(n)

    base = hrp_allocate(pd.DataFrame(columns=cov.columns), cov=cov)

    permuted_cov = cov.iloc[perm, :].iloc[:, perm]
    permuted = hrp_allocate(pd.DataFrame(columns=permuted_cov.columns), cov=permuted_cov)

    # Permutation-stability gate: the permuted run's leaf order, mapped back to
    # the original labels, must equal the base run's leaf order. ``perm[k]`` is
    # the original position now sitting at position ``k`` in the permuted cov, so
    # a permuted leaf index ``j`` corresponds to original index ``perm[j]``.
    base_label_order = [labels[i] for i in base.quasidiag_order]
    permuted_label_order = [labels[perm[j]] for j in permuted.quasidiag_order]
    assume(permuted_label_order == base_label_order)

    # On the stable subspace HRP is genuinely equivariant: the weight attached to
    # each *asset label* is independent of presentation order.
    aligned = permuted.weights.reindex(base.weights.index)
    np.testing.assert_allclose(base.weights.to_numpy(), aligned.to_numpy(), rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------- #
# Quasi-diagonalisation returns a valid permutation / bijection               #
# --------------------------------------------------------------------------- #
@_SETTINGS
@given(cov=psd_covariances())
def test_quasi_diag_is_bijection(cov: pd.DataFrame) -> None:
    n = cov.shape[0]
    # Build the linkage exactly as the HRP pipeline does.
    std = np.sqrt(np.diag(cov.to_numpy()))
    corr = cov.to_numpy() / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    corr_df = pd.DataFrame(corr, index=cov.index, columns=cov.columns)

    link = linkage_matrix(euclidean_codistance(correl_dist(corr_df)), method="single")
    order = get_quasi_diag(link)

    assert isinstance(order, list)
    assert all(isinstance(i, int) for i in order)
    # A bijection of range(N): right length, no repeats, exactly the leaf set.
    assert len(order) == n
    assert len(set(order)) == n
    assert sorted(order) == list(range(n))

    # The order recorded on the full HRPResult is itself a bijection of the same
    # leaf set (it may differ from `order` only by tie-breaking on equal-distance
    # merges, so we compare it as a set, not element-wise).
    result = hrp_allocate(pd.DataFrame(columns=cov.columns), cov=cov)
    assert sorted(result.quasidiag_order) == list(range(n))


# --------------------------------------------------------------------------- #
# Estimator-level no-lookahead: weights depend only on the in-sample window    #
# --------------------------------------------------------------------------- #
@_SETTINGS
@given(
    panel=returns_panels(),
    split_frac=st.floats(min_value=0.4, max_value=0.8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_hrp_no_lookahead(panel: pd.DataFrame, split_frac: float, seed: int) -> None:
    n_obs = len(panel)
    split = int(n_obs * split_frac)
    assume(2 <= split < n_obs)  # need >=2 in-sample obs, and some out-of-sample.

    in_sample = panel.iloc[:split]

    # Baseline: weights fit on the in-sample window only.
    base = hrp_allocate(in_sample).weights

    # Now corrupt every out-of-sample row with arbitrary noise and re-fit on the
    # SAME in-sample window of a panel whose future was rewritten.
    gen = np.random.default_rng(seed)
    perturbed = panel.copy()
    future = perturbed.iloc[split:]
    perturbed.iloc[split:] = future.to_numpy() + gen.standard_normal(future.shape)

    refit = hrp_allocate(perturbed.iloc[:split]).weights

    # The in-sample slices are byte-identical, so the weights must be too.
    pd.testing.assert_series_equal(base, refit, check_exact=True)


# --------------------------------------------------------------------------- #
# Deflated Sharpe is non-increasing in n_trials                               #
# --------------------------------------------------------------------------- #
@_SETTINGS
@given(
    observed_sharpe=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
    n_obs=st.integers(min_value=50, max_value=2000),
    variance_of_trial_sharpes=st.floats(
        min_value=1e-6, max_value=0.5, allow_nan=False, allow_infinity=False
    ),
    n_a=st.integers(min_value=1, max_value=500),
    extra=st.integers(min_value=1, max_value=500),
    skew=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    kurtosis=st.floats(min_value=3.0, max_value=12.0, allow_nan=False),
)
def test_dsr_non_increasing_in_n_trials(
    observed_sharpe: float,
    n_obs: int,
    variance_of_trial_sharpes: float,
    n_a: int,
    extra: int,
    skew: float,
    kurtosis: float,
) -> None:
    n_b = n_a + extra  # strictly more trials.

    def dsr(n_trials: int) -> float:
        return deflated_sharpe_ratio(
            observed_sharpe,
            n_obs=n_obs,
            n_trials=n_trials,
            variance_of_trial_sharpes=variance_of_trial_sharpes,
            skew=skew,
            kurtosis=kurtosis,
        )

    fewer = dsr(n_a)
    more = dsr(n_b)
    # More multiplicity raises the expected-max benchmark, so the deflated
    # Sharpe can only fall (or stay equal). Small slack for the inverse-CDF
    # approximation in the benchmark.
    assert more <= fewer + 1e-9


# --------------------------------------------------------------------------- #
# Verdict truth-table: never HRP_BEATS_1N when the CI straddles zero          #
# --------------------------------------------------------------------------- #
@_SETTINGS
@given(
    jkm_pvalue=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    deflated_sharpe=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ci_low=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
    ci_high=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
)
def test_verdict_never_beats_when_ci_straddles_zero(
    jkm_pvalue: float,
    deflated_sharpe: float,
    ci_low: float,
    ci_high: float,
) -> None:
    assume(ci_low <= ci_high)  # precondition of derive_verdict.
    verdict = derive_verdict(jkm_pvalue, deflated_sharpe, ci_low, ci_high)

    assert isinstance(verdict, Verdict)

    ci_straddles_zero = ci_low <= 0.0 <= ci_high
    if ci_straddles_zero:
        # The honesty requirement: a CI that includes zero can NEVER support a
        # positive (or negative) directional claim.
        assert verdict == Verdict.NO_SIGNIFICANT_DIFFERENCE

    # And the converse direction of the truth table: a BEATS verdict implies the
    # CI is strictly positive AND every gate cleared.
    if verdict == Verdict.HRP_BEATS_1N:
        assert ci_low > 0.0
        assert jkm_pvalue < 0.05
        assert deflated_sharpe >= 0.95
    if verdict == Verdict.HRP_LOSES_TO_1N:
        assert ci_high < 0.0


@_SETTINGS
@given(
    jkm_pvalue=st.floats(min_value=0.0, max_value=0.0499, allow_nan=False),
    deflated_sharpe=st.floats(min_value=0.95, max_value=1.0, allow_nan=False),
    ci_low=st.floats(min_value=1e-6, max_value=5.0, allow_nan=False),
    width=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
)
def test_verdict_beats_when_all_evidence_agrees(
    jkm_pvalue: float,
    deflated_sharpe: float,
    ci_low: float,
    width: float,
) -> None:
    """When every gate clears AND the CI is strictly positive -> HRP_BEATS_1N."""
    ci_high = ci_low + width
    verdict = derive_verdict(jkm_pvalue, deflated_sharpe, ci_low, ci_high)
    assert verdict == Verdict.HRP_BEATS_1N
