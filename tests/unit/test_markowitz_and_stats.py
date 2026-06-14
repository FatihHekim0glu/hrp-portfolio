"""Unit tests for the Markowitz adapter and the backtest stats kernels.

These exercise the REAL behavior of
:mod:`hrp.allocate.markowitz_adapter` and :mod:`hrp.backtest.stats` against
hand-computed expected values on tiny, deterministic inputs (no network, no
heavy fixtures beyond ``de_prado_example``). cvxpy-dependent paths are guarded
with ``pytest.importorskip``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hrp.allocate.markowitz_adapter import max_sharpe_weights, min_var_weights
from hrp.backtest.stats import (
    annualized_vol,
    max_drawdown,
    sharpe_ratio,
    turnover,
)

# --------------------------------------------------------------------------- #
# min_var_weights
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_min_var_diagonal_cov_is_inverse_variance() -> None:
    """On a diagonal covariance, min-var weights are proportional to 1/variance.

    With diag([1, 2, 4]) the inverse variances are [1, 0.5, 0.25] (sum 1.75),
    so the closed-form (all-positive, no QP needed) weights are [4/7, 2/7, 1/7].
    """
    cov = pd.DataFrame(
        np.diag([1.0, 2.0, 4.0]),
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    w = min_var_weights(cov)

    assert isinstance(w, pd.Series)
    assert list(w.index) == ["A", "B", "C"]
    expected = np.array([4.0, 2.0, 1.0]) / 7.0
    np.testing.assert_allclose(w.to_numpy(), expected, rtol=1e-10, atol=1e-12)
    # The defining proportionality: w_i * variance_i is constant across assets.
    products = w.to_numpy() * np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(products, products[0], rtol=1e-10)


@pytest.mark.unit
def test_min_var_weights_sum_to_one_and_nonnegative() -> None:
    """Diagonal-cov min-var weights form a valid simplex (sum 1, all >= 0)."""
    cov = pd.DataFrame(
        np.diag([0.04, 0.01, 0.09, 0.25]),
        index=list("WXYZ"),
        columns=list("WXYZ"),
    )
    w = min_var_weights(cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-10)
    assert (w.to_numpy() >= -1e-12).all()
    # Smallest variance (X, 0.01) must receive the largest weight.
    assert w.idxmax() == "X"


@pytest.mark.unit
def test_min_var_identity_cov_is_equal_weight() -> None:
    """An identity covariance gives the equal-weight portfolio (all variances equal)."""
    n = 5
    cov = pd.DataFrame(np.eye(n))
    w = min_var_weights(cov)
    np.testing.assert_allclose(w.to_numpy(), np.full(n, 1.0 / n), rtol=1e-10)


@pytest.mark.unit
def test_min_var_de_prado_example_is_valid_simplex(de_prado_example: pd.DataFrame) -> None:
    """On the de Prado correlation matrix, min-var returns a valid simplex.

    This well-conditioned, all-positive-correlation matrix yields an all-positive
    closed-form solution, so the result is labelled by asset, sums to one, and
    is non-negative.
    """
    w = min_var_weights(de_prado_example)
    assert list(w.index) == list(de_prado_example.columns)
    assert w.sum() == pytest.approx(1.0, abs=1e-10)
    assert (w.to_numpy() >= -1e-12).all()
    assert len(w) == de_prado_example.shape[0]


@pytest.mark.unit
def test_min_var_numpy_input_uses_integer_labels() -> None:
    """A bare ndarray cov produces integer-range asset labels (0..n-1)."""
    cov = np.diag([1.0, 4.0])
    w = min_var_weights(cov)
    assert list(w.index) == [0, 1]
    np.testing.assert_allclose(w.to_numpy(), np.array([4.0, 1.0]) / 5.0, rtol=1e-10)


@pytest.mark.unit
def test_min_var_non_square_raises_validation_error() -> None:
    """A non-square covariance is rejected with ValidationError."""
    from hrp._exceptions import ValidationError

    bad = np.ones((2, 3))
    with pytest.raises(ValidationError):
        min_var_weights(bad)


@pytest.mark.unit
def test_min_var_long_only_qp_clips_short_closed_form() -> None:
    """When the closed form goes short, the long-only QP returns a valid simplex.

    With a strong POSITIVE covariance between two assets of very unequal variance
    (var 1 and var 4, correlation 0.95), the unconstrained closed form puts a
    negative weight on the high-variance asset; the long-only branch (cvxpy QP)
    must repair it to a non-negative simplex.
    """
    pytest.importorskip("cvxpy")
    cov = pd.DataFrame(
        [
            [1.0, 1.9, 0.0],
            [1.9, 4.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        index=list("ABC"),
        columns=list("ABC"),
    )
    # Confirm the unconstrained closed form is indeed infeasible long-only.
    w_short = min_var_weights(cov, long_only=False)
    assert (w_short.to_numpy() < 0.0).any()

    w = min_var_weights(cov, long_only=True)
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    assert (w.to_numpy() >= -1e-8).all()


# --------------------------------------------------------------------------- #
# max_sharpe_weights
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_max_sharpe_returns_valid_simplex() -> None:
    """Long-only max-Sharpe returns asset-labelled weights on the budget simplex."""
    pytest.importorskip("cvxpy")
    cov = pd.DataFrame(
        np.diag([0.04, 0.04, 0.04]),
        index=list("ABC"),
        columns=list("ABC"),
    )
    mu = pd.Series([0.02, 0.01, 0.005], index=list("ABC"))
    w = max_sharpe_weights(cov, mu)

    assert isinstance(w, pd.Series)
    assert list(w.index) == list("ABC")
    assert w.sum() == pytest.approx(1.0, abs=1e-7)
    assert (w.to_numpy() >= -1e-7).all()


@pytest.mark.unit
def test_max_sharpe_tilts_toward_highest_return_when_risk_equal() -> None:
    """With equal, uncorrelated variances, the asset with the largest mu wins.

    Identical diagonal variances mean the tangency tilt is driven entirely by the
    expected-return vector, so the top-mu asset must carry the largest weight.
    """
    pytest.importorskip("cvxpy")
    cov = pd.DataFrame(
        np.diag([0.04, 0.04, 0.04]),
        index=list("ABC"),
        columns=list("ABC"),
    )
    mu = pd.Series([0.03, 0.01, 0.005], index=list("ABC"))
    w = max_sharpe_weights(cov, mu)
    assert w.idxmax() == "A"


@pytest.mark.unit
def test_max_sharpe_aligns_mu_to_cov_columns() -> None:
    """mu given in a shuffled order is reindexed to the covariance's columns."""
    pytest.importorskip("cvxpy")
    cov = pd.DataFrame(
        np.diag([0.04, 0.04, 0.04]),
        index=list("ABC"),
        columns=list("ABC"),
    )
    mu_ordered = pd.Series([0.03, 0.01, 0.005], index=list("ABC"))
    mu_shuffled = pd.Series([0.005, 0.03, 0.01], index=list("CAB"))
    w_ordered = max_sharpe_weights(cov, mu_ordered)
    w_shuffled = max_sharpe_weights(cov, mu_shuffled)
    np.testing.assert_allclose(
        w_ordered.to_numpy(), w_shuffled.to_numpy(), rtol=1e-6, atol=1e-7
    )


@pytest.mark.unit
def test_max_sharpe_no_positive_excess_raises() -> None:
    """If no asset beats the risk-free rate, the tangency portfolio is undefined."""
    pytest.importorskip("cvxpy")
    from hrp._exceptions import ValidationError

    cov = pd.DataFrame(np.diag([0.04, 0.04]), index=list("AB"), columns=list("AB"))
    mu = pd.Series([0.01, 0.02], index=list("AB"))
    with pytest.raises(ValidationError):
        max_sharpe_weights(cov, mu, risk_free=0.05)


@pytest.mark.unit
def test_max_sharpe_mu_length_mismatch_raises() -> None:
    """A mu vector whose length does not match cov is rejected."""
    pytest.importorskip("cvxpy")
    from hrp._exceptions import ValidationError

    cov = pd.DataFrame(np.diag([0.04, 0.04, 0.04]))
    mu = pd.Series([0.02, 0.01])  # length 2, cov is 3x3, integer-aligned -> NaN
    with pytest.raises(ValidationError):
        max_sharpe_weights(cov, mu)


# --------------------------------------------------------------------------- #
# sharpe_ratio
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_sharpe_ratio_hand_computed() -> None:
    """Sharpe on [0.01, 0.03]: mean 0.02, std(ddof=1)=sqrt(2e-4), * sqrt(4)=2.8284."""
    r = pd.Series([0.01, 0.03])
    sr = sharpe_ratio(r, periods_per_year=4)
    expected = (0.02 / math.sqrt(2e-4)) * math.sqrt(4)
    assert sr == pytest.approx(expected, rel=1e-12)
    assert sr == pytest.approx(2.8284271247461903, rel=1e-12)


@pytest.mark.unit
def test_sharpe_ratio_subtracts_risk_free() -> None:
    """The per-period risk-free rate is subtracted from each return before the ratio."""
    r = pd.Series([0.04, 0.06])  # excess [0.01, 0.03] after rf=0.03
    sr = sharpe_ratio(r, risk_free=0.03, periods_per_year=4)
    # Same excess series as the hand-computed test above.
    assert sr == pytest.approx(2.8284271247461903, rel=1e-12)


@pytest.mark.unit
def test_sharpe_ratio_zero_vol_is_nan() -> None:
    """A perfectly flat (zero-volatility) series has an undefined Sharpe (NaN)."""
    r = pd.Series([0.01, 0.01, 0.01, 0.01])
    assert math.isnan(sharpe_ratio(r))


@pytest.mark.unit
def test_sharpe_ratio_all_negative_is_negative() -> None:
    """A series with negative mean excess returns a negative Sharpe ratio."""
    r = pd.Series([-0.01, -0.03])  # mean -0.02 < 0, finite vol
    sr = sharpe_ratio(r, periods_per_year=4)
    expected = (-0.02 / math.sqrt(2e-4)) * math.sqrt(4)
    assert sr == pytest.approx(expected, rel=1e-12)
    assert sr < 0.0


# --------------------------------------------------------------------------- #
# annualized_vol
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_annualized_vol_hand_computed() -> None:
    """Annualized vol of [0.10, -0.05, 0.10, -0.05]: std(ddof=1) * sqrt(252)."""
    r = pd.Series([0.10, -0.05, 0.10, -0.05])
    sigma = r.std(ddof=1)  # 0.08660254...
    expected = sigma * math.sqrt(252)
    av = annualized_vol(r)
    assert av == pytest.approx(expected, rel=1e-12)
    assert av == pytest.approx(1.374772708486752, rel=1e-12)


@pytest.mark.unit
def test_annualized_vol_zero_for_constant_series() -> None:
    """A constant series has zero sample std and therefore zero annualized vol."""
    r = pd.Series([0.02, 0.02, 0.02])
    assert annualized_vol(r) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.unit
def test_annualized_vol_scales_with_periods_per_year() -> None:
    """Annualized vol scales as sqrt(periods_per_year)."""
    r = pd.Series([0.01, -0.01, 0.02, -0.02])
    v252 = annualized_vol(r, periods_per_year=252)
    v63 = annualized_vol(r, periods_per_year=63)
    assert v252 / v63 == pytest.approx(math.sqrt(252 / 63), rel=1e-12)


# --------------------------------------------------------------------------- #
# turnover
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_turnover_full_rotation_is_one() -> None:
    """Rotating fully out of one asset and into another is turnover 1.0."""
    prev = pd.Series({"A": 1.0, "B": 0.0})
    new = pd.Series({"A": 0.0, "B": 1.0})
    assert turnover(prev, new) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_turnover_identical_weights_is_zero() -> None:
    """No change in weights yields zero turnover."""
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert turnover(w, w) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.unit
def test_turnover_aligns_on_union_with_zero_fill() -> None:
    """Disjoint/partial labels are aligned on the union with missing assets as 0.

    prev = {A: 0.6, B: 0.4}, new = {B: 0.4, C: 0.6}. Aligned on {A, B, C}:
    |dA|=0.6, |dB|=0.0, |dC|=0.6 -> 0.5 * 1.2 = 0.6.
    """
    prev = pd.Series({"A": 0.6, "B": 0.4})
    new = pd.Series({"B": 0.4, "C": 0.6})
    assert turnover(prev, new) == pytest.approx(0.6, abs=1e-12)


@pytest.mark.unit
def test_turnover_partial_rebalance_hand_computed() -> None:
    """prev {A:0.5,B:0.5} -> new {A:0.7,B:0.3}: 0.5*(0.2+0.2)=0.2."""
    prev = pd.Series({"A": 0.5, "B": 0.5})
    new = pd.Series({"A": 0.7, "B": 0.3})
    assert turnover(prev, new) == pytest.approx(0.2, abs=1e-12)


# --------------------------------------------------------------------------- #
# max_drawdown
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_max_drawdown_hand_computed() -> None:
    """[0.10, -0.20, 0.05] -> wealth [1.1, 0.88, 0.924], peak 1.1, mdd = -0.2."""
    r = pd.Series([0.10, -0.20, 0.05])
    mdd = max_drawdown(r)
    assert mdd == pytest.approx(-0.2, abs=1e-12)


@pytest.mark.unit
def test_max_drawdown_monotonic_increase_is_zero() -> None:
    """A series that never declines has zero drawdown."""
    r = pd.Series([0.01, 0.02, 0.03])
    assert max_drawdown(r) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.unit
def test_max_drawdown_all_negative() -> None:
    """[-0.1, -0.1] -> wealth [0.9, 0.81], peak 0.9, worst dd = 0.81/0.9 - 1 = -0.1."""
    r = pd.Series([-0.1, -0.1])
    mdd = max_drawdown(r)
    assert mdd == pytest.approx(-0.1, abs=1e-12)
    assert mdd <= 0.0


@pytest.mark.unit
def test_max_drawdown_is_non_positive() -> None:
    """Max drawdown is always <= 0 by construction (clamped at the peak)."""
    r = pd.Series([0.05, -0.02, 0.03, -0.10, 0.04])
    assert max_drawdown(r) <= 0.0
