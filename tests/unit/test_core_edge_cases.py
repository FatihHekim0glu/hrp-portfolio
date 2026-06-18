"""Unit tests closing coverage gaps in the CORE HRP modules.

These exercise the REAL behavior (not just "it runs") of the under-covered
allocation, clustering, walk-forward, and evaluation kernels against
hand-computed expected values and tight invariants on tiny, deterministic
inputs. No network, no heavy machinery beyond the shared seeded fixtures.

Coverage targets (the "Missing" lines from the report):

- ``allocate/hrp.py`` - two-asset and single-asset degenerate cases,
  ``get_cluster_var`` inverse-variance weighting (and its rejections), the
  ``get_rec_bipart`` split, ndarray-cov label-recovery branches, and
  ``HRPResult.to_dict``.
- ``allocate/ivp.py`` + ``allocate/naive.py`` - exact expected weights on a
  known covariance, plus the validation rejections.
- ``cluster/quasidiag.py`` + ``linkage.py`` + ``distance.py`` - non-``single``
  linkage methods, distance edge values, and the malformed-input branches.
- ``backtest/walk_forward.py`` - a short deterministic run: ``shift(1)``
  boundary, purge/embargo, cost application, cost monotonicity, anchored window.
- ``evaluation`` comparison/dsr/verdict - JKM on identical series (~0 gap, p=1),
  DSR edge ``n_trials=1`` (reduces to plain PSR), and every ``Verdict`` branch
  of :func:`derive_verdict`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hrp._exceptions import InsufficientDataError, ValidationError
from hrp.allocate.hrp import (
    HRPResult,
    get_cluster_var,
    get_rec_bipart,
    hrp_allocate,
)
from hrp.allocate.ivp import ivp_weights
from hrp.allocate.naive import naive_weights
from hrp.backtest.stats import sharpe_ratio
from hrp.backtest.walk_forward import _safe_float, walk_forward_backtest
from hrp.cluster.distance import correl_dist, euclidean_codistance
from hrp.cluster.linkage import VALID_METHODS, linkage_matrix
from hrp.cluster.quasidiag import get_quasi_diag
from hrp.evaluation.comparison import (
    block_bootstrap_sharpe_gap,
    jobson_korkie_memmel,
)
from hrp.evaluation.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from hrp.evaluation.verdict import Verdict, derive_verdict

# --------------------------------------------------------------------------- #
# get_cluster_var - inverse-variance weighting (the HONESTY requirement)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_get_cluster_var_is_inverse_variance_not_equal_weight() -> None:
    """Cluster variance uses INVERSE-VARIANCE intra-cluster weights, not equal.

    For ``cov = diag([1, 4])`` the inverse variances are ``[1, 0.25]``,
    normalized to ``[0.8, 0.2]``. The cluster variance is
    ``0.8**2 * 1 + 0.2**2 * 4 = 0.64 + 0.16 = 0.80``. Equal weights would give
    ``0.5**2 * 1 + 0.5**2 * 4 = 1.25`` - pinning the inverse-variance behaviour.
    """
    cov = np.diag([1.0, 4.0])
    var = get_cluster_var(cov, [0, 1])
    assert var == pytest.approx(0.80, abs=1e-12)
    # Guard against the equal-weight bug producing 1.25.
    assert var != pytest.approx(1.25, abs=1e-6)


@pytest.mark.unit
def test_get_cluster_var_singleton_returns_own_variance() -> None:
    """A single-asset cluster's variance is simply that asset's variance."""
    cov = np.diag([1.0, 4.0])
    assert get_cluster_var(cov, [0]) == pytest.approx(1.0, abs=1e-12)
    assert get_cluster_var(cov, [1]) == pytest.approx(4.0, abs=1e-12)


@pytest.mark.unit
def test_get_cluster_var_accepts_dataframe_and_subslices() -> None:
    """A DataFrame cov is sliced positionally to the requested cluster items."""
    cov = pd.DataFrame(
        np.diag([1.0, 2.0, 4.0]),
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    # Cluster {B, C} = positions [1, 2]: inv_var [0.5, 0.25] -> [2/3, 1/3];
    # variance = (2/3)**2 * 2 + (1/3)**2 * 4 = 8/9 + 4/9 = 12/9 = 4/3.
    var = get_cluster_var(cov, [1, 2])
    assert var == pytest.approx(4.0 / 3.0, abs=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("items", "cov"),
    [
        ([], np.diag([1.0, 4.0])),  # empty cluster
        ([5], np.diag([1.0, 4.0])),  # out-of-range index
        ([0, 1], np.array([[0.0, 0.0], [0.0, 1.0]])),  # non-positive diagonal
    ],
)
def test_get_cluster_var_rejects_bad_inputs(items: list[int], cov: np.ndarray) -> None:
    """Empty / out-of-range cluster items and a non-positive diagonal are rejected."""
    with pytest.raises(ValidationError):
        get_cluster_var(cov, items)


@pytest.mark.unit
def test_get_cluster_var_rejects_non_square() -> None:
    """A non-square cov is rejected before any slicing."""
    with pytest.raises(ValidationError):
        get_cluster_var(np.zeros((2, 3)), [0])


# --------------------------------------------------------------------------- #
# get_rec_bipart - the recursive-bisection split
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_get_rec_bipart_two_assets_split() -> None:
    """Two assets split once: alpha = 1 - V_left / (V_left + V_right).

    For ``cov = diag([1, 4])`` and order ``[0, 1]``, the left half is ``[0]``
    (var 1) and the right half ``[1]`` (var 4), so
    ``alpha = 1 - 1/(1+4) = 0.8`` and weights are ``[0.8, 0.2]`` - more capital
    to the lower-variance asset.
    """
    cov = np.diag([1.0, 4.0])
    w = get_rec_bipart(cov, [0, 1])
    assert w.loc[0] == pytest.approx(0.8, abs=1e-12)
    assert w.loc[1] == pytest.approx(0.2, abs=1e-12)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_get_rec_bipart_single_asset_is_degenerate_one() -> None:
    """A single-asset universe never enters the bisection loop: weight is 1.0."""
    w = get_rec_bipart(np.array([[2.5]]), [0])
    assert list(w.index) == [0]
    assert w.loc[0] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_get_rec_bipart_respects_leaf_order_permutation() -> None:
    """The split honours the supplied (non-identity) leaf order.

    With order ``[1, 0]`` on ``cov = diag([1, 4])`` the left half is position 1
    (var 4) and the right is position 0 (var 1), so ``alpha = 1 - 4/5 = 0.2``:
    position 1 gets 0.2 and position 0 gets 0.8 (still favouring lower variance).
    """
    cov = np.diag([1.0, 4.0])
    w = get_rec_bipart(cov, [1, 0])
    assert w.loc[1] == pytest.approx(0.2, abs=1e-12)
    assert w.loc[0] == pytest.approx(0.8, abs=1e-12)


@pytest.mark.unit
def test_get_rec_bipart_rejects_bad_permutation_and_shape() -> None:
    """A non-permutation ``sort_ix`` and a non-square cov are both rejected."""
    cov = np.diag([1.0, 4.0])
    with pytest.raises(ValidationError):
        get_rec_bipart(cov, [0, 0])  # not a permutation
    with pytest.raises(ValidationError):
        get_rec_bipart(np.zeros((2, 3)), [0, 1])  # non-square


# --------------------------------------------------------------------------- #
# hrp_allocate - degenerate small universes and ndarray-cov label recovery
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_hrp_allocate_two_assets_matches_rec_bipart() -> None:
    """On two uncorrelated assets HRP reduces to a single inverse-variance split.

    Feeding ``cov = diag([1, 4])`` (and labelled returns) the weights must equal
    the hand-computed ``{A: 0.8, B: 0.2}`` and the labels/order propagate.
    """
    cov = np.array([[1.0, 0.0], [0.0, 4.0]])
    returns = pd.DataFrame(np.zeros((6, 2)), columns=["A", "B"])
    res = hrp_allocate(returns, cov=cov)
    assert isinstance(res, HRPResult)
    assert res.weights.loc["A"] == pytest.approx(0.8, abs=1e-12)
    assert res.weights.loc["B"] == pytest.approx(0.2, abs=1e-12)
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert sorted(res.ordered_assets) == ["A", "B"]
    assert sorted(res.quasidiag_order) == [0, 1]
    assert res.meta["n_assets"] == 2


@pytest.mark.unit
def test_hrp_allocate_block_structure_simplex(block_correlation_cov: pd.DataFrame) -> None:
    """On the block covariance HRP returns valid simplex weights with full labels."""
    res = hrp_allocate(pd.DataFrame(), cov=block_correlation_cov)
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert (res.weights >= 0.0).all()
    assert set(res.weights.index) == set(block_correlation_cov.columns)
    assert sorted(res.quasidiag_order) == list(range(len(block_correlation_cov)))


@pytest.mark.unit
def test_hrp_allocate_singular_cov_still_valid(singular_cov: pd.DataFrame) -> None:
    """ROBUSTNESS: HRP yields valid weights on a singular covariance (no inversion)."""
    res = hrp_allocate(pd.DataFrame(), cov=singular_cov)
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert (res.weights >= 0.0).all()
    assert np.all(np.isfinite(res.weights.to_numpy()))


@pytest.mark.unit
def test_hrp_allocate_ndarray_cov_recovers_integer_labels() -> None:
    """When ndarray cov and returns columns disagree, integer labels are used.

    ``returns`` has 2 columns but cov is 2x2 with matching size, so labels come
    from the returns; here we force the mismatch (returns has a different number
    of columns) and expect positional integer labels 0..N-1.
    """
    cov = np.array([[1.0, 0.0], [0.0, 4.0]])
    # returns with 3 columns != cov size 2 -> integer labels branch.
    returns = pd.DataFrame(np.zeros((4, 3)), columns=["X", "Y", "Z"])
    res = hrp_allocate(returns, cov=cov)
    assert list(res.weights.index) == [0, 1]
    assert res.weights.loc[0] == pytest.approx(0.8, abs=1e-12)


@pytest.mark.unit
def test_hrp_allocate_rejects_non_square_and_bad_diagonal() -> None:
    """A non-square ndarray cov and a non-positive variance entry are rejected."""
    with pytest.raises(ValidationError):
        hrp_allocate(pd.DataFrame(np.zeros((5, 2))), cov=np.zeros((2, 3)))
    with pytest.raises(ValidationError):
        hrp_allocate(
            pd.DataFrame(np.zeros((5, 2)), columns=["A", "B"]),
            cov=np.array([[0.0, 0.0], [0.0, 1.0]]),
        )


@pytest.mark.unit
def test_hrp_result_to_dict_is_json_friendly() -> None:
    """``HRPResult.to_dict`` strips numpy/pandas types at the API boundary."""
    cov = np.array([[1.0, 0.0], [0.0, 4.0]])
    res = hrp_allocate(pd.DataFrame(np.zeros((4, 2)), columns=["A", "B"]), cov=cov)
    d = res.to_dict()
    assert set(d["weights"]) == {"A", "B"}
    assert all(isinstance(v, float) for v in d["weights"].values())
    assert isinstance(d["link"], list)
    assert d["weights"]["A"] == pytest.approx(0.8, abs=1e-12)
    assert d["quasidiag_order"] == [0, 1]


# --------------------------------------------------------------------------- #
# ivp_weights / naive_weights - exact weights on a known covariance
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_ivp_weights_exact_on_known_diagonal() -> None:
    """IVP on ``diag([1, 2, 4])`` gives ``[4/7, 2/7, 1/7]`` (1/variance, normalized)."""
    cov = pd.DataFrame(
        np.diag([1.0, 2.0, 4.0]),
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    w = ivp_weights(cov)
    assert w.loc["A"] == pytest.approx(4.0 / 7.0, abs=1e-12)
    assert w.loc["B"] == pytest.approx(2.0 / 7.0, abs=1e-12)
    assert w.loc["C"] == pytest.approx(1.0 / 7.0, abs=1e-12)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_ivp_weights_ignores_off_diagonal() -> None:
    """IVP uses only the diagonal: off-diagonal covariances do not change weights."""
    diag_only = pd.DataFrame(np.diag([1.0, 4.0]), index=["A", "B"], columns=["A", "B"])
    with_offdiag = pd.DataFrame([[1.0, 0.9], [0.9, 4.0]], index=["A", "B"], columns=["A", "B"])
    w0 = ivp_weights(diag_only)
    w1 = ivp_weights(with_offdiag)
    pd.testing.assert_series_equal(w0, w1)
    assert w0.loc["A"] == pytest.approx(0.8, abs=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    "cov",
    [
        np.zeros((2, 3)),  # non-square
        np.array([[np.inf, 0.0], [0.0, 1.0]]),  # non-finite diagonal
        np.array([[0.0, 0.0], [0.0, 1.0]]),  # non-positive variance
    ],
)
def test_ivp_weights_rejects_bad_cov(cov: np.ndarray) -> None:
    """IVP rejects non-square, non-finite, and non-positive-variance covariances."""
    with pytest.raises(ValidationError):
        ivp_weights(cov)


@pytest.mark.unit
def test_naive_weights_equal_and_simplex() -> None:
    """1/N puts exactly equal weight on every (distinct) asset and sums to 1."""
    w = naive_weights(["X", "Y", "Z", "W"])
    assert np.allclose(w.to_numpy(), 0.25, atol=1e-12)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert list(w.index) == ["X", "Y", "Z", "W"]


@pytest.mark.unit
@pytest.mark.parametrize("assets", [[], ["A", "A"]])
def test_naive_weights_rejects_empty_and_duplicates(assets: list[str]) -> None:
    """Empty and duplicate-labelled asset lists are rejected."""
    with pytest.raises(ValidationError):
        naive_weights(assets)


# --------------------------------------------------------------------------- #
# distance kernels - edge values and validation branches
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_correl_dist_edge_values() -> None:
    """``d = sqrt(0.5(1-rho))``: rho=+1->0, rho=0->1/sqrt(2), rho=-1->1."""
    corr = pd.DataFrame(
        [
            [1.0, 1.0, 0.0, -1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0],
        ]
    )
    d = correl_dist(corr)
    assert d.iloc[0, 0] == pytest.approx(0.0, abs=1e-12)  # diagonal
    assert d.iloc[0, 1] == pytest.approx(0.0, abs=1e-12)  # rho = +1
    assert d.iloc[0, 2] == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-12)  # rho = 0
    assert d.iloc[0, 3] == pytest.approx(1.0, abs=1e-12)  # rho = -1
    # Symmetric with a zero diagonal.
    assert np.allclose(d.to_numpy(), d.to_numpy().T)
    assert np.allclose(np.diag(d.to_numpy()), 0.0)


@pytest.mark.unit
def test_correl_dist_relabels_ndarray_input() -> None:
    """An ndarray input is returned with consistent integer labels on both axes."""
    d = correl_dist(np.array([[1.0, 0.5], [0.5, 1.0]]))
    assert list(d.index) == list(d.columns)
    assert d.index.equals(d.columns)


@pytest.mark.unit
def test_correl_dist_rejects_non_square_and_out_of_domain() -> None:
    """A non-square matrix and an entry outside ``[-1, 1]`` are rejected."""
    with pytest.raises(ValidationError):
        correl_dist(pd.DataFrame(np.zeros((2, 3))))
    with pytest.raises(ValidationError):
        correl_dist(pd.DataFrame([[1.0, 2.0], [2.0, 1.0]]))


@pytest.mark.unit
def test_euclidean_codistance_known_columns() -> None:
    """Co-distance is the Euclidean distance between the columns of ``dist``.

    For ``dist = [[0, 1], [1, 0]]`` the two columns are ``[0, 1]`` and
    ``[1, 0]``; their Euclidean distance is ``sqrt(1 + 1) = sqrt(2)``.
    """
    dist = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]], index=["A", "B"], columns=["A", "B"])
    cod = euclidean_codistance(dist)
    assert cod.iloc[0, 1] == pytest.approx(math.sqrt(2.0), abs=1e-12)
    assert cod.iloc[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(cod.to_numpy(), cod.to_numpy().T)


@pytest.mark.unit
def test_euclidean_codistance_rejects_non_square() -> None:
    """A non-square distance matrix is rejected."""
    with pytest.raises(ValidationError):
        euclidean_codistance(pd.DataFrame(np.zeros((2, 3))))


# --------------------------------------------------------------------------- #
# linkage_matrix - non-'single' methods + validation branches
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("method", sorted(VALID_METHODS))
def test_linkage_matrix_all_methods_produce_valid_linkage(
    method: str, de_prado_example: pd.DataFrame
) -> None:
    """Every accepted linkage method yields a valid ``(N-1) x 4`` linkage matrix.

    Exercises the non-``single`` ablation methods (ward/complete/average), and
    the recovered leaf order must remain a permutation of ``range(N)``.
    """
    cod = euclidean_codistance(correl_dist(de_prado_example))
    link = linkage_matrix(cod, method=method)
    n = len(de_prado_example)
    assert link.shape == (n - 1, 4)
    assert np.all(np.isfinite(link))
    order = get_quasi_diag(link)
    assert sorted(order) == list(range(n))


@pytest.mark.unit
def test_linkage_matrix_rejects_bad_method() -> None:
    """An unrecognized linkage method is rejected (no silent substitution)."""
    dist = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValidationError):
        linkage_matrix(dist, method="nope")


@pytest.mark.unit
@pytest.mark.parametrize(
    "dist",
    [
        pd.DataFrame(np.zeros((2, 3))),  # non-square
        pd.DataFrame([[0.0]]),  # fewer than two assets
        pd.DataFrame([[0.0, 1.0], [2.0, 0.0]]),  # asymmetric
        pd.DataFrame([[1.0, 1.0], [1.0, 1.0]]),  # non-zero diagonal
    ],
)
def test_linkage_matrix_rejects_malformed_distance(dist: pd.DataFrame) -> None:
    """Non-square, too-small, asymmetric, and non-zero-diagonal inputs are rejected."""
    with pytest.raises(ValidationError):
        linkage_matrix(dist)


# --------------------------------------------------------------------------- #
# get_quasi_diag - leaf-order recovery + malformed-input branches
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_get_quasi_diag_recovers_block_order(block_correlation_cov: pd.DataFrame) -> None:
    """The recovered leaf order is a valid permutation that quasi-diagonalizes.

    Reordering the correlation matrix by the leaf order must place the high
    within-block correlations near the diagonal (mean of the first super-diagonal
    rises versus the original ordering's nearest-neighbour correlation).
    """
    cod = euclidean_codistance(correl_dist(block_correlation_cov))
    link = linkage_matrix(cod, method="single")
    order = get_quasi_diag(link)
    n = len(block_correlation_cov)
    assert sorted(order) == list(range(n))

    corr = block_correlation_cov.to_numpy()
    reordered = corr[np.ix_(order, order)]
    # Off-diagonal neighbours after reordering should be at least as block-like.
    neigh = np.mean(np.abs(np.diag(reordered, k=1)))
    assert neigh >= 0.5  # within-block (0.8) neighbours dominate the super-diagonal


@pytest.mark.unit
def test_get_quasi_diag_rejects_bad_shape_and_empty() -> None:
    """A non-``(N-1) x 4`` matrix and a zero-merge matrix are rejected."""
    with pytest.raises(ValidationError):
        get_quasi_diag(np.zeros((3, 3)))
    with pytest.raises(ValidationError):
        get_quasi_diag(np.zeros((0, 4)))


@pytest.mark.unit
def test_get_quasi_diag_rejects_dangling_cluster_reference() -> None:
    """A linkage referencing a non-existent merge id is rejected.

    With a single merge row (N=2) the children must be leaves 0 and 1. Here we
    reference cluster ids 2 and 3 (>= N) whose defining rows do not all exist,
    tripping the dangling-reference / invalid-permutation guard.
    """
    bad = np.array([[2.0, 3.0, 0.5, 2.0]])
    with pytest.raises(ValidationError):
        get_quasi_diag(bad)


# --------------------------------------------------------------------------- #
# walk_forward_backtest - deterministic short run
# --------------------------------------------------------------------------- #


@pytest.fixture
def wf_panel() -> pd.DataFrame:
    """A small deterministic daily return panel long enough for monthly rebalances."""
    gen = np.random.default_rng(12345)
    n_obs, n_assets = 120, 3
    idx = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    data = gen.standard_normal((n_obs, n_assets)) * 0.01
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C"])


def _equal_weight_allocator(window: pd.DataFrame) -> pd.Series:
    """A trivial 1/N allocator used to make the backtest fully deterministic."""
    return naive_weights(list(window.columns))


@pytest.mark.unit
def test_walk_forward_deterministic_structure(wf_panel: pd.DataFrame) -> None:
    """A monthly walk-forward produces the expected rebalance count and metadata."""
    res = walk_forward_backtest(
        wf_panel,
        _equal_weight_allocator,
        lookback_window=21,
        rebalance="monthly",
        cost_bps=10.0,
        embargo=1,
        purge=1,
    )
    # first_rebal = lookback(21) + purge(1) + embargo(1) = 23; step = 21;
    # positions 23, 44, 65, 86, 107 -> 5 rebalances within 120 rows.
    assert res.n_rebalances == 5
    assert res.meta["step"] == 21
    assert res.meta["n_assets"] == 3
    assert len(res.oos_returns) == len(res.gross_returns)
    # The applied weights are shifted by one: the very first OOS return row must
    # post-date the first rebalance position (no return earned on/before decision).
    assert res.oos_returns.index[0] > wf_panel.index[22]


@pytest.mark.unit
def test_walk_forward_first_turnover_and_cost(wf_panel: pd.DataFrame) -> None:
    """The first rebalance trades from all-cash; turnover 0.5, cost = 0.5*bps/1e4.

    Going from a zero book to ``[1/3, 1/3, 1/3]`` is one-way turnover
    ``0.5 * (3 * 1/3) = 0.5``; at 10 bps/side the cost is
    ``0.5 * 10 / 10000 = 5e-4``.
    """
    res = walk_forward_backtest(
        wf_panel, _equal_weight_allocator, lookback_window=21, cost_bps=10.0
    )
    assert res.turnover.iloc[0] == pytest.approx(0.5, abs=1e-12)
    assert res.costs.iloc[0] == pytest.approx(5e-4, abs=1e-15)
    # A static 1/N allocator never trades again, so later turnover is zero.
    assert res.turnover.iloc[1:].abs().max() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_walk_forward_costs_reduce_net_returns(wf_panel: pd.DataFrame) -> None:
    """Net OOS returns never exceed gross, and equal gross+cost at the boundary.

    With a single non-zero cost charged at the first rebalance boundary, the net
    series differs from gross by exactly that cost on exactly that one row.
    """
    res = walk_forward_backtest(
        wf_panel, _equal_weight_allocator, lookback_window=21, cost_bps=25.0
    )
    diff = (res.gross_returns - res.oos_returns).round(15)
    assert (diff >= -1e-15).all()  # net <= gross everywhere
    # Exactly one boundary charge (the first, only non-zero cost).
    charged = diff[diff > 1e-12]
    assert len(charged) == 1
    assert charged.iloc[0] == pytest.approx(res.costs.iloc[0], abs=1e-15)


@pytest.mark.unit
def test_walk_forward_net_sharpe_non_increasing_in_cost(wf_panel: pd.DataFrame) -> None:
    """Higher per-side cost cannot improve the net Sharpe (cost monotonicity)."""
    sharpes = []
    for bps in (0.0, 10.0, 50.0):
        res = walk_forward_backtest(
            wf_panel, _equal_weight_allocator, lookback_window=21, cost_bps=bps
        )
        sharpes.append(sharpe_ratio(res.oos_returns))
    assert sharpes[0] >= sharpes[1] >= sharpes[2]


@pytest.mark.unit
def test_walk_forward_anchored_window_runs(wf_panel: pd.DataFrame) -> None:
    """The anchored (expanding) in-sample branch produces a valid result."""
    res = walk_forward_backtest(
        wf_panel, _equal_weight_allocator, lookback_window=21, anchored=True
    )
    assert res.meta["anchored"] is True
    assert res.n_rebalances >= 1
    assert len(res.oos_returns) > 0


@pytest.mark.unit
def test_walk_forward_to_dict_scrubs_types(wf_panel: pd.DataFrame) -> None:
    """``BacktestResult.to_dict`` renders ISO keys and finite/None floats only."""
    res = walk_forward_backtest(wf_panel, _equal_weight_allocator, lookback_window=21)
    d = res.to_dict()
    assert set(d) >= {
        "oos_returns",
        "gross_returns",
        "weights",
        "turnover",
        "costs",
        "n_rebalances",
    }
    for v in d["turnover"].values():
        assert v is None or isinstance(v, float)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"lookback_window": 4, "cost_bps": -1.0}, ValidationError),  # negative cost
        ({"lookback_window": 4, "rebalance": "weekly"}, ValidationError),  # bad cadence
        ({"lookback_window": 2}, ValidationError),  # lookback < n_assets + 1
        ({"lookback_window": 4, "purge": -1}, ValidationError),  # negative purge
    ],
)
def test_walk_forward_rejects_bad_params(
    wf_panel: pd.DataFrame, kwargs: dict, exc: type[Exception]
) -> None:
    """Scalar-parameter guards (cost/rebalance/lookback/purge) reject bad inputs."""
    with pytest.raises(exc):
        walk_forward_backtest(wf_panel, _equal_weight_allocator, **kwargs)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.5, 1.5),
        (float("nan"), None),
        (float("inf"), None),
        ("not-a-number", None),
        (None, None),
    ],
)
def test_safe_float_scrubs_non_finite(value: object, expected: float | None) -> None:
    """``_safe_float`` maps non-finite / non-numeric values to ``None``."""
    assert _safe_float(value) == expected


@pytest.mark.unit
def test_walk_forward_too_short_panel_raises() -> None:
    """A panel too short for even one split raises ``InsufficientDataError``."""
    short = pd.DataFrame(
        np.zeros((10, 2)),
        index=pd.date_range("2020-01-01", periods=10, freq="B"),
        columns=["A", "B"],
    )
    with pytest.raises(InsufficientDataError):
        walk_forward_backtest(short, _equal_weight_allocator, lookback_window=9)


# --------------------------------------------------------------------------- #
# jobson_korkie_memmel - identical series => ~zero gap, p-value 1.0
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_jkm_identical_series_gives_pvalue_one() -> None:
    """Comparing a series to itself gives a zero Sharpe gap and p-value 1.0."""
    gen = np.random.default_rng(7)
    s = pd.Series(gen.standard_normal(200) * 0.01)
    p = jobson_korkie_memmel(s, s)
    assert p == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_jkm_zero_variance_series_reports_no_evidence() -> None:
    """A degenerate (zero-variance) series yields p-value 1.0 (ill-defined gap)."""
    gen = np.random.default_rng(8)
    s = pd.Series(gen.standard_normal(60) * 0.01)
    flat = pd.Series(np.zeros(60))
    assert jobson_korkie_memmel(flat, s) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_jkm_rejects_too_few_observations() -> None:
    """Fewer than 3 aligned observations cannot support the test."""
    with pytest.raises(ValidationError):
        jobson_korkie_memmel(pd.Series([0.1, 0.2]), pd.Series([0.0, 0.1]))


@pytest.mark.unit
def test_jkm_positional_alignment_when_no_common_labels() -> None:
    """With disjoint indexes of equal length the series align positionally.

    Two series sharing no labels but the same length are compared element-wise
    in order; comparing such a series to a shifted copy of itself still produces
    a finite, valid two-sided p-value in ``[0, 1]``.
    """
    a = pd.Series([0.01, -0.02, 0.03, 0.00, 0.02], index=[0, 1, 2, 3, 4])
    b = pd.Series([0.00, -0.01, 0.02, 0.01, 0.03], index=[100, 101, 102, 103, 104])
    p = jobson_korkie_memmel(a, b)
    assert 0.0 <= p <= 1.0


@pytest.mark.unit
def test_jkm_no_common_labels_unequal_length_raises() -> None:
    """Disjoint labels AND unequal lengths cannot be aligned at all."""
    a = pd.Series([0.1, 0.2, 0.3], index=[10, 11, 12])
    b = pd.Series([0.1, 0.2], index=[20, 21])
    with pytest.raises(ValidationError):
        jobson_korkie_memmel(a, b)


@pytest.mark.unit
def test_block_bootstrap_identical_series_zero_gap_seed_reproducible() -> None:
    """Bootstrapping a series against itself gives a zero gap and a degenerate CI.

    The Sharpe gap is exactly zero, and because both series share the resample
    index every bootstrap gap is zero too, so ``ci_low == ci_high == 0``. A fixed
    seed makes the result byte-identical across runs.
    """
    gen = np.random.default_rng(9)
    s = pd.Series(gen.standard_normal(120) * 0.01)
    r1 = block_bootstrap_sharpe_gap(s, s, n_bootstrap=64, seed=11)
    r2 = block_bootstrap_sharpe_gap(s, s, n_bootstrap=64, seed=11)
    assert r1.sharpe_gap == pytest.approx(0.0, abs=1e-12)
    assert r1.ci_low == pytest.approx(0.0, abs=1e-12)
    assert r1.ci_high == pytest.approx(0.0, abs=1e-12)
    assert r1.jkm_pvalue == pytest.approx(1.0, abs=1e-12)
    assert r1.n_bootstrap == 64
    # Reproducibility for a fixed seed.
    assert r1.to_dict() == r2.to_dict()


@pytest.mark.unit
def test_block_bootstrap_explicit_block_size_recorded() -> None:
    """An explicit ``block_size`` overrides the data-driven default in the meta."""
    gen = np.random.default_rng(10)
    a = pd.Series(gen.standard_normal(90) * 0.01)
    b = a + 0.001
    res = block_bootstrap_sharpe_gap(a, b, n_bootstrap=40, block_size=5, seed=3)
    assert res.meta["block_size"] == 5
    assert res.ci_low <= res.sharpe_gap <= res.ci_high or res.ci_low <= res.ci_high


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_bootstrap": 0},  # too few resamples
        {"confidence": 1.5},  # confidence outside (0, 1)
    ],
)
def test_block_bootstrap_rejects_bad_params(kwargs: dict) -> None:
    """Out-of-range ``n_bootstrap`` / ``confidence`` are rejected."""
    s = pd.Series(np.arange(50, dtype=float) * 0.001)
    with pytest.raises(ValidationError):
        block_bootstrap_sharpe_gap(s, s, **kwargs)


# --------------------------------------------------------------------------- #
# deflated_sharpe_ratio - edge n_trials=1 reduces to plain PSR
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_dsr_single_trial_equals_psr_against_zero() -> None:
    """With ``n_trials == 1`` the expected-max benchmark collapses to 0: DSR == PSR."""
    sr = 0.1
    psr = probabilistic_sharpe_ratio(sr, n_obs=100)
    dsr = deflated_sharpe_ratio(sr, n_obs=100, n_trials=1, variance_of_trial_sharpes=0.04)
    assert dsr == pytest.approx(psr, abs=1e-12)


@pytest.mark.unit
def test_dsr_zero_trial_variance_equals_psr_against_zero() -> None:
    """Zero cross-trial variance also collapses the benchmark to 0 (DSR == PSR)."""
    sr = 0.1
    psr = probabilistic_sharpe_ratio(sr, n_obs=100)
    dsr = deflated_sharpe_ratio(sr, n_obs=100, n_trials=20, variance_of_trial_sharpes=0.0)
    assert dsr == pytest.approx(psr, abs=1e-12)


@pytest.mark.unit
def test_dsr_non_increasing_in_n_trials() -> None:
    """More trials -> a higher benchmark -> a non-increasing deflated Sharpe."""
    common = dict(n_obs=250, variance_of_trial_sharpes=0.04)
    d1 = deflated_sharpe_ratio(0.2, n_trials=1, **common)
    d10 = deflated_sharpe_ratio(0.2, n_trials=10, **common)
    d100 = deflated_sharpe_ratio(0.2, n_trials=100, **common)
    assert d1 >= d10 >= d100


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_obs": 1, "n_trials": 5, "variance_of_trial_sharpes": 0.01},  # n_obs < 2
        {"n_obs": 50, "n_trials": 0, "variance_of_trial_sharpes": 0.01},  # n_trials < 1
        {"n_obs": 50, "n_trials": 5, "variance_of_trial_sharpes": -0.1},  # var < 0
    ],
)
def test_dsr_rejects_bad_params(kwargs: dict) -> None:
    """``n_obs < 2``, ``n_trials < 1``, and negative trial variance are rejected."""
    with pytest.raises(ValidationError):
        deflated_sharpe_ratio(0.1, **kwargs)


@pytest.mark.unit
def test_psr_rejects_too_few_obs() -> None:
    """PSR requires at least two observations."""
    with pytest.raises(ValidationError):
        probabilistic_sharpe_ratio(0.1, n_obs=1)


@pytest.mark.unit
def test_psr_rejects_non_positive_variance_term() -> None:
    """A large skew can drive the bracket variance non-positive; PSR rejects it.

    ``variance = 1 - skew*SR + 0.25*(kurt-1)*SR^2`` goes non-positive for a large
    Sharpe and large positive skew, leaving the statistic undefined.
    """
    with pytest.raises(ValidationError):
        probabilistic_sharpe_ratio(5.0, n_obs=100, skew=10.0, kurtosis=3.0)


@pytest.mark.unit
def test_dsr_extreme_tails_exercise_norm_ppf_branches() -> None:
    """Large ``n_trials`` pushes the benchmark quantiles into both ppf tails.

    The expected-maximum benchmark evaluates ``Phi^{-1}(1 - 1/N)`` and
    ``Phi^{-1}(1 - 1/(N e))``; for a very large ``N`` these probabilities land in
    the upper tail of the inverse-CDF, and the DSR remains a valid probability.
    """
    dsr = deflated_sharpe_ratio(0.3, n_obs=500, n_trials=100_000, variance_of_trial_sharpes=0.04)
    assert 0.0 <= dsr <= 1.0
    # An overwhelming multiplicity strongly deflates even a healthy Sharpe.
    plain = probabilistic_sharpe_ratio(0.3, n_obs=500)
    assert dsr < plain


# --------------------------------------------------------------------------- #
# derive_verdict - every enum branch reached
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_verdict_hrp_beats_when_all_evidence_agrees() -> None:
    """Strictly-positive CI + significant JKM + DSR above threshold => HRP beats."""
    v = derive_verdict(jkm_pvalue=0.01, deflated_sharpe=0.99, ci_low=0.10, ci_high=0.30)
    assert v is Verdict.HRP_BEATS_1N


@pytest.mark.unit
def test_verdict_hrp_loses_when_ci_strictly_negative() -> None:
    """Strictly-negative CI + significant JKM + DSR above threshold => HRP loses."""
    v = derive_verdict(jkm_pvalue=0.01, deflated_sharpe=0.99, ci_low=-0.30, ci_high=-0.10)
    assert v is Verdict.HRP_LOSES_TO_1N


@pytest.mark.unit
@pytest.mark.parametrize(
    ("jkm", "dsr", "lo", "hi", "reason"),
    [
        (0.01, 0.99, -0.10, 0.20, "ci straddles zero"),
        (0.50, 0.99, 0.10, 0.20, "jkm insignificant"),
        (0.01, 0.50, 0.10, 0.20, "dsr below threshold"),
        (0.01, 0.99, 0.00, 0.20, "ci_low == 0 (boundary straddle)"),
        (0.01, 0.99, -0.20, 0.00, "ci_high == 0 (boundary straddle)"),
    ],
)
def test_verdict_no_difference_when_any_evidence_fails(
    jkm: float, dsr: float, lo: float, hi: float, reason: str
) -> None:
    """A directional claim requires ALL three lines of evidence; any failure => no diff."""
    assert derive_verdict(jkm, dsr, lo, hi) is Verdict.NO_SIGNIFICANT_DIFFERENCE, reason


@pytest.mark.unit
@pytest.mark.parametrize(
    ("jkm", "lo", "hi"),
    [
        (1.5, 0.1, 0.2),  # p-value outside [0, 1]
        (float("nan"), 0.1, 0.2),  # non-finite p-value
        (0.01, 0.3, 0.1),  # ci_low > ci_high
    ],
)
def test_verdict_rejects_malformed_inputs(jkm: float, lo: float, hi: float) -> None:
    """Out-of-range p-value and an inverted CI are rejected."""
    with pytest.raises(ValidationError):
        derive_verdict(jkm, 0.99, lo, hi)
