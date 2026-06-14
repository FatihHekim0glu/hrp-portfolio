"""Unit tests for data loading and the low-level helper modules.

Covers:

- ``hrp.data`` — forced synthetic-GBM fallback for :func:`get_prices` (no
  network; the ``data`` extra is absent so the lazy fetchers raise and we land on
  the deterministic synthetic panel), :func:`compute_returns`
  (``pct_change(fill_method=None)`` + leading-row drop + NaN-gap handling), and
  :func:`get_risk_free` (synthetic 2% annual fallback, deannualized).
- ``hrp._validation`` — error paths and inner alignment.
- ``hrp._rng`` — seeded determinism and substream independence.
- ``hrp._manifest`` — config-hash determinism and ``to_dict`` round-trips.

All inputs are synthetic/seeded; nothing touches the network.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from hrp import data as hrp_data
from hrp._exceptions import InsufficientDataError, ValidationError
from hrp._manifest import RunManifest, config_hash
from hrp._rng import make_rng, spawn_substreams
from hrp._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    validate_min_obs,
)
from hrp.data import compute_returns, get_prices, get_risk_free

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# hrp.data: get_prices — forced synthetic fallback                            #
# --------------------------------------------------------------------------- #


def test_get_prices_falls_back_to_synthetic_offline() -> None:
    """With the ``data`` extra absent, the fetch chain raises and we get synthetic.

    The lazy ``import yfinance`` / ``pandas_datareader`` inside the fetchers raise
    :class:`ImportError`, which :func:`get_prices` swallows, landing on the
    deterministic synthetic panel. No network is touched.
    """
    tickers = ["AAA", "BBB", "CCC"]
    start, end = date(2021, 1, 1), date(2021, 3, 1)

    frame, source = get_prices(tickers, start, end)

    assert source == "synthetic"
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == tickers
    # Business-day (Mon-Fri) index spanning the inclusive range.
    expected_index = pd.date_range(start=start, end=end, freq="B")
    pd.testing.assert_index_equal(frame.index, expected_index)
    assert frame.shape == (len(expected_index), len(tickers))
    assert str(frame.to_numpy().dtype) == "float64"
    # GBM prices are strictly positive and finite.
    assert np.isfinite(frame.to_numpy()).all()
    assert (frame.to_numpy() > 0).all()


def test_get_prices_synthetic_is_deterministic() -> None:
    """The same request yields byte-identical synthetic prices (seeded off request)."""
    tickers = ["AAA", "BBB"]
    start, end = date(2021, 1, 1), date(2021, 2, 1)

    frame_a, src_a = get_prices(tickers, start, end)
    frame_b, src_b = get_prices(tickers, start, end)

    assert src_a == src_b == "synthetic"
    pd.testing.assert_frame_equal(frame_a, frame_b)


def test_get_prices_synthetic_anchored_first_row() -> None:
    """Row 0 is the GBM start price (log-return anchored at 0): in [20, 200)."""
    frame, _ = get_prices(["AAA", "BBB", "CCC"], date(2021, 1, 1), date(2021, 2, 1))
    first_row = frame.iloc[0].to_numpy()
    assert np.all(first_row >= 20.0)
    assert np.all(first_row < 200.0)


def test_get_prices_explicit_source_pref_still_falls_back() -> None:
    """A single-source preference whose fetcher raises also lands on synthetic."""
    frame, source = get_prices(
        ["AAA"], date(2021, 1, 1), date(2021, 2, 1), source_pref="stooq"
    )
    assert source == "synthetic"
    assert list(frame.columns) == ["AAA"]


def test_get_prices_rejects_empty_tickers() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        get_prices([], date(2021, 1, 1), date(2021, 2, 1))


def test_get_prices_rejects_end_not_after_start() -> None:
    with pytest.raises(ValidationError, match="must be after start"):
        get_prices(["AAA"], date(2021, 2, 1), date(2021, 1, 1))


def test_get_prices_does_not_touch_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Belt-and-suspenders: stub the fetchers to assert the synthetic path is used."""

    def _boom(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise RuntimeError("network access attempted")

    monkeypatch.setattr(hrp_data, "_fetch_yfinance", _boom)
    monkeypatch.setattr(hrp_data, "_fetch_stooq", _boom)

    frame, source = get_prices(["AAA", "BBB"], date(2021, 1, 1), date(2021, 2, 1))
    assert source == "synthetic"
    assert not frame.empty


# --------------------------------------------------------------------------- #
# hrp.data: compute_returns                                                    #
# --------------------------------------------------------------------------- #


def test_compute_returns_hand_computed() -> None:
    """Simple returns equal price ratios minus one, with the leading row dropped."""
    idx = pd.date_range("2021-01-01", periods=4, freq="B")
    prices = pd.DataFrame(
        {"X": [100.0, 110.0, 99.0, 99.0], "Y": [50.0, 25.0, 50.0, 75.0]}, index=idx
    )

    returns = compute_returns(prices)

    # First (all-NaN) row dropped -> 3 rows remain.
    assert returns.shape == (3, 2)
    pd.testing.assert_index_equal(returns.index, idx[1:])
    # X: 110/100-1=0.10, 99/110-1=-0.10, 99/99-1=0.0
    np.testing.assert_allclose(returns["X"].to_numpy(), [0.10, -0.10, 0.0])
    # Y: 25/50-1=-0.50, 50/25-1=1.0, 75/50-1=0.5
    np.testing.assert_allclose(returns["Y"].to_numpy(), [-0.50, 1.0, 0.5])


def test_compute_returns_no_forward_fill_across_nan_gap() -> None:
    """A NaN price must NOT be forward-filled before differencing.

    ``pct_change(fill_method=None)`` differences each column on its observed
    values: the return at the NaN row and the row after are NaN, never a
    manufactured 0.0 from ffill-then-diff.
    """
    idx = pd.date_range("2021-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"X": [100.0, np.nan, 121.0, 121.0]}, index=idx)

    returns = compute_returns(prices)

    vals = returns["X"].to_numpy()
    # Row 1 (the NaN price) -> NaN return; row 2 (after NaN) -> NaN return.
    assert np.isnan(vals[0])
    assert np.isnan(vals[1])
    # Row 3: 121/121 - 1 = 0.0 (both prices observed).
    assert vals[2] == pytest.approx(0.0)
    # Crucially, no spurious zero return was manufactured across the gap.
    assert not np.any(vals == pytest.approx(0.0)) or vals[2] == pytest.approx(0.0)


def test_compute_returns_accepts_ndarray() -> None:
    """A 2-D ndarray is coerced via ensure_dataframe and differenced correctly."""
    arr = np.array([[1.0, 2.0], [2.0, 4.0], [4.0, 4.0]])
    returns = compute_returns(arr)
    assert returns.shape == (2, 2)
    np.testing.assert_allclose(returns.iloc[0].to_numpy(), [1.0, 1.0])
    np.testing.assert_allclose(returns.iloc[1].to_numpy(), [1.0, 0.0])


def test_compute_returns_roundtrips_synthetic_prices() -> None:
    """End-to-end: synthetic prices -> returns has one fewer row and is float64."""
    prices, _ = get_prices(["AAA", "BBB"], date(2021, 1, 1), date(2021, 3, 1))
    returns = compute_returns(prices)
    assert returns.shape == (prices.shape[0] - 1, prices.shape[1])
    assert str(returns.to_numpy().dtype) == "float64"


# --------------------------------------------------------------------------- #
# hrp.data: get_risk_free                                                      #
# --------------------------------------------------------------------------- #


def test_get_risk_free_synthetic_flat_2pct() -> None:
    """Offline fallback is a flat 2% annual rate, deannualized per-period."""
    start, end = date(2021, 1, 1), date(2021, 2, 1)
    rf = get_risk_free(start, end)

    expected_index = pd.date_range(start=start, end=end, freq="B")
    pd.testing.assert_index_equal(rf.index, expected_index)
    assert rf.name == "risk_free"

    # Per-period rate consistent with 252: (1.02)^(1/252) - 1, flat across the grid.
    expected = (1.02) ** (1.0 / 252.0) - 1.0
    np.testing.assert_allclose(rf.to_numpy(), expected)
    # All entries identical (flat synthetic rate).
    assert rf.nunique() == 1
    # Tiny positive daily rate.
    assert 0.0 < rf.iloc[0] < 1e-3


def test_get_risk_free_periods_per_year_changes_deannualization() -> None:
    """A monthly annualization (12) yields a larger per-period rate than daily (252)."""
    start, end = date(2021, 1, 1), date(2021, 6, 1)
    rf_daily = get_risk_free(start, end, periods_per_year=252)
    rf_monthly = get_risk_free(start, end, periods_per_year=12)

    expected_monthly = (1.02) ** (1.0 / 12.0) - 1.0
    np.testing.assert_allclose(rf_monthly.to_numpy(), expected_monthly)
    assert rf_monthly.iloc[0] > rf_daily.iloc[0]


def test_get_risk_free_rejects_end_not_after_start() -> None:
    with pytest.raises(ValidationError, match="must be after start"):
        get_risk_free(date(2021, 2, 1), date(2021, 1, 1))


def test_get_risk_free_aligned_to_returns_index() -> None:
    """The rf series shares the business-day grid used by the synthetic prices."""
    start, end = date(2021, 1, 1), date(2021, 3, 1)
    prices, _ = get_prices(["AAA"], start, end)
    rf = get_risk_free(start, end)
    pd.testing.assert_index_equal(rf.index, prices.index)


# --------------------------------------------------------------------------- #
# hrp._validation                                                              #
# --------------------------------------------------------------------------- #


def test_ensure_dataframe_passthrough_and_copy() -> None:
    """A clean DataFrame is returned as a float64 copy (input not mutated)."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = ensure_dataframe(df, name="x")
    assert str(out.to_numpy().dtype) == "float64"
    out.iloc[0, 0] = 999.0
    assert df.iloc[0, 0] == 1  # original untouched


def test_ensure_dataframe_rejects_non_2d_ndarray() -> None:
    with pytest.raises(ValidationError, match="2-dimensional"):
        ensure_dataframe(np.arange(5), name="x")


def test_ensure_dataframe_rejects_nan_by_default() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan]})
    with pytest.raises(ValidationError, match="contains NaN"):
        ensure_dataframe(df, name="x")


def test_ensure_dataframe_allows_nan_when_requested() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan]})
    out = ensure_dataframe(df, name="x", allow_nan=True)
    assert bool(out.isna().to_numpy().any())


def test_ensure_dataframe_rejects_empty() -> None:
    with pytest.raises(ValidationError, match="at least one row and one column"):
        ensure_dataframe(pd.DataFrame(), name="x")


def test_ensure_dataframe_applies_columns_to_ndarray() -> None:
    out = ensure_dataframe(np.ones((2, 2)), name="x", columns=["p", "q"])
    assert list(out.columns) == ["p", "q"]


def test_ensure_series_rejects_non_1d_ndarray() -> None:
    with pytest.raises(ValidationError, match="1-dimensional"):
        ensure_series(np.ones((2, 2)), name="s")


def test_ensure_series_accepts_1d_ndarray() -> None:
    out = ensure_series(np.array([1.0, 2.0, 3.0]), name="s")
    assert isinstance(out, pd.Series)
    np.testing.assert_array_equal(out.to_numpy(), [1.0, 2.0, 3.0])


def test_ensure_series_accepts_list() -> None:
    out = ensure_series([1.0, 2.0], name="s")
    assert str(out.dtype) == "float64"
    assert len(out) == 2


def test_ensure_series_allows_nan_when_requested() -> None:
    out = ensure_series(pd.Series([1.0, np.nan]), name="s", allow_nan=True)
    assert bool(out.isna().any())


def test_ensure_series_rejects_nan_and_empty() -> None:
    with pytest.raises(ValidationError, match="contains NaN"):
        ensure_series(pd.Series([1.0, np.nan]), name="s")
    with pytest.raises(ValidationError, match="non-empty"):
        ensure_series(pd.Series([], dtype="float64"), name="s")


def test_validate_min_obs_passes_and_raises() -> None:
    df = pd.DataFrame(np.ones((5, 2)))
    # Exactly enough rows: no error.
    validate_min_obs(df, 5, name="panel")
    # One short of the requirement: InsufficientDataError (a ValidationError).
    with pytest.raises(InsufficientDataError, match="at least 6 are required"):
        validate_min_obs(df, 6, name="panel")
    assert issubclass(InsufficientDataError, ValidationError)


def test_align_inner_intersects_sorted() -> None:
    """align_inner reindexes both frames to the sorted common index."""
    idx_l = pd.date_range("2021-01-04", periods=5, freq="B")  # Mon..Fri
    idx_r = pd.date_range("2021-01-06", periods=5, freq="B")  # Wed..next Tue
    left = pd.DataFrame({"l": range(5)}, index=idx_l)
    right = pd.DataFrame({"r": range(5)}, index=idx_r)

    aligned_left, aligned_right = align_inner(left, right)

    common = idx_l.intersection(idx_r).sort_values()
    pd.testing.assert_index_equal(aligned_left.index, common)
    pd.testing.assert_index_equal(aligned_right.index, common)
    # Columns are preserved on each side.
    assert list(aligned_left.columns) == ["l"]
    assert list(aligned_right.columns) == ["r"]
    # Monotonic increasing after the inner join.
    assert aligned_left.index.is_monotonic_increasing


def test_align_inner_raises_on_empty_intersection() -> None:
    left = pd.DataFrame({"l": [1]}, index=pd.to_datetime(["2021-01-01"]))
    right = pd.DataFrame({"r": [1]}, index=pd.to_datetime(["2022-01-01"]))
    with pytest.raises(ValidationError, match="no common index labels"):
        align_inner(left, right)


# --------------------------------------------------------------------------- #
# hrp._rng                                                                      #
# --------------------------------------------------------------------------- #


def test_make_rng_is_deterministic_for_seed() -> None:
    a = make_rng(12345).standard_normal(16)
    b = make_rng(12345).standard_normal(16)
    np.testing.assert_array_equal(a, b)


def test_make_rng_different_seeds_diverge() -> None:
    a = make_rng(1).standard_normal(16)
    b = make_rng(2).standard_normal(16)
    assert not np.array_equal(a, b)


def test_make_rng_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_rng(-1)


def test_spawn_substreams_count_and_type() -> None:
    children = spawn_substreams(7, 4)
    assert len(children) == 4
    assert all(isinstance(c, np.random.Generator) for c in children)


def test_spawn_substreams_children_are_independent() -> None:
    """Distinct substreams produce distinct draws (pairwise different)."""
    children = spawn_substreams(7, 3)
    draws = [c.standard_normal(32) for c in children]
    assert not np.array_equal(draws[0], draws[1])
    assert not np.array_equal(draws[0], draws[2])
    assert not np.array_equal(draws[1], draws[2])


def test_spawn_substreams_is_deterministic() -> None:
    """The same (seed, n) reproduces identical children byte-for-byte."""
    a = [c.standard_normal(8) for c in spawn_substreams(99, 3)]
    b = [c.standard_normal(8) for c in spawn_substreams(99, 3)]
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_spawn_substreams_zero_children() -> None:
    assert spawn_substreams(0, 0) == []


def test_spawn_substreams_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        spawn_substreams(-1, 2)
    with pytest.raises(ValueError, match="non-negative"):
        spawn_substreams(2, -1)


# --------------------------------------------------------------------------- #
# hrp._manifest                                                                #
# --------------------------------------------------------------------------- #


def test_config_hash_is_deterministic() -> None:
    cfg = {"a": 1, "b": [1, 2, 3], "c": "x"}
    assert config_hash(cfg) == config_hash(cfg)


def test_config_hash_is_key_order_invariant() -> None:
    """Logically-equal configs differing only in key order hash identically."""
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_config_hash_changes_with_values() -> None:
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_config_hash_length_and_hexness() -> None:
    digest = config_hash({"seed": 7})
    assert len(digest) == 32  # 16-byte BLAKE2b -> 32 hex chars
    int(digest, 16)  # parses as hex (raises ValueError otherwise)


def test_config_hash_handles_non_json_native_via_default_str() -> None:
    """``default=str`` lets date-like values hash without raising."""
    digest = config_hash({"start": date(2021, 1, 1)})
    assert len(digest) == 32


def test_run_manifest_to_dict_roundtrips() -> None:
    """to_dict returns every field as a plain JSON-serializable dict."""
    manifest = RunManifest(
        git_sha="abc123",
        dirty=True,
        config_hash="deadbeef" * 4,
        seed=42,
        extra={"note": "demo"},
    )
    d = manifest.to_dict()
    assert d == {
        "git_sha": "abc123",
        "dirty": True,
        "config_hash": "deadbeef" * 4,
        "seed": 42,
        "extra": {"note": "demo"},
    }
    # Reconstructible from its own dict.
    assert RunManifest(**d) == manifest


def test_run_manifest_capture_hashes_config_and_records_seed() -> None:
    """capture() embeds config_hash(config) and the int seed; git fields are strings/bools."""
    cfg = {"method": "hrp", "lookback": 252}
    manifest = RunManifest.capture(cfg, seed=2026)

    assert manifest.config_hash == config_hash(cfg)
    assert manifest.seed == 2026
    assert isinstance(manifest.git_sha, str)
    assert isinstance(manifest.dirty, bool)


def test_run_manifest_capture_handles_git_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When git shelling-out fails, capture() degrades to 'unknown'/False, not raise."""
    import subprocess as _subprocess

    from hrp import _manifest as manifest_mod

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise _subprocess.SubprocessError("no git here")

    monkeypatch.setattr(manifest_mod.subprocess, "run", _boom)

    manifest = RunManifest.capture({"k": "v"}, seed=5)
    assert manifest.git_sha == "unknown"
    assert manifest.dirty is False
    assert manifest.config_hash == config_hash({"k": "v"})
    assert manifest.seed == 5


def test_run_manifest_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    manifest = RunManifest(git_sha="x", dirty=False, config_hash="y", seed=1)
    with pytest.raises(FrozenInstanceError):
        manifest.seed = 2  # type: ignore[misc]
