"""Unit tests for the Plotly figure builders and the Typer CLI.

Covers:

- ``hrp.plots`` — each of the five figure builders (dendrogram, quasi-diagonal
  heatmap before/after, weights, OOS equity, Sharpe-gap bootstrap). Every builder
  must return a plain ``{"data", "layout"}`` mapping whose contents are
  JSON-serializable (no numpy/pandas/Plotly object leaks across the API boundary),
  and we assert real numerical structure (heatmap diagonal, weight values,
  CI-marker positions, trace types) rather than merely "it runs".
- ``hrp.cli`` — the ``demo`` command, invoked via ``typer.testing.CliRunner``,
  runs the full synthetic horse-race pipeline offline and exits ``0``.

All inputs are synthetic/seeded; nothing touches the network. The dendrogram input
is a real :class:`hrp.allocate.hrp.HRPResult` constructed from a seeded panel.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from hrp import plots
from hrp._rng import make_rng
from hrp.allocate.hrp import HRPResult, hrp_allocate
from hrp.cli import build_app

pytestmark = pytest.mark.unit


def _assert_figure_dict(fig: object) -> dict:
    """Assert ``fig`` is a ``{"data", "layout"}`` mapping with JSON-safe contents.

    Returns the figure so callers can make further structural assertions. The
    ``json.dumps`` round-trip is the load-bearing check: it fails loudly if any
    numpy scalar/array, pandas object, or Plotly graph-object leaked through.
    """
    assert isinstance(fig, dict)
    assert set(fig) == {"data", "layout"}
    assert isinstance(fig["data"], list)
    assert isinstance(fig["layout"], dict)
    # Round-trips to JSON and back to identical native containers.
    encoded = json.dumps(fig)
    assert json.loads(encoded) == fig
    return fig


@pytest.fixture
def hrp_result(one_factor_returns: pd.DataFrame) -> HRPResult:
    """A real ``HRPResult`` from the seeded one-factor panel (no network)."""
    return hrp_allocate(one_factor_returns)


# --------------------------------------------------------------------------- #
# dendrogram_figure                                                            #
# --------------------------------------------------------------------------- #
def test_dendrogram_figure_shape_and_title(
    hrp_result: HRPResult, one_factor_returns: pd.DataFrame
) -> None:
    """Dendrogram builder returns a serializable figure with the default title."""
    labels = list(one_factor_returns.columns)
    fig = _assert_figure_dict(
        plots.dendrogram_figure(hrp_result.link, labels)
    )

    # A dendrogram of N leaves draws N-1 merge links -> non-empty trace list.
    assert len(fig["data"]) > 0
    assert fig["layout"]["title"] == {"text": "HRP dendrogram"}


def test_dendrogram_figure_accepts_list_link() -> None:
    """The linkage matrix may be passed as a nested list; output stays JSON-safe."""
    # A minimal 3-leaf linkage: merge leaves 0&1, then that cluster with leaf 2.
    link = [
        [0.0, 1.0, 0.5, 2.0],
        [2.0, 3.0, 1.0, 3.0],
    ]
    fig = _assert_figure_dict(
        plots.dendrogram_figure(np.asarray(link), ["A", "B", "C"])
    )
    assert len(fig["data"]) > 0


# --------------------------------------------------------------------------- #
# quasidiag_heatmap_figure (before / after)                                   #
# --------------------------------------------------------------------------- #
def test_quasidiag_heatmap_before_is_raw_order(
    block_correlation_cov: pd.DataFrame,
) -> None:
    """Identity order renders the raw correlation: z matches the input matrix."""
    corr = block_correlation_cov
    n = corr.shape[0]
    identity_order = list(range(n))

    fig = _assert_figure_dict(
        plots.quasidiag_heatmap_figure(corr, identity_order)
    )
    trace = fig["data"][0]
    assert trace["type"] == "heatmap"
    assert trace["zmin"] == -1.0 and trace["zmax"] == 1.0

    z = np.asarray(trace["z"])
    # Identity reorder is a no-op: z equals the original correlation matrix.
    np.testing.assert_allclose(z, corr.to_numpy())
    # Unit diagonal is preserved.
    np.testing.assert_allclose(np.diag(z), np.ones(n))
    # Axis labels follow the (identity) permutation = the original columns.
    assert trace["x"] == [str(c) for c in corr.columns]
    assert trace["y"] == [str(c) for c in corr.columns]


def test_quasidiag_heatmap_after_reorders_rows_and_cols(
    de_prado_example: pd.DataFrame,
) -> None:
    """A non-trivial leaf order permutes rows AND columns symmetrically."""
    corr = de_prado_example
    # Group the two correlated pairs (0,1) and (2,3) by reversing block order.
    order = [2, 3, 0, 1]

    fig = _assert_figure_dict(
        plots.quasidiag_heatmap_figure(corr, order, title="after")
    )
    trace = fig["data"][0]
    z = np.asarray(trace["z"])
    expected = corr.to_numpy()[np.ix_(order, order)]
    np.testing.assert_allclose(z, expected)

    labels = [str(c) for c in corr.columns]
    assert trace["x"] == [labels[i] for i in order]
    assert trace["y"] == [labels[i] for i in order]
    # The custom title propagates into the layout.
    assert fig["layout"]["title"] == {"text": "after"}


def test_quasidiag_heatmap_accepts_plain_ndarray() -> None:
    """A bare ndarray (no labels) yields integer-string axis labels."""
    mat = np.array([[1.0, 0.3], [0.3, 1.0]])
    fig = _assert_figure_dict(
        plots.quasidiag_heatmap_figure(mat, [1, 0])
    )
    trace = fig["data"][0]
    # order=[1,0] swaps the labels derived from positions.
    assert trace["x"] == ["1", "0"]
    np.testing.assert_allclose(
        np.asarray(trace["z"]), mat[np.ix_([1, 0], [1, 0])]
    )


# --------------------------------------------------------------------------- #
# weights_figure                                                              #
# --------------------------------------------------------------------------- #
def test_weights_figure_horizontal_bar_values(hrp_result: HRPResult) -> None:
    """Weights builder emits a horizontal bar whose x-values are the weights."""
    weights = hrp_result.weights
    fig = _assert_figure_dict(plots.weights_figure(weights))

    trace = fig["data"][0]
    assert trace["type"] == "bar"
    assert trace["orientation"] == "h"
    assert trace["y"] == [str(idx) for idx in weights.index]
    np.testing.assert_allclose(
        np.asarray(trace["x"]), weights.to_numpy()
    )
    # Simplex weights sum to one; the bar values preserve that.
    assert sum(trace["x"]) == pytest.approx(1.0)
    assert fig["layout"]["title"] == {"text": "Portfolio weights"}


def test_weights_figure_accepts_mapping() -> None:
    """A plain dict of weights is coerced to a Series and rendered by label order."""
    fig = _assert_figure_dict(
        plots.weights_figure(
            pd.Series({"X": 0.25, "Y": 0.75}), title="custom"
        )
    )
    trace = fig["data"][0]
    assert trace["y"] == ["X", "Y"]
    np.testing.assert_allclose(np.asarray(trace["x"]), [0.25, 0.75])
    assert fig["layout"]["title"] == {"text": "custom"}


# --------------------------------------------------------------------------- #
# oos_equity_figure                                                           #
# --------------------------------------------------------------------------- #
def test_oos_equity_figure_one_trace_per_column() -> None:
    """One line trace per allocator column, x-axis = ISO-formatted dates."""
    index = pd.date_range("2021-01-01", periods=4, freq="B")
    curves = pd.DataFrame(
        {"hrp": [1.0, 1.02, 1.05, 1.04], "naive_1n": [1.0, 0.99, 1.01, 1.03]},
        index=index,
    )
    fig = _assert_figure_dict(plots.oos_equity_figure(curves))

    assert len(fig["data"]) == 2
    assert {t["name"] for t in fig["data"]} == {"hrp", "naive_1n"}
    for trace in fig["data"]:
        assert trace["type"] == "scatter"
        assert trace["mode"] == "lines"
        col = trace["name"]
        np.testing.assert_allclose(
            np.asarray(trace["y"]), curves[col].to_numpy()
        )
    # The shared x-axis carries ISO date strings (timestamps did not leak).
    hrp_trace = next(t for t in fig["data"] if t["name"] == "hrp")
    assert hrp_trace["x"][0] == index[0].isoformat()
    assert all(isinstance(v, str) for v in hrp_trace["x"])


def test_oos_equity_figure_empty_frame_has_no_traces() -> None:
    """An empty curve frame yields no traces but a valid serializable layout."""
    fig = _assert_figure_dict(plots.oos_equity_figure(pd.DataFrame()))
    assert fig["data"] == []


# --------------------------------------------------------------------------- #
# sharpe_gap_bootstrap_figure                                                 #
# --------------------------------------------------------------------------- #
def test_sharpe_gap_bootstrap_figure_histogram_and_markers() -> None:
    """Histogram of gap samples plus three vertical markers (0, CI low, CI high)."""
    samples = np.array([-0.10, -0.02, 0.0, 0.03, 0.08, 0.12, 0.20])
    ci_low, ci_high = -0.05, 0.15

    fig = _assert_figure_dict(
        plots.sharpe_gap_bootstrap_figure(
            samples, ci_low=ci_low, ci_high=ci_high
        )
    )
    trace = fig["data"][0]
    assert trace["type"] == "histogram"
    np.testing.assert_allclose(np.asarray(trace["x"]), samples)

    shapes = fig["layout"]["shapes"]
    assert len(shapes) == 3
    # Markers sit at zero (the null) and at both CI bounds; each is a vertical
    # line (x0 == x1) spanning the full paper height.
    marker_x = [s["x0"] for s in shapes]
    assert marker_x == [0.0, ci_low, ci_high]
    for shape in shapes:
        assert shape["type"] == "line"
        assert shape["x0"] == shape["x1"]
        assert shape["yref"] == "paper"
        assert (shape["y0"], shape["y1"]) == (0.0, 1.0)


def test_sharpe_gap_bootstrap_figure_accepts_series() -> None:
    """A pandas Series of samples is flattened to the histogram x-values."""
    series = pd.Series([0.01, -0.01, 0.02, 0.0])
    fig = _assert_figure_dict(
        plots.sharpe_gap_bootstrap_figure(series, ci_low=-0.01, ci_high=0.02)
    )
    np.testing.assert_allclose(
        np.asarray(fig["data"][0]["x"]), series.to_numpy()
    )


def test_sharpe_gap_bootstrap_figure_accepts_2d_array() -> None:
    """A 2-D array of samples is raveled to a flat histogram input."""
    arr = np.array([[0.01, -0.01], [0.02, 0.03]])
    fig = _assert_figure_dict(
        plots.sharpe_gap_bootstrap_figure(arr, ci_low=-0.01, ci_high=0.03)
    )
    np.testing.assert_allclose(
        np.asarray(fig["data"][0]["x"]), arr.ravel()
    )


# --------------------------------------------------------------------------- #
# CLI: demo command (synthetic pipeline, no network)                          #
# --------------------------------------------------------------------------- #
def test_cli_demo_runs_offline_and_exits_zero() -> None:
    """The ``demo`` command runs the synthetic horse race offline and exits 0."""
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(
        build_app(), ["demo", "--n-assets", "6", "--n-bootstrap", "50"]
    )

    assert result.exit_code == 0, result.output
    # The summary header and the synthetic-data marker confirm the pipeline ran
    # entirely on the deterministic synthetic panel (no network fetch).
    assert "HRP walk-forward horse race" in result.stdout
    assert "data source        : synthetic" in result.stdout
    assert "verdict" in result.stdout


def test_cli_demo_default_options_exit_zero() -> None:
    """The ``demo`` command with default options also runs offline and exits 0."""
    from typer.testing import CliRunner

    # Keep the bootstrap small for speed; assets default to 8.
    result = CliRunner().invoke(build_app(), ["demo", "--n-bootstrap", "30"])
    assert result.exit_code == 0, result.output
    assert "assets             : 8" in result.stdout


def test_cli_no_args_shows_help() -> None:
    """Invoking with no arguments prints help (no_args_is_help) and lists demo."""
    from typer.testing import CliRunner

    result = CliRunner().invoke(build_app(), [])
    # no_args_is_help exits with Typer's usage code (2) and lists the commands.
    assert result.exit_code == 2
    assert "demo" in result.output
    assert "run" in result.output


def test_build_app_is_isolated_instance() -> None:
    """build_app returns a fresh Typer app each call (no shared mutable state)."""
    import typer

    app_a = build_app()
    app_b = build_app()
    assert isinstance(app_a, typer.Typer)
    assert app_a is not app_b


def test_make_rng_is_available_for_seeded_inputs() -> None:
    """Sanity check that the seeded RNG helper used elsewhere stays importable."""
    gen = make_rng(123)
    assert isinstance(gen.standard_normal(3), np.ndarray)
