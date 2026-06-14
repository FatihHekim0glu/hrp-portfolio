"""Unit tests for the real Polygon EOD data provider.

These tests MOCK the HTTP layer end-to-end and assert that NO real network call
ever happens:

- the stdlib path (used when ``httpx`` is absent) is exercised by monkeypatching
  ``urllib.request.urlopen``;
- the ``httpx`` path is exercised by injecting a fake ``httpx`` module into
  ``sys.modules`` so the provider's lazy ``import httpx`` resolves to a stub.

Coverage:

- correct parsing of a sample Polygon aggregates JSON response into a wide
  ``date x ticker`` DataFrame of adjusted closes (inner-joined across tickers);
- HTTP ``429`` retry-with-backoff behaviour on both the urllib and httpx paths
  (``time.sleep`` is patched out so the tests are fast);
- API-key resolution from an explicit arg / env var / ``.env``;
- a hard guarantee that no socket is ever opened (any real network attempt is
  turned into an immediate failure).

Nothing here touches the network.
"""

from __future__ import annotations

import json
import sys
import types
import urllib.error
from datetime import date
from typing import Any, ClassVar

import pandas as pd
import pytest

from hrp import data as hrp_data
from hrp._exceptions import ValidationError
from hrp.data_providers.polygon import PolygonProvider

pytestmark = pytest.mark.unit

_API_KEY = "test-key-123"


# --------------------------------------------------------------------------- #
# Sample Polygon aggregates payloads                                          #
# --------------------------------------------------------------------------- #
# Polygon bar dicts: ``t`` = epoch ms (UTC midnight), ``c`` = adjusted close.
_DAY_MS = 86_400_000
_T0 = pd.Timestamp("2021-01-04", tz="UTC").value // 1_000_000  # epoch ms


def _payload(closes: list[float], *, start_ms: int = _T0) -> dict[str, Any]:
    """A well-formed Polygon aggregates payload with one bar per consecutive day."""
    return {
        "ticker": "X",
        "status": "OK",
        "queryCount": len(closes),
        "resultsCount": len(closes),
        "adjusted": True,
        "results": [
            {
                "t": start_ms + i * _DAY_MS,
                "o": c - 1.0,
                "h": c + 1.0,
                "l": c - 2.0,
                "c": c,
                "v": 1_000 + i,
            }
            for i, c in enumerate(closes)
        ],
    }


# --------------------------------------------------------------------------- #
# Network guard + fakes                                                        #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Belt-and-suspenders: make any *real* network attempt fail loudly.

    Tests install their own fakes for ``urlopen`` / ``httpx``; this fixture
    guarantees that if some path slipped through to a real socket the test would
    error rather than silently reach out to the internet.
    """
    import socket

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("real network access attempted")

    monkeypatch.setattr(socket.socket, "connect", _boom)
    # Patch the urllib opener too, so the default is a hard failure unless a test
    # explicitly replaces it with its own fake.
    monkeypatch.setattr("urllib.request.urlopen", _boom)
    # No real sleeping during 429 backoff.
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)


class _FakeUrlopenResponse:
    """Minimal context-manager stand-in for ``urllib.request.urlopen``'s result."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeUrlopenResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _install_urlopen(
    monkeypatch: pytest.MonkeyPatch, handler: Any
) -> list[str]:
    """Install a fake ``urllib.request.urlopen`` and record the URLs it is called with.

    ``handler(url) -> dict`` returns the JSON payload to serve for that URL, or
    may raise an ``urllib.error.HTTPError`` to simulate an HTTP status.
    """
    calls: list[str] = []

    def _fake_urlopen(url: str, *args: object, **kwargs: object) -> _FakeUrlopenResponse:
        calls.append(url)
        payload = handler(url)
        return _FakeUrlopenResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    return calls


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.polygon.io", code=code, msg="boom", hdrs=None, fp=None
    )


# --- Fake httpx -------------------------------------------------------------- #


class _FakeHttpxResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHttpxClient:
    """Serves a scripted queue of responses; records requested URLs."""

    #: Class-level script: list of (status_code, payload) consumed per .get().
    script: ClassVar[list[tuple[int, dict[str, Any]]]] = []
    urls: ClassVar[list[str]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> _FakeHttpxClient:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def get(self, url: str, *args: object, **kwargs: object) -> _FakeHttpxResponse:
        type(self).urls.append(url)
        status, payload = type(self).script.pop(0)
        return _FakeHttpxResponse(status, payload)


def _install_fake_httpx(
    monkeypatch: pytest.MonkeyPatch, script: list[tuple[int, dict[str, Any]]]
) -> type[_FakeHttpxClient]:
    """Inject a fake ``httpx`` module whose Client serves ``script`` responses."""
    _FakeHttpxClient.script = list(script)
    _FakeHttpxClient.urls = []
    fake_httpx = types.ModuleType("httpx")
    fake_httpx.Client = _FakeHttpxClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    return _FakeHttpxClient


# --------------------------------------------------------------------------- #
# Parsing (urllib path)                                                        #
# --------------------------------------------------------------------------- #


def test_fetch_parses_single_ticker_via_urllib(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single-ticker payload parses into a date-indexed, float64 close panel."""
    calls = _install_urlopen(monkeypatch, lambda _url: _payload([100.0, 101.0, 102.5]))

    provider = PolygonProvider(api_key=_API_KEY)
    frame = provider.fetch(["AAPL"], date(2021, 1, 4), date(2021, 1, 8))

    assert list(frame.columns) == ["AAPL"]
    assert frame.shape == (3, 1)
    assert str(frame.to_numpy().dtype) == "float64"
    assert frame["AAPL"].tolist() == [100.0, 101.0, 102.5]
    # Index is the normalized (midnight, tz-naive) bar dates.
    assert list(frame.index) == [
        pd.Timestamp("2021-01-04"),
        pd.Timestamp("2021-01-05"),
        pd.Timestamp("2021-01-06"),
    ]
    # One request was made, to the documented aggregates endpoint, with the key.
    assert len(calls) == 1
    assert "api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2021-01-04/2021-01-08" in calls[0]
    assert "adjusted=true" in calls[0]
    assert "sort=asc" in calls[0]
    assert "limit=50000" in calls[0]
    assert f"apiKey={_API_KEY}" in calls[0]


def test_fetch_inner_joins_across_tickers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple tickers are inner-joined on date; only common dates survive."""

    def _handler(url: str) -> dict[str, Any]:
        if "ticker/AAA/" in url:
            # 3 days starting 2021-01-04.
            return _payload([10.0, 11.0, 12.0], start_ms=_T0)
        # BBB: 3 days but starting one day later -> overlap is 2 days.
        return _payload([20.0, 21.0, 22.0], start_ms=_T0 + _DAY_MS)

    _install_urlopen(monkeypatch, _handler)

    provider = PolygonProvider(api_key=_API_KEY)
    frame = provider.fetch(["AAA", "BBB"], date(2021, 1, 4), date(2021, 1, 8))

    # Columns preserve request order.
    assert list(frame.columns) == ["AAA", "BBB"]
    # Inner join: only the two overlapping dates (Jan 5 and Jan 6).
    assert list(frame.index) == [pd.Timestamp("2021-01-05"), pd.Timestamp("2021-01-06")]
    assert frame.shape == (2, 2)
    assert frame["AAA"].tolist() == [11.0, 12.0]
    assert frame["BBB"].tolist() == [20.0, 21.0]
    # No NaNs survive an inner join.
    assert not frame.isna().to_numpy().any()


def test_fetch_raises_on_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """A payload with no ``results`` raises (caller decides on fallback)."""
    _install_urlopen(
        monkeypatch, lambda _url: {"ticker": "X", "status": "OK", "results": []}
    )
    provider = PolygonProvider(api_key=_API_KEY)
    with pytest.raises(ValueError, match="no results"):
        provider.fetch(["AAPL"], date(2021, 1, 4), date(2021, 1, 8))


# --------------------------------------------------------------------------- #
# 429 retry-with-backoff                                                       #
# --------------------------------------------------------------------------- #


def test_urllib_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two 429s then a 200 -> succeeds after exactly two retries (urllib path)."""
    state = {"n": 0}

    def _handler(_url: str) -> dict[str, Any]:
        state["n"] += 1
        if state["n"] <= 2:
            raise _http_error(429)
        return _payload([100.0, 101.0])

    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    _install_urlopen(monkeypatch, _handler)

    provider = PolygonProvider(api_key=_API_KEY, max_retries=3, backoff_base=0.5)
    frame = provider.fetch(["AAPL"], date(2021, 1, 4), date(2021, 1, 8))

    assert frame["AAPL"].tolist() == [100.0, 101.0]
    assert state["n"] == 3  # two 429s + one success
    # Exponential backoff: 0.5 * 2**0, then 0.5 * 2**1.
    assert sleeps == [0.5, 1.0]


def test_urllib_raises_after_exhausting_429_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent 429 beyond the retry budget surfaces the HTTPError (urllib path)."""

    def _handler(_url: str) -> dict[str, Any]:
        raise _http_error(429)

    _install_urlopen(monkeypatch, _handler)

    provider = PolygonProvider(api_key=_API_KEY, max_retries=2, backoff_base=0.0)
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        provider.fetch(["AAPL"], date(2021, 1, 4), date(2021, 1, 8))
    assert excinfo.value.code == 429


def test_urllib_does_not_retry_non_429(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-429 HTTP error is raised immediately, with no retries."""
    state = {"n": 0}

    def _handler(_url: str) -> dict[str, Any]:
        state["n"] += 1
        raise _http_error(500)

    _install_urlopen(monkeypatch, _handler)

    provider = PolygonProvider(api_key=_API_KEY, max_retries=3, backoff_base=0.0)
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        provider.fetch(["AAPL"], date(2021, 1, 4), date(2021, 1, 8))
    assert excinfo.value.code == 500
    assert state["n"] == 1  # no retries on a 500


# --------------------------------------------------------------------------- #
# httpx path (lazy import resolves to an injected fake)                        #
# --------------------------------------------------------------------------- #


def test_fetch_uses_httpx_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``httpx`` imports, the provider uses it and parses the response."""
    fake = _install_fake_httpx(
        monkeypatch, script=[(200, _payload([55.0, 56.0, 57.0]))]
    )

    provider = PolygonProvider(api_key=_API_KEY)
    frame = provider.fetch(["MSFT"], date(2021, 1, 4), date(2021, 1, 8))

    assert frame["MSFT"].tolist() == [55.0, 56.0, 57.0]
    assert len(fake.urls) == 1
    assert "api.polygon.io/v2/aggs/ticker/MSFT/range/1/day/" in fake.urls[0]
    assert f"apiKey={_API_KEY}" in fake.urls[0]


def test_httpx_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 then a 200 over the httpx path retries once and succeeds."""
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    fake = _install_fake_httpx(
        monkeypatch,
        script=[(429, {}), (200, _payload([1.0, 2.0]))],
    )

    provider = PolygonProvider(api_key=_API_KEY, max_retries=2, backoff_base=0.25)
    frame = provider.fetch(["NVDA"], date(2021, 1, 4), date(2021, 1, 8))

    assert frame["NVDA"].tolist() == [1.0, 2.0]
    assert len(fake.urls) == 2  # one 429 + one success
    assert sleeps == [0.25]  # 0.25 * 2**0


def test_httpx_raises_after_exhausting_429_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent 429 over httpx exhausts retries and raises a RuntimeError."""
    _install_fake_httpx(monkeypatch, script=[(429, {}), (429, {}), (429, {})])

    provider = PolygonProvider(api_key=_API_KEY, max_retries=2, backoff_base=0.0)
    with pytest.raises(RuntimeError, match="429"):
        provider.fetch(["AAPL"], date(2021, 1, 4), date(2021, 1, 8))


# --------------------------------------------------------------------------- #
# API-key resolution                                                          #
# --------------------------------------------------------------------------- #


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no explicit key, the env var is used."""
    monkeypatch.setenv("POLYGON_API_KEY", "env-key")
    calls = _install_urlopen(monkeypatch, lambda _url: _payload([1.0, 2.0]))

    provider = PolygonProvider()  # no explicit key
    provider.fetch(["AAPL"], date(2021, 1, 4), date(2021, 1, 8))

    assert "apiKey=env-key" in calls[0]


def test_explicit_key_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "env-key")
    calls = _install_urlopen(monkeypatch, lambda _url: _payload([1.0, 2.0]))

    provider = PolygonProvider(api_key="explicit-key")
    provider.fetch(["AAPL"], date(2021, 1, 4), date(2021, 1, 8))

    assert "apiKey=explicit-key" in calls[0]


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """No explicit key, no env var, and no .env -> ValidationError.

    The polygon module's ``__file__`` is monkeypatched into an isolated tmp dir
    so the repo-root ``.env`` is not discovered during this test.
    """
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    from hrp.data_providers import polygon as polygon_mod

    fake_file = tmp_path / "src" / "hrp" / "data_providers" / "polygon.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(polygon_mod, "__file__", str(fake_file))

    with pytest.raises(ValidationError, match="no API key"):
        PolygonProvider()


# --------------------------------------------------------------------------- #
# Input validation                                                            #
# --------------------------------------------------------------------------- #


def test_fetch_rejects_empty_tickers(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = PolygonProvider(api_key=_API_KEY)
    with pytest.raises(ValidationError, match="non-empty"):
        provider.fetch([], date(2021, 1, 4), date(2021, 1, 8))


def test_fetch_rejects_end_not_after_start(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = PolygonProvider(api_key=_API_KEY)
    with pytest.raises(ValidationError, match="must be after start"):
        provider.fetch(["AAPL"], date(2021, 1, 8), date(2021, 1, 4))


# --------------------------------------------------------------------------- #
# Wiring into hrp.data.get_prices                                             #
# --------------------------------------------------------------------------- #


def test_get_prices_source_pref_polygon_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """``source_pref='polygon'`` routes through the provider and reports provenance."""
    monkeypatch.setenv("POLYGON_API_KEY", _API_KEY)
    _install_urlopen(monkeypatch, lambda _url: _payload([100.0, 101.0, 102.0]))

    frame, source = hrp_data.get_prices(
        ["AAPL"], date(2021, 1, 4), date(2021, 1, 8), source_pref="polygon"
    )

    assert source == "polygon"
    assert list(frame.columns) == ["AAPL"]
    assert frame["AAPL"].tolist() == [100.0, 101.0, 102.0]


def test_get_prices_polygon_failure_falls_through_to_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If Polygon raises, get_prices falls through to yfinance->stooq->synthetic."""

    def _boom(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise RuntimeError("polygon down")

    # Force every real fetcher to fail; the only survivor is the synthetic panel.
    monkeypatch.setattr(hrp_data, "_fetch_polygon", _boom)
    monkeypatch.setattr(hrp_data, "_fetch_yfinance", _boom)
    monkeypatch.setattr(hrp_data, "_fetch_stooq", _boom)

    frame, source = hrp_data.get_prices(
        ["AAA", "BBB"], date(2021, 1, 1), date(2021, 2, 1), source_pref="polygon"
    )

    assert source == "synthetic"
    assert not frame.empty
    assert list(frame.columns) == ["AAA", "BBB"]
