"""Alpaca 分足クライアントのテスト（HTTP モック・ネットワーク/APIキー不要）。"""

import pandas as pd
import pytest

from daytrade.data.alpaca import AlpacaClient, AlpacaError, to_regular_session


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, pages, status=200):
        self.pages = pages
        self.status = status
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if self.status >= 400:
            return FakeResponse(self.status, {"message": "subscription does not permit"})
        return FakeResponse(200, self.pages[len(self.calls) - 1])


def _bar(ts, close, volume=100):
    return {"t": ts, "o": close, "h": close + 0.5, "l": close - 0.5, "c": close,
            "v": volume, "n": 10, "vw": close}


def _client(session):
    return AlpacaClient("id", "secret", session=session)


def test_from_env_requires_keys(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    with pytest.raises(AlpacaError):
        AlpacaClient.from_env()


def test_get_bars_paginates_and_sends_auth():
    pages = [
        {"bars": [_bar("2026-09-01T13:30:00Z", 100)], "symbol": "AAPL", "next_page_token": "p1"},
        {"bars": [_bar("2026-09-01T13:31:00Z", 101)], "symbol": "AAPL", "next_page_token": None},
    ]
    session = FakeSession(pages)
    df = _client(session).get_bars("aapl", start="2026-09-01", end="2026-09-01")

    assert len(df) == 2
    assert str(df.index.tz) == "UTC"
    url, params, headers = session.calls[0]
    assert url.endswith("/v2/stocks/AAPL/bars")              # シンボルは大文字化
    assert params["timeframe"] == "1Min" and params["adjustment"] == "split"
    assert headers["APCA-API-KEY-ID"] == "id"
    assert session.calls[1][1]["page_token"] == "p1"          # 2ページ目


def test_get_bars_empty():
    df = _client(FakeSession([{"bars": None, "next_page_token": None}])).get_bars(
        "AAPL", start="2026-09-05", end="2026-09-05")
    assert df.empty


def test_sip_permission_error_suggests_iex():
    with pytest.raises(AlpacaError, match="iex"):
        _client(FakeSession([], status=403)).get_bars("AAPL", start="2026-09-01", end="2026-09-01")


def test_regular_session_filters_and_converts_to_ny():
    # 2026-09-01 は夏時間（EDT=UTC-4）。13:29Z=09:29 / 13:30Z=09:30 / 19:59Z=15:59 / 20:00Z=16:00
    pages = [{"bars": [_bar("2026-09-01T13:29:00Z", 99), _bar("2026-09-01T13:30:00Z", 100),
                       _bar("2026-09-01T19:59:00Z", 101), _bar("2026-09-01T20:00:00Z", 102)],
              "next_page_token": None}]
    raw = _client(FakeSession(pages)).get_bars("AAPL", start="2026-09-01", end="2026-09-01")
    out = to_regular_session(raw)

    assert str(out.index.tz) == "America/New_York"
    assert [t.strftime("%H:%M") for t in out.index] == ["09:30", "15:59"]
    assert list(out.columns[:5]) == ["open", "high", "low", "close", "volume"]


def test_regular_session_runs_through_backtest():
    """Alpaca 形式のデータが米国株設定のバックテストにそのまま流れること。"""
    from daytrade.backtest.engine import BacktestConfig, run_backtest
    from daytrade.core.markets import US

    idx = pd.date_range("2026-09-01 13:30", periods=390, freq="1min", tz="UTC")
    bars = [_bar(t.strftime("%Y-%m-%dT%H:%M:%SZ"), 100 + (i % 30) * 0.1, 100 + i)
            for i, t in enumerate(idx)]
    raw = _client(FakeSession([{"bars": bars, "next_page_token": None}])).get_bars(
        "AAPL", start="2026-09-01", end="2026-09-01")
    df = to_regular_session(raw)
    result = run_backtest(df, config=BacktestConfig.for_market(US, initial_cash=2000.0))
    assert len(result.equity_curve) == len(df)
    assert result.enriched["bar_index"].iloc[0] == 0          # NY の寄り付きが0本目
