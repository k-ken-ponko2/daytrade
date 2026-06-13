"""市場プロファイル（日本株/米国株）と海外データ取得のテスト。"""

import numpy as np
import pandas as pd
import pytest

from daytrade.backtest.engine import BacktestConfig, run_backtest
from daytrade.backtest.sample_data import make_intraday_ohlcv
from daytrade.core.markets import JAPAN, US, get_profile
from daytrade.core.risk import position_size
from daytrade.core.types import StrategyParams
from daytrade.data.overseas import StooqClient, load_ohlcv_csv


# --- MarketProfile --- #

def test_presets_lot_size_and_currency():
    assert JAPAN.lot_size == 100 and JAPAN.currency == "JPY"
    assert US.lot_size == 1 and US.currency == "USD"


def test_min_position_value_shows_lot_advantage():
    # 同じ価格でも、米国株(1株)は日本株(100株)の1/100の最低注文額
    price = 3000.0
    assert JAPAN.min_position_value(price) == 300_000.0   # 単元株の壁
    assert US.min_position_value(price) == 3_000.0


def test_max_names_diversification():
    # 30万円・$150 銘柄なら米国株は複数銘柄持てる。日本株(単元100=1.5万/単位…)と比較
    cap = 300_000.0
    assert US.max_names(cap, 150.0) == 2000        # 1株150 → 2000株ぶん
    # 価格2000円の日本株は1単元=20万 → 1銘柄しか持てない
    assert JAPAN.max_names(cap, 2000.0) == 1


def test_get_profile_and_unknown():
    assert get_profile("us") is US
    assert get_profile("JP") is JAPAN
    with pytest.raises(ValueError):
        get_profile("eu")


# --- position_size with lot_size=1 --- #

def test_position_size_us_allows_small_account():
    # 小資金（$2,000）・$150 銘柄・損切り $5。
    # 100株単位だと最低でも約$15,000必要で建玉できない(0株)が、1株単位なら買える。
    jp = position_size(2000, 150.0, 145.0, risk_fraction=0.05, lot_size=100)
    us = position_size(2000, 150.0, 145.0, risk_fraction=0.05, lot_size=1)
    assert jp == 0          # 単元(100株)の壁で建玉できない
    assert us > 0 and isinstance(us, int)


# --- BacktestConfig.for_market --- #

def test_for_market_sets_lot_and_costs():
    cfg = BacktestConfig.for_market(US, initial_cash=2000.0)
    assert cfg.lot_size == 1
    assert cfg.initial_cash == 2000.0
    assert cfg.commission_rate == US.commission_rate


def test_for_market_overrides():
    cfg = BacktestConfig.for_market(US, initial_cash=2000.0, commission_rate=0.00495)
    assert cfg.commission_rate == 0.00495   # 証券会社の実料率で上書き


def test_engine_runs_us_profile():
    # $150 帯の合成データを 1株単位で回せる（最低注文の壁がない）
    df = make_intraday_ohlcv(n_days=10, base_price=150.0)
    cfg = BacktestConfig.for_market(US, initial_cash=2000.0)
    res = run_backtest(df, StrategyParams(fast_period=3, slow_period=10, force_close_bar=55), cfg)
    assert "final_equity" in res.metrics
    # 建玉できていれば1株単位（端株なし）
    for t in res.trades:
        assert t.shares >= 1 and float(t.shares).is_integer()


# --- overseas data --- #

class FakeResponse:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


class FakeSession:
    def __init__(self, text, status=200):
        self.text = text
        self.status = status
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return FakeResponse(self.status, self.text)


_CSV = "Date,Open,High,Low,Close,Volume\n2024-01-02,100,102,99,101,1000\n2024-01-03,101,105,100,104,1200\n"


def test_stooq_appends_us_suffix_and_parses():
    s = FakeSession(_CSV)
    df = StooqClient(session=s).get_daily("AAPL")
    # シンボルに .us が付く
    assert s.calls[0][1]["s"] == "aapl.us"
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].iloc[-1] == 104
    assert df.index.is_monotonic_increasing


def test_stooq_keeps_explicit_suffix():
    s = FakeSession(_CSV)
    StooqClient(session=s).get_daily("7203.jp")
    assert s.calls[0][1]["s"] == "7203.jp"


def test_stooq_no_data_returns_empty():
    df = StooqClient(session=FakeSession("No data")).get_daily("ZZZZ")
    assert df.empty


def test_load_ohlcv_csv(tmp_path):
    p = tmp_path / "aapl.csv"
    p.write_text(_CSV)
    df = load_ohlcv_csv(str(p))
    assert len(df) == 2 and df["high"].iloc[0] == 102
