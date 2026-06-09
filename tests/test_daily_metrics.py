"""日利メトリクス（目標2.5%に対する実測）のテスト。"""

import numpy as np
import pandas as pd
import pytest

from daytrade.backtest.engine import BacktestConfig, daily_returns, run_backtest
from daytrade.backtest.sample_data import make_intraday_ohlcv
from daytrade.core.types import StrategyParams


def test_daily_returns_collapses_to_per_session():
    # 2営業日・各3バー。日次の最終評価額の前日比をとる
    idx = pd.to_datetime([
        "2024-04-01 09:00", "2024-04-01 09:01", "2024-04-01 09:02",
        "2024-04-02 09:00", "2024-04-02 09:01", "2024-04-02 09:02",
    ])
    equity = pd.Series([100, 101, 110, 110, 120, 121], index=idx, dtype=float)
    dr = daily_returns(equity)
    # 1日目終値110 → 2日目終値121 = +10%
    assert len(dr) == 1
    assert dr.iloc[0] == pytest.approx(0.1)


def test_daily_returns_empty_for_non_datetime_index():
    assert daily_returns(pd.Series([1.0, 2.0])).empty


def test_metrics_include_daily_target_fields():
    df = make_intraday_ohlcv(n_days=10)
    res = run_backtest(df, StrategyParams(fast_period=3, slow_period=10, force_close_bar=55),
                       BacktestConfig(daily_return_target=0.025))
    m = res.metrics
    for key in ["num_days", "avg_daily_return", "geom_daily_return",
                "daily_target", "daily_target_hit_rate", "daily_target_gap"]:
        assert key in m
    assert m["daily_target"] == 0.025
    assert 0.0 <= m["daily_target_hit_rate"] <= 1.0
    # gap = 平均日利 - 目標 の整合
    assert abs(m["daily_target_gap"] - (m["avg_daily_return"] - 0.025)) < 1e-12


def test_geom_daily_return_matches_compounding():
    # 既知のエクイティで複利日利を検算：3営業日で 300k→...、final/initial の (1/n_days) 乗
    df = make_intraday_ohlcv(n_days=5, regime="trend_up")
    cfg = BacktestConfig()
    res = run_backtest(df, StrategyParams(fast_period=3, slow_period=10, force_close_bar=55), cfg)
    m = res.metrics
    expected = (m["final_equity"] / cfg.initial_cash) ** (1.0 / m["num_days"]) - 1.0
    assert abs(m["geom_daily_return"] - expected) < 1e-9


def test_target_hit_rate_with_strong_daily_gains():
    # 各日 +3%（目標2.5%超）で確実に上がるエクイティを直接検証
    days = pd.date_range("2024-04-01", periods=6, freq="D")
    idx = pd.DatetimeIndex(days)
    equity = pd.Series(300000 * (1.03 ** np.arange(6)), index=idx)
    dr = daily_returns(equity)
    assert (dr >= 0.025).all()
    assert len(dr) == 5
