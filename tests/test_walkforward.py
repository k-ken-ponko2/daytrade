"""ウォークフォワード検証とサンプルデータの相場局面のテスト。"""

import pandas as pd
import pytest

from daytrade.backtest.engine import BacktestConfig, run_backtest
from daytrade.backtest.sample_data import REGIME_DRIFT, make_intraday_ohlcv
from daytrade.backtest.walkforward import (
    optimize,
    param_grid,
    rolling_windows,
    run_walk_forward,
    total_return_objective,
)
from daytrade.core.types import StrategyParams


# --- サンプルデータの局面 --- #

def test_regimes_produce_expected_trend_direction():
    up = make_intraday_ohlcv(n_days=10, regime="trend_up", seed=1)
    down = make_intraday_ohlcv(n_days=10, regime="trend_down", seed=1)
    # 上昇局面は終値が始値を上回り、下落局面は下回る傾向
    assert up["close"].iloc[-1] > up["close"].iloc[0]
    assert down["close"].iloc[-1] < down["close"].iloc[0]


def test_unknown_regime_raises():
    with pytest.raises(ValueError):
        make_intraday_ohlcv(regime="moon")


def test_all_regimes_generate_data():
    for regime in REGIME_DRIFT:
        df = make_intraday_ohlcv(n_days=3, regime=regime)
        assert len(df) == 3 * 60
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]


# --- param_grid --- #

def test_param_grid_skips_invalid_combos():
    grid = param_grid(fast_period=[3, 20], slow_period=[10])
    # fast=20, slow=10 は無効 → 除外され fast=3 のみ残る
    assert len(grid) == 1
    assert grid[0].fast_period == 3


def test_param_grid_applies_base():
    grid = param_grid(base=dict(force_close_bar=55), fast_period=[3], slow_period=[10])
    assert all(p.force_close_bar == 55 for p in grid)


# --- objective / optimize --- #

def test_objective_disqualifies_too_few_trades():
    df = make_intraday_ohlcv(n_days=5)
    # 取引が出にくいパラメータ＋高い min_trades → 失格(-inf)
    res = run_backtest(df, StrategyParams(volume_surge_mult=100.0), BacktestConfig())
    assert total_return_objective(res, min_trades=5) == float("-inf")


def test_optimize_picks_a_candidate():
    df = make_intraday_ohlcv(n_days=10, regime="trend_up")
    candidates = param_grid(
        base=dict(force_close_bar=55),
        fast_period=[3, 5], slow_period=[10, 20], volume_surge_mult=[1.3],
    )
    best, score = optimize(
        df, candidates, BacktestConfig(),
        objective=lambda r: total_return_objective(r, min_trades=1),
    )
    assert best in candidates


# --- rolling_windows --- #

def test_rolling_windows_split_by_days_no_overlap():
    df = make_intraday_ohlcv(n_days=20)
    windows = rolling_windows(df, train_days=10, test_days=5)
    assert len(windows) == 2  # (0-14), (5-19) ... step=test=5 → 開始0,5
    for w in windows:
        train_days = pd.DatetimeIndex(w.train.index).normalize().nunique()
        test_days = pd.DatetimeIndex(w.test.index).normalize().nunique()
        assert train_days == 10 and test_days == 5
        # train と test は日付が重ならない
        td = set(pd.DatetimeIndex(w.train.index).normalize())
        te = set(pd.DatetimeIndex(w.test.index).normalize())
        assert td.isdisjoint(te)


def test_rolling_windows_empty_when_not_enough_days():
    df = make_intraday_ohlcv(n_days=5)
    assert rolling_windows(df, train_days=10, test_days=5) == []


# --- run_walk_forward --- #

def test_walk_forward_runs_and_summarizes():
    df = make_intraday_ohlcv(n_days=30, regime="mixed")
    candidates = param_grid(
        base=dict(force_close_bar=55, max_hold_bars=20),
        fast_period=[3, 5], slow_period=[10, 20], volume_surge_mult=[1.3, 1.5],
    )
    report = run_walk_forward(
        df, candidates, config=BacktestConfig(),
        train_days=10, test_days=5,
    )
    assert len(report.windows) >= 2
    s = report.summary
    for key in ["num_windows", "is_mean_return", "oos_mean_return",
                "overfit_gap", "oos_hit_rate", "total_oos_trades"]:
        assert key in s
    # overfit_gap = IS平均 - OOS平均 の整合
    assert s["overfit_gap"] == pytest.approx(s["is_mean_return"] - s["oos_mean_return"])


def test_walk_forward_each_window_uses_a_valid_param():
    df = make_intraday_ohlcv(n_days=24, regime="trend_up")
    candidates = param_grid(
        base=dict(force_close_bar=55),
        fast_period=[3, 5], slow_period=[10, 20], volume_surge_mult=[1.3],
    )
    report = run_walk_forward(df, candidates, train_days=8, test_days=4)
    for w in report.windows:
        assert w.best_params.fast_period < w.best_params.slow_period
