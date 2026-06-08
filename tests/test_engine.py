"""バックテストエンジンとリスク管理（株数計算）のテスト。"""

import pandas as pd
import pytest

from daytrade.backtest.engine import BacktestConfig, run_backtest
from daytrade.backtest.sample_data import make_intraday_ohlcv
from daytrade.core.risk import position_size
from daytrade.core.types import StrategyParams


# --- position_size --- #

def test_position_size_rounds_to_lot():
    # 余力30万・損切り幅5円・リスク5% → 予算15000/5=3000株、ただし余力で買えるのは…
    size = position_size(300_000, 1000.0, 995.0, risk_fraction=0.05, lot_size=100)
    assert size % 100 == 0
    assert size > 0


def test_position_size_capped_by_cash():
    # 高い株価では余力が効く（リスク予算より余力が先に効く）
    size = position_size(300_000, 5000.0, 4900.0, risk_fraction=0.5, lot_size=100)
    assert size * 5000.0 <= 300_000


def test_position_size_zero_on_bad_stop():
    assert position_size(300_000, 1000.0, 1000.0) == 0   # 損切り幅0
    assert position_size(300_000, 1000.0, 1100.0) == 0   # ストップが上（負の幅）


def test_position_size_zero_on_no_cash():
    assert position_size(0, 1000.0, 990.0) == 0


# --- run_backtest --- #

def test_backtest_runs_on_sample_data():
    df = make_intraday_ohlcv(n_days=5, seed=42)
    params = StrategyParams(force_close_bar=55, max_hold_bars=20)
    result = run_backtest(df, params, BacktestConfig())

    assert len(result.equity_curve) == len(df)
    assert "action" in result.enriched.columns
    assert set(result.enriched["position"].unique()).issubset({0, 1})
    # メトリクスが揃っている
    for key in ["initial_cash", "final_equity", "total_return", "num_trades",
                "win_rate", "max_drawdown"]:
        assert key in result.metrics


def test_backtest_no_position_carried_overnight():
    # force_close_bar を入れれば、各トレードは同一日内で完結する
    df = make_intraday_ohlcv(n_days=5, bars_per_day=60, seed=7)
    params = StrategyParams(force_close_bar=55, max_hold_bars=60)
    result = run_backtest(df, params, BacktestConfig())
    for t in result.trades:
        assert t.entry_time.normalize() == t.exit_time.normalize()


def test_backtest_trades_have_consistent_fields():
    df = make_intraday_ohlcv(n_days=8, seed=3)
    result = run_backtest(df, StrategyParams(force_close_bar=55), BacktestConfig())
    for t in result.trades:
        assert t.shares > 0 and t.shares % 100 == 0
        assert t.exit_time >= t.entry_time
        # return_pct と pnl の符号は一致する
        assert (t.pnl > 0) == (t.return_pct > 0) or t.pnl == 0


def test_zero_commission_zero_slippage_is_more_profitable():
    df = make_intraday_ohlcv(n_days=10, seed=11)
    params = StrategyParams(force_close_bar=55)
    with_cost = run_backtest(df, params, BacktestConfig(
        commission_rate=0.001, slippage_rate=0.001))
    no_cost = run_backtest(df, params, BacktestConfig(
        commission_rate=0.0, slippage_rate=0.0))
    if with_cost.metrics["num_trades"] > 0:
        # 手数料・スリッページは必ず利益を削る（README 4章）
        assert no_cost.metrics["final_equity"] >= with_cost.metrics["final_equity"]
