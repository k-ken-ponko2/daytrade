"""決済ロジック拡張のテスト（トレーリング／段階利確／撤退ライン）。"""

import numpy as np
import pandas as pd
import pytest

from daytrade.backtest.engine import BacktestConfig, run_backtest
from daytrade.backtest.sample_data import make_intraday_ohlcv
from daytrade.core.risk import retreat_triggered
from daytrade.core.signals import decide, open_position, update_trailing_stop
from daytrade.core.types import Action, Features, PositionState, StrategyParams


def _f(**ov):
    base = dict(
        close=105.0, high=106.0, low=104.0, volume=3000.0,
        sma_fast=104.0, sma_slow=103.0, prev_sma_fast=102.0, prev_sma_slow=103.0,
        vwap=100.0, atr=2.0, volume_avg=1000.0, day_high=106.0, bar_index=10,
    )
    base.update(ov)
    return Features(**base)


# --- open_position --- #

def test_open_position_sets_stops_and_scale_target():
    p = StrategyParams(atr_stop_mult=1.0, atr_take_mult=3.0, scale_out_atr_mult=1.5)
    pos = open_position(100.0, 2.0, p)
    assert pos.is_open
    assert pos.stop_price == 98.0          # 100 - 1.0*2
    assert pos.take_price == 106.0         # 100 + 3.0*2
    assert pos.scale_out_price == 103.0    # 100 + 1.5*2
    assert pos.high_water == 100.0


def test_open_position_no_scale_target_when_disabled():
    pos = open_position(100.0, 2.0, StrategyParams())
    assert pos.scale_out_price == 0.0      # 無効時は 0


# --- update_trailing_stop --- #

def test_trailing_stop_ratchets_up():
    p = StrategyParams(trailing_stop_atr_mult=1.0)
    pos = open_position(100.0, 2.0, p)      # 初期ストップ 98
    update_trailing_stop(pos, _f(high=110.0, atr=2.0), p)
    # high_water=110 → ストップ候補 108、初期98より上なので引き上がる
    assert pos.high_water == 110.0
    assert pos.stop_price == 108.0


def test_trailing_stop_never_lowers():
    p = StrategyParams(trailing_stop_atr_mult=1.0)
    pos = open_position(100.0, 2.0, p)
    update_trailing_stop(pos, _f(high=110.0, atr=2.0), p)   # ストップ 108 に上昇
    update_trailing_stop(pos, _f(high=105.0, atr=2.0), p)   # 高値が下がってもストップは下げない
    assert pos.stop_price == 108.0


def test_trailing_disabled_keeps_initial_stop():
    p = StrategyParams()  # trailing 無効
    pos = open_position(100.0, 2.0, p)
    update_trailing_stop(pos, _f(high=120.0, atr=2.0), p)
    assert pos.stop_price == 98.0          # 変わらない
    assert pos.high_water == 120.0         # high_water は更新される


# --- decide: SCALE_OUT --- #

def _open(**ov):
    base = dict(is_open=True, entry_price=100.0, stop_price=98.0, take_price=120.0,
                scale_out_price=104.0, high_water=100.0, scaled_out=False, bars_held=0)
    base.update(ov)
    return PositionState(**base)


def test_decide_scale_out_when_first_target_hit():
    p = StrategyParams(scale_out_atr_mult=2.0)
    f = _f(high=105.0, low=100.0)  # 104(scale)到達、120(take)未満、98(stop)に触れず
    assert decide(f, _open(), p) == Action.SCALE_OUT


def test_decide_scale_out_only_once():
    p = StrategyParams(scale_out_atr_mult=2.0)
    f = _f(high=105.0, low=100.0)
    assert decide(f, _open(scaled_out=True), p) == Action.HOLD


def test_decide_full_take_precedes_scale_out():
    p = StrategyParams(scale_out_atr_mult=2.0)
    f = _f(high=125.0, low=100.0)  # take(120)到達 → 全利確が優先
    assert decide(f, _open(), p) == Action.EXIT


def test_decide_stop_precedes_scale_out():
    p = StrategyParams(scale_out_atr_mult=2.0)
    f = _f(high=105.0, low=97.0)  # stop(98)割れ → 損切り優先
    assert decide(f, _open(), p) == Action.EXIT


def test_decide_no_scale_out_when_disabled():
    f = _f(high=105.0, low=100.0)
    assert decide(f, _open(), StrategyParams()) == Action.HOLD


# --- retreat_triggered --- #

def test_retreat_triggered_threshold():
    assert retreat_triggered(150_000, 300_000, 0.5) is True
    assert retreat_triggered(150_001, 300_000, 0.5) is False


def test_retreat_disabled_when_none():
    assert retreat_triggered(1.0, 300_000, None) is False


# --- engine: scale-out produces a partial trade --- #

def _dip_then_rise_df():
    """フラットなウォームアップ → 短い押し目 → ゴールデンクロスで建玉 → 素直な上昇。

    上昇に押し目を作らないので、ストップに触れず段階利確→全利確まで到達する。
    """
    flat = np.full(26, 1000.0)
    dip = np.array([998.0, 996.0, 994.0, 994.0])
    rise = np.linspace(994.0, 1060.0, 40)
    close = np.concatenate([flat, dip, rise])
    idx = pd.date_range("2024-04-01 09:00", periods=len(close), freq="1min")
    df = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0,
                       "close": close, "volume": np.full(len(close), 700.0)}, index=idx)
    df.iloc[30:36, df.columns.get_loc("volume")] = 4000  # クロス時に出来高急増
    return df


def test_engine_scale_out_records_partial_then_full():
    df = _dip_then_rise_df()
    p = StrategyParams(fast_period=3, slow_period=10, atr_stop_mult=3.0, atr_take_mult=8.0,
                       scale_out_atr_mult=1.0, scale_out_fraction=0.5, max_hold_bars=500)
    res = run_backtest(df, p, BacktestConfig())
    reasons = [t.exit_reason for t in res.trades]

    # 一部利確 → 残りを全利確、の2件が記録される
    assert reasons == ["scale_out", "take_profit"]
    scale, final = res.trades
    # 段階利確は保有の一部（半分・単元丸め）、残りが全利確
    assert scale.shares % 100 == 0 and scale.shares > 0
    assert final.shares == scale.shares * 2          # 300株を100利確→残り200
    # 同一エントリーからの分割なので建玉時刻は一致
    assert scale.entry_time == final.entry_time


def _ride_down_df():
    """寄り付き付近でエントリーさせ、その後ストップに触れず評価額が落ちていく相場。

    オシレーションで指標ウォームアップ後にゴールデンクロス＋出来高急増を作りエントリー、
    続く緩やかな下げでポジションが含み損を抱えていく（撤退ライン検証用）。
    """
    osc = 1000 + 6 * np.sin(np.arange(36) / 2.0)
    decline = np.linspace(osc[-1], 480, 50)
    close = np.concatenate([osc, decline])
    idx = pd.date_range("2024-04-01 09:00", periods=len(close), freq="1min")
    df = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0,
                       "close": close, "volume": np.full(len(close), 700.0)}, index=idx)
    df.iloc[20:36, df.columns.get_loc("volume")] = 5000
    return df


def test_engine_retreat_cuts_loss_and_halts():
    df = _ride_down_df()
    # 損切りを遠くに置き（ライドさせる）、ほぼ全力ポジションにして撤退ラインを試す
    p = StrategyParams(fast_period=3, slow_period=10,
                       atr_stop_mult=200.0, atr_take_mult=200.0, max_hold_bars=500)

    no_retreat = run_backtest(df, p, BacktestConfig(risk_fraction=0.9))
    with_retreat = run_backtest(df, p, BacktestConfig(risk_fraction=0.9, retreat_drawdown=0.3))

    # 撤退なし：最後までポジションを抱えたまま（含み損が膨らむ）
    assert no_retreat.enriched["position"].iloc[-1] == 1
    # 撤退あり：強制クローズで損切りし、以後ノーポジ。理由は "retreat"
    assert with_retreat.enriched["position"].iloc[-1] == 0
    assert any(t.exit_reason == "retreat" for t in with_retreat.trades)
    # 撤退が損失を早めに止めるので、評価額は撤退なしより高い
    assert with_retreat.metrics["final_equity"] > no_retreat.metrics["final_equity"]
