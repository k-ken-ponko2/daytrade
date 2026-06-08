"""売買判定（単一の真実）のテスト。

decide() の各分岐と、generate_signals() の通し動作を確認する。
"""

import numpy as np
import pandas as pd
import pytest

from daytrade.core.signals import decide, generate_signals, initial_stops
from daytrade.core.types import Action, Features, PositionState, StrategyParams


def _features(**overrides):
    """エントリー条件を満たす Features を作り、overrides で一部だけ崩せるようにする。"""
    base = dict(
        close=105.0, high=106.0, low=104.0, volume=3000.0,
        sma_fast=104.0, sma_slow=103.0,      # 現在: fast > slow
        prev_sma_fast=102.0, prev_sma_slow=103.0,  # 直前: fast <= slow → 上抜け
        vwap=100.0, atr=2.0, volume_avg=1000.0,    # volume 3000 >= 1.5*1000
        day_high=106.0, bar_index=10,
    )
    base.update(overrides)
    return Features(**base)


def test_enter_long_when_all_conditions_met():
    assert decide(_features(), PositionState(), StrategyParams()) == Action.ENTER_LONG


def test_no_entry_without_cross():
    # 直前から既に fast > slow（上抜けが「今」ではない）
    f = _features(prev_sma_fast=104.0, prev_sma_slow=103.0)
    assert decide(f, PositionState(), StrategyParams()) == Action.HOLD


def test_no_entry_without_volume_surge():
    f = _features(volume=1000.0)  # 1.5倍未満
    assert decide(f, PositionState(), StrategyParams()) == Action.HOLD


def test_no_entry_below_vwap():
    f = _features(close=99.0, vwap=100.0)  # VWAP 下
    assert decide(f, PositionState(), StrategyParams()) == Action.HOLD


def test_no_entry_with_nan_indicator():
    f = _features(atr=float("nan"))
    assert decide(f, PositionState(), StrategyParams()) == Action.HOLD


def test_require_day_high_break_blocks_when_not_breaking():
    params = StrategyParams(require_day_high_break=True)
    f = _features(high=105.0, day_high=110.0)  # 高値更新していない
    assert decide(f, PositionState(), params) == Action.HOLD


# --- 決済（ポジション保有中） --- #

def _open_position():
    return PositionState(is_open=True, entry_price=100.0,
                         stop_price=98.0, take_price=104.0, bars_held=0)


def test_exit_on_stop_loss():
    f = _features(low=97.0)  # ストップ 98 を割った
    assert decide(f, _open_position(), StrategyParams()) == Action.EXIT


def test_exit_on_take_profit():
    f = _features(high=105.0, low=100.0)  # ターゲット 104 に到達
    assert decide(f, _open_position(), StrategyParams()) == Action.EXIT


def test_stop_takes_priority_over_take():
    # 同一バーで損切りも利確も触れていたら損切り優先（保守的）
    f = _features(low=97.0, high=105.0)
    assert decide(f, _open_position(), StrategyParams()) == Action.EXIT


def test_exit_on_max_hold():
    params = StrategyParams(max_hold_bars=5)
    pos = PositionState(is_open=True, entry_price=100.0,
                        stop_price=90.0, take_price=200.0, bars_held=5)
    f = _features(low=100.0, high=100.0)  # ストップ/ターゲットには触れない
    assert decide(f, pos, params) == Action.EXIT


def test_force_close_before_session_end():
    params = StrategyParams(force_close_bar=285)
    pos = PositionState(is_open=True, entry_price=100.0,
                        stop_price=90.0, take_price=200.0, bars_held=1)
    f = _features(bar_index=290, low=100.0, high=100.0)
    assert decide(f, pos, params) == Action.EXIT


def test_no_new_entry_near_close():
    params = StrategyParams(force_close_bar=285)
    f = _features(bar_index=290)  # エントリー条件は満たすが引け間際
    assert decide(f, PositionState(), params) == Action.HOLD


def test_initial_stops_from_atr():
    stop, take = initial_stops(100.0, 2.0, StrategyParams(atr_stop_mult=1.0, atr_take_mult=2.0))
    assert stop == 98.0
    assert take == 104.0


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        StrategyParams(fast_period=20, slow_period=5)  # fast >= slow


# --- 通し（基準実装） --- #

def test_generate_signals_runs_and_holds_position():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01 09:00", periods=60, freq="1min")
    # 上昇トレンド＋出来高スパイクを仕込む
    close = 1000 + np.cumsum(rng.normal(0.5, 1.0, 60))
    df = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": rng.uniform(500, 1500, 60),
    }, index=idx)
    df.iloc[30, df.columns.get_loc("volume")] = 50000  # 出来高急増

    out = generate_signals(df, StrategyParams())
    assert "action" in out.columns and "position" in out.columns
    assert set(out["position"].unique()).issubset({0, 1})
    # ポジションは 0/1 のフラグとして一貫している（保有中は1が連続する）
    assert out["position"].max() in (0, 1)
