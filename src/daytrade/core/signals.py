"""売買判定 — 単一の真実（Single Source of Truth）。

このモジュールの decide() が売買判定の中核。Backtrader / NautilusTrader の
アダプタは、各エンジンのバーから Features を組み立てて decide() を呼ぶだけにする。
ロジック修正はここ1箇所で済み、両エンジンの差分が「エンジンの差」だけに絞れる。

注意（README 8章）:
- ここでは未来のバーを参照しない（ルックアヘッドバイアスを作らない）。
- 手数料・スリッページはアダプタ／エンジン側で必ず加味する。
"""

from __future__ import annotations

import math

import pandas as pd

from daytrade.core.indicators import compute_indicators
from daytrade.core.types import Action, Features, PositionState, StrategyParams


def _is_nan(*values: float) -> bool:
    return any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values)


def _crossed_up(f: Features) -> bool:
    """短期SMAが長期SMAを上抜けた（ゴールデンクロス）か。"""
    return f.prev_sma_fast <= f.prev_sma_slow and f.sma_fast > f.sma_slow


def _entry_conditions_met(f: Features, params: StrategyParams) -> bool:
    """エントリー条件（A かつ B かつ C …）。

    1. 短期SMAが長期SMAを上抜け（トレンド転換）
    2. 出来高が平常時の volume_surge_mult 倍以上（注目度）
    3. 価格が VWAP を vwap_dev_min 以上上回る（地合いが買い優勢）
    4. （任意）当日高値を更新中
    """
    if _is_nan(f.sma_fast, f.sma_slow, f.prev_sma_fast, f.prev_sma_slow,
               f.vwap, f.atr, f.volume_avg):
        return False

    if not _crossed_up(f):
        return False

    if f.volume_avg <= 0 or f.volume < params.volume_surge_mult * f.volume_avg:
        return False

    if f.vwap <= 0:
        return False
    vwap_dev = (f.close - f.vwap) / f.vwap
    if vwap_dev < params.vwap_dev_min:
        return False

    if params.require_day_high_break and f.high < f.day_high:
        return False

    return True


def initial_stops(entry_price: float, atr_value: float, params: StrategyParams) -> tuple[float, float]:
    """エントリー時の損切り・利確価格を ATR から計算する。"""
    stop = entry_price - params.atr_stop_mult * atr_value
    take = entry_price + params.atr_take_mult * atr_value
    return stop, take


def decide(f: Features, position: PositionState, params: StrategyParams) -> Action:
    """1バー分の売買判定。状態は変更せず Action を返すだけ（副作用なし）。

    ポジションを持っている間の決済は次の優先順で評価する:
      1. 損切り（最優先・機械的に実行）
      2. 引け前の強制クローズ（持ち越さない）
      3. 保有時間の上限
      4. 利確
    ポジションがなければエントリー条件を評価する。
    """
    if position.is_open:
        # 1. 損切り（安値がストップに触れたら）
        if f.low <= position.stop_price:
            return Action.EXIT

        # 2. 引け前の強制クローズ
        if params.force_close_bar is not None and f.bar_index >= params.force_close_bar:
            return Action.EXIT

        # 3. 保有時間の上限
        if position.bars_held >= params.max_hold_bars:
            return Action.EXIT

        # 4. 利確（高値がターゲットに触れたら）
        if f.high >= position.take_price:
            return Action.EXIT

        return Action.HOLD

    # 引け間際は新規エントリーしない
    if params.force_close_bar is not None and f.bar_index >= params.force_close_bar:
        return Action.HOLD

    if _entry_conditions_met(f, params):
        return Action.ENTER_LONG

    return Action.HOLD


def _row_to_features(row: pd.Series) -> Features:
    return Features(
        close=row["close"],
        high=row["high"],
        low=row["low"],
        volume=row["volume"],
        sma_fast=row["sma_fast"],
        sma_slow=row["sma_slow"],
        prev_sma_fast=row["prev_sma_fast"],
        prev_sma_slow=row["prev_sma_slow"],
        vwap=row["vwap"],
        atr=row["atr"],
        volume_avg=row["volume_avg"],
        day_high=row["day_high"],
        bar_index=int(row["bar_index"]),
    )


def generate_signals(df: pd.DataFrame, params: StrategyParams | None = None) -> pd.DataFrame:
    """OHLCV の DataFrame を受け取り、バー単位で decide() を回した結果を返す。

    これは「共通ロジックを素のループで回した基準実装」。Backtrader / NautilusTrader
    の結果をこの基準と突き合わせれば、ズレが「エンジンの差」なのか「ロジックの差」
    なのかを切り分けられる。約定は終値で行う簡易モデル（手数料・スリッページ抜き）。

    戻り値: 入力にカラム action / position を足した DataFrame。
    """
    params = params or StrategyParams()
    enriched = compute_indicators(df, params)

    position = PositionState()
    actions: list[str] = []
    positions: list[int] = []

    for _, row in enriched.iterrows():
        f = _row_to_features(row)
        action = decide(f, position, params)

        if action == Action.ENTER_LONG and not position.is_open:
            stop, take = initial_stops(f.close, f.atr, params)
            position = PositionState(
                is_open=True,
                entry_price=f.close,
                stop_price=stop,
                take_price=take,
                bars_held=0,
            )
        elif action == Action.EXIT and position.is_open:
            position = PositionState()
        elif position.is_open:
            position.bars_held += 1

        actions.append(action.value)
        positions.append(1 if position.is_open else 0)

    enriched["action"] = actions
    enriched["position"] = positions
    return enriched
