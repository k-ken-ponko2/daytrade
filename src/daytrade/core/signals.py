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


def open_position(entry_price: float, atr_value: float, params: StrategyParams) -> PositionState:
    """エントリー時のポジション状態を組み立てる（損切り・利確・段階利確目標を設定）。

    段階利確が有効なら scale_out_price を、無効なら 0.0（=無効）を入れる。
    high_water は取得価格から開始する（トレーリングストップの基準）。
    """
    stop, take = initial_stops(entry_price, atr_value, params)
    scale_out = (entry_price + params.scale_out_atr_mult * atr_value
                 if params.scale_out_atr_mult is not None else 0.0)
    return PositionState(
        is_open=True, entry_price=entry_price, stop_price=stop, take_price=take,
        scale_out_price=scale_out, high_water=entry_price, scaled_out=False, bars_held=0,
    )


def update_trailing_stop(position: PositionState, f: Features, params: StrategyParams) -> PositionState:
    """高値(high_water)を更新し、トレーリングストップで損切り価格を引き上げる。

    decide() は副作用なしに保つため、状態更新はこの関数（ループ側が各バーで呼ぶ）に集約する。
    トレーリングが無効なら high_water の更新のみ。ストップは下げない（ラチェット）。
    """
    if not position.is_open:
        return position
    position.high_water = max(position.high_water, f.high)
    if params.trailing_stop_atr_mult is not None and not _is_nan(f.atr):
        candidate = position.high_water - params.trailing_stop_atr_mult * f.atr
        position.stop_price = max(position.stop_price, candidate)
    return position


def decide(f: Features, position: PositionState, params: StrategyParams) -> Action:
    """1バー分の売買判定。状態は変更せず Action を返すだけ（副作用なし）。

    ポジションを持っている間の決済は次の優先順で評価する:
      1. 損切り（最優先・機械的に実行。トレーリングで引き上がったストップも含む）
      2. 引け前の強制クローズ（持ち越さない）
      3. 保有時間の上限
      4. 全利確
      5. 段階利確（一部だけ利確して原資回収、残りを伸ばす）
    ポジションがなければエントリー条件を評価する。
    トレーリングの状態更新は update_trailing_stop() がループ側で先に行う前提。
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

        # 4. 全利確（高値がターゲットに触れたら）
        if f.high >= position.take_price:
            return Action.EXIT

        # 5. 段階利確（まだ一部利確しておらず、第1目標に到達）
        if (params.scale_out_atr_mult is not None and not position.scaled_out
                and f.high >= position.scale_out_price):
            return Action.SCALE_OUT

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
        if position.is_open:
            update_trailing_stop(position, f, params)
        action = decide(f, position, params)

        if action == Action.ENTER_LONG and not position.is_open:
            position = open_position(f.close, f.atr, params)
        elif action == Action.EXIT and position.is_open:
            position = PositionState()
        elif action == Action.SCALE_OUT and position.is_open:
            # 基準実装は数量を持たないため、一部利確は「フラグだけ」立てて保有継続
            position.scaled_out = True
            position.bars_held += 1
        elif position.is_open:
            position.bars_held += 1

        actions.append(action.value)
        positions.append(1 if position.is_open else 0)

    enriched["action"] = actions
    enriched["position"] = positions
    return enriched
