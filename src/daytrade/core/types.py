"""共通データ型。

エンジン（Backtrader / NautilusTrader）に依存しない素のデータ構造だけを置く。
ここに外部ライブラリ固有の型を持ち込まないこと（共通化が崩れるため）。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    """売買判定の結果。アダプタ層がこれを各エンジンの注文に変換する。"""

    HOLD = "hold"
    ENTER_LONG = "enter_long"
    EXIT = "exit"


@dataclass(frozen=True)
class StrategyParams:
    """戦略パラメータ。

    過剰最適化（カーブフィッティング）を避けるため、初期値は素直な値にしている。
    最適化はこのパラメータ集合の範囲で、複数の相場局面に対して行う。
    """

    # --- 指標の期間 ---
    fast_period: int = 5
    slow_period: int = 20
    atr_period: int = 14
    volume_avg_period: int = 20

    # --- エントリー条件 ---
    volume_surge_mult: float = 1.5  # 出来高が平常時の何倍で「急増」とみなすか
    vwap_dev_min: float = 0.0       # VWAP からの最小乖離率（0.0 = VWAP より上であればよい）
    require_day_high_break: bool = False  # 当日高値更新を必須にするか

    # --- 決済条件 ---
    atr_stop_mult: float = 1.0   # 損切り幅 = ATR × この倍数
    atr_take_mult: float = 2.0   # 利確幅   = ATR × この倍数
    max_hold_bars: int = 30      # 保有時間の上限（バー数）。超えたら手仕舞い
    force_close_bar: int | None = None  # このバー以降は強制クローズ（引け前クローズ用）

    def __post_init__(self) -> None:
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period は slow_period より小さくする必要があります")
        if self.atr_stop_mult <= 0 or self.atr_take_mult <= 0:
            raise ValueError("ATR 倍数は正の値にしてください")


@dataclass(frozen=True)
class Features:
    """1バー分の特徴量（指標計算済み）。

    アダプタ層が各エンジンのバーからこの構造を組み立てて decide() に渡す。
    NaN（指標のウォームアップ期間）は呼び出し側で除外しておくこと。
    """

    close: float
    high: float
    low: float
    volume: float

    sma_fast: float
    sma_slow: float
    prev_sma_fast: float
    prev_sma_slow: float

    vwap: float
    atr: float
    volume_avg: float
    day_high: float

    bar_index: int  # 当日の寄り付きを0としたバー連番（引け前クローズ判定に使う）


@dataclass
class PositionState:
    """ポジションの状態。decide() はこれを読むだけで、更新はアダプタ側で行う。"""

    is_open: bool = False
    entry_price: float = 0.0
    stop_price: float = 0.0
    take_price: float = 0.0
    bars_held: int = 0
