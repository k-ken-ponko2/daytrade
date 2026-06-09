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
    SCALE_OUT = "scale_out"   # 段階的利確：保有の一部だけを利確する
    EXIT = "exit"             # 全数手仕舞い


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
    atr_take_mult: float = 2.0   # 全利確幅 = ATR × この倍数
    max_hold_bars: int = 30      # 保有時間の上限（バー数）。超えたら手仕舞い
    force_close_bar: int | None = None  # このバー以降は強制クローズ（引け前クローズ用）

    # --- 決済の高度化（README 6・7章。None で無効＝従来動作） ---
    trailing_stop_atr_mult: float | None = None  # 高値からこの ATR 倍ぶん下げたら損切り（利を伸ばす）
    scale_out_atr_mult: float | None = None      # この ATR 倍に達したら一部利確（原資回収）
    scale_out_fraction: float = 0.5              # 段階利確で手仕舞う割合（残りを伸ばす）

    def __post_init__(self) -> None:
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period は slow_period より小さくする必要があります")
        if self.atr_stop_mult <= 0 or self.atr_take_mult <= 0:
            raise ValueError("ATR 倍数は正の値にしてください")
        if self.trailing_stop_atr_mult is not None and self.trailing_stop_atr_mult <= 0:
            raise ValueError("trailing_stop_atr_mult は正の値にしてください")
        if self.scale_out_atr_mult is not None and self.scale_out_atr_mult <= 0:
            raise ValueError("scale_out_atr_mult は正の値にしてください")
        if not 0.0 < self.scale_out_fraction < 1.0:
            raise ValueError("scale_out_fraction は 0 と 1 の間にしてください")


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
    """ポジションの状態。decide() はこれを読むだけで、更新はアダプタ側で行う。

    トレーリング／段階利確のため、取得後の最高値(high_water)と一部利確済みフラグ
    (scaled_out)、段階利確の目標(scale_out_price)も保持する。
    """

    is_open: bool = False
    entry_price: float = 0.0
    stop_price: float = 0.0
    take_price: float = 0.0
    bars_held: int = 0

    scale_out_price: float = 0.0   # 段階利確の目標価格（0=無効）
    high_water: float = 0.0        # 取得後の最高値（トレーリングストップ用）
    scaled_out: bool = False       # 既に一部利確したか
