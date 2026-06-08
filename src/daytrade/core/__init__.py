"""共通ロジック層（エンジン非依存・純Python）。

- indicators: 指標計算（pandas/numpy。後で pandas-ta / TA-Lib に差し替え可能）
- signals:    売買判定（単一の真実）。バー単位の decide() を中核とする
- types:      共通データ型（Action / Features / PositionState / StrategyParams）
"""

from daytrade.core.signals import decide, generate_signals
from daytrade.core.types import (
    Action,
    Features,
    PositionState,
    StrategyParams,
)

__all__ = [
    "Action",
    "Features",
    "PositionState",
    "StrategyParams",
    "decide",
    "generate_signals",
]
