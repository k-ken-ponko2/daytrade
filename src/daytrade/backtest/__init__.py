"""バックテスト層。

- engine:      共通ロジックを素のループで回す基準バックテスト（手数料・スリッページ込み）
- sample_data: プラン不要で動かせる合成分足データ（デモ・テスト用）

基準実装の結果を Backtrader / NautilusTrader と突き合わせて二重検証する（README 5章）。
"""

from daytrade.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    Trade,
    daily_returns,
    run_backtest,
)
from daytrade.backtest.walkforward import (
    WalkForwardReport,
    WindowResult,
    param_grid,
    rolling_windows,
    run_walk_forward,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Trade",
    "WalkForwardReport",
    "WindowResult",
    "daily_returns",
    "param_grid",
    "rolling_windows",
    "run_backtest",
    "run_walk_forward",
]
