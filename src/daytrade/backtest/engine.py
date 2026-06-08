"""基準バックテストエンジン（エンジン非依存）。

共通ロジック core.signals.decide() を素のループで回し、手数料・スリッページ込みで
損益を計算する。これが二重検証の「基準値」になる。Backtrader / NautilusTrader の
結果をこの基準と突き合わせ、ズレが許容範囲かを確認する（README 5・8章）。

約定モデル（シンプル・前提を明示）:
- decide() が ENTER_LONG / EXIT を返したバーの「終値」で約定する。
- スリッページは終値に対し、買いは不利な方向（高く）・売りは不利な方向（安く）に乗せる。
- 手数料は約定代金 × commission_rate を売買の両側で引く。
- 株数は core.risk.position_size（損切り幅からの逆算＋単元株丸め）で決める。
ルックアヘッドはしない（その時点で確定している指標のみ参照）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from daytrade.core.indicators import compute_indicators
from daytrade.core.risk import position_size
from daytrade.core.signals import _row_to_features, decide, initial_stops
from daytrade.core.types import Action, PositionState, StrategyParams


@dataclass(frozen=True)
class BacktestConfig:
    """バックテストの執行・コスト設定。"""

    initial_cash: float = 300_000.0   # 元手30万円（README 0章）
    commission_rate: float = 0.0005   # 片道手数料（約定代金比）。実際の料率に合わせる
    slippage_rate: float = 0.0002     # 片道スリッページ（価格比）
    risk_fraction: float = 0.05       # 1トレード最大損失（資金比。README: 5〜7%）
    lot_size: int = 100               # 単元株


@dataclass
class Trade:
    """1往復のトレード記録。"""

    entry_time: pd.Timestamp
    entry_price: float       # スリッページ込みの実効約定価格
    exit_time: pd.Timestamp
    exit_price: float        # スリッページ込みの実効約定価格
    shares: int
    pnl: float               # 手数料込みの純損益（円）
    return_pct: float        # 約定代金に対する純損益率
    exit_reason: str


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series          # 各バーのマーク・トゥ・マーケット評価額
    enriched: pd.DataFrame           # 指標＋action＋position 付きの入力
    metrics: dict = field(default_factory=dict)


def _exit_reason(f, position: PositionState, params: StrategyParams) -> str:
    """EXIT の理由を decide() と同じ優先順で判定する（記録用）。"""
    if f.low <= position.stop_price:
        return "stop_loss"
    if params.force_close_bar is not None and f.bar_index >= params.force_close_bar:
        return "force_close"
    if position.bars_held >= params.max_hold_bars:
        return "max_hold"
    if f.high >= position.take_price:
        return "take_profit"
    return "signal"


def _compute_metrics(trades: list[Trade], equity: pd.Series, config: BacktestConfig) -> dict:
    if equity.empty:
        return {}
    final_equity = float(equity.iloc[-1])
    total_return = final_equity / config.initial_cash - 1.0

    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    return {
        "initial_cash": config.initial_cash,
        "final_equity": final_equity,
        "total_return": total_return,
        "num_trades": len(trades),
        "win_rate": float(len(wins) / len(trades)) if trades else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
    }


def run_backtest(
    df: pd.DataFrame,
    params: StrategyParams | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """OHLCV を受け取り、共通ロジックを回して損益とエクイティ曲線を返す。"""
    params = params or StrategyParams()
    config = config or BacktestConfig()
    enriched = compute_indicators(df, params)

    position = PositionState()
    cash = config.initial_cash       # 余力
    shares_held = 0
    entry_price_eff = 0.0
    entry_time: pd.Timestamp | None = None

    trades: list[Trade] = []
    equity_points: list[float] = []
    actions: list[str] = []
    positions: list[int] = []

    for ts, row in enriched.iterrows():
        f = _row_to_features(row)
        action = decide(f, position, params)

        if action == Action.ENTER_LONG and not position.is_open:
            stop, take = initial_stops(f.close, f.atr, params)
            entry_price_eff = f.close * (1.0 + config.slippage_rate)
            size = position_size(
                cash, entry_price_eff, stop,
                risk_fraction=config.risk_fraction, lot_size=config.lot_size,
            )
            if size > 0:
                notional = entry_price_eff * size
                commission = notional * config.commission_rate
                cash -= notional + commission
                shares_held = size
                entry_time = ts
                position = PositionState(
                    is_open=True, entry_price=entry_price_eff,
                    stop_price=stop, take_price=take, bars_held=0,
                )

        elif action == Action.EXIT and position.is_open:
            reason = _exit_reason(f, position, params)
            exit_price_eff = f.close * (1.0 - config.slippage_rate)
            notional = exit_price_eff * shares_held
            commission = notional * config.commission_rate
            cash += notional - commission

            entry_notional = position.entry_price * shares_held
            pnl = (exit_price_eff - position.entry_price) * shares_held \
                - commission - entry_notional * config.commission_rate
            trades.append(Trade(
                entry_time=entry_time, entry_price=position.entry_price,
                exit_time=ts, exit_price=exit_price_eff, shares=shares_held,
                pnl=pnl, return_pct=pnl / entry_notional if entry_notional else 0.0,
                exit_reason=reason,
            ))
            shares_held = 0
            position = PositionState()

        elif position.is_open:
            position.bars_held += 1

        # マーク・トゥ・マーケット評価額
        equity_points.append(cash + shares_held * f.close)
        actions.append(action.value)
        positions.append(1 if position.is_open else 0)

    enriched = enriched.copy()
    enriched["action"] = actions
    enriched["position"] = positions
    equity = pd.Series(equity_points, index=enriched.index, name="equity")

    return BacktestResult(
        trades=trades,
        equity_curve=equity,
        enriched=enriched,
        metrics=_compute_metrics(trades, equity, config),
    )
