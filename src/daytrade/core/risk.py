"""リスク管理（株数計算）— エンジン非依存。

README 7章のルールを実装する:
- 1トレードの最大損失を資金の一定割合に固定し、損切り幅から逆算して株数を決める。
- 日本株は単元株（通常100株）単位でしか売買できないため、ロット単位に丸める。

両エンジン（基準実装・Backtrader アダプタ）が同じ株数計算を使うことで、
二重検証時のズレを「サイズの差」ではなく「エンジンの差」に絞れる。
"""

from __future__ import annotations

import math


def position_size(
    cash: float,
    entry_price: float,
    stop_price: float,
    *,
    risk_fraction: float = 0.05,
    lot_size: int = 100,
) -> int:
    """1トレードの株数を返す（単元株に丸めた整数）。

    Args:
        cash:          現在の余力（円）。
        entry_price:   想定エントリー価格。
        stop_price:    損切り価格。entry_price との差が1株あたりの想定損失。
        risk_fraction: 1トレードで許容する損失の資金に対する割合（README: 5〜7%）。
        lot_size:      単元株数（日本株は通常100）。

    Returns:
        購入株数。条件を満たせない（損切り幅0以下・余力不足）場合は0。
    """
    if cash <= 0 or entry_price <= 0:
        return 0

    risk_per_share = entry_price - stop_price
    if risk_per_share <= 0:
        return 0

    # 損失上限から逆算した株数
    risk_budget = cash * risk_fraction
    shares_by_risk = risk_budget / risk_per_share

    # 余力で買える株数（手数料・余裕は呼び出し側のコストモデルで吸収）
    shares_by_cash = cash / entry_price

    shares = min(shares_by_risk, shares_by_cash)

    # 単元株に切り下げ
    lots = math.floor(shares / lot_size)
    return max(0, lots * lot_size)


def retreat_triggered(
    equity: float,
    initial_cash: float,
    drawdown: float | None,
) -> bool:
    """撤退ライン判定（README 7章）。

    評価額が初期資金から drawdown ぶん減ったら True。例: drawdown=0.5 なら半減で撤退。
    drawdown が None なら無効（常に False）。発火したら一旦すべて手仕舞い、手法を見直す。
    """
    if drawdown is None:
        return False
    return equity <= initial_cash * (1.0 - drawdown)
