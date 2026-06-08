#!/usr/bin/env python3
"""二重検証ハーネス：基準実装 vs Backtrader（README 5章フェーズ4）。

同一の合成分足データに同一パラメータを与え、
  (A) core の基準バックテスト run_backtest()
  (B) Backtrader アダプタ（共通ロジック decide() を呼ぶ）
の最終評価額・トレード数を突き合わせ、ズレが許容範囲かを判定する。

「完全一致はしない前提」（README 5章）：約定タイミングや端数処理が違うため、
差を許容相対誤差 --tol 以内で評価する。Backtrader は cheat-on-close を有効化して
基準実装の「終値約定」に寄せることで差を最小化している。

実行:
    pip install backtrader            # 未導入なら
    PYTHONPATH=src python scripts/compare_engines.py
    PYTHONPATH=src python scripts/compare_engines.py --tol 0.02 --days 5
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from daytrade.backtest.engine import BacktestConfig, run_backtest
from daytrade.backtest.sample_data import make_intraday_ohlcv
from daytrade.core.types import StrategyParams


def run_backtrader(df: pd.DataFrame, params: StrategyParams, config: BacktestConfig):
    """Backtrader で同じロジックを回し、(最終評価額, トレード数) を返す。"""
    import backtrader as bt

    from daytrade.adapters.backtrader_adapter import make_strategy

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(config.initial_cash)
    cerebro.broker.setcommission(commission=config.commission_rate)
    cerebro.broker.set_slippage_perc(config.slippage_rate)
    cerebro.broker.set_coc(True)  # cheat-on-close: 終値約定に寄せる

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(make_strategy(
        params, risk_fraction=config.risk_fraction, lot_size=config.lot_size))
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    strat = cerebro.run()[0]
    final_value = cerebro.broker.getvalue()
    ta = strat.analyzers.trades.get_analysis()
    num_trades = ta.get("total", {}).get("closed", 0)
    return final_value, num_trades


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="基準実装 vs Backtrader 二重検証")
    parser.add_argument("--days", type=int, default=5, help="合成データの日数")
    parser.add_argument("--tol", type=float, default=0.02,
                        help="最終評価額の許容相対誤差（既定2%）")
    args = parser.parse_args(argv)

    df = make_intraday_ohlcv(n_days=args.days)
    # run_demo.py と同じ取引頻度が高めの設定で、複数トレードを跨いで突き合わせる
    params = StrategyParams(
        fast_period=3, slow_period=10, volume_surge_mult=1.3,
        force_close_bar=55, max_hold_bars=20,
    )
    config = BacktestConfig()

    base = run_backtest(df, params, config)
    base_equity = base.metrics.get("final_equity", config.initial_cash)
    base_trades = base.metrics.get("num_trades", 0)

    try:
        bt_equity, bt_trades = run_backtrader(df, params, config)
    except ImportError:
        print("[skip] backtrader が未導入です。`pip install backtrader` 後に再実行してください。",
              file=sys.stderr)
        print(f"基準実装のみ: 最終評価額={base_equity:,.0f} 円 / トレード数={base_trades}")
        return 0

    denom = config.initial_cash
    rel_diff = abs(bt_equity - base_equity) / denom

    print("\n=== 二重検証：基準実装 vs Backtrader ===")
    print(f"{'':16}{'基準実装':>16}{'Backtrader':>16}")
    print(f"{'最終評価額(円)':16}{base_equity:>16,.0f}{bt_equity:>16,.0f}")
    print(f"{'トレード数':16}{base_trades:>16d}{bt_trades:>16d}")
    print(f"\n評価額の相対差: {rel_diff*100:.3f} %（許容 {args.tol*100:.1f} %）")

    ok = rel_diff <= args.tol
    print("判定: ✅ 許容範囲内（エンジン差として説明可能）" if ok
          else "判定: ⚠️ 許容超過。ロジック/指標/約定モデルの差を要調査")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
