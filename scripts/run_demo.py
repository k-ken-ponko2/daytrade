#!/usr/bin/env python3
"""end-to-end デモ：データ → 指標 → 売買判定 → 損益 → 可視化。

有料の分足アドオンがなくても動くよう、既定では同梱の合成分足データを使う。
J-Quants free プランで取れる日足を使いたい場合は --jquants CODE を指定する
（free は12週間遅延・日足のみ。デイトレ用の分足は有料アドオンが必要）。

実行例:
    PYTHONPATH=src python scripts/run_demo.py                 # 合成分足でデモ
    PYTHONPATH=src python scripts/run_demo.py --out demo.png  # チャート保存
    PYTHONPATH=src python scripts/run_demo.py --jquants 7203 --from 2024-01-01 --to 2024-03-31
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from daytrade.backtest.engine import BacktestConfig, run_backtest
from daytrade.backtest.sample_data import make_intraday_ohlcv
from daytrade.core.types import StrategyParams


def _load_jquants_daily(code: str, from_date: str | None, to_date: str | None) -> pd.DataFrame:
    """J-Quants free プランの日足を取得し、調整後 OHLCV に整形して返す（キャッシュ利用）。"""
    from daytrade.data.jquants import JQuantsClient
    from daytrade.data.loader import DataStore, to_ohlcv

    client = JQuantsClient.from_env()
    raw = DataStore().get_daily(client, code, from_date=from_date, to_date=to_date)
    if raw.empty:
        return raw
    return to_ohlcv(raw, adjusted=True)


def _print_metrics(metrics: dict, n_bars: int) -> None:
    if not metrics:
        print("トレードが発生しませんでした（条件が厳しすぎるかデータが短い）。")
        return
    print("\n=== バックテスト結果（手数料・スリッページ込み） ===")
    print(f"バー数            : {n_bars}")
    print(f"初期資金          : {metrics['initial_cash']:>12,.0f} 円")
    print(f"最終評価額        : {metrics['final_equity']:>12,.0f} 円")
    print(f"トータルリターン  : {metrics['total_return']*100:>11.2f} %")
    print(f"トレード数        : {metrics['num_trades']:>12d}")
    print(f"勝率              : {metrics['win_rate']*100:>11.2f} %")
    print(f"平均利益          : {metrics['avg_win']:>12,.0f} 円")
    print(f"平均損失          : {metrics['avg_loss']:>12,.0f} 円")
    pf = metrics["profit_factor"]
    print(f"プロフィットファクタ: {pf:>10.2f}" if pf != float("inf") else "プロフィットファクタ:        inf")
    print(f"最大ドローダウン  : {metrics['max_drawdown']*100:>11.2f} %")
    print("\n※ 合成データでの配線確認用。優位性の検証ではない（README 8章）。")


def _save_chart(result, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = result.enriched
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(df.index, df["close"], lw=0.8, color="black", label="close")
    ax1.plot(df.index, df["vwap"], lw=0.7, color="orange", alpha=0.7, label="VWAP")
    entries = df[df["action"] == "enter_long"]
    exits = df[df["action"] == "exit"]
    ax1.scatter(entries.index, entries["close"], marker="^", color="green", s=60, label="entry", zorder=5)
    ax1.scatter(exits.index, exits["close"], marker="v", color="red", s=60, label="exit", zorder=5)
    ax1.set_ylabel("price")
    ax1.legend(loc="upper left")
    ax1.set_title("Daytrade strategy demo (synthetic intraday)")

    ax2.plot(result.equity_curve.index, result.equity_curve.values, color="navy", lw=1.0)
    ax2.axhline(result.metrics.get("initial_cash", 0), color="grey", ls="--", lw=0.8)
    ax2.set_ylabel("equity (JPY)")
    ax2.set_xlabel("time")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nチャートを保存しました: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="デイトレ戦略 end-to-end デモ")
    parser.add_argument("--jquants", metavar="CODE", default=None,
                        help="J-Quants free の日足を使う銘柄コード（省略時は合成分足）")
    parser.add_argument("--from", dest="from_date", default=None, help="開始日 YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", default=None, help="終了日 YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=20, help="合成データの日数")
    parser.add_argument("--out", default=None, help="チャートPNGの保存先（省略時は保存しない）")
    args = parser.parse_args(argv)

    if args.jquants:
        print(f"J-Quants(free) 日足を取得: code={args.jquants}")
        df = _load_jquants_daily(args.jquants, args.from_date, args.to_date)
        if df.empty:
            print("データが0件でした。free は12週間遅延のため、古い日付範囲を指定してください。")
            return 0
        # 日足では VWAP/当日高値/引け前クローズは意味を持たない（1日=1バー）。
        # 設定を日足向けに緩め、SMAクロス＋出来高でのみ判定する。
        params = StrategyParams(
            fast_period=5, slow_period=20, atr_period=14, volume_avg_period=20,
            force_close_bar=None, max_hold_bars=10,
        )
    else:
        print(f"合成分足データを生成: {args.days} 日分")
        df = make_intraday_ohlcv(n_days=args.days)
        # デモが視覚的に分かりやすいよう、短めのMAで取引頻度を上げている
        # （優位性の主張ではなく配線確認用。実検証では実データで再調整する）
        # トレーリング・段階利確も有効化して決済ロジックの拡張を見せる（README 6・7章）
        params = StrategyParams(
            fast_period=3, slow_period=10, volume_surge_mult=1.3,
            force_close_bar=55, max_hold_bars=20,
            trailing_stop_atr_mult=1.5, scale_out_atr_mult=1.0, scale_out_fraction=0.5,
        )

    result = run_backtest(df, params, BacktestConfig())
    _print_metrics(result.metrics, len(df))

    if args.out:
        try:
            _save_chart(result, args.out)
        except ImportError:
            print("matplotlib 未導入のためチャートは省略（pip install matplotlib）。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
