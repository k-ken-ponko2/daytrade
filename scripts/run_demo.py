#!/usr/bin/env python3
"""end-to-end デモ：データ → 指標 → 売買判定 → 損益 → 可視化。

データソースを選べる:
- （指定なし）      合成分足。APIキー不要で配線確認ができる。
- --alpaca SYMBOL   Alpaca の米国株分足（無料キー）。実データ。
- --jquants-minute CODE  J-Quants V2 の日本株分足（分足アドオン契約が必要）。実データ。
- --jquants CODE    J-Quants の日足（参考。デイトレ戦略は日中の動きが前提なので結果に意味は薄い）。

実データの分足は data/intraday/ にキャッシュし、2回目以降は API を呼ばない。

実行例:
    PYTHONPATH=src python scripts/run_demo.py --out demo.png
    PYTHONPATH=src python scripts/run_demo.py --alpaca AAPL --from 2026-09-01 --to 2026-09-30 --out aapl.png
    PYTHONPATH=src python scripts/run_demo.py --jquants-minute 7203 --from 2026-09-01 --to 2026-09-30 --out 7203.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from daytrade.backtest.engine import BacktestConfig, run_backtest
from daytrade.backtest.sample_data import make_intraday_ohlcv
from daytrade.core.markets import JAPAN, US, MarketProfile, get_profile
from daytrade.core.types import StrategyParams

INITIAL_CASH = {"JPY": 300_000.0, "USD": 2_000.0}   # 元手30万円（米国株は約$2,000換算）

# 実データ分足用の素直な初期パラメータ（最適化していない。README 8章）。
# force_close_bar は「その日の何本目以降は持ち越さず手仕舞うか」。
#   日本株: 前場151本＋後場176本 ≒ 327本/日 → 315本目 ≒ 15:10 前後
#   米国株: 390本/日 → 380本目 ≒ 15:50
# 約定のない分は欠落するため、閑散銘柄では実際の時刻より遅くなる点に注意。
_REAL_EXITS = dict(max_hold_bars=30, trailing_stop_atr_mult=1.5,
                   scale_out_atr_mult=1.0, scale_out_fraction=0.5)
REAL_PARAMS = {
    "JPY": StrategyParams(force_close_bar=315, **_REAL_EXITS),
    "USD": StrategyParams(force_close_bar=380, **_REAL_EXITS),
}


@dataclass
class Scenario:
    df: pd.DataFrame
    params: StrategyParams
    config: BacktestConfig
    market: MarketProfile
    title: str
    synthetic: bool


def _synthetic(args) -> Scenario:
    market = get_profile(args.market)
    base_price = 150.0 if market is US else 1000.0
    print(f"合成分足データを生成: {args.days} 日分 / market={args.market}")
    df = make_intraday_ohlcv(n_days=args.days, base_price=base_price)
    # デモが視覚的に分かりやすいよう短めのMAで取引頻度を上げている（配線確認用）
    params = StrategyParams(
        fast_period=3, slow_period=10, volume_surge_mult=1.3,
        force_close_bar=55, max_hold_bars=20,
        trailing_stop_atr_mult=1.5, scale_out_atr_mult=1.0, scale_out_fraction=0.5,
    )
    config = BacktestConfig.for_market(market, initial_cash=INITIAL_CASH[market.currency])
    return Scenario(df, params, config, market, "synthetic intraday", synthetic=True)


def _real_intraday(source: str, symbol: str, args) -> Scenario:
    from daytrade.data.intraday import load_intraday

    market = US if source == "alpaca" else JAPAN
    print(f"実データ分足を取得: source={source} symbol={symbol} {args.from_date}〜{args.to_date}")
    df = load_intraday(source, symbol, from_date=args.from_date, to_date=args.to_date,
                       feed=args.feed, refresh=args.refresh)
    config = BacktestConfig.for_market(market, initial_cash=INITIAL_CASH[market.currency])
    return Scenario(df, REAL_PARAMS[market.currency], config, market,
                    f"{symbol.upper()} 1min ({source})", synthetic=False)


def _jquants_daily(code: str, args) -> Scenario:
    from daytrade.data.jquants import JQuantsClient
    from daytrade.data.loader import DataStore, to_ohlcv

    print(f"J-Quants 日足を取得: code={code}（参考。デイトレ戦略の検証には分足を使う）")
    raw = DataStore().get_daily(JQuantsClient.from_env(), code,
                                from_date=args.from_date, to_date=args.to_date)
    df = to_ohlcv(raw, adjusted=True) if not raw.empty else raw
    # 日足では VWAP/当日高値/引け前クローズは意味を持たない（1日=1バー）
    params = StrategyParams(force_close_bar=None, max_hold_bars=10)
    return Scenario(df, params, BacktestConfig(), JAPAN, f"{code} daily (J-Quants)", synthetic=False)


def _print_metrics(metrics: dict, n_bars: int, currency: str) -> None:
    cur = {"JPY": "円", "USD": "USD"}.get(currency, currency)
    print("\n=== バックテスト結果（手数料・スリッページ込み） ===")
    print(f"バー数            : {n_bars}")
    print(f"初期資金          : {metrics['initial_cash']:>12,.0f} {cur}")
    print(f"最終評価額        : {metrics['final_equity']:>12,.0f} {cur}")
    print(f"トータルリターン  : {metrics['total_return']*100:>11.2f} %")
    print(f"トレード数        : {metrics['num_trades']:>12d}")
    print(f"勝率              : {metrics['win_rate']*100:>11.2f} %")
    print(f"平均利益          : {metrics['avg_win']:>12,.2f} {cur}")
    print(f"平均損失          : {metrics['avg_loss']:>12,.2f} {cur}")
    pf = metrics["profit_factor"]
    print(f"プロフィットファクタ: {pf:>10.2f}" if pf != float("inf") else "プロフィットファクタ:        inf")
    print(f"最大ドローダウン  : {metrics['max_drawdown']*100:>11.2f} %")

    # 日利 vs 目標（README 0章：目標 日利2.5%）
    tgt = metrics.get("daily_target", 0.025)
    print(f"\n--- 日利（目標 {tgt*100:.1f}%）---")
    print(f"営業日数          : {metrics.get('num_days', 0):>12d}")
    print(f"実効日利(複利)    : {metrics.get('geom_daily_return', 0)*100:>11.2f} %")
    print(f"平均日利          : {metrics.get('avg_daily_return', 0)*100:>11.2f} %")
    print(f"目標達成日の割合  : {metrics.get('daily_target_hit_rate', 0)*100:>11.1f} %")
    print(f"目標との差(平均)  : {metrics.get('daily_target_gap', 0)*100:>11.2f} pt")


def _warn_if_untradeable(sc: Scenario, num_trades: int) -> None:
    """日本株で「1単元が元手を超えて1回も買えない」ケースを明示する（README 0章の壁）。"""
    if num_trades or sc.df.empty:
        return
    unit = sc.market.min_position_value(float(sc.df["close"].median()))
    if unit > sc.config.initial_cash:
        print(f"\n[注意] 1単元の必要資金 ≒ {unit:,.0f} が元手 {sc.config.initial_cash:,.0f} を超えるため、"
              "1回も買えませんでした。より低位の銘柄を選ぶか、米国株（--alpaca）で試してください。")


def _save_chart(result, path: str, title: str, currency: str) -> None:
    """バー連番を横軸にして描く（夜間・休日の空白を詰め、日中の売買を見やすくする）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = result.enriched
    x = np.arange(len(df))
    days = pd.Index(df.index.date)
    day_starts = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
    step = max(1, len(day_starts) // 12)   # 日付ラベルは最大12個程度に間引く

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax1.plot(x, df["close"], lw=0.8, color="black", label="close")
    ax1.plot(x, df["vwap"], lw=0.7, color="orange", alpha=0.8, label="VWAP")
    entries = np.flatnonzero(df["action"].to_numpy() == "enter_long")
    exits = np.flatnonzero(df["action"].to_numpy() == "exit")
    ax1.scatter(entries, df["close"].iloc[entries], marker="^", color="green", s=50, label="entry", zorder=5)
    ax1.scatter(exits, df["close"].iloc[exits], marker="v", color="red", s=50, label="exit", zorder=5)
    ax1.set_ylabel("price")
    ax1.legend(loc="upper left")
    ax1.set_title(f"Daytrade strategy demo ({title})")

    ax2.plot(x, result.equity_curve.to_numpy(), color="navy", lw=1.0)
    ax2.axhline(result.metrics.get("initial_cash", 0), color="grey", ls="--", lw=0.8)
    ax2.set_ylabel(f"equity ({currency})")

    for ax in (ax1, ax2):
        for s in day_starts:
            ax.axvline(s, color="#dddddd", lw=0.6, zorder=0)
    ax2.set_xticks(day_starts[::step])
    ax2.set_xticklabels([str(days[i]) for i in day_starts[::step]], rotation=45, ha="right")
    ax2.set_xlabel("session (gaps between days removed)")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nチャートを保存しました: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="デイトレ戦略 end-to-end デモ")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--alpaca", metavar="SYMBOL", help="Alpaca の米国株分足で実行（例: AAPL）")
    src.add_argument("--jquants-minute", metavar="CODE", help="J-Quants の日本株分足で実行（例: 7203）")
    src.add_argument("--jquants", metavar="CODE", help="J-Quants の日足で実行（参考）")
    parser.add_argument("--from", dest="from_date", default=None, help="開始日 YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", default=None, help="終了日 YYYY-MM-DD")
    parser.add_argument("--feed", choices=["sip", "iex"], default="sip",
                        help="Alpaca のデータソース（既定 sip。権限エラー時は iex）")
    parser.add_argument("--refresh", action="store_true", help="分足キャッシュを無視して再取得")
    parser.add_argument("--days", type=int, default=20, help="合成データの日数")
    parser.add_argument("--market", choices=["jp", "us"], default="jp",
                        help="合成データの市場プロファイル（jp=単元株100/円、us=1株単位/USD）")
    parser.add_argument("--out", default=None, help="チャートPNGの保存先（省略時は保存しない）")
    args = parser.parse_args(argv)

    real_source = ("alpaca", args.alpaca) if args.alpaca else \
        ("jquants", args.jquants_minute) if args.jquants_minute else None
    if real_source and not (args.from_date and args.to_date):
        parser.error("実データの分足には --from と --to が必要です（例: --from 2026-09-01 --to 2026-09-30）")

    try:
        if real_source:
            sc = _real_intraday(*real_source, args)
        elif args.jquants:
            sc = _jquants_daily(args.jquants, args)
        else:
            sc = _synthetic(args)
    except RuntimeError as exc:   # JQuantsError / AlpacaError（認証・権限・レート制限）
        print(f"[エラー] {exc}", file=sys.stderr)
        return 1

    if sc.df.empty:
        print("データが0件でした。期間（休場日のみでないか）・銘柄コード・契約プランを確認してください。")
        return 0

    result = run_backtest(sc.df, sc.params, sc.config)
    _print_metrics(result.metrics, len(sc.df), sc.market.currency)
    _warn_if_untradeable(sc, result.metrics["num_trades"])
    print("\n※ " + ("合成データでの配線確認用。優位性の検証ではない（README 8章）。" if sc.synthetic else
                    "実データ・1銘柄・単一期間の結果。優位性の判断は複数期間のウォークフォワードで（README 8章）。"))

    if args.out:
        try:
            _save_chart(result, args.out, sc.title, sc.market.currency)
        except ImportError:
            print("matplotlib 未導入のためチャートは省略（pip install matplotlib）。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
