#!/usr/bin/env python3
"""ウォークフォワード検証（過剰最適化の検出）と相場局面別ロバストネス。

README 8章の「どの相場で壊れるか探し」「過剰最適化が最大の罠」を実地で見るための
スクリプト。プラン不要の合成分足で動く。J-Quants 実データに差し替える場合は、
df を to_ohlcv() の出力に置き換えればよい（日足なら params の分足前提を外すこと）。

実行例:
    PYTHONPATH=src python scripts/walk_forward.py                 # mixed 局面でWF
    PYTHONPATH=src python scripts/walk_forward.py --regime trend_up
    PYTHONPATH=src python scripts/walk_forward.py --all-regimes   # 局面横断の要約
    PYTHONPATH=src python scripts/walk_forward.py --days 40 --train 10 --test 5
"""

from __future__ import annotations

import argparse

from daytrade.backtest.engine import BacktestConfig
from daytrade.backtest.sample_data import REGIME_DRIFT, make_intraday_ohlcv
from daytrade.backtest.walkforward import param_grid, run_walk_forward


def _candidates():
    """探索するパラメータ群（小さめのグリッド）。"""
    return param_grid(
        base=dict(force_close_bar=55, max_hold_bars=20),
        fast_period=[3, 5],
        slow_period=[10, 20],
        volume_surge_mult=[1.3, 1.5],
        atr_take_mult=[1.5, 2.5],
    )


def _run_one(regime: str, args, config) -> dict:
    df = make_intraday_ohlcv(n_days=args.days, regime=regime, seed=args.seed)
    report = run_walk_forward(
        df, _candidates(), config=config,
        train_days=args.train, test_days=args.test, step_days=args.step,
    )
    return {"regime": regime, "report": report}


def _print_report(regime: str, report) -> None:
    print(f"\n===== ウォークフォワード（regime={regime}）=====")
    if not report.windows:
        print("区間を作れませんでした。--days を増やすか train/test を小さくしてください。")
        return
    print(f"{'区間':52}{'IS%':>8}{'OOS%':>8}{'OOS取引':>8}{'勝率%':>8}")
    for w in report.windows:
        p = w.best_params
        print(f"{w.label:52}{w.is_return*100:>8.2f}{w.oos_return*100:>8.2f}"
              f"{w.oos_trades:>8d}{w.oos_win_rate*100:>8.1f}")
        print(f"  └ best: fast={p.fast_period} slow={p.slow_period} "
              f"vol={p.volume_surge_mult} take={p.atr_take_mult}")
    s = report.summary
    print(f"\n  IS平均 {s['is_mean_return']*100:+.2f}% / OOS平均 {s['oos_mean_return']*100:+.2f}%"
          f" / 過剰最適化ギャップ {s['overfit_gap']*100:+.2f}pt")
    print(f"  OOSプラス区間 {s['oos_positive_windows']}/{s['num_windows']}"
          f"（{s['oos_hit_rate']*100:.0f}%） / OOS総取引 {s['total_oos_trades']}")
    _verdict(s)


def _verdict(s: dict) -> None:
    gap = s["overfit_gap"]
    oos = s["oos_mean_return"]
    if s["total_oos_trades"] == 0:
        print("  判定: ❔ OOS取引ゼロ。区間が短すぎるか条件が厳しすぎる（評価不能）")
    elif oos > 0 and gap < 0.02:
        print("  判定: ✅ OOSでもプラスかつIS-OOS差が小さい（過剰最適化の兆候は弱い）")
    elif gap >= 0.02 and oos <= 0:
        print("  判定: ⚠️ ISは良いがOOSで崩れる＝過剰最適化の典型。採用しない")
    else:
        print("  判定: ➖ 判断保留。区間数を増やして再確認（合成データは目安）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ウォークフォワード検証")
    parser.add_argument("--regime", default="mixed", choices=list(REGIME_DRIFT),
                        help="相場局面（既定 mixed）")
    parser.add_argument("--all-regimes", action="store_true", help="全局面を横断で要約")
    parser.add_argument("--days", type=int, default=40, help="生成する日数")
    parser.add_argument("--train", type=int, default=10, help="IS（最適化）日数")
    parser.add_argument("--test", type=int, default=5, help="OOS（評価）日数")
    parser.add_argument("--step", type=int, default=None, help="前進日数（既定=test）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    config = BacktestConfig()
    regimes = list(REGIME_DRIFT) if args.all_regimes else [args.regime]

    summaries = []
    for regime in regimes:
        out = _run_one(regime, args, config)
        _print_report(regime, out["report"])
        if out["report"].summary:
            summaries.append((regime, out["report"].summary))

    if args.all_regimes and summaries:
        print("\n===== 局面横断サマリ（OOS平均リターン）=====")
        for regime, s in summaries:
            print(f"  {regime:12}: OOS {s['oos_mean_return']*100:+.2f}%"
                  f" / ギャップ {s['overfit_gap']*100:+.2f}pt"
                  f" / OOS取引 {s['total_oos_trades']}")
        print("\n※ 局面ごとに挙動が大きく変わるなら、その戦略は局面依存。"
              "実運用前に局面フィルタや停止条件を検討する（README 8章）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
