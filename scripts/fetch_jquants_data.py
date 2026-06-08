#!/usr/bin/env python3
"""J-Quants データ取得パイプライン（free 日足）。

日足を取得 → ローカルにキャッシュ(parquet/CSV) → バックテスト用 OHLCV に整形、までを
一括で行う。流動性スクリーニング（売買代金上位）と、取得漏れの営業日チェックも兼ねる。

事前準備は fetch_jquants_minimal.py と同じ（.env に認証情報）。free は12週間遅延。

実行例:
    # 1銘柄を取得・キャッシュして OHLCV を表示
    PYTHONPATH=src python scripts/fetch_jquants_data.py fetch 7203 --from 2024-01-01 --to 2024-03-31

    # 流動性（売買代金）上位を、指定営業日の平均でスクリーニング
    PYTHONPATH=src python scripts/fetch_jquants_data.py screen --dates 2024-01-15 2024-01-16 --top 30

    # キャッシュ済み日足に対する欠損営業日チェック
    PYTHONPATH=src python scripts/fetch_jquants_data.py check 7203 --from 2024-01-01 --to 2024-03-31
"""

from __future__ import annotations

import argparse
import sys

from daytrade.data.calendar import missing_sessions
from daytrade.data.jquants import JQuantsClient, JQuantsError
from daytrade.data.loader import DataStore, to_ohlcv
from daytrade.data.screening import screen_liquid_stocks


def _client() -> JQuantsClient:
    client = JQuantsClient.from_env()
    client.authenticate()
    return client


def cmd_fetch(args) -> int:
    store = DataStore(args.cache_dir)
    client = _client()
    raw = store.get_daily(
        client, args.code, from_date=args.from_date, to_date=args.to_date, refresh=args.refresh
    )
    if raw.empty:
        print("データが0件でした（free は12週間遅延。古い日付範囲を指定）。")
        return 0
    df = to_ohlcv(raw, adjusted=not args.raw)
    kind = "生" if args.raw else "調整後"
    print(f"取得 {len(raw)} 行 → OHLCV({kind}) {len(df)} 行  [cache: {store._path(args.code)}]")
    print(df.tail(args.rows).to_string())
    return 0


def cmd_screen(args) -> int:
    client = _client()
    print(f"売買代金スクリーニング: dates={args.dates} top={args.top}")
    ranked = screen_liquid_stocks(
        client, args.dates, top_n=args.top, min_turnover=args.min_turnover
    )
    if ranked.empty:
        print("該当なし（日付が新しすぎる/データ未取得の可能性）。")
        return 0
    cols = [c for c in ["Code", "CompanyName", "MarketCodeName", "avg_turnover", "days_count"]
            if c in ranked.columns]
    show = ranked[cols].copy()
    if "avg_turnover" in show.columns:
        show["avg_turnover"] = show["avg_turnover"].map(lambda v: f"{v:,.0f}")
    print(show.to_string(index=False))
    return 0


def cmd_check(args) -> int:
    store = DataStore(args.cache_dir)
    client = _client()
    raw = store.get_daily(client, args.code, from_date=args.from_date, to_date=args.to_date)
    if raw.empty:
        print("データが0件でした。")
        return 0
    df = to_ohlcv(raw)
    cal = client.get_trading_calendar(from_date=args.from_date, to_date=args.to_date)
    missing = missing_sessions(df.index, cal)
    if len(missing) == 0:
        print(f"欠損なし：{len(df)} 営業日ぶん揃っています。")
    else:
        print(f"欠損営業日 {len(missing)} 件（取得漏れ/上場前後の可能性）:")
        for d in missing:
            print(" ", d.date())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="J-Quants データ取得パイプライン")
    parser.add_argument("--cache-dir", default="data", help="キャッシュ保存先（既定 ./data）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="日足を取得・キャッシュして OHLCV を表示")
    p_fetch.add_argument("code")
    p_fetch.add_argument("--from", dest="from_date", default=None)
    p_fetch.add_argument("--to", dest="to_date", default=None)
    p_fetch.add_argument("--raw", action="store_true", help="調整後でなく生の価格を使う")
    p_fetch.add_argument("--refresh", action="store_true", help="キャッシュを無視して再取得")
    p_fetch.add_argument("--rows", type=int, default=10)
    p_fetch.set_defaults(func=cmd_fetch)

    p_screen = sub.add_parser("screen", help="売買代金で流動性上位を絞り込む")
    p_screen.add_argument("--dates", nargs="+", required=True, help="対象営業日（複数可）")
    p_screen.add_argument("--top", type=int, default=50)
    p_screen.add_argument("--min-turnover", type=float, default=None)
    p_screen.set_defaults(func=cmd_screen)

    p_check = sub.add_parser("check", help="欠損営業日チェック")
    p_check.add_argument("code")
    p_check.add_argument("--from", dest="from_date", default=None)
    p_check.add_argument("--to", dest="to_date", default=None)
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except JQuantsError as exc:
        print(f"[エラー] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
