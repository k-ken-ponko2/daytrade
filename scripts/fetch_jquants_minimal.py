#!/usr/bin/env python3
"""フェーズ1：J-Quants からデータ取得する最小コード（接続確認用）。

README 9章のフェーズ1に対応。J-Quants Free（0円）で認証と日足取得の疎通を確認する。
Free はデータが12週間遅延するため、取得できる最新日付が数ヶ月前になるのは正常。

事前準備:
    1. https://jpx-jquants.com/ で登録（まずは Free）
    2. .env.example を .env にコピーし JQUANTS_EMAIL / JQUANTS_PASSWORD を記入
    3. pip install -r requirements.txt

実行:
    # リポジトリ直下から
    PYTHONPATH=src python scripts/fetch_jquants_minimal.py 7203
    PYTHONPATH=src python scripts/fetch_jquants_minimal.py 7203 --from 2024-01-01 --to 2024-03-31
"""

from __future__ import annotations

import argparse
import sys

from daytrade.data.jquants import JQuantsClient, JQuantsError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="J-Quants 接続確認（日足取得）")
    parser.add_argument("code", nargs="?", default="7203",
                        help="銘柄コード（例: 7203 = トヨタ）。省略時は 7203")
    parser.add_argument("--from", dest="from_date", default=None, help="開始日 YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", default=None, help="終了日 YYYY-MM-DD")
    parser.add_argument("--rows", type=int, default=10, help="表示する行数（末尾から）")
    args = parser.parse_args(argv)

    try:
        client = JQuantsClient.from_env()
        print("認証中 …")
        client.authenticate()
        print("認証成功。")

        print(f"日足を取得中: code={args.code} from={args.from_date} to={args.to_date}")
        df = client.get_daily_quotes(
            code=args.code, from_date=args.from_date, to_date=args.to_date
        )
    except JQuantsError as exc:
        print(f"[エラー] {exc}", file=sys.stderr)
        print("→ .env の認証情報、プランのデータ範囲（Free は12週間遅延）を確認してください。",
              file=sys.stderr)
        return 1

    if df.empty:
        print("データが0件でした。日付範囲や銘柄コードを見直してください。")
        print("（Free プランは直近12週間のデータが取得できない点に注意）")
        return 0

    print(f"\n取得件数: {len(df)} 行")
    cols = [c for c in ["Date", "Code", "Open", "High", "Low", "Close", "Volume"]
            if c in df.columns]
    print(df[cols].tail(args.rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
