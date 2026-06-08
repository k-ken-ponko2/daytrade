"""取引カレンダーのユーティリティ。

J-Quants の取引カレンダー（/markets/trading_calendar）を使い、
- 営業日リストの取得
- 価格データの欠損営業日チェック（取得漏れの検出）
を行う。日付範囲指定や、休場日を挟んだ連続性の確認に使う。
"""

from __future__ import annotations

import pandas as pd

# HolidayDivision: "1"=営業日, "2"=東証半日立会日（どちらも立会あり）
BUSINESS_DIVISIONS = {"1", "2"}


def business_days(calendar: pd.DataFrame) -> pd.DatetimeIndex:
    """取引カレンダーDataFrameから立会日（半日立会含む）の DatetimeIndex を返す。"""
    if calendar.empty or "HolidayDivision" not in calendar.columns:
        return pd.DatetimeIndex([])
    div = calendar["HolidayDivision"].astype(str)
    days = calendar.loc[div.isin(BUSINESS_DIVISIONS), "Date"]
    return pd.DatetimeIndex(pd.to_datetime(days)).sort_values()


def missing_sessions(price_index: pd.Index, calendar: pd.DataFrame) -> pd.DatetimeIndex:
    """立会日のうち、価格データに存在しない日（取得漏れの疑い）を返す。

    price_index は OHLCV の DatetimeIndex を想定。範囲はカレンダーの範囲に限定して比較する。
    """
    biz = business_days(calendar)
    if len(biz) == 0:
        return pd.DatetimeIndex([])

    have = pd.DatetimeIndex(pd.to_datetime(price_index)).normalize()
    biz = biz.normalize()
    # 価格データが存在する範囲だけで比較（データ期間外の営業日は欠損扱いしない）
    if len(have) > 0:
        biz = biz[(biz >= have.min()) & (biz <= have.max())]
    return biz.difference(have)
