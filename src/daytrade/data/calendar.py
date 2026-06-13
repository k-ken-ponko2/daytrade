"""取引カレンダーのユーティリティ。

- 日本株：J-Quants の取引カレンダー（/markets/trading_calendar）DataFrame から立会日を抽出。
- 米国株：NYSE の祝日ルール（pandas の祝日プリミティブ）から立会日を生成。
いずれも「価格データの欠損営業日チェック（取得漏れ検出）」に使える。
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)

# HolidayDivision: "1"=営業日, "2"=東証半日立会日（どちらも立会あり）
BUSINESS_DIVISIONS = {"1", "2"}


def _missing_against(price_index: pd.Index, sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """立会日 sessions のうち、価格データに無い日（取得漏れ疑い）を返す。

    価格データが存在する範囲内に限定して比較する（データ期間外の営業日は欠損扱いしない）。
    """
    if len(sessions) == 0:
        return pd.DatetimeIndex([])
    have = pd.DatetimeIndex(pd.to_datetime(price_index)).normalize()
    biz = pd.DatetimeIndex(sessions).normalize()
    if len(have) > 0:
        biz = biz[(biz >= have.min()) & (biz <= have.max())]
    return biz.difference(have)


# --------------------------------------------------------------------------- #
# 日本株（J-Quants カレンダー）
# --------------------------------------------------------------------------- #
def business_days(calendar: pd.DataFrame) -> pd.DatetimeIndex:
    """取引カレンダーDataFrameから立会日（半日立会含む）の DatetimeIndex を返す。"""
    if calendar.empty or "HolidayDivision" not in calendar.columns:
        return pd.DatetimeIndex([])
    div = calendar["HolidayDivision"].astype(str)
    days = calendar.loc[div.isin(BUSINESS_DIVISIONS), "Date"]
    return pd.DatetimeIndex(pd.to_datetime(days)).sort_values()


def missing_sessions(price_index: pd.Index, calendar: pd.DataFrame) -> pd.DatetimeIndex:
    """立会日のうち、価格データに存在しない日（取得漏れの疑い）を返す。"""
    return _missing_against(price_index, business_days(calendar))


# --------------------------------------------------------------------------- #
# 米国株（NYSE カレンダー）
# --------------------------------------------------------------------------- #
class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """NYSE の通年休場日（定例祝日）。

    対象は毎年繰り返す祝日のみ：元日・MLK・大統領の日・グッドフライデー・メモリアルデー・
    ジューンティーンス（2022〜）・独立記念日・レイバーデー・感謝祭・クリスマス。
    観測規則は NYSE 実務に合わせ、元日のみ「土曜は前倒ししない（日→月のみ）」、他は
    nearest_workday（土→金、日→月）。短縮立会（半日）や一度きりの臨時休場（追悼日等）は含まない。
    """

    rules = [
        # 元日は土曜でも前倒ししない（年末最終営業日を開けるため）。日曜のみ月曜へ。
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def nyse_holidays(start, end) -> pd.DatetimeIndex:
    """期間内の NYSE 休場日（定例祝日）を返す。"""
    return NYSEHolidayCalendar().holidays(pd.Timestamp(start), pd.Timestamp(end))


def nyse_sessions(start, end) -> pd.DatetimeIndex:
    """期間内の NYSE 立会日（平日 − 定例祝日）を返す。短縮立会は通常の立会日として含む。"""
    weekdays = pd.bdate_range(start, end)  # 月〜金
    return weekdays.difference(nyse_holidays(start, end))


def missing_us_sessions(price_index: pd.Index) -> pd.DatetimeIndex:
    """米国株データの欠損立会日を返す（価格データの範囲で NYSE 立会日と突き合わせ）。"""
    have = pd.DatetimeIndex(pd.to_datetime(price_index)).normalize()
    if len(have) == 0:
        return pd.DatetimeIndex([])
    sessions = nyse_sessions(have.min(), have.max())
    return _missing_against(price_index, sessions)
