"""米国株対応：タイムゾーン別の当日リセットと NYSE カレンダーのテスト。"""

import numpy as np
import pandas as pd

from daytrade.core.indicators import (
    bar_index_intraday,
    compute_indicators,
    vwap_intraday,
)
from daytrade.core.types import StrategyParams
from daytrade.data.calendar import (
    missing_us_sessions,
    nyse_holidays,
    nyse_sessions,
)


# --------------------------------------------------------------------------- #
# タイムゾーン別の当日リセット
# --------------------------------------------------------------------------- #
def _us_utc_df():
    """UTC インデックスの米国株分足。ET の取引日をまたぐ時刻を含める。

    2024-03-05 の延長時間も含め、UTC では 00〜01時台＝ET 前日夜になる点を突く。
    具体的に：2024-03-06 00:30 UTC = 2024-03-05 19:30 ET（ET では 3/5）。
    """
    ts = pd.to_datetime([
        "2024-03-05 18:00",  # 13:00 ET 3/5
        "2024-03-05 20:30",  # 15:30 ET 3/5
        "2024-03-06 00:30",  # 19:30 ET 3/5（UTCでは日付が 3/6 に変わる）
        "2024-03-06 14:30",  # 09:30 ET 3/6（翌取引日の寄り）
        "2024-03-06 18:00",  # 13:00 ET 3/6
    ]).tz_localize("UTC")
    return pd.DataFrame({
        "open": [100, 101, 102, 110, 111], "high": [101, 102, 103, 111, 112],
        "low": [99, 100, 101, 109, 110], "close": [100.5, 101.5, 102.5, 110.5, 111.5],
        "volume": [1000, 1000, 1000, 1000, 1000],
    }, index=ts)


def test_bar_index_resets_on_et_trading_day_not_utc():
    df = _us_utc_df()
    # UTC 基準だと 3/6 00:30 が翌日扱いになり区切りがズレる
    utc_idx = bar_index_intraday(df)            # tz 指定なし＝UTC日付で区切り
    et_idx = bar_index_intraday(df, "America/New_York")
    # ET 基準では最初の3本が同一取引日(3/5)で 0,1,2、次の2本(3/6)が 0,1
    assert list(et_idx) == [0, 1, 2, 0, 1]
    # UTC 基準では 00:30 UTC で日付が変わり 0,1,0,... とズレる（ET と異なる）
    assert list(utc_idx) != list(et_idx)


def test_vwap_resets_on_et_trading_day():
    df = _us_utc_df()
    vwap = vwap_intraday(df, "America/New_York")
    # ET 3/6 の最初のバー（index 3）では VWAP ≒ そのバーの typical price（当日初）
    typical = (df["high"].iloc[3] + df["low"].iloc[3] + df["close"].iloc[3]) / 3.0
    assert abs(vwap.iloc[3] - typical) < 1e-9


def test_compute_indicators_accepts_session_tz():
    df = _us_utc_df()
    out = compute_indicators(df, StrategyParams(fast_period=2, slow_period=3),
                             session_tz="America/New_York")
    assert "bar_index" in out.columns
    assert list(out["bar_index"]) == [0, 1, 2, 0, 1]


def test_naive_index_unaffected_by_tz():
    # naive（取引所ローカル想定）なデータは tz 指定でも挙動が変わらない
    idx = pd.date_range("2024-04-01 09:30", periods=4, freq="1min")
    df = pd.DataFrame({"high": [1, 2, 3, 4], "low": [1, 1, 1, 1],
                      "close": [1, 2, 3, 4], "volume": [1, 1, 1, 1]}, index=idx)
    a = bar_index_intraday(df)
    b = bar_index_intraday(df, "America/New_York")
    assert list(a) == list(b) == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# NYSE カレンダー
# --------------------------------------------------------------------------- #
def test_nyse_2024_holidays():
    hols = set(nyse_holidays("2024-01-01", "2024-12-31").normalize())
    expected = {pd.Timestamp(d) for d in [
        "2024-01-01",  # 元日
        "2024-01-15",  # MLK
        "2024-02-19",  # 大統領の日
        "2024-03-29",  # グッドフライデー
        "2024-05-27",  # メモリアルデー
        "2024-06-19",  # ジューンティーンス
        "2024-07-04",  # 独立記念日
        "2024-09-02",  # レイバーデー
        "2024-11-28",  # 感謝祭
        "2024-12-25",  # クリスマス
    ]}
    assert hols == expected


def test_new_years_saturday_not_observed_on_friday():
    # 2022-01-01 は土曜 → NYSE は前倒し休場しない（前年12/31は立会）
    hols = nyse_holidays("2021-12-25", "2022-01-05").normalize()
    assert pd.Timestamp("2021-12-31") not in hols


def test_christmas_saturday_observed_on_friday():
    # 2021-12-25 は土曜 → クリスマスは前倒しで 12/24 休場
    hols = nyse_holidays("2021-12-20", "2021-12-31").normalize()
    assert pd.Timestamp("2021-12-24") in hols


def test_nyse_sessions_excludes_weekends_and_holidays():
    sessions = nyse_sessions("2024-07-01", "2024-07-07").normalize()
    # 7/4(木)休場・7/6,7/7は週末。立会は 1,2,3,5 日
    assert pd.Timestamp("2024-07-04") not in sessions
    assert pd.Timestamp("2024-07-05") in sessions
    assert pd.Timestamp("2024-07-06") not in sessions   # 土
    assert pd.Timestamp("2024-07-01") in sessions


def test_missing_us_sessions_detects_gap():
    # 7/1,7/2,7/5 を持ち、7/3 が欠損（7/4は休場なので欠損扱いしない）
    have = pd.to_datetime(["2024-07-01", "2024-07-02", "2024-07-05"])
    missing = missing_us_sessions(have)
    assert list(missing) == [pd.Timestamp("2024-07-03")]
