"""J-Quants 連携のテスト（HTTP モック・ネットワーク/認証情報不要）。

FakeSession で API レスポンスを差し替え、認証フロー・ページネーション・整形・
キャッシュ・カレンダー・スクリーニングを検証する。CI でもそのまま回せる。
"""

import pandas as pd
import pytest

from daytrade.data.calendar import business_days, missing_sessions
from daytrade.data.jquants import JQuantsClient
from daytrade.data.loader import DataStore, to_ohlcv
from daytrade.data.screening import attach_listed_info, rank_by_turnover, screen_liquid_stocks


# --------------------------------------------------------------------------- #
# HTTP モック
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def _daily_row(date, code, close, volume, turnover):
    """daily_quotes の1行（生＋調整後）。調整後は分かりやすく生の2倍にしておく。"""
    return {
        "Date": date, "Code": code,
        "Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": volume,
        "TurnoverValue": turnover,
        "AdjustmentFactor": 1.0,
        "AdjustmentOpen": close * 2, "AdjustmentHigh": (close + 1) * 2,
        "AdjustmentLow": (close - 1) * 2, "AdjustmentClose": close * 2,
        "AdjustmentVolume": volume * 2,
    }


class FakeSession:
    def __init__(self, *, daily_pages=None, by_date=None, calendar=None, listed=None):
        self.daily_pages = daily_pages or []        # code 指定: 順に返すページ
        self.by_date = by_date or {}                # date 指定: {date: [rows]}
        self.calendar = calendar or []
        self.listed = listed or []
        self.get_calls = []
        self._daily_idx = 0

    def post(self, url, **kw):
        if url.endswith("/token/auth_user"):
            return FakeResponse(200, {"refreshToken": "refresh-token"})
        if url.endswith("/token/auth_refresh"):
            return FakeResponse(200, {"idToken": "id-token"})
        return FakeResponse(404, {"message": "unknown"})

    def get(self, url, headers=None, params=None, timeout=None):
        params = params or {}
        self.get_calls.append((url, dict(params)))

        if "/prices/daily_quotes" in url:
            if "date" in params:
                return FakeResponse(200, {"daily_quotes": self.by_date.get(params["date"], [])})
            page = self.daily_pages[self._daily_idx]
            self._daily_idx += 1
            return FakeResponse(200, page)

        if "/markets/trading_calendar" in url:
            return FakeResponse(200, {"trading_calendar": self.calendar})

        if "/listed/info" in url:
            return FakeResponse(200, {"info": self.listed})

        return FakeResponse(404, {"message": "unknown"})


def _client(session):
    return JQuantsClient(email="a@b.c", password="pw", session=session)


# --------------------------------------------------------------------------- #
# 認証・取得・ページネーション
# --------------------------------------------------------------------------- #
def test_authenticate_fetches_id_token():
    client = _client(FakeSession())
    client.authenticate()
    assert client._id_token == "id-token"
    assert client._refresh_token == "refresh-token"


def test_get_daily_quotes_paginates_and_sorts():
    pages = [
        {"daily_quotes": [_daily_row("2024-01-02", "7203", 102, 100, 1000)],
         "pagination_key": "k1"},
        {"daily_quotes": [_daily_row("2024-01-01", "7203", 101, 90, 900)]},  # 末尾・古い日付
    ]
    client = _client(FakeSession(daily_pages=pages))
    df = client.get_daily_quotes(code="7203")
    assert len(df) == 2
    # ページをまたいで結合され、Date 昇順に並ぶ
    assert list(df["Date"]) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]


def test_get_trading_calendar_parses_dates():
    cal = [{"Date": "2024-01-01", "HolidayDivision": "0"},
           {"Date": "2024-01-04", "HolidayDivision": "1"}]
    client = _client(FakeSession(calendar=cal))
    df = client.get_trading_calendar(from_date="2024-01-01", to_date="2024-01-04")
    assert len(df) == 2
    assert df["Date"].dtype.kind == "M"  # datetime


# --------------------------------------------------------------------------- #
# 整形（調整後 / 生）
# --------------------------------------------------------------------------- #
def _raw_df():
    rows = [_daily_row("2024-01-01", "7203", 100, 1000, 1_000_000),
            _daily_row("2024-01-02", "7203", 110, 1100, 1_100_000)]
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def test_to_ohlcv_adjusted_default():
    out = to_ohlcv(_raw_df())
    assert list(out.columns[:5]) == ["open", "high", "low", "close", "volume"]
    # 調整後 = 生の2倍（このフィクスチャ定義）
    assert out["close"].iloc[0] == 200
    assert out["volume"].iloc[0] == 2000
    assert "turnover" in out.columns
    assert out.index.is_monotonic_increasing


def test_to_ohlcv_raw():
    out = to_ohlcv(_raw_df(), adjusted=False)
    assert out["close"].iloc[0] == 100
    assert out["volume"].iloc[0] == 1000


def test_to_ohlcv_drops_missing_rows():
    df = _raw_df()
    df.loc[0, ["AdjustmentClose"]] = None
    out = to_ohlcv(df, adjusted=True)
    assert len(out) == 1  # 欠損行は除外


def test_to_ohlcv_falls_back_to_raw_when_no_adjusted():
    df = _raw_df().drop(columns=[c for c in _raw_df().columns if c.startswith("Adjustment")])
    out = to_ohlcv(df, adjusted=True)  # 調整後が無ければ生にフォールバック
    assert out["close"].iloc[0] == 100


# --------------------------------------------------------------------------- #
# キャッシュ DataStore
# --------------------------------------------------------------------------- #
class CountingClient:
    """get_daily_quotes の呼び出し回数を数えるダミー。"""

    def __init__(self, df):
        self._df = df
        self.calls = 0

    def get_daily_quotes(self, code=None, date=None, from_date=None, to_date=None):
        self.calls += 1
        return self._df.copy()


def test_datastore_caches_and_avoids_refetch(tmp_path):
    store = DataStore(tmp_path)
    client = CountingClient(_raw_df())

    first = store.get_daily(client, "7203")
    assert client.calls == 1 and len(first) == 2
    # 2回目はキャッシュから読むので API は呼ばれない
    second = store.get_daily(client, "7203")
    assert client.calls == 1
    assert len(second) == 2


def test_datastore_slices_cached_by_date(tmp_path):
    store = DataStore(tmp_path)
    client = CountingClient(_raw_df())
    store.get_daily(client, "7203")  # キャッシュ作成
    sliced = store.get_daily(client, "7203", from_date="2024-01-02")
    assert len(sliced) == 1
    assert sliced["Date"].iloc[0] == pd.Timestamp("2024-01-02")


def test_datastore_refresh_forces_refetch(tmp_path):
    store = DataStore(tmp_path)
    client = CountingClient(_raw_df())
    store.get_daily(client, "7203")
    store.get_daily(client, "7203", refresh=True)
    assert client.calls == 2


# --------------------------------------------------------------------------- #
# カレンダー
# --------------------------------------------------------------------------- #
def test_business_days_filters_holidays():
    cal = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-04", "2024-01-05"]),
        "HolidayDivision": ["0", "1", "2"],  # 非営業 / 営業 / 半日立会
    })
    biz = business_days(cal)
    assert list(biz) == [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")]


def test_missing_sessions_detects_gap():
    cal = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-09"]),
        "HolidayDivision": ["1", "1", "1"],
    })
    price_index = pd.to_datetime(["2024-01-04", "2024-01-09"])  # 01-05 が欠損
    missing = missing_sessions(price_index, cal)
    assert list(missing) == [pd.Timestamp("2024-01-05")]


# --------------------------------------------------------------------------- #
# スクリーニング
# --------------------------------------------------------------------------- #
def test_rank_by_turnover_orders_and_filters():
    day1 = pd.DataFrame({"Code": ["A", "B", "C"], "TurnoverValue": [300, 200, 100]})
    day2 = pd.DataFrame({"Code": ["A", "B", "C"], "TurnoverValue": [100, 200, 50]})
    ranked = rank_by_turnover([day1, day2], top_n=2, min_turnover=120)
    # 平均: A=200, B=200, C=75 → C は min 未満で除外、top_n=2
    assert list(ranked["Code"]) == ["A", "B"]
    assert (ranked["avg_turnover"] >= 120).all()
    assert (ranked["days_count"] == 2).all()


def test_attach_listed_info_merges_names():
    ranked = pd.DataFrame({"Code": ["A"], "avg_turnover": [200.0], "days_count": [2]})
    listed = pd.DataFrame({"Code": ["A"], "CompanyName": ["AlphaCorp"],
                           "MarketCodeName": ["Prime"]})
    merged = attach_listed_info(ranked, listed)
    assert merged.loc[0, "CompanyName"] == "AlphaCorp"


def test_screen_liquid_stocks_end_to_end():
    by_date = {
        "2024-01-15": [_daily_row("2024-01-15", "7203", 100, 1000, 5_000_000),
                       _daily_row("2024-01-15", "6758", 200, 500, 1_000_000)],
        "2024-01-16": [_daily_row("2024-01-16", "7203", 101, 1100, 6_000_000),
                       _daily_row("2024-01-16", "6758", 201, 400, 900_000)],
    }
    listed = [{"Code": "7203", "CompanyName": "Toyota", "MarketCodeName": "Prime"},
              {"Code": "6758", "CompanyName": "Sony", "MarketCodeName": "Prime"}]
    client = _client(FakeSession(by_date=by_date, listed=listed))
    ranked = screen_liquid_stocks(client, ["2024-01-15", "2024-01-16"], top_n=5)
    assert list(ranked["Code"]) == ["7203", "6758"]  # トヨタの方が代金大
    assert ranked.loc[0, "CompanyName"] == "Toyota"


def test_to_ohlcv_empty():
    assert to_ohlcv(pd.DataFrame()).empty
