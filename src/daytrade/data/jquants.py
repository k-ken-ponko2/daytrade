"""J-Quants API クライアント（V2・APIキー方式）。

V1（メール＋パスワード → リフレッシュトークン → IDトークン）は 2026-06-01 に廃止された。
V2 はダッシュボードで発行した APIキーを `x-api-key` ヘッダに付けるだけで使える。

V2 はレスポンスのカラム名が短縮されている（Open→O、AdjustmentClose→AdjC など）。
このクライアントは取得結果を **V1 と同じ長いカラム名に正規化して返す**。こうすることで
loader / calendar / screening やキャッシュ済みデータはそのまま使い続けられる。

分足（/equities/bars/minute）は「株価 分足・ティック」アドオン（Light プラン以上）が必要。
分足は分割未調整・JST・約定のない分は欠落、という仕様なので to_intraday_ohlcv() で整える。

参考: https://jpx-jquants.com/ja/spec
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests

API_BASE = "https://api.jquants.com/v2"
DEFAULT_TIMEOUT = 30

# V2 の短縮名 → 既存コードが使う V1 の長い名前
_DAILY_COLUMNS = {
    "O": "Open", "H": "High", "L": "Low", "C": "Close",
    "Vo": "Volume", "Va": "TurnoverValue",
    "AdjFactor": "AdjustmentFactor",
    "AdjO": "AdjustmentOpen", "AdjH": "AdjustmentHigh", "AdjL": "AdjustmentLow",
    "AdjC": "AdjustmentClose", "AdjVo": "AdjustmentVolume",
}
_MINUTE_COLUMNS = {
    "O": "Open", "H": "High", "L": "Low", "C": "Close",
    "Vo": "Volume", "Va": "TurnoverValue",
}
_CALENDAR_COLUMNS = {"HolDiv": "HolidayDivision"}
_MASTER_COLUMNS = {
    "CoName": "CompanyName", "CoNameEn": "CompanyNameEnglish",
    "S17": "Sector17Code", "S17Nm": "Sector17CodeName",
    "S33": "Sector33Code", "S33Nm": "Sector33CodeName",
    "ScaleCat": "ScaleCategory", "Mkt": "MarketCode", "MktNm": "MarketCodeName",
    "Mrgn": "MarginCode", "MrgnNm": "MarginCodeName",
}

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class JQuantsError(RuntimeError):
    """J-Quants API 呼び出しの失敗。"""


def _date_param(value: str | None) -> str | None:
    """'YYYY-MM-DD' / 'YYYYMMDD' のどちらでも受ける（V2 は両方可）。"""
    return value or None


def _to_frame(rows: list[dict], columns: dict[str, str], sort_by: list[str]) -> pd.DataFrame:
    """レスポンス行を DataFrame にし、カラム名を正規化して日付で昇順に並べる。"""
    df = pd.DataFrame(rows).rename(columns=columns)
    if df.empty or "Date" not in df.columns:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    keys = [k for k in sort_by if k in df.columns]
    return df.sort_values(keys).reset_index(drop=True)


class JQuantsClient:
    """J-Quants API V2 の薄いラッパ。

    使い方:
        client = JQuantsClient.from_env()            # .env の JQUANTS_API_KEY を使う
        df = client.get_daily_quotes("7203")         # 日足（カラム名は V1 互換）
        mdf = client.get_minute_bars("7203", from_date="2026-09-01", to_date="2026-09-05")
    """

    def __init__(self, api_key: str, *, session: requests.Session | None = None) -> None:
        if not api_key:
            raise JQuantsError("APIキーが空です。")
        self._api_key = api_key
        self._session = session or requests.Session()

    # ------------------------------------------------------------------ #
    # 構築
    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls) -> "JQuantsClient":
        """環境変数（.env 含む）の JQUANTS_API_KEY から生成する。"""
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass  # python-dotenv 未導入でも環境変数が直接あれば動く

        api_key = os.getenv("JQUANTS_API_KEY")
        if not api_key:
            hint = ""
            if os.getenv("JQUANTS_EMAIL") or os.getenv("JQUANTS_REFRESH_TOKEN"):
                hint = "（メール/パスワード・リフレッシュトークン方式の V1 は 2026-06-01 に廃止済み）"
            raise JQuantsError(
                "APIキーが見つかりません。J-Quants のダッシュボード「設定 » APIキー」で発行し、"
                f".env に JQUANTS_API_KEY を設定してください。{hint}"
            )
        return cls(api_key)

    def authenticate(self) -> None:
        """疎通確認。V2 はトークン発行が不要なので、APIキーでカレンダーを1週間分だけ取得してみる。

        Free プランは「直近12週間を除く2年分」なので、約半年前の日付を使う。
        """
        start = pd.Timestamp.today().normalize() - pd.Timedelta(days=180)
        self._get_paginated("/markets/calendar", {
            "from": start.strftime("%Y-%m-%d"),
            "to": (start + pd.Timedelta(days=6)).strftime("%Y-%m-%d"),
        })

    # ------------------------------------------------------------------ #
    # データ取得
    # ------------------------------------------------------------------ #
    def _get_paginated(self, path: str, params: dict[str, Any]) -> list[dict]:
        """ページネーション対応の GET。V2 は全エンドポイントで data / pagination_key。"""
        results: list[dict] = []
        query = {k: v for k, v in params.items() if v is not None}
        while True:
            resp = self._session.get(
                f"{API_BASE}{path}",
                headers={"x-api-key": self._api_key},
                params=query,
                timeout=DEFAULT_TIMEOUT,
            )
            self._raise_for_status(resp)
            payload = resp.json()
            results.extend(payload.get("data", []))
            pagination_key = payload.get("pagination_key")
            if not pagination_key:
                return results
            query["pagination_key"] = pagination_key

    def get_daily_quotes(
        self,
        code: str | None = None,
        *,
        date: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> pd.DataFrame:
        """日足株価 (/equities/bars/daily)。

        code 指定で1銘柄の期間取得、date 指定で全銘柄の1日分が取れる。
        戻り値は Date 昇順。カラム名は V1 互換（Open / AdjustmentClose / TurnoverValue …）。
        """
        rows = self._get_paginated("/equities/bars/daily", {
            "code": code, "date": _date_param(date),
            "from": _date_param(from_date), "to": _date_param(to_date),
        })
        return _to_frame(rows, _DAILY_COLUMNS, ["Date", "Code"])

    def get_minute_bars(
        self,
        code: str | None = None,
        *,
        date: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> pd.DataFrame:
        """分足 (/equities/bars/minute)。「株価 分足・ティック」アドオンが必要。

        code か date のどちらかが必須。保存期間は過去2年。価格は分割未調整。
        戻り値は Date / Time / Code / Open / High / Low / Close / Volume / TurnoverValue。
        V2 は行の順序を保証しないので、Date→Time で並べ替えて返す。
        """
        if not code and not date:
            raise JQuantsError("分足の取得には code か date の指定が必要です。")
        rows = self._get_paginated("/equities/bars/minute", {
            "code": code, "date": _date_param(date),
            "from": _date_param(from_date), "to": _date_param(to_date),
        })
        return _to_frame(rows, _MINUTE_COLUMNS, ["Code", "Date", "Time"])

    def get_listed_info(self, code: str | None = None, *, date: str | None = None) -> pd.DataFrame:
        """上場銘柄一覧 (/equities/master)。code 省略で全銘柄。"""
        rows = self._get_paginated("/equities/master", {"code": code, "date": _date_param(date)})
        return pd.DataFrame(rows).rename(columns=_MASTER_COLUMNS)

    def get_trading_calendar(
        self,
        *,
        from_date: str | None = None,
        to_date: str | None = None,
        holiday_division: str | None = None,
    ) -> pd.DataFrame:
        """取引カレンダー (/markets/calendar)。

        各日付の HolidayDivision（V2 では HolDiv）を返す:
            "0" 非営業日 / "1" 営業日 / "2" 東証半日立会日 / "3" 祝日取引あり非営業日
        """
        rows = self._get_paginated("/markets/calendar", {
            "from": _date_param(from_date), "to": _date_param(to_date),
            "hol_div": holiday_division,
        })
        return _to_frame(rows, _CALENDAR_COLUMNS, ["Date"])

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            detail = resp.json().get("message", "")
        except Exception:  # noqa: BLE001 - 本文がJSONでないことがある
            detail = resp.text[:200]
        hints = {
            401: "APIキーが無効です。ダッシュボードで再発行してください。",
            403: "契約プランでは利用できないデータです（分足はアドオン契約が必要）。",
            429: "レート制限に達しました。少し待ってから再実行してください。",
        }
        hint = hints.get(resp.status_code, "")
        raise JQuantsError(f"HTTP {resp.status_code}: {detail} {hint}".strip())


def to_intraday_ohlcv(minute: pd.DataFrame, *, code: str | None = None) -> pd.DataFrame:
    """get_minute_bars() の結果をバックテスト用 OHLCV（JSTの naive DatetimeIndex）に整える。

    - index は「日付＋バー開始時刻」。naive のまま返すと、指標計算は取引所ローカル時刻として
      日付ごとに VWAP / 当日高値 / バー連番をリセットする（core.indicators）。
    - 複数銘柄が混ざっている場合は code で1銘柄に絞る（4桁指定なら末尾0を補う）。
    - 約定のない分は J-Quants 側で欠落している。埋めずにそのまま使う（出来高0の偽バーを作らない）。
    - 価格は分割未調整。分割日をまたぐ期間では SMA/ATR が不連続になるので期間を区切って使う。
    """
    if minute.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    df = minute
    if code is not None and "Code" in df.columns:
        target = code if len(code) == 5 else f"{code}0"
        df = df[df["Code"].astype(str) == target]
        if df.empty:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

    stamp = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d") + " " + df["Time"].astype(str)
    out = pd.DataFrame(
        {std: pd.to_numeric(df[src].values, errors="coerce")
         for std, src in (("open", "Open"), ("high", "High"), ("low", "Low"),
                          ("close", "Close"), ("volume", "Volume"))},
        index=pd.DatetimeIndex(pd.to_datetime(stamp.values), name="datetime"),
    )
    if "TurnoverValue" in df.columns:
        out["turnover"] = pd.to_numeric(df["TurnoverValue"].values, errors="coerce")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=OHLCV_COLUMNS)
