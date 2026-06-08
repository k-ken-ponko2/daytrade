"""J-Quants API クライアント（最小実装）。

認証フロー（公式）:
  1. メール＋パスワード → リフレッシュトークン (/token/auth_user)
  2. リフレッシュトークン → IDトークン (/token/auth_refresh)  ※有効期限が短い
  3. IDトークンを Authorization ヘッダに付けて各APIを叩く

Free プランでも認証と日足取得は動く（データは12週間遅延）。分足は有償アドオン。
まず Free で疎通を確認し、骨格ができてから Standard＋分足を契約する方針（README 3章）。

参考: https://jpx.gitbook.io/j-quants-ja/
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests

API_BASE = "https://api.jquants.com/v1"
DEFAULT_TIMEOUT = 30


class JQuantsError(RuntimeError):
    """J-Quants API 呼び出しの失敗。"""


class JQuantsClient:
    """J-Quants API の薄いラッパ。

    使い方:
        client = JQuantsClient.from_env()      # .env / 環境変数から認証
        df = client.get_daily_quotes("7203")   # トヨタの日足
    """

    def __init__(
        self,
        email: str | None = None,
        password: str | None = None,
        refresh_token: str | None = None,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self._email = email
        self._password = password
        self._refresh_token = refresh_token
        self._id_token: str | None = None
        self._session = session or requests.Session()

    # ------------------------------------------------------------------ #
    # 構築
    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls) -> "JQuantsClient":
        """環境変数（.env 含む）から認証情報を読み込む。

        JQUANTS_REFRESH_TOKEN があればそれを優先。なければ EMAIL/PASSWORD を使う。
        """
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass  # python-dotenv 未導入でも環境変数が直接あれば動く

        refresh_token = os.getenv("JQUANTS_REFRESH_TOKEN")
        email = os.getenv("JQUANTS_EMAIL")
        password = os.getenv("JQUANTS_PASSWORD")

        if not refresh_token and not (email and password):
            raise JQuantsError(
                "認証情報が見つかりません。.env に JQUANTS_EMAIL / JQUANTS_PASSWORD "
                "（または JQUANTS_REFRESH_TOKEN）を設定してください。"
            )
        return cls(email=email, password=password, refresh_token=refresh_token)

    # ------------------------------------------------------------------ #
    # 認証
    # ------------------------------------------------------------------ #
    def _fetch_refresh_token(self) -> str:
        if not (self._email and self._password):
            raise JQuantsError("リフレッシュトークン取得にはメールとパスワードが必要です。")
        resp = self._session.post(
            f"{API_BASE}/token/auth_user",
            json={"mailaddress": self._email, "password": self._password},
            timeout=DEFAULT_TIMEOUT,
        )
        self._raise_for_status(resp)
        token = resp.json().get("refreshToken")
        if not token:
            raise JQuantsError("リフレッシュトークンの取得に失敗しました。")
        return token

    def _fetch_id_token(self, refresh_token: str) -> str:
        resp = self._session.post(
            f"{API_BASE}/token/auth_refresh",
            params={"refreshtoken": refresh_token},
            timeout=DEFAULT_TIMEOUT,
        )
        self._raise_for_status(resp)
        token = resp.json().get("idToken")
        if not token:
            raise JQuantsError("IDトークンの取得に失敗しました。")
        return token

    def authenticate(self, force: bool = False) -> None:
        """IDトークンを取得して内部に保持する。"""
        if self._id_token and not force:
            return
        if not self._refresh_token:
            self._refresh_token = self._fetch_refresh_token()
        self._id_token = self._fetch_id_token(self._refresh_token)

    @property
    def _auth_headers(self) -> dict[str, str]:
        if not self._id_token:
            self.authenticate()
        return {"Authorization": f"Bearer {self._id_token}"}

    # ------------------------------------------------------------------ #
    # データ取得
    # ------------------------------------------------------------------ #
    def _get_paginated(self, path: str, params: dict[str, Any], key: str) -> list[dict]:
        """ページネーション対応の GET。全ページを結合して返す。"""
        results: list[dict] = []
        query = dict(params)
        while True:
            resp = self._session.get(
                f"{API_BASE}{path}",
                headers=self._auth_headers,
                params=query,
                timeout=DEFAULT_TIMEOUT,
            )
            self._raise_for_status(resp)
            payload = resp.json()
            results.extend(payload.get(key, []))
            pagination_key = payload.get("pagination_key")
            if not pagination_key:
                break
            query["pagination_key"] = pagination_key
        return results

    def get_daily_quotes(
        self,
        code: str | None = None,
        *,
        date: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> pd.DataFrame:
        """日足株価 (/prices/daily_quotes)。

        code 指定で1銘柄の期間取得、date 指定で全銘柄の1日分が取れる。
        日付は "YYYY-MM-DD" 形式。戻り値は Date 昇順の DataFrame。
        """
        params: dict[str, Any] = {}
        if code:
            params["code"] = code
        if date:
            params["date"] = date
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        rows = self._get_paginated("/prices/daily_quotes", params, key="daily_quotes")
        df = pd.DataFrame(rows)
        if not df.empty and "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
        return df

    def get_listed_info(self, code: str | None = None) -> pd.DataFrame:
        """上場銘柄一覧 (/listed/info)。code 省略で全銘柄。"""
        params = {"code": code} if code else {}
        rows = self._get_paginated("/listed/info", params, key="info")
        return pd.DataFrame(rows)

    def get_trading_calendar(
        self,
        *,
        from_date: str | None = None,
        to_date: str | None = None,
        holiday_division: str | None = None,
    ) -> pd.DataFrame:
        """取引カレンダー (/markets/trading_calendar)。

        各日付の HolidayDivision を返す:
            "0" 非営業日 / "1" 営業日 / "2" 東証半日立会日 / "3" 祝日取引あり非営業日
        戻り値は Date 昇順、Date は datetime に変換済み。
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if holiday_division:
            params["holidaydivision"] = holiday_division

        rows = self._get_paginated(
            "/markets/trading_calendar", params, key="trading_calendar"
        )
        df = pd.DataFrame(rows)
        if not df.empty and "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code >= 400:
            detail = ""
            try:
                detail = resp.json().get("message", "")
            except Exception:  # noqa: BLE001 - 本文がJSONでないことがある
                detail = resp.text[:200]
            raise JQuantsError(f"HTTP {resp.status_code}: {detail}")
