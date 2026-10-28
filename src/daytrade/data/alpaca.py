"""Alpaca Market Data API（米国株の分足）クライアント。過去検証専用。

無料（Basic）プランの APIキーで、米国株の分足を長期間ぶん取得できる。発注には使わない
（実運用ブローカーは IBKR の方針。README 0章）。

データソース（feed）:
- "sip": 全米の取引所を統合したデータ（出来高100%）。既定。無料プランでは「直近15分より前」
         の過去データなら取得できる。
- "iex": IEX 取引所1か所のみ（全体の出来高の数%）。サブスクリプションなしで常に使える。
         SIP が権限エラーになる場合の代替。出来高の絶対値が小さい点に注意。

出力は tz-aware（America/New_York）の DatetimeIndex を持つ OHLCV。バックテストは
BacktestConfig.for_market(US, ...) の session_tz で NY の取引日ごとに日次リセットする。

参考: https://docs.alpaca.markets/reference/stockbars
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests

DATA_BASE = "https://data.alpaca.markets/v2"
DEFAULT_TIMEOUT = 30
MAX_LIMIT = 10_000          # 1リクエストあたりの最大バー数（API上限）
NY_TZ = "America/New_York"
REGULAR_OPEN = "09:30"
REGULAR_CLOSE = "16:00"     # この時刻のバーは含めない（15:59 が最後の通常取引バー）

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
_BAR_COLUMNS = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
                "n": "trade_count", "vw": "bar_vwap"}


class AlpacaError(RuntimeError):
    """Alpaca API 呼び出しの失敗。"""


class AlpacaClient:
    """Alpaca Market Data API の薄いラッパ（分足取得用）。

    使い方:
        client = AlpacaClient.from_env()
        raw = client.get_bars("AAPL", start="2026-09-01", end="2026-09-30")
        df = to_regular_session(raw)        # 通常取引時間のみの OHLCV（NY時間）
    """

    def __init__(self, key_id: str, secret_key: str, *,
                 session: requests.Session | None = None) -> None:
        if not (key_id and secret_key):
            raise AlpacaError("Alpaca の API Key ID と Secret Key が必要です。")
        self._headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key}
        self._session = session or requests.Session()

    @classmethod
    def from_env(cls) -> "AlpacaClient":
        """環境変数（.env 含む）の ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY から生成する。"""
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        key_id = os.getenv("ALPACA_API_KEY_ID")
        secret = os.getenv("ALPACA_API_SECRET_KEY")
        if not (key_id and secret):
            raise AlpacaError(
                "Alpaca の認証情報が見つかりません。.env に ALPACA_API_KEY_ID / "
                "ALPACA_API_SECRET_KEY を設定してください（https://alpaca.markets/ で無料発行）。"
            )
        return cls(key_id, secret)

    def get_bars(
        self,
        symbol: str,
        *,
        start: str,
        end: str,
        timeframe: str = "1Min",
        feed: str = "sip",
        adjustment: str = "split",
    ) -> pd.DataFrame:
        """1銘柄のバーを全ページ取得して返す（UTC の tz-aware index、時刻はバーの開始）。

        adjustment="split" で株式分割を遡って補正する（分割日をまたいでも SMA/ATR が壊れない）。
        start / end は "YYYY-MM-DD" または RFC-3339。どちらも含む。
        """
        params: dict[str, Any] = {
            "timeframe": timeframe, "start": start, "end": end, "feed": feed,
            "adjustment": adjustment, "limit": MAX_LIMIT,
        }
        bars: list[dict] = []
        while True:
            resp = self._session.get(
                f"{DATA_BASE}/stocks/{symbol.upper()}/bars",
                headers=self._headers, params=params, timeout=DEFAULT_TIMEOUT,
            )
            self._raise_for_status(resp, feed)
            payload = resp.json()
            bars.extend(payload.get("bars") or [])
            token = payload.get("next_page_token")
            if not token:
                break
            params["page_token"] = token

        if not bars:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df = pd.DataFrame(bars).rename(columns=_BAR_COLUMNS)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("t"), utc=True), name="datetime")
        return df[~df.index.duplicated(keep="last")].sort_index()

    @staticmethod
    def _raise_for_status(resp: requests.Response, feed: str) -> None:
        if resp.status_code < 400:
            return
        try:
            detail = resp.json().get("message", "")
        except Exception:  # noqa: BLE001 - 本文がJSONでないことがある
            detail = resp.text[:200]
        hints = {
            401: "APIキーが無効です。.env の ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY を確認してください。",
            403: ("このプランでは取得できません。直近15分を含めない期間にするか、"
                  "--feed iex を試してください。" if feed == "sip" else "権限がありません。"),
            429: "レート制限に達しました。少し待ってから再実行してください。",
        }
        raise AlpacaError(f"HTTP {resp.status_code}: {detail} {hints.get(resp.status_code, '')}".strip())


def to_regular_session(bars: pd.DataFrame, *, tz: str = NY_TZ) -> pd.DataFrame:
    """取引所時間（NY）に変換し、通常取引時間 09:30–15:59 のバーだけを残す。

    デイトレ戦略の VWAP / 当日高値 / 引け前クローズは通常取引時間を前提にしているため、
    プレ・アフターマーケットは除く。短縮立会日（半日）は 13:00 以降が時間外取引になるが、
    ここでは時刻だけで判定するので、その日は時間外のバーが残る点に注意（年に数日）。
    """
    if bars.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    local = bars.tz_convert(tz)
    regular = local.between_time(REGULAR_OPEN, REGULAR_CLOSE, inclusive="left")
    cols = OHLCV_COLUMNS + [c for c in ("trade_count", "bar_vwap") if c in regular.columns]
    out = regular[cols].astype(float).dropna(subset=OHLCV_COLUMNS)
    return out.rename_axis("datetime")
