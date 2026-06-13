"""海外株（米国株）の過去データ取得。

J-Quants は日本株専用なので、海外株は別ソースを使う。ここでは無料・APIキー不要の
Stooq（日足CSV）と、任意のCSVを取り込む汎用ローダを用意する。出力はバックテストが
そのまま使える OHLCV（open/high/low/close/volume、DatetimeIndex）。

実運用のリアルタイム/発注は別途ブローカーAPI（IBKR / Alpaca 等）が必要で、ここは
過去検証用。ネットワーク非依存に保つため requests.Session を注入可能にし、モックでテストする。

Stooq 例: https://stooq.com/q/d/l/?s=aapl.us&i=d  （米国株はシンボルに .us を付ける）
"""

from __future__ import annotations

import io

import pandas as pd
import requests

STOOQ_URL = "https://stooq.com/q/d/l/"
DEFAULT_TIMEOUT = 30

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Date/Open/.../Volume を持つ表を標準 OHLCV（DatetimeIndex）に整える。"""
    if df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols:
        raise ValueError("Date 列が見つかりません。")
    out = pd.DataFrame(index=pd.to_datetime(df[cols["date"]]))
    for std in OHLCV_COLUMNS:
        if std not in cols:
            raise ValueError(f"{std} 列が見つかりません。")
        out[std] = pd.to_numeric(df[cols[std]].values, errors="coerce")
    out.index.name = "datetime"
    return out.sort_index().dropna(subset=OHLCV_COLUMNS)


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """ローカルCSV（Date,Open,High,Low,Close,Volume）を OHLCV として読み込む。

    証券会社や任意サイトからエクスポートした日足CSVをそのまま検証に使える汎用入口。
    """
    return _normalize(pd.read_csv(path))


class StooqClient:
    """Stooq の無料日足CSVクライアント（APIキー不要）。

    使い方:
        df = StooqClient().get_daily("AAPL")          # 米国株は自動で .us を付与
        df = StooqClient().get_daily("aapl.us", from_date="2023-01-01")
    """

    def __init__(self, *, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    @staticmethod
    def _symbol(symbol: str, market: str) -> str:
        """Stooq 形式のシンボルに整える（米国株はサフィックス .us）。"""
        s = symbol.lower()
        if "." in s:
            return s
        return f"{s}.{market.lower()}"

    def get_daily(
        self,
        symbol: str,
        *,
        market: str = "us",
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> pd.DataFrame:
        """日足 OHLCV を取得して返す。Stooq は EOD（場中遅延あり）。"""
        params = {"s": self._symbol(symbol, market), "i": "d"}
        if from_date:
            params["d1"] = from_date.replace("-", "")
        if to_date:
            params["d2"] = to_date.replace("-", "")
        resp = self._session.get(STOOQ_URL, params=params, timeout=DEFAULT_TIMEOUT)
        if resp.status_code >= 400:
            raise RuntimeError(f"Stooq HTTP {resp.status_code}")
        text = resp.text.strip()
        # 取得不能時は "No data" 等のテキストが返る
        if not text or "," not in text.splitlines()[0]:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        return _normalize(pd.read_csv(io.StringIO(text)))
