"""実データ分足の取得窓口（J-Quants 日本株 / Alpaca 米国株）＋ローカルキャッシュ。

デモ・ウォークフォワードなどのスクリプトはここだけを呼べばよい。取得結果はバックテストが
そのまま使える OHLCV（open/high/low/close/volume、DatetimeIndex）に整えて返す。

- source="jquants": J-Quants V2 分足（アドオン契約が必要）。JST の naive index。分割未調整。
- source="alpaca":  Alpaca 分足（無料）。通常取引時間のみ、NY 時間の tz-aware index。分割調整済み。

同じ銘柄・期間の再取得を避けるため、整形済みの OHLCV を data/intraday/ に保存する。
J-Quants はレート制限（Light 60回/分）と従量的な待ち時間があるので、キャッシュの効果が大きい。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCES = ("jquants", "alpaca")


def _cache_path(root: Path, source: str, symbol: str, from_date: str, to_date: str,
                feed: str | None) -> Path:
    suffix = f"_{feed}" if feed else ""
    name = f"{source}{suffix}_{symbol.upper()}_{from_date}_{to_date}.csv"
    return root / "intraday" / name


def _read_cache(path: Path, source: str) -> pd.DataFrame | None:
    """キャッシュを読む。Alpaca はオフセット付きで保存しているので NY 時間に戻す。"""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    stamps = df.pop("datetime")
    index = (pd.to_datetime(stamps, utc=True).dt.tz_convert("America/New_York")
             if source == "alpaca" else pd.to_datetime(stamps))
    df.index = pd.DatetimeIndex(index, name="datetime")
    return df


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.rename_axis("datetime").reset_index().to_csv(path, index=False)


def load_intraday(
    source: str,
    symbol: str,
    *,
    from_date: str,
    to_date: str,
    feed: str = "sip",
    cache_root: str | Path = "data",
    refresh: bool = False,
    client=None,
) -> pd.DataFrame:
    """実データの分足 OHLCV を返す（キャッシュがあればそれを使う）。

    Args:
        source:    "jquants"（日本株）か "alpaca"（米国株）。
        symbol:    銘柄コード（例 "7203"）またはティッカー（例 "AAPL"）。
        from_date / to_date: "YYYY-MM-DD"（両端を含む）。
        feed:      Alpaca のデータソース（"sip" / "iex"）。J-Quants では無視。
        refresh:   True でキャッシュを無視して再取得。
        client:    テスト用に注入するクライアント。省略時は .env から生成。
    """
    if source not in SOURCES:
        raise ValueError(f"未知のデータソース: {source}（{SOURCES} のいずれか）")

    path = _cache_path(Path(cache_root), source, symbol, from_date, to_date,
                       feed if source == "alpaca" else None)
    if not refresh:
        cached = _read_cache(path, source)
        if cached is not None and not cached.empty:
            return cached

    if source == "jquants":
        from daytrade.data.jquants import JQuantsClient, to_intraday_ohlcv

        client = client or JQuantsClient.from_env()
        raw = client.get_minute_bars(symbol, from_date=from_date, to_date=to_date)
        df = to_intraday_ohlcv(raw, code=symbol)
    else:
        from daytrade.data.alpaca import AlpacaClient, to_regular_session

        client = client or AlpacaClient.from_env()
        raw = client.get_bars(symbol, start=from_date, end=to_date, feed=feed)
        df = to_regular_session(raw)

    if not df.empty:
        _write_cache(df, path)
    return df
