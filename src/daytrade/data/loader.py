"""J-Quants データの整形とローカルキャッシュ。

- to_ohlcv():  J-Quants の日足レスポンスをバックテスト用の OHLCV に整形する。
               調整後（株式分割等を遡って補正・連続）と生のどちらかを選べる。既定は調整後。
- DataStore:   取得した「生のレスポンス」をローカルに保存し、再取得を避ける。
               parquet が使えれば parquet、無ければ CSV にフォールバックする。

設計：キャッシュには J-Quants の生レスポンス（全カラム）を保存し、調整後／生の選択は
読み出し時の to_ohlcv() で行う。こうすると後から方針を変えても再取得が不要。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# バックテストが期待する標準カラム
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

_RAW_MAP = {
    "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
}
_ADJ_MAP = {
    "open": "AdjustmentOpen", "high": "AdjustmentHigh", "low": "AdjustmentLow",
    "close": "AdjustmentClose", "volume": "AdjustmentVolume",
}


def to_ohlcv(raw: pd.DataFrame, *, adjusted: bool = True, keep_turnover: bool = True) -> pd.DataFrame:
    """J-Quants の daily_quotes レスポンスを OHLCV(DataFrame) に整形する。

    Args:
        raw:           get_daily_quotes() が返す DataFrame（Date / Open / ... / Adjustment*）。
        adjusted:      True で調整後価格（既定・指標/バックテスト向き）、False で生の約定価格。
        keep_turnover: True なら売買代金(turnover)列も残す（流動性の目安）。

    Returns:
        Date を昇順の DatetimeIndex にした open/high/low/close/volume(+turnover) の DataFrame。
        値が欠損する行（その日売買なし等）は除外する。
    """
    if raw.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    colmap = _ADJ_MAP if adjusted else _RAW_MAP
    missing = [src for src in colmap.values() if src not in raw.columns]
    if missing:
        # 調整後列が無いプラン/レスポンスのときは生にフォールバック
        if adjusted and all(v in raw.columns for v in _RAW_MAP.values()):
            colmap = _RAW_MAP
        else:
            raise ValueError(f"必要な価格カラムがありません: {missing}")

    out = pd.DataFrame(index=pd.to_datetime(raw["Date"]))
    for std, src in colmap.items():
        out[std] = pd.to_numeric(raw[src].values, errors="coerce")

    if keep_turnover and "TurnoverValue" in raw.columns:
        out["turnover"] = pd.to_numeric(raw["TurnoverValue"].values, errors="coerce")

    out.index.name = "datetime"
    out = out.sort_index()
    return out.dropna(subset=OHLCV_COLUMNS)


# --------------------------------------------------------------------------- #
# ローカルキャッシュ
# --------------------------------------------------------------------------- #
def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


class DataStore:
    """J-Quants の生レスポンスを銘柄ごとにローカル保存するキャッシュ。

    使い方:
        store = DataStore()                       # 既定: ./data 配下
        raw = store.get_daily(client, "7203")     # 無ければ取得して保存、あれば読むだけ
        df = to_ohlcv(raw, adjusted=True)         # バックテスト用に整形
    """

    def __init__(self, root: str | Path = "data") -> None:
        self.root = Path(root)
        self._ext = "parquet" if _parquet_available() else "csv"

    def _path(self, code: str, kind: str = "daily") -> Path:
        return self.root / kind / f"{code}.{self._ext}"

    # --- 低レベル I/O --- #
    def _read(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df

    def _write(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)

    # --- 公開 API --- #
    def load_raw(self, code: str, kind: str = "daily") -> pd.DataFrame | None:
        """キャッシュ済みの生レスポンスを返す。無ければ None。"""
        return self._read(self._path(code, kind))

    def save_raw(self, code: str, df: pd.DataFrame, kind: str = "daily") -> None:
        self._write(df, self._path(code, kind))

    def get_daily(
        self,
        client,
        code: str,
        *,
        from_date: str | None = None,
        to_date: str | None = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """日足の生レスポンスを返す。キャッシュがあればそれを使い、無ければ取得して保存する。

        refresh=True で強制再取得。free は12週間遅延のため、最新日付は数ヶ月前になる。
        """
        if not refresh:
            cached = self.load_raw(code)
            if cached is not None and not cached.empty:
                return self._slice(cached, from_date, to_date)

        fetched = client.get_daily_quotes(code=code, from_date=from_date, to_date=to_date)
        if not fetched.empty:
            self.save_raw(code, fetched)
        return fetched

    @staticmethod
    def _slice(df: pd.DataFrame, from_date: str | None, to_date: str | None) -> pd.DataFrame:
        if "Date" not in df.columns:
            return df
        mask = pd.Series(True, index=df.index)
        if from_date:
            mask &= df["Date"] >= pd.Timestamp(from_date)
        if to_date:
            mask &= df["Date"] <= pd.Timestamp(to_date)
        return df.loc[mask].reset_index(drop=True)
