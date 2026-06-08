"""指標計算（エンジン非依存）。

pandas / numpy のみで実装し、外部Cライブラリへの依存をなくしている。
高速化・網羅性が必要になったら、各関数の中身を pandas-ta / TA-Lib 呼び出しに
差し替えればよい（入出力の形は変えないこと）。

入力 DataFrame に期待するカラム: open, high, low, close, volume
インデックスは時刻（分足）を想定。VWAP と当日高値は「日付ごと」に集計する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _require_columns(df: pd.DataFrame) -> None:
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"必要なカラムがありません: {missing}")


def sma(series: pd.Series, period: int) -> pd.Series:
    """単純移動平均。"""
    return series.rolling(window=period, min_periods=period).mean()


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range（Wilder の平滑化）。"""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    # Wilder の平滑化 = alpha 1/period の EMA
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _session_key(index: pd.DatetimeIndex) -> pd.Index:
    """日付（立会日）ごとのグループキー。VWAP / 当日高値の日次リセットに使う。"""
    return pd.Index(index.normalize())


def vwap_intraday(df: pd.DataFrame) -> pd.Series:
    """当日始まりからの出来高加重平均価格（日付ごとに毎朝リセット）。"""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    key = _session_key(df.index)
    pv = (typical * df["volume"]).groupby(key).cumsum()
    vol = df["volume"].groupby(key).cumsum()
    return pv / vol.replace(0, np.nan)


def day_high_so_far(df: pd.DataFrame) -> pd.Series:
    """当日寄り付きからその時点までの高値（日付ごとにリセット）。"""
    key = _session_key(df.index)
    return df["high"].groupby(key).cummax()


def bar_index_intraday(df: pd.DataFrame) -> pd.Series:
    """当日の寄り付きを0としたバー連番。引け前の強制クローズ判定に使う。"""
    key = _session_key(df.index)
    return df.groupby(key).cumcount()


def compute_indicators(df: pd.DataFrame, params) -> pd.DataFrame:
    """全指標を計算して列を追加した新しい DataFrame を返す。

    params は core.types.StrategyParams を想定（期間パラメータのみ参照）。
    戻り値の列: sma_fast, sma_slow, prev_sma_fast, prev_sma_slow,
                atr, vwap, volume_avg, day_high, bar_index
    """
    _require_columns(df)
    out = df.copy()

    out["sma_fast"] = sma(out["close"], params.fast_period)
    out["sma_slow"] = sma(out["close"], params.slow_period)
    out["prev_sma_fast"] = out["sma_fast"].shift(1)
    out["prev_sma_slow"] = out["sma_slow"].shift(1)

    out["atr"] = atr(out, params.atr_period)
    out["vwap"] = vwap_intraday(out)
    out["volume_avg"] = sma(out["volume"], params.volume_avg_period)
    out["day_high"] = day_high_so_far(out)
    out["bar_index"] = bar_index_intraday(out)

    return out
