"""指標計算のテスト。"""

import numpy as np
import pandas as pd

from daytrade.core import indicators
from daytrade.core.types import StrategyParams


def _make_intraday_df(n_per_day=30, n_days=2, seed=0):
    """2日分の分足ダミーデータ（VWAP/当日高値の日次リセット確認用）。"""
    rng = np.random.default_rng(seed)
    frames = []
    for d in range(n_days):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d) + pd.Timedelta(hours=9)
        idx = pd.date_range(start, periods=n_per_day, freq="1min")
        close = 1000 + np.cumsum(rng.normal(0, 2, n_per_day))
        high = close + rng.uniform(0, 2, n_per_day)
        low = close - rng.uniform(0, 2, n_per_day)
        frames.append(pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close,
             "volume": rng.uniform(100, 1000, n_per_day)},
            index=idx,
        ))
    return pd.concat(frames)


def test_sma_basic():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = indicators.sma(s, 3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == 2.0
    assert out.iloc[4] == 4.0


def test_atr_positive_and_warmup():
    df = _make_intraday_df()
    a = indicators.atr(df, 14)
    valid = a.dropna()
    assert (valid > 0).all()


def test_vwap_resets_each_day():
    df = _make_intraday_df()
    vwap = indicators.vwap_intraday(df)
    # 各日の最初のバーでは VWAP ≒ typical price（その日唯一の点）
    day_starts = df.groupby(df.index.normalize()).head(1).index
    for ts in day_starts:
        typical = (df.loc[ts, "high"] + df.loc[ts, "low"] + df.loc[ts, "close"]) / 3.0
        assert abs(vwap.loc[ts] - typical) < 1e-6


def test_bar_index_resets_each_day():
    df = _make_intraday_df(n_per_day=30, n_days=2)
    bidx = indicators.bar_index_intraday(df)
    # 各日 0..29 で始まり最大29
    assert bidx.groupby(df.index.normalize()).min().eq(0).all()
    assert bidx.groupby(df.index.normalize()).max().eq(29).all()


def test_compute_indicators_adds_columns():
    df = _make_intraday_df()
    out = indicators.compute_indicators(df, StrategyParams())
    for col in ["sma_fast", "sma_slow", "prev_sma_fast", "prev_sma_slow",
                "atr", "vwap", "volume_avg", "day_high", "bar_index"]:
        assert col in out.columns
