"""デモ・テスト用の合成分足データ。

J-Quants の有料分足アドオンがなくても end-to-end を動かせるよう、現実的な性質
（日中のトレンド・寄り付きの出来高増・日次のギャップ）を持つダミー分足を生成する。
あくまで配線の確認用であり、戦略の優位性検証には実データ（README 9章）を使うこと。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_intraday_ohlcv(
    n_days: int = 5,
    bars_per_day: int = 60,
    start_date: str = "2024-04-01",
    base_price: float = 1000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """合成分足 OHLCV を返す（インデックスは時刻、カラムは open/high/low/close/volume）。

    各日 09:00 始まりの bars_per_day 本。日ごとに小さなギャップとトレンドを持たせ、
    寄り付き付近と日中数カ所に出来高スパイクを入れて、エントリー条件が時々成立するようにする。
    """
    rng = np.random.default_rng(seed)
    frames = []
    price = base_price

    for d in range(n_days):
        day = pd.Timestamp(start_date) + pd.Timedelta(days=d)
        idx = pd.date_range(day + pd.Timedelta(hours=9), periods=bars_per_day, freq="1min")

        # その日のドリフト（トレンド）。日ごとに上げ下げが変わる
        drift = rng.normal(0, 0.05)
        steps = rng.normal(drift, 1.2, bars_per_day)
        close = price + np.cumsum(steps)

        spread = rng.uniform(0.3, 1.2, bars_per_day)
        high = close + spread
        low = close - spread
        open_ = np.empty(bars_per_day)
        open_[0] = close[0] - steps[0]
        open_[1:] = close[:-1]

        # 出来高：寄り付き付近を厚めにし、ランダムなスパイクを混ぜる
        volume = rng.uniform(300, 1200, bars_per_day)
        volume[:5] *= rng.uniform(2.0, 4.0)
        for _ in range(rng.integers(1, 4)):
            j = rng.integers(5, bars_per_day)
            volume[j] *= rng.uniform(3.0, 6.0)

        frames.append(pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=idx,
        ))
        # 翌日へのギャップ
        price = close[-1] + rng.normal(0, 3.0)

    df = pd.concat(frames)
    df.index.name = "datetime"
    return df
