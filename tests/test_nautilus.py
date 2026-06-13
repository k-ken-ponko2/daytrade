"""NautilusTrader アダプタのテスト。

nautilus_trader 未導入の環境では自動スキップ（CI は requirements.txt のみ＝スキップ）。
ここでは Nautilus エンジンを起動せず、純粋部品の NautilusFeatureBuilder ＋ decide() の
配線を検証する。発注グルー部は Nautilus 実行コンテキストが要るため別途。
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("nautilus_trader")

from daytrade.adapters.nautilus_adapter import (  # noqa: E402
    NautilusFeatureBuilder,
    make_nautilus_strategy,
)
from daytrade.core.signals import decide, generate_signals  # noqa: E402
from daytrade.core.types import Action, StrategyParams  # noqa: E402


def _dip_then_rise_df():
    flat = np.full(26, 1000.0)
    dip = np.array([998.0, 996.0, 994.0, 994.0])
    rise = np.linspace(994.0, 1060.0, 40)
    close = np.concatenate([flat, dip, rise])
    idx = pd.date_range("2024-04-01 09:00", periods=len(close), freq="1min")
    df = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0,
                       "close": close, "volume": np.full(len(close), 700.0)}, index=idx)
    df.iloc[30:36, df.columns.get_loc("volume")] = 4000
    return df


def test_feature_builder_warms_up_then_produces_values():
    params = StrategyParams(fast_period=3, slow_period=10)
    builder = NautilusFeatureBuilder(params)
    df = _dip_then_rise_df()

    feats = []
    for ts, row in df.iterrows():
        feats.append(builder.update(row["high"], row["low"], row["close"], row["volume"],
                                    session=ts.normalize()))

    # ウォームアップ中（最初のバー）は SMA が NaN
    assert feats[0].sma_slow != feats[0].sma_slow  # NaN
    # 十分バーが進めば SMA/ATR/VWAP が実数になる
    last = feats[-1]
    assert last.sma_fast == last.sma_fast and last.atr == last.atr
    assert last.bar_index == len(df) - 1  # 全バー同一立会日なので連番が伸びる


def test_builder_plus_decide_enters_like_baseline():
    """ビルダー＋decide() の建玉タイミングが、基準実装の建玉と概ね一致する。

    SMA/出来高平均/VWAP は core と同じ定義なのでエントリー条件は揃うはず
    （ATR の平滑化方式だけは差があり得るが、エントリー判定には効かない）。
    """
    params = StrategyParams(fast_period=3, slow_period=10, max_hold_bars=500,
                            atr_stop_mult=3.0)
    df = _dip_then_rise_df()

    # Nautilus 部品経由（エンジンなし・状態は持たず建玉シグナルの有無だけ見る）
    builder = NautilusFeatureBuilder(params)
    from daytrade.core.types import PositionState
    flat = PositionState()  # 常にノーポジ前提でエントリーシグナルを拾う
    nautilus_entry_bars = []
    for i, (ts, row) in enumerate(df.iterrows()):
        f = builder.update(row["high"], row["low"], row["close"], row["volume"], ts.normalize())
        if decide(f, flat, params) == Action.ENTER_LONG:
            nautilus_entry_bars.append(i)

    # 基準実装のエントリーバー
    base = generate_signals(df, params)
    base_entry_bars = [i for i, a in enumerate(base["action"]) if a == "enter_long"]

    assert nautilus_entry_bars, "ビルダー経由でエントリーが出ていない"
    # 最初のエントリーバーが一致（指標定義が同じなので）
    assert nautilus_entry_bars[0] == base_entry_bars[0]


def test_make_nautilus_strategy_builds_classes():
    strat_cls, cfg_cls = make_nautilus_strategy(StrategyParams())
    # Strategy / StrategyConfig のサブクラスが返る（構造的に妥当）
    from nautilus_trader.trading.strategy import Strategy, StrategyConfig
    assert issubclass(strat_cls, Strategy)
    assert issubclass(cfg_cls, StrategyConfig)
