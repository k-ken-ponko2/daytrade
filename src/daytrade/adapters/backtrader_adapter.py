"""Backtrader アダプタ（骨格）。

共通ロジック core.signals.decide() を Backtrader の Strategy から呼ぶだけにする。
backtrader をインストール（requirements.txt のコメントを外す）してから使う。

使い方の例:
    import backtrader as bt
    from daytrade.adapters.backtrader_adapter import make_strategy
    from daytrade.core.types import StrategyParams

    cerebro = bt.Cerebro()
    cerebro.addstrategy(make_strategy(StrategyParams(force_close_bar=290)))
    cerebro.broker.setcommission(commission=0.0005)  # 手数料は必ず入れる
    # data feed を adddata して run()
"""

from __future__ import annotations

from daytrade.core.indicators import atr as _atr_series  # noqa: F401  (将来利用)
from daytrade.core.signals import decide, initial_stops
from daytrade.core.types import Action, Features, PositionState, StrategyParams


def make_strategy(params: StrategyParams):
    """StrategyParams を束ねた Backtrader Strategy クラスを返すファクトリ。

    backtrader を遅延 import するため、関数内で定義している。
    指標は backtrader 組み込みの indicator を使い、Features に詰めて decide() に渡す。
    """
    import backtrader as bt

    class _Strategy(bt.Strategy):
        def __init__(self) -> None:
            self.p_params = params
            self.sma_fast = bt.ind.SMA(self.data.close, period=params.fast_period)
            self.sma_slow = bt.ind.SMA(self.data.close, period=params.slow_period)
            self.atr = bt.ind.ATR(self.data, period=params.atr_period)
            self.vol_avg = bt.ind.SMA(self.data.volume, period=params.volume_avg_period)
            # VWAP / day_high / bar_index は当日リセットが必要。
            # ここでは簡略化のため day_high を全体高値で近似している。
            # 実装を詰める際は core.indicators の日次集計に合わせること。
            self.state = PositionState()
            self._session_day = None
            self._bar_index = -1
            self._cum_pv = 0.0
            self._cum_vol = 0.0
            self._day_high = float("-inf")

        def _update_session(self) -> None:
            day = self.data.datetime.date(0)
            if day != self._session_day:
                self._session_day = day
                self._bar_index = 0
                self._cum_pv = 0.0
                self._cum_vol = 0.0
                self._day_high = float("-inf")
            else:
                self._bar_index += 1
            typical = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3.0
            self._cum_pv += typical * self.data.volume[0]
            self._cum_vol += self.data.volume[0]
            self._day_high = max(self._day_high, self.data.high[0])

        def _features(self) -> Features:
            vwap = self._cum_pv / self._cum_vol if self._cum_vol else float("nan")
            return Features(
                close=self.data.close[0],
                high=self.data.high[0],
                low=self.data.low[0],
                volume=self.data.volume[0],
                sma_fast=self.sma_fast[0],
                sma_slow=self.sma_slow[0],
                prev_sma_fast=self.sma_fast[-1],
                prev_sma_slow=self.sma_slow[-1],
                vwap=vwap,
                atr=self.atr[0],
                volume_avg=self.vol_avg[0],
                day_high=self._day_high,
                bar_index=self._bar_index,
            )

        def next(self) -> None:
            self._update_session()
            self.state.is_open = bool(self.position)
            f = self._features()
            action = decide(f, self.state, self.p_params)

            if action == Action.ENTER_LONG and not self.position:
                stop, take = initial_stops(f.close, f.atr, self.p_params)
                self.state = PositionState(
                    is_open=True, entry_price=f.close,
                    stop_price=stop, take_price=take, bars_held=0,
                )
                self.buy()
            elif action == Action.EXIT and self.position:
                self.close()
                self.state = PositionState()
            elif self.position:
                self.state.bars_held += 1

    return _Strategy
