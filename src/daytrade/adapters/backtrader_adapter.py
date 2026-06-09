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

import math

from daytrade.core.risk import position_size
from daytrade.core.signals import decide, open_position, update_trailing_stop
from daytrade.core.types import Action, Features, PositionState, StrategyParams


def make_strategy(params: StrategyParams, *, risk_fraction: float = 0.05, lot_size: int = 100):
    """StrategyParams を束ねた Backtrader Strategy クラスを返すファクトリ。

    backtrader を遅延 import するため、関数内で定義している。
    指標は backtrader 組み込みの indicator を使い、Features に詰めて decide() に渡す。
    株数は基準実装と同じ core.risk.position_size で決める（二重検証のズレを抑えるため）。
    トレーリング・段階利確も基準実装と同じ core 関数（open_position / update_trailing_stop）
    を使い、両エンジンの差を「エンジンの差」だけに絞る。
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

            # トレーリングストップの状態更新（基準実装と同じ core 関数）
            if self.state.is_open:
                update_trailing_stop(self.state, f, self.p_params)

            action = decide(f, self.state, self.p_params)

            if action == Action.ENTER_LONG and not self.position:
                prospective = open_position(f.close, f.atr, self.p_params)
                size = position_size(
                    self.broker.getcash(), f.close, prospective.stop_price,
                    risk_fraction=risk_fraction, lot_size=lot_size,
                )
                if size > 0:
                    self.state = prospective
                    self.buy(size=size)
            elif action == Action.SCALE_OUT and self.position:
                held = self.position.size
                sc = int(math.floor(held * self.p_params.scale_out_fraction / lot_size) * lot_size)
                if 0 < sc < held:
                    self.sell(size=sc)
                self.state.scaled_out = True
                self.state.bars_held += 1
            elif action == Action.EXIT and self.position:
                self.close()
                self.state = PositionState()
            elif self.position:
                self.state.bars_held += 1

    return _Strategy
