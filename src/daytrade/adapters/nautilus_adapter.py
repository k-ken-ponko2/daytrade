"""NautilusTrader アダプタ。

Backtrader アダプタと同じ方針で、売買判定そのものは共通ロジック core.signals.decide()
に委譲する。Nautilus 固有の責務は (1) バー → Features の組み立て、(2) Action → 注文 への
変換、の2点だけに絞る（README 5章「単一の真実」）。

構成:
- NautilusFeatureBuilder: Nautilus の指標（SMA/ATR）と当日リセット系（VWAP/当日高値/バー連番）
  を内部に持ち、1バーから Features を作る純粋な部品。エンジン非依存なので単体テスト可能。
- make_nautilus_strategy(): on_bar() で上記ビルダーと decide() を使う Strategy を返すファクトリ。
  注文発注部（buy/sell/close）は Nautilus の実行コンテキストが要るため、ここを薄く保つ。

nautilus_trader は重い依存のため遅延 import。未導入でも core 層のテストは動く。
"""

from __future__ import annotations

import math

from daytrade.core.signals import decide, open_position, update_trailing_stop
from daytrade.core.types import Action, Features, PositionState, StrategyParams

_NAN = float("nan")


class NautilusFeatureBuilder:
    """Nautilus のバー列から Features を組み立てる（純粋・エンジン非依存）。

    指標は Nautilus 組み込みの SimpleMovingAverage / AverageTrueRange を使い、ウォームアップ
    未完了の指標は NaN を返す（decide() 側で除外される）。VWAP・当日高値・バー連番は
    立会日ごとにリセットする（core.indicators の日次集計と定義を揃える）。
    """

    def __init__(self, params: StrategyParams) -> None:
        from nautilus_trader.indicators.averages import SimpleMovingAverage
        from nautilus_trader.indicators.volatility import AverageTrueRange

        self.params = params
        self._sma_fast = SimpleMovingAverage(params.fast_period)
        self._sma_slow = SimpleMovingAverage(params.slow_period)
        self._atr = AverageTrueRange(params.atr_period)
        self._vol_avg = SimpleMovingAverage(params.volume_avg_period)

        self._prev_fast = _NAN
        self._prev_slow = _NAN

        self._session = None
        self._bar_index = -1
        self._cum_pv = 0.0
        self._cum_vol = 0.0
        self._day_high = float("-inf")

    @staticmethod
    def _val(indicator) -> float:
        return indicator.value if indicator.initialized else _NAN

    def update(self, high: float, low: float, close: float, volume: float, session) -> Features:
        """1バーを取り込み、その時点の Features を返す。session は立会日（日付）の識別子。"""
        # 当日リセット（VWAP / 当日高値 / バー連番）
        if session != self._session:
            self._session = session
            self._bar_index = 0
            self._cum_pv = 0.0
            self._cum_vol = 0.0
            self._day_high = float("-inf")
        else:
            self._bar_index += 1

        typical = (high + low + close) / 3.0
        self._cum_pv += typical * volume
        self._cum_vol += volume
        self._day_high = max(self._day_high, high)
        vwap = self._cum_pv / self._cum_vol if self._cum_vol else _NAN

        # クロス判定のため、更新前の SMA 値を prev として退避
        self._prev_fast = self._val(self._sma_fast)
        self._prev_slow = self._val(self._sma_slow)

        self._sma_fast.update_raw(close)
        self._sma_slow.update_raw(close)
        self._atr.update_raw(high, low, close)
        self._vol_avg.update_raw(volume)

        return Features(
            close=close, high=high, low=low, volume=volume,
            sma_fast=self._val(self._sma_fast), sma_slow=self._val(self._sma_slow),
            prev_sma_fast=self._prev_fast, prev_sma_slow=self._prev_slow,
            vwap=vwap, atr=self._val(self._atr), volume_avg=self._val(self._vol_avg),
            day_high=self._day_high, bar_index=self._bar_index,
        )


def make_nautilus_strategy(params: StrategyParams, *, risk_fraction: float = 0.05, lot_size: int = 100):
    """on_bar() で decide() を呼ぶ Nautilus Strategy クラスを返すファクトリ。

    判定は core.signals.decide() に委譲し、トレーリング／段階利確も基準実装と同じ
    core 関数を使う。発注は Nautilus の MarketOrder で行う薄いグルーに留める。
    nautilus_trader を遅延 import するため関数内でクラス定義している。
    """
    from nautilus_trader.model.enums import OrderSide, TimeInForce
    from nautilus_trader.trading.strategy import Strategy, StrategyConfig

    class _Config(StrategyConfig, frozen=True):
        instrument_id: str
        bar_type: str

    class _Strategy(Strategy):
        def __init__(self, config: _Config) -> None:
            super().__init__(config)
            self.p_params = params
            self.builder = NautilusFeatureBuilder(params)
            self.state = PositionState()

        def on_start(self) -> None:
            from nautilus_trader.model.data import BarType
            self.instrument = self.cache.instrument(self.config.instrument_id)
            self.subscribe_bars(BarType.from_str(self.config.bar_type))

        def _position_qty(self) -> int:
            pos = self.portfolio.net_position(self.instrument.id)
            return int(abs(pos)) if pos is not None else 0

        def _submit(self, side, qty: int) -> None:
            from nautilus_trader.model.objects import Quantity
            order = self.order_factory.market(
                instrument_id=self.instrument.id,
                order_side=side,
                quantity=Quantity.from_int(qty),
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)

        def on_bar(self, bar) -> None:
            session = bar.ts_event  # 立会日は実装時に日付へ正規化する（下記 NOTE 参照）
            f = self.builder.update(
                high=float(bar.high), low=float(bar.low),
                close=float(bar.close), volume=float(bar.volume), session=session,
            )
            held = self._position_qty()
            self.state.is_open = held > 0
            if self.state.is_open:
                update_trailing_stop(self.state, f, self.p_params)

            action = decide(f, self.state, self.p_params)

            if action == Action.ENTER_LONG and held == 0:
                prospective = open_position(f.close, f.atr, self.p_params)
                from daytrade.core.risk import position_size
                cash = self.portfolio.account(self.instrument.venue).balance_total().as_double()
                size = position_size(cash, f.close, prospective.stop_price,
                                     risk_fraction=risk_fraction, lot_size=lot_size)
                if size > 0:
                    self.state = prospective
                    self._submit(OrderSide.BUY, size)
            elif action == Action.SCALE_OUT and held > 0:
                sc = int(math.floor(held * self.p_params.scale_out_fraction / lot_size) * lot_size)
                if 0 < sc < held:
                    self._submit(OrderSide.SELL, sc)
                self.state.scaled_out = True
                self.state.bars_held += 1
            elif action == Action.EXIT and held > 0:
                self.close_all_positions(self.instrument.id)
                self.state = PositionState()
            elif held > 0:
                self.state.bars_held += 1

    # NOTE: session は「立会日」を表す識別子であれば何でもよい（VWAP等の日次リセット用）。
    # 実運用では bar.ts_event をその日の 0時 UTC（または JST 取引日）へ丸めて渡すこと。
    # 例: pd.Timestamp(bar.ts_event, unit="ns", tz="UTC").tz_convert("Asia/Tokyo").normalize()
    return _Strategy, _Config
