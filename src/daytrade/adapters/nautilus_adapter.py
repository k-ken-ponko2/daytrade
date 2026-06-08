"""NautilusTrader アダプタ（骨格）。

Backtrader アダプタと同じく、共通ロジック core.signals.decide() を呼ぶだけにする。
NautilusTrader は API が厚い（Strategy / Indicator / OrderFactory 等）ため、ここでは
「どこで Features を組み立て、どこで decide() を呼ぶか」の枠だけを示す骨格に留める。
nautilus_trader をインストール（requirements.txt のコメントを外す）してから肉付けする。

実装時のポイント:
- on_bar() の中で Features を作り decide() を呼ぶ（判定ロジックはここに書かない）。
- VWAP / day_high / bar_index は当日リセットを自前で管理する（core.indicators と同じ定義に揃える）。
- 手数料・スリッページは instrument / fill model に必ず設定する（README 4・8章）。
- 二重検証では generate_signals() の基準実装と突き合わせ、許容ズレ内かを確認する。
"""

from __future__ import annotations

from daytrade.core.signals import decide, initial_stops  # noqa: F401  (実装時に使用)
from daytrade.core.types import Action, Features, PositionState, StrategyParams  # noqa: F401


def build_strategy(params: StrategyParams):
    """NautilusTrader の Strategy を構築する（未実装の骨格）。

    nautilus_trader 導入後、以下の流れで実装する:
        class DaytradeStrategy(Strategy):
            def on_bar(self, bar):
                f = self._features(bar)            # バー → Features
                action = decide(f, self.state, params)
                # action を submit_order / close_position に変換
    """
    raise NotImplementedError(
        "NautilusTrader アダプタは骨格です。nautilus_trader 導入後に on_bar() を実装し、"
        "判定は core.signals.decide() に委譲してください。"
    )
