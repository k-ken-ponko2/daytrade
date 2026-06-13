"""市場プロファイル（市場ごとの執行パラメータ）。

中核ロジック（decide / 指標 / リスク管理 / バックテストエンジン）は市場非依存。
市場ごとに変わるのは「最低注文単位・通貨・手数料・スリッページ」だけなので、
それらをこのプロファイルに閉じ込める。30万円の小資金では、単元株100の日本株より
1株単位の米国株のほうが最低注文額が小さく分散しやすい（README 0章の動機）。

手数料は証券会社依存なので既定値はあくまで目安。実際の料率に合わせて上書きすること。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketProfile:
    """市場ごとの執行パラメータ。

    lot_size:        最低注文単位（日本株=100、米国株=1）。
    currency:        建玉通貨（"JPY" / "USD" など）。資金・価格はこの通貨で一貫させる。
    commission_rate: 片道手数料（約定代金比）。証券会社依存の目安。
    slippage_rate:   片道スリッページ（価格比）。
    allow_fractional: 端株（小数株）注文の可否。現状エンジンは整数株のみ。
    """

    name: str
    currency: str
    lot_size: int = 1
    commission_rate: float = 0.0005
    slippage_rate: float = 0.0002
    allow_fractional: bool = False
    session_tz: str = "Asia/Tokyo"   # 立会日の境界に使う取引所タイムゾーン

    def min_position_value(self, price: float) -> float:
        """1単位（最低注文）の必要資金 ＝ 価格 × lot_size。

        「最低注文数の壁」を数値で表す。日本株は price×100、米国株は price×1。
        """
        return price * self.lot_size

    def max_names(self, capital: float, price: float) -> int:
        """与えられた資金で、その価格帯の銘柄を最低単位で何銘柄持てるか（分散の目安）。"""
        unit = self.min_position_value(price)
        return int(capital // unit) if unit > 0 else 0


# --- プリセット（手数料は目安。実際の証券会社の料率で上書きする） --- #

# 日本株：単元株100。三菱UFJ eスマート証券（kabuステーション）想定。
JAPAN = MarketProfile(
    name="Japan", currency="JPY", lot_size=100,
    commission_rate=0.0005, slippage_rate=0.0002, session_tz="Asia/Tokyo",
)

# 米国株：1株単位。手数料は証券会社で大きく異なる（例：Alpaca/IBKR ≒ 0〜極小、
# SBI/楽天の外国株 ≒ 約定代金の 0.495%・上限あり）。既定は控えめに置き、要上書き。
US = MarketProfile(
    name="US", currency="USD", lot_size=1,
    commission_rate=0.0, slippage_rate=0.0005, allow_fractional=False,
    session_tz="America/New_York",
)

PROFILES = {"jp": JAPAN, "us": US}


def get_profile(key: str) -> MarketProfile:
    """キー（"jp" / "us"）からプロファイルを取得。"""
    try:
        return PROFILES[key.lower()]
    except KeyError:
        raise ValueError(f"未知の市場キー: {key}（{list(PROFILES)} のいずれか）") from None
