"""kabuステーションAPI クライアント（ローカル REST）。

kabuステーション（三菱UFJ eスマート証券・Windows専用アプリ）が起動していると、
ローカルに REST/WebSocket サーバが立つ。本番は :18080、検証は :18081。
認証は API パスワードを POST /token に渡してトークンを得て、以後 X-API-KEY に載せる。

このモジュールは「実行インターフェースの骨格」。本番発注は不可逆・外向きなので、
まず検証ポートとドライラン(dry_run=True)で挙動を確認してから本番に切り替えること。
ネットワーク/アプリ非依存に保つため requests.Session を注入可能にし、モックでテストする。

参考: https://kabucom.github.io/kabusapi/ptal/
"""

from __future__ import annotations

import os
from typing import Any

import requests

from daytrade.core.types import Action

DEFAULT_TIMEOUT = 10


class KabuError(RuntimeError):
    """kabuステーションAPI 呼び出しの失敗。"""


# --------------------------------------------------------------------------- #
# 注文の組み立て
# --------------------------------------------------------------------------- #
def side_for_action(action: Action) -> str:
    """core の Action を kabu の Side（"1"=売, "2"=買）に変換する。

    ENTER_LONG は新規買い、EXIT / SCALE_OUT は売り（現物の手仕舞い）。
    """
    if action == Action.ENTER_LONG:
        return "2"  # 買
    if action in (Action.EXIT, Action.SCALE_OUT):
        return "1"  # 売
    raise ValueError(f"発注に変換できない Action: {action}")


def build_market_order(
    symbol: str,
    side: str,
    qty: int,
    trade_password: str,
    *,
    exchange: int = 1,        # 1=東証
    account_type: int = 4,    # 4=特定（README 1章：特定口座）
    cash_margin: int = 1,     # 1=現物
    deliv_type: int = 2,      # 2=お預り金（現物買のとき必須）
) -> dict[str, Any]:
    """成行注文のリクエストボディを組み立てる（現物・当日有効）。

    FrontOrderType=10（成行）、Price=0、ExpireDay=0（当日）。
    売り（手仕舞い）では deliv_type は不要なので 0 にする。
    """
    return {
        "Password": trade_password,
        "Symbol": symbol,
        "Exchange": exchange,
        "SecurityType": 1,            # 1=株式
        "Side": side,                 # "1"=売 / "2"=買
        "CashMargin": cash_margin,
        "DelivType": deliv_type if side == "2" else 0,
        "AccountType": account_type,
        "Qty": qty,
        "FrontOrderType": 10,         # 10=成行
        "Price": 0,
        "ExpireDay": 0,               # 0=当日
    }


# --------------------------------------------------------------------------- #
# クライアント
# --------------------------------------------------------------------------- #
class KabuStationClient:
    """kabuステーションAPI の薄いラッパ。

    使い方:
        client = KabuStationClient.from_env()      # API/取引パスワードを環境変数から
        client.authenticate()
        board = client.get_board("7203", exchange=1)
        # 発注は不可逆。まずは dry_run で確認:
        client.send_order(build_market_order("7203","2",100, client.trade_password), dry_run=True)
    """

    def __init__(
        self,
        api_password: str | None = None,
        trade_password: str | None = None,
        *,
        is_test: bool = True,
        host: str = "localhost",
        session: requests.Session | None = None,
    ) -> None:
        self.api_password = api_password
        self.trade_password = trade_password
        port = 18081 if is_test else 18080  # 既定は検証ポート（事故防止）
        self.base_url = f"http://{host}:{port}/kabusapi"
        self._token: str | None = None
        self._session = session or requests.Session()

    @classmethod
    def from_env(cls, *, is_test: bool = True) -> "KabuStationClient":
        """環境変数（.env 含む）から認証情報を読み込む。

        KABU_API_PASSWORD … API パスワード（/token 用）
        KABU_TRADE_PASSWORD … 取引パスワード（発注用）
        既定は検証環境(is_test=True)。本番は明示的に is_test=False を渡す。
        """
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        api_pw = os.getenv("KABU_API_PASSWORD")
        trade_pw = os.getenv("KABU_TRADE_PASSWORD")
        if not api_pw:
            raise KabuError("KABU_API_PASSWORD が設定されていません（.env を確認）。")
        return cls(api_password=api_pw, trade_password=trade_pw, is_test=is_test)

    # --- 認証 --- #
    def authenticate(self) -> str:
        """API パスワードでトークンを取得し、内部に保持する。"""
        if not self.api_password:
            raise KabuError("API パスワードが未設定です。")
        resp = self._session.post(
            f"{self.base_url}/token",
            json={"APIPassword": self.api_password},
            timeout=DEFAULT_TIMEOUT,
        )
        self._raise_for_status(resp)
        token = resp.json().get("Token")
        if not token:
            raise KabuError("トークンの取得に失敗しました。")
        self._token = token
        return token

    @property
    def _headers(self) -> dict[str, str]:
        if not self._token:
            self.authenticate()
        return {"X-API-KEY": self._token, "Content-Type": "application/json"}

    # --- 参照系 --- #
    def get_board(self, symbol: str, *, exchange: int = 1) -> dict:
        """板・気配・現在値 (/board/{symbol}@{exchange})。リアルタイム判定の入力。"""
        resp = self._session.get(
            f"{self.base_url}/board/{symbol}@{exchange}",
            headers=self._headers, timeout=DEFAULT_TIMEOUT,
        )
        self._raise_for_status(resp)
        return resp.json()

    def get_positions(self) -> list[dict]:
        """保有ポジション一覧 (/positions)。"""
        resp = self._session.get(
            f"{self.base_url}/positions", headers=self._headers, timeout=DEFAULT_TIMEOUT,
        )
        self._raise_for_status(resp)
        return resp.json()

    def get_orders(self) -> list[dict]:
        """注文一覧・約定状況 (/orders)。"""
        resp = self._session.get(
            f"{self.base_url}/orders", headers=self._headers, timeout=DEFAULT_TIMEOUT,
        )
        self._raise_for_status(resp)
        return resp.json()

    def get_wallet_cash(self) -> dict:
        """現物の余力 (/wallet/cash)。発注前の資金確認に使う。"""
        resp = self._session.get(
            f"{self.base_url}/wallet/cash", headers=self._headers, timeout=DEFAULT_TIMEOUT,
        )
        self._raise_for_status(resp)
        return resp.json()

    # --- 発注系（不可逆・外向き） --- #
    def send_order(self, order: dict, *, dry_run: bool = True) -> dict:
        """発注 (/sendorder)。dry_run=True なら送信せず内容を返すだけ（事故防止の既定）。

        本番で実発注する場合のみ dry_run=False を明示する。
        """
        if dry_run:
            return {"dry_run": True, "would_send": order}
        resp = self._session.post(
            f"{self.base_url}/sendorder", headers=self._headers,
            json=order, timeout=DEFAULT_TIMEOUT,
        )
        self._raise_for_status(resp)
        return resp.json()

    def cancel_order(self, order_id: str, *, dry_run: bool = True) -> dict:
        """注文取消 (/cancelorder)。dry_run の扱いは send_order と同じ。"""
        if not self.trade_password:
            raise KabuError("取消には取引パスワードが必要です。")
        body = {"OrderId": order_id, "Password": self.trade_password}
        if dry_run:
            return {"dry_run": True, "would_cancel": body}
        resp = self._session.put(
            f"{self.base_url}/cancelorder", headers=self._headers,
            json=body, timeout=DEFAULT_TIMEOUT,
        )
        self._raise_for_status(resp)
        return resp.json()

    # --- 内部 --- #
    @staticmethod
    def _raise_for_status(resp: requests.Response) -> None:
        if resp.status_code >= 400:
            detail = ""
            try:
                payload = resp.json()
                detail = payload.get("Message") or payload.get("message") or str(payload)
            except Exception:  # noqa: BLE001 - 本文がJSONでないことがある
                detail = resp.text[:200]
            raise KabuError(f"HTTP {resp.status_code}: {detail}")
