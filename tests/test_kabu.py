"""kabuステーションAPI クライアントのテスト（HTTPモック・アプリ/ネットワーク不要）。"""

import pytest

from daytrade.core.types import Action
from daytrade.live.kabu import (
    KabuError,
    KabuStationClient,
    build_market_order,
    side_for_action,
)


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, *, board=None, positions=None, wallet=None, fail=False):
        self.board = board or {}
        self.positions = positions or []
        self.wallet = wallet or {}
        self.fail = fail
        self.posts = []
        self.puts = []

    def post(self, url, json=None, timeout=None, headers=None):
        self.posts.append((url, json))
        if url.endswith("/token"):
            return FakeResponse(200, {"ResultCode": 0, "Token": "tok-123"})
        if url.endswith("/sendorder"):
            if self.fail:
                return FakeResponse(400, {"Code": 4, "Message": "発注エラー"})
            return FakeResponse(200, {"Result": 0, "OrderId": "ORD-1"})
        return FakeResponse(404, {"Message": "unknown"})

    def get(self, url, headers=None, timeout=None):
        if "/board/" in url:
            return FakeResponse(200, self.board)
        if url.endswith("/positions"):
            return FakeResponse(200, self.positions)
        if url.endswith("/wallet/cash"):
            return FakeResponse(200, self.wallet)
        if url.endswith("/orders"):
            return FakeResponse(200, [])
        return FakeResponse(404, {"Message": "unknown"})

    def put(self, url, headers=None, json=None, timeout=None):
        self.puts.append((url, json))
        return FakeResponse(200, {"Result": 0})


def _client(session, **kw):
    return KabuStationClient(api_password="api-pw", trade_password="trade-pw",
                             session=session, **kw)


# --- 注文組み立て --- #

def test_side_for_action():
    assert side_for_action(Action.ENTER_LONG) == "2"   # 買
    assert side_for_action(Action.EXIT) == "1"         # 売
    assert side_for_action(Action.SCALE_OUT) == "1"    # 売
    with pytest.raises(ValueError):
        side_for_action(Action.HOLD)


def test_build_market_order_buy_sets_deliv_type():
    order = build_market_order("7203", "2", 100, "trade-pw")
    assert order["Symbol"] == "7203" and order["Qty"] == 100
    assert order["FrontOrderType"] == 10 and order["Price"] == 0   # 成行
    assert order["DelivType"] == 2     # 現物買はお預り金


def test_build_market_order_sell_no_deliv_type():
    order = build_market_order("7203", "1", 100, "trade-pw")
    assert order["DelivType"] == 0     # 売りは不要


# --- 認証・参照 --- #

def test_authenticate_sets_token_and_header():
    s = FakeSession()
    c = _client(s)
    assert c.authenticate() == "tok-123"
    assert c._headers["X-API-KEY"] == "tok-123"


def test_default_uses_test_port():
    c = _client(FakeSession())
    assert ":18081/" in c.base_url     # 既定は検証ポート（事故防止）


def test_production_port_when_not_test():
    c = _client(FakeSession(), is_test=False)
    assert ":18080/" in c.base_url


def test_get_board_returns_payload():
    s = FakeSession(board={"Symbol": "7203", "CurrentPrice": 2500.0, "BidPrice": 2499.0})
    c = _client(s)
    board = c.get_board("7203")
    assert board["CurrentPrice"] == 2500.0


def test_get_wallet_cash():
    c = _client(FakeSession(wallet={"StockAccountWallet": 300000.0}))
    assert c.get_wallet_cash()["StockAccountWallet"] == 300000.0


# --- 発注（dry_run 既定で安全） --- #

def test_send_order_dry_run_does_not_post():
    s = FakeSession()
    c = _client(s)
    c.authenticate()
    order = build_market_order("7203", "2", 100, "trade-pw")
    result = c.send_order(order)            # 既定 dry_run=True
    assert result["dry_run"] is True
    assert result["would_send"] == order
    # /sendorder へは飛んでいない（/token のみ）
    assert not any(u.endswith("/sendorder") for u, _ in s.posts)


def test_send_order_real_posts():
    s = FakeSession()
    c = _client(s)
    c.authenticate()
    order = build_market_order("7203", "2", 100, "trade-pw")
    result = c.send_order(order, dry_run=False)
    assert result["OrderId"] == "ORD-1"
    assert any(u.endswith("/sendorder") for u, _ in s.posts)


def test_send_order_error_raises():
    s = FakeSession(fail=True)
    c = _client(s)
    c.authenticate()
    order = build_market_order("7203", "2", 100, "trade-pw")
    with pytest.raises(KabuError):
        c.send_order(order, dry_run=False)


def test_cancel_order_dry_run():
    c = _client(FakeSession())
    c.authenticate()
    res = c.cancel_order("ORD-9")
    assert res["dry_run"] is True
    assert res["would_cancel"]["OrderId"] == "ORD-9"


def test_from_env_requires_api_password(monkeypatch):
    monkeypatch.delenv("KABU_API_PASSWORD", raising=False)
    with pytest.raises(KabuError):
        KabuStationClient.from_env()
