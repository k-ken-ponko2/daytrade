"""実データ分足の窓口（load_intraday）とデモスクリプトの結合テスト（ネットワーク不要）。"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daytrade.data.intraday import load_intraday

ROOT = Path(__file__).resolve().parents[1]


def _fake_alpaca_raw(n_days=3):
    """Alpaca get_bars() が返す形（UTC・時間外を含む）の分足。"""
    rng = np.random.default_rng(0)
    frames = []
    for d in pd.bdate_range("2026-09-01", periods=n_days):
        idx = pd.date_range(d + pd.Timedelta(hours=13), d + pd.Timedelta(hours=20, minutes=30),
                            freq="1min", tz="UTC", inclusive="left")  # 09:00–16:29 NY（夏時間）
        close = 150 + rng.normal(0, 0.05, len(idx)).cumsum()
        frames.append(pd.DataFrame({"open": close, "high": close + 0.05, "low": close - 0.05,
                                    "close": close, "volume": rng.integers(100, 1000, len(idx))},
                                   index=idx))
    return pd.concat(frames)


def _fake_jquants_raw():
    """J-Quants get_minute_bars() が返す形（V1互換カラム・Date/Time 別）。"""
    times = [f"{h:02d}:{m:02d}" for h in range(9, 15) for m in range(60)]
    return pd.DataFrame({
        "Date": pd.to_datetime(["2026-09-01"] * len(times)), "Time": times, "Code": "72030",
        "Open": 3000.0, "High": 3001.0, "Low": 2999.0, "Close": 3000.0, "Volume": 1000,
        "TurnoverValue": 3_000_000,
    })


class CountingAlpaca:
    def __init__(self):
        self.calls = 0

    def get_bars(self, symbol, *, start, end, feed="sip"):
        self.calls += 1
        return _fake_alpaca_raw()


class CountingJQuants:
    def __init__(self):
        self.calls = 0

    def get_minute_bars(self, code, *, from_date=None, to_date=None):
        self.calls += 1
        return _fake_jquants_raw()


def test_alpaca_cache_roundtrip_keeps_ny_timezone(tmp_path):
    client = CountingAlpaca()
    first = load_intraday("alpaca", "aapl", from_date="2026-09-01", to_date="2026-09-03",
                          cache_root=tmp_path, client=client)
    second = load_intraday("alpaca", "aapl", from_date="2026-09-01", to_date="2026-09-03",
                           cache_root=tmp_path, client=client)
    assert client.calls == 1                                   # 2回目はキャッシュ
    assert len(first) == 3 * 390                               # 通常取引時間のみ
    assert str(second.index.tz) == "America/New_York"
    pd.testing.assert_index_equal(first.index, second.index)


def test_jquants_cache_roundtrip_is_naive_jst(tmp_path):
    client = CountingJQuants()
    first = load_intraday("jquants", "7203", from_date="2026-09-01", to_date="2026-09-01",
                          cache_root=tmp_path, client=client)
    second = load_intraday("jquants", "7203", from_date="2026-09-01", to_date="2026-09-01",
                           cache_root=tmp_path, client=client)
    assert client.calls == 1
    assert second.index.tz is None
    assert second.index[0] == pd.Timestamp("2026-09-01 09:00")
    assert np.allclose(first["close"], second["close"])


def test_unknown_source():
    with pytest.raises(ValueError):
        load_intraday("yahoo", "AAPL", from_date="2026-09-01", to_date="2026-09-02")


def _load_run_demo():
    spec = importlib.util.spec_from_file_location("run_demo", ROOT / "scripts" / "run_demo.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module   # dataclass の解決に必要
    spec.loader.exec_module(module)
    return module


def test_run_demo_with_alpaca_data(tmp_path, monkeypatch, capsys):
    """--alpaca 経路がデータ取得→バックテスト→表示まで通ること（取得はモック）。"""
    import daytrade.data.intraday as intraday

    real = intraday.load_intraday

    def fake_load(source, symbol, **kw):
        kw.update(cache_root=tmp_path, client=CountingAlpaca())
        return real(source, symbol, **kw)

    monkeypatch.setattr(intraday, "load_intraday", fake_load)
    demo = _load_run_demo()
    assert demo.main(["--alpaca", "AAPL", "--from", "2026-09-01", "--to", "2026-09-03"]) == 0
    out = capsys.readouterr().out
    assert "バー数            : 1170" in out
    assert "USD" in out


def test_run_demo_requires_dates_for_real_data():
    demo = _load_run_demo()
    with pytest.raises(SystemExit):
        demo.main(["--alpaca", "AAPL"])
