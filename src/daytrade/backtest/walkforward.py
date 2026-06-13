"""ウォークフォワード検証 — 過剰最適化（カーブフィッティング）の検出。

考え方（README 8章）:
- パラメータを過去区間（イン・サンプル / IS）で最適化し、その「先」の未知区間
  （アウト・オブ・サンプル / OOS）で評価する。
- IS では良いのに OOS で崩れるなら、それは優位性ではなく過剰最適化。
- これを区間をずらしながら繰り返し、IS と OOS の差（ギャップ）を見る。

注意:
- 各区間は独立にバックテストするため、区間先頭の slow_period 本前後は指標の
  ウォームアップで取引が出にくい。test_days は十分に取ること。
- 合成データでの検証は「配線とロジックの確認」であり、優位性の証明ではない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from daytrade.backtest.engine import BacktestConfig, run_backtest
from daytrade.core.types import StrategyParams


# --------------------------------------------------------------------------- #
# パラメータ探索
# --------------------------------------------------------------------------- #
def param_grid(base: dict | None = None, **axes) -> list[StrategyParams]:
    """各軸の値の直積から StrategyParams 候補リストを作る。

    例: param_grid(fast_period=[3,5], slow_period=[10,20], volume_surge_mult=[1.3,1.5])
    無効な組み合わせ（fast>=slow など）は自動的に除外する。
    """
    base = base or {}
    keys = list(axes)
    candidates: list[StrategyParams] = []
    for combo in product(*axes.values()):
        kwargs = dict(base)
        kwargs.update(dict(zip(keys, combo)))
        try:
            candidates.append(StrategyParams(**kwargs))
        except ValueError:
            continue  # fast>=slow などの無効組み合わせ
    return candidates


def total_return_objective(result, *, min_trades: int = 3) -> float:
    """最適化の目的関数：トータルリターン。ただし取引が少なすぎる解は失格にする。

    取引数の下限を設けることで、「たまたま1勝しただけ」の過剰適合を弾く。
    """
    m = result.metrics
    if not m or m.get("num_trades", 0) < min_trades:
        return float("-inf")
    return float(m.get("total_return", float("-inf")))


def optimize(
    df: pd.DataFrame,
    candidates: list[StrategyParams],
    config: BacktestConfig,
    objective=total_return_objective,
) -> tuple[StrategyParams | None, float]:
    """候補の中から目的関数が最大の StrategyParams を選ぶ（グリッドサーチ）。"""
    best: StrategyParams | None = None
    best_score = float("-inf")
    for p in candidates:
        score = objective(run_backtest(df, p, config))
        if score > best_score:
            best_score, best = score, p
    return best, best_score


# --------------------------------------------------------------------------- #
# 区間分割
# --------------------------------------------------------------------------- #
@dataclass
class Window:
    label: str
    train: pd.DataFrame
    test: pd.DataFrame


def _session_dates(df: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(pd.DatetimeIndex(df.index).normalize().unique())


def rolling_windows(
    df: pd.DataFrame,
    *,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
) -> list[Window]:
    """立会日ベースで (train, test) の区間を前進させながら生成する。

    train_days 日で最適化し、続く test_days 日で評価。step_days ずつ前進（既定は test_days
    ＝重複なし）。デイトレ前提で「日」単位に分割する。
    """
    step = step_days or test_days
    days = _session_dates(df)
    norm = pd.DatetimeIndex(df.index).normalize()

    windows: list[Window] = []
    i = 0
    while i + train_days + test_days <= len(days):
        train_dates = set(days[i:i + train_days])
        test_dates = set(days[i + train_days:i + train_days + test_days])
        train = df[norm.isin(train_dates)]
        test = df[norm.isin(test_dates)]
        label = (f"{min(train_dates).date()}→{max(train_dates).date()} | "
                 f"OOS {min(test_dates).date()}→{max(test_dates).date()}")
        windows.append(Window(label=label, train=train, test=test))
        i += step
    return windows


# --------------------------------------------------------------------------- #
# ウォークフォワード実行
# --------------------------------------------------------------------------- #
@dataclass
class WindowResult:
    label: str
    best_params: StrategyParams
    is_return: float
    oos_return: float
    oos_trades: int
    oos_win_rate: float


@dataclass
class WalkForwardReport:
    windows: list[WindowResult]
    summary: dict = field(default_factory=dict)


def run_walk_forward(
    df: pd.DataFrame,
    candidates: list[StrategyParams],
    *,
    config: BacktestConfig | None = None,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
    objective=total_return_objective,
) -> WalkForwardReport:
    """各区間で IS 最適化 → OOS 評価を行い、過剰最適化のギャップを集計する。"""
    config = config or BacktestConfig()
    windows = rolling_windows(df, train_days=train_days, test_days=test_days, step_days=step_days)

    results: list[WindowResult] = []
    for w in windows:
        best, is_score = optimize(w.train, candidates, config, objective)
        if best is None:
            continue
        is_res = run_backtest(w.train, best, config)
        oos_res = run_backtest(w.test, best, config)
        results.append(WindowResult(
            label=w.label,
            best_params=best,
            is_return=is_res.metrics.get("total_return", 0.0),
            oos_return=oos_res.metrics.get("total_return", 0.0),
            oos_trades=oos_res.metrics.get("num_trades", 0),
            oos_win_rate=oos_res.metrics.get("win_rate", 0.0),
        ))

    return WalkForwardReport(windows=results, summary=_summarize(results))


def _summarize(results: list[WindowResult]) -> dict:
    if not results:
        return {}
    is_returns = np.array([r.is_return for r in results])
    oos_returns = np.array([r.oos_return for r in results])
    oos_positive = int((oos_returns > 0).sum())

    return {
        "num_windows": len(results),
        "is_mean_return": float(is_returns.mean()),
        "oos_mean_return": float(oos_returns.mean()),
        # ギャップが大きい（IS>>OOS）ほど過剰最適化の疑いが濃い
        "overfit_gap": float(is_returns.mean() - oos_returns.mean()),
        "oos_positive_windows": oos_positive,
        "oos_hit_rate": float(oos_positive / len(results)),
        "total_oos_trades": int(sum(r.oos_trades for r in results)),
    }
