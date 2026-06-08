"""売買代金（流動性）による銘柄スクリーニング。

デイトレは流動性が命（板が薄いとスリッページで負ける）。売買代金 TurnoverValue の
大きい銘柄＝出来高・約定が厚い銘柄を機械的に絞り込み、デイトレ対象候補にする。

free プランでも /prices/daily_quotes に date を指定すれば「その日の全銘柄」が取れる
（12週間遅延）。複数営業日ぶんを平均して、日々のブレを均した順位にする。
"""

from __future__ import annotations

import pandas as pd


def rank_by_turnover(
    daily_frames: list[pd.DataFrame],
    *,
    top_n: int = 50,
    min_turnover: float | None = None,
) -> pd.DataFrame:
    """日次の全銘柄データ（複数日）から、平均売買代金で上位銘柄をランキングする。

    Args:
        daily_frames: 各要素が1営業日の全銘柄 DataFrame（Code / TurnoverValue を含む）。
        top_n:        上位何銘柄を返すか。
        min_turnover: 平均売買代金の下限（円）。これ未満は除外。

    Returns:
        Code / avg_turnover / days_count を平均売買代金の降順で並べた DataFrame。
    """
    usable = [f[["Code", "TurnoverValue"]].copy()
              for f in daily_frames if not f.empty and "TurnoverValue" in f.columns]
    if not usable:
        return pd.DataFrame(columns=["Code", "avg_turnover", "days_count"])

    combined = pd.concat(usable, ignore_index=True)
    combined["TurnoverValue"] = pd.to_numeric(combined["TurnoverValue"], errors="coerce")
    combined = combined.dropna(subset=["TurnoverValue"])

    grouped = combined.groupby("Code")["TurnoverValue"]
    ranked = pd.DataFrame({
        "avg_turnover": grouped.mean(),
        "days_count": grouped.size(),
    }).reset_index()

    if min_turnover is not None:
        ranked = ranked[ranked["avg_turnover"] >= min_turnover]

    ranked = ranked.sort_values("avg_turnover", ascending=False).reset_index(drop=True)
    return ranked.head(top_n)


def attach_listed_info(ranked: pd.DataFrame, listed: pd.DataFrame) -> pd.DataFrame:
    """ランキングに銘柄名・市場区分・業種を付与する（listed/info をマージ）。"""
    if ranked.empty or listed.empty or "Code" not in listed.columns:
        return ranked
    keep = [c for c in ["Code", "CompanyName", "MarketCode",
                        "MarketCodeName", "Sector33CodeName"] if c in listed.columns]
    return ranked.merge(listed[keep], on="Code", how="left")


def screen_liquid_stocks(
    client,
    dates: list[str],
    *,
    top_n: int = 50,
    min_turnover: float | None = None,
    attach_info: bool = True,
) -> pd.DataFrame:
    """指定営業日ぶんの全銘柄を取得し、平均売買代金で流動性上位を絞り込む。

    Args:
        client:   JQuantsClient。
        dates:    対象営業日のリスト（"YYYY-MM-DD"）。free では12週間以上前を指定する。
        top_n:    上位何銘柄を返すか。
        min_turnover: 平均売買代金の下限（円）。
        attach_info:  True なら銘柄名・市場区分を付与する。

    Returns:
        流動性上位銘柄の DataFrame（avg_turnover 降順）。
    """
    frames = [client.get_daily_quotes(date=d) for d in dates]
    ranked = rank_by_turnover(frames, top_n=top_n, min_turnover=min_turnover)
    if attach_info and not ranked.empty:
        ranked = attach_listed_info(ranked, client.get_listed_info())
    return ranked
