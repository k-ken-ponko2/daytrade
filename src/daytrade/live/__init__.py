"""ライブ実行層（kabuステーションAPI）。

実運用の発注・約定・リアルタイム板取得を担う（README 1・2章）。kabuステーションアプリ
（Windows専用）が起動した状態のローカル REST(localhost) に対して動く。バックテストの
J-Quants/合成データとは役割が異なり、ここは「本番の手足」。

注意:
- 取引パスワード等の秘匿情報は環境変数／.env から読む（コードに直書きしない）。
- 実発注は外向き・不可逆。まずは検証環境(ポート18081)とドライランで確認すること。
"""

from daytrade.live.kabu import KabuStationClient, KabuError, build_market_order, side_for_action

__all__ = ["KabuStationClient", "KabuError", "build_market_order", "side_for_action"]
