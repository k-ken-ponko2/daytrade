"""アダプタ層。

各バックテストエンジンへの接続部分だけをここに置く。売買判定そのものは
core.signals.decide() に委譲し、ここではエンジンのバー → Features への変換と、
返ってきた Action → エンジンの注文への変換だけを担う（README 5章）。

backtrader / nautilus_trader は重い依存なので、本体は遅延 import している。
未インストールでも core 層のテストは動く。
"""
