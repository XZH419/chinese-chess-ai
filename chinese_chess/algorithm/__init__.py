"""算法层：AI 搜索引擎 + 局面评估 + 开局库。

本包是项目参考实现中 ``algorithm/`` 目录的 Python 对应物，
包含 MinimaxAI、MCTSAI、RandomAI 三种搜索引擎，
Evaluation 静态评估模块，以及 Zobrist 开局库。

**依赖规则**：仅依赖模型层 (``chinese_chess.model.*``)，严禁依赖 GUI 代码。
"""
