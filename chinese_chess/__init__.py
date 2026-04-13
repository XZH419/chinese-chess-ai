"""中国象棋（Xiangqi）项目顶层包。

本包按照 MVC 架构组织，与参考 Java 项目的分层结构对齐：
- ``model``: 模型层（棋盘状态、棋子实体、规则引擎、Zobrist 哈希）
- ``algorithm``: 算法层（MinimaxAI、MCTSAI、MCTSMinimaxAI、RandomAI、评估函数、开局库）
- ``control``: 控制器层（游戏流程编排、AI 调度、终局判定）
- ``view``: 视图层（PyQt5 GUI / CLI 前端）
"""
