"""随机走子 AI（基线对照组）。

作为最低智能水平的 AI 实现，主要用途：
- 与 MinimaxAI / MCTSAI 进行胜率对比测试，验证搜索算法的有效性。
- 快速测试 GUI 和 Controller 的走棋流程是否正常。
- 作为 MCTS 轻量级模拟的概念参照（模拟阶段的随机走子逻辑类似）。

遵守 Searcher 统一接口：
- 从 ``Rules.get_legal_moves`` 获取所有合法走法。
- 若无合法走法：返回 ``None``（视为认输 / 被将死）。
- 否则用可选的固定随机种子均匀随机选择一步。
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

from engine.rules import Rules


class RandomAI:
    """随机走子 AI：从合法走法中均匀随机选择。

    Attributes:
        last_stats: 上一次走棋的统计信息字典（供 GUI 日志显示）。
    """

    def __init__(self, seed: Optional[int] = None):
        """初始化随机 AI。

        Args:
            seed: 可选的随机种子。指定后走棋序列可复现（用于测试）。
        """
        self._rng = random.Random(seed)
        self.last_stats = {}

    def choose_move(
        self,
        board,
        time_limit: Optional[float] = None,
        **kwargs: object,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Searcher 统一接口：为当前行棋方随机选择一步合法走法。

        Args:
            board: 当前棋盘状态。
            time_limit: 时间限制（随机 AI 不需要搜索，此参数忽略）。
            **kwargs: 兼容其他 AI 的额外关键字参数（如 game_history）。

        Returns:
            随机选中的走法四元组 ``(src_r, src_c, dst_r, dst_c)``，
            或 ``None``（无合法走法，即被将死或困毙）。
        """
        self.last_stats = {"random": True, "time_taken": 0.0}
        moves = Rules.get_legal_moves(board, board.current_player)
        if not moves:
            return None
        return self._rng.choice(list(moves))

    def get_best_move(
        self,
        board,
        time_limit: Optional[float] = None,
        **kwargs,
    ) -> Optional[Tuple[int, int, int, int]]:
        """兼容 ``MinimaxAI`` / ``MCTSAI`` 的 ``get_best_move`` 接口。

        Args:
            board: 当前棋盘状态。
            time_limit: 时间限制（忽略）。
            **kwargs: 额外关键字参数。

        Returns:
            随机选中的走法四元组，或 ``None``。
        """
        return self.choose_move(board, time_limit=time_limit, **kwargs)
