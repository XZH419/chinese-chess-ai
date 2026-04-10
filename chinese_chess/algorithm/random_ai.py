"""Random AI (baseline).

需求：
- 遵守 Searcher 接口：从 `Rules.get_legal_moves` 取合法走法
- 若无合法走法：返回 None（视为认输/无法行动）
- 否则随机选择一步
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

from chinese_chess.model.rules import Rules


class RandomAI:
    """最基础的测试 AI：随机走子。"""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self.last_stats = {}

    # Searcher-style API (recommended)
    def choose_move(self, board, time_limit: Optional[float] = None, **kwargs) -> Optional[Tuple[int, int, int, int]]:
        self.last_stats = {"depth": 0, "time_taken": 0.0, "nodes_evaluated": 0}
        moves = Rules.get_legal_moves(board, board.current_player)
        if not moves:
            return None
        return self._rng.choice(list(moves))

    # Compatibility with existing AI API in this repo
    def get_best_move(self, board, time_limit: Optional[float] = None, **kwargs) -> Optional[Tuple[int, int, int, int]]:
        return self.choose_move(board, time_limit=time_limit)

