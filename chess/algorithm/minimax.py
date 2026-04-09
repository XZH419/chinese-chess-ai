"""Minimax AI with alpha-beta pruning.

Physical migration from `chinese-chess/ai/ai_minimax.py` with only import-path
and API updates (Board.make_move -> Board.apply_move, Board.*rules -> Rules.*).
"""

from __future__ import annotations

import time
from typing import Optional, Tuple, Dict, Any

from chess.model.rules import Rules

from .evaluation import Evaluation


class MinimaxAI:
    def __init__(self, depth=3):
        # Minimax搜索的深度限制。
        self.depth = depth
        self._nodes = 0
        # 供 GUI Dashboard/日志读取的最近一次搜索统计
        self.last_stats: Dict[str, Any] = {}

    def choose_move(self, board, time_limit: Optional[float] = 10.0) -> Optional[Tuple[int, int, int, int]]:
        """Searcher 接口：为当前 board.current_player 选择一步。"""
        return self.get_best_move(board, time_limit=time_limit)

    def get_best_move(self, board, time_limit: Optional[float] = 10.0):
        """在允许时间内选择最优走法（Alpha-Beta Minimax）。"""
        start = time.time()
        self._nodes = 0

        player = board.current_player
        maximizing = (player == "red")  # 评估函数基础分为 red-black
        best_move = None
        best_value = float("-inf") if maximizing else float("inf")

        moves = Rules.get_legal_moves(board, player)
        for move in moves:
            if time_limit is not None and (time.time() - start) > time_limit:
                break

            captured = board.apply_move(*move)
            value = self._alphabeta(
                board=board,
                depth=self.depth - 1,
                alpha=float("-inf"),
                beta=float("inf"),
                maximizing=not maximizing,
                start_time=start,
                time_limit=time_limit,
                maximizing_color=player,
            )
            board.undo_move(*move, captured)

            if maximizing:
                if value > best_value:
                    best_value = value
                    best_move = move
            else:
                if value < best_value:
                    best_value = value
                    best_move = move

        elapsed = time.time() - start
        self.last_stats = {
            "depth": int(self.depth),
            "time_taken": float(elapsed),
            "nodes_evaluated": int(self._nodes),
        }
        print(f"本次搜索深度: {self.depth}")
        print(f"搜索耗时 (秒): {elapsed:.3f}")
        print(f"评估的节点总数: {self._nodes}")
        return best_move

    def _alphabeta(
        self,
        board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        start_time: float,
        time_limit: Optional[float],
        maximizing_color: str,
    ) -> float:
        """标准 Alpha-Beta 剪枝递归。"""
        if time_limit is not None and (time.time() - start_time) > time_limit:
            # 超时：返回当前评估（不再扩展）
            self._nodes += 1
            return Evaluation.evaluate(board, maximizing_color=maximizing_color)

        if depth == 0 or Rules.is_game_over(board):
            self._nodes += 1
            return Evaluation.evaluate(board, maximizing_color=maximizing_color)

        player = board.current_player
        moves = Rules.get_legal_moves(board, player)

        if maximizing:
            value = float("-inf")
            for move in moves:
                captured = board.apply_move(*move)
                child = self._alphabeta(
                    board=board,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    maximizing=False,
                    start_time=start_time,
                    time_limit=time_limit,
                    maximizing_color=maximizing_color,
                )
                board.undo_move(*move, captured)
                value = max(value, child)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = float("inf")
            for move in moves:
                captured = board.apply_move(*move)
                child = self._alphabeta(
                    board=board,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    maximizing=True,
                    start_time=start_time,
                    time_limit=time_limit,
                    maximizing_color=maximizing_color,
                )
                board.undo_move(*move, captured)
                value = min(value, child)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

