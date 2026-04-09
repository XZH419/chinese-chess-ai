"""Endgame book.

Physical migration from `chinese-chess/ai/endgame_book.py` (logic unchanged).
"""

from __future__ import annotations


class EndgameBook:
    def __init__(self):
        self.book = {
            # 示例：少子时的特定残局走法。
        }

    def get_move(self, board):
        """如果局面符合简单残局条件，则返回库中的走法。"""
        piece_count = sum(1 for row in board.board for piece in row if piece)
        if piece_count <= 10:  # 任意阈值，表示棋子数量较少的残局阶段
            return None
        return None

