"""Opening book.

Physical migration from `chinese-chess/ai/opening_book.py` (logic unchanged).
"""

from __future__ import annotations

import random


class OpeningBook:
    def __init__(self):
        self.book = {
            # 初始局面对应的常见开局走法。
            "initial": [(9, 6, 7, 8), (9, 2, 7, 0), (9, 7, 7, 6)]
        }

    def get_move(self, board):
        """根据当前棋盘状态返回预定义的开局走法。"""
        if "initial" in self.book:
            return random.choice(self.book["initial"])
        return None

