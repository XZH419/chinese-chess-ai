"""Piece entity (Model).

Physical migration target from `chinese-chess/ai/pieces.py`.
This module should remain free of rule logic; it only represents piece data
and a display representation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Color, PieceType


@dataclass(slots=True)
class Piece:
    """表示一个中国象棋棋子。

    This is a direct migration of the existing `Piece` data model.
    """

    color: str  # kept as 'red'/'black' to avoid changing existing logic
    piece_type: str  # kept as string like 'jiang','shi',...

    def __str__(self) -> str:
        # 将棋子转换为可显示的中文符号。
        symbols = {
            "red": {
                "jiang": "帅",
                "shi": "仕",
                "xiang": "相",
                "ma": "马",
                "che": "车",
                "pao": "炮",
                "bing": "兵",
            },
            "black": {
                "jiang": "将",
                "shi": "士",
                "xiang": "象",
                "ma": "马",
                "che": "车",
                "pao": "炮",
                "bing": "卒",
            },
        }
        return symbols[self.color][self.piece_type]

    def __repr__(self) -> str:
        return f"{self.color}_{self.piece_type}"

