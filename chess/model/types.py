"""Shared types for the chess model layer.

This file is intentionally lightweight to avoid import cycles:
- It must not import `board` or `rules`.
- Higher layers (controller/ai/view) can import from here safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class Color(str, Enum):
    """Piece side / player side.

    Kept compatible with the existing implementation that uses 'red'/'black'.
    """

    RED = "red"
    BLACK = "black"

    def opponent(self) -> "Color":
        return Color.BLACK if self == Color.RED else Color.RED


class PieceType(str, Enum):
    """Xiangqi piece types.

    Kept compatible with the existing implementation that uses these strings.
    """

    JIANG = "jiang"
    SHI = "shi"
    XIANG = "xiang"
    MA = "ma"
    CHE = "che"
    PAO = "pao"
    BING = "bing"


@dataclass(frozen=True, slots=True)
class Move:
    """A move represented as (start_row, start_col, end_row, end_col).

    We keep the tuple shape identical to the current project so the migration
    can be mechanical. Higher layers can later wrap/convert as needed.
    """

    start_row: int
    start_col: int
    end_row: int
    end_col: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.start_row, self.start_col, self.end_row, self.end_col)

