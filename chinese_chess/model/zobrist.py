"""Zobrist 随机数表：供 Board 增量维护局面哈希、Minimax 置换表键。

布局：90 格 × 14 种（红/黑 × 7 兵种）+ 单独一个「黑方走子」键。
红方行棋时不异或该键；黑方行棋时异或。与 ``Board.apply_move`` 翻转回合同步。
"""

from __future__ import annotations

import random

N_SQUARES = 90
N_PIECE_SLOT = 14  # 7 红 + 7 黑

_PTYPE_OFF = {"jiang": 0, "shi": 1, "xiang": 2, "ma": 3, "che": 4, "pao": 5, "bing": 6}


def _build_tables() -> tuple[list[int], int]:
    rng = random.Random(0x7F81A53)
    piece_table = [rng.getrandbits(64) for _ in range(N_SQUARES * N_PIECE_SLOT)]
    black_to_move = rng.getrandbits(64)
    return piece_table, black_to_move


PIECE_TABLE, BLACK_TO_MOVE = _build_tables()


def piece_slot(piece) -> int:
    return _PTYPE_OFF[piece.piece_type] + (0 if piece.color == "red" else 7)


def piece_key(square_index: int, piece) -> int:
    return PIECE_TABLE[square_index * N_PIECE_SLOT + piece_slot(piece)]


def full_hash(board) -> int:
    """全量计算（仅 init_board / 调试用）；对局中用 Board 增量更新。"""
    h = 0
    mult = N_PIECE_SLOT
    b = board.board
    for r in range(10):
        row = b[r]
        base = r * 9 * mult
        for c in range(9):
            p = row[c]
            if p is not None:
                h ^= PIECE_TABLE[base + c * mult + piece_slot(p)]
    if board.current_player == "black":
        h ^= BLACK_TO_MOVE
    return h
