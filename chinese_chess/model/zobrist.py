"""Zobrist 随机数表：供 ``Board`` 增量维护局面哈希，同时作为置换表键。

**布局**：90 格 × 14 种棋子槽位（红方 7 兵种 + 黑方 7 兵种）
+ 单独一个「黑方走子」键 ``BLACK_TO_MOVE``。

**使用方式**：
- 红方行棋时不异或 ``BLACK_TO_MOVE``；黑方行棋时异或。
  与 ``Board.apply_move`` 的回合翻转严格同步。
- 每次走子 / 撤销只需做 2~3 次异或操作即可增量更新哈希，无需全量重算。
- 伪随机种子固定为 ``0x7F81A53``，确保不同运行之间哈希表一致，
  方便调试和开局库的 Zobrist 键预计算。
"""

from __future__ import annotations

import random

# 棋盘总格数：10 行 × 9 列 = 90
N_SQUARES = 90
# 棋子槽位总数：红方 7 种 + 黑方 7 种 = 14
N_PIECE_SLOT = 14

# 兵种 → 槽位偏移量映射（红方偏移 0~6，黑方偏移 7~13）
_PTYPE_OFF = {"jiang": 0, "shi": 1, "xiang": 2, "ma": 3, "che": 4, "pao": 5, "bing": 6}


def _build_tables() -> tuple[list[int], int]:
    """构建 Zobrist 随机数表（模块加载时调用一次）。

    使用固定种子的伪随机数生成器，保证每次启动程序时表内容相同。

    Returns:
        ``(piece_table, black_to_move)`` 元组：
        - ``piece_table``: 长度 90×14 = 1260 的 64 位随机数列表。
        - ``black_to_move``: 黑方走子时异或的 64 位随机数。
    """
    rng = random.Random(0x7F81A53)
    piece_table = [rng.getrandbits(64) for _ in range(N_SQUARES * N_PIECE_SLOT)]
    black_to_move = rng.getrandbits(64)
    return piece_table, black_to_move


PIECE_TABLE, BLACK_TO_MOVE = _build_tables()


def piece_slot(piece) -> int:
    """计算棋子在 Zobrist 表中的槽位索引。

    Args:
        piece: ``Piece`` 实例（需有 ``piece_type`` 和 ``color`` 属性）。

    Returns:
        0~13 的槽位索引（红方 0~6，黑方 7~13）。
    """
    return _PTYPE_OFF[piece.piece_type] + (0 if piece.color == "red" else 7)


def piece_key(square_index: int, piece) -> int:
    """获取指定格子上指定棋子的 Zobrist 键值。

    Args:
        square_index: 一维格子索引（``row * 9 + col``）。
        piece: ``Piece`` 实例。

    Returns:
        64 位 Zobrist 随机数，用于异或到局面哈希中。
    """
    return PIECE_TABLE[square_index * N_PIECE_SLOT + piece_slot(piece)]


def full_hash(board) -> int:
    """全量计算棋盘的 Zobrist 哈希值。

    遍历整个 10×9 棋盘，对每个非空格子的棋子异或对应的 Zobrist 键，
    并根据当前行棋方决定是否异或 ``BLACK_TO_MOVE``。

    仅在 ``Board.init_board()`` 初始化和调试校验时使用；
    正常对局中由 ``Board.apply_move`` / ``Board.undo_move`` 增量更新。

    Args:
        board: ``Board`` 实例（需有 ``board``、``current_player`` 属性）。

    Returns:
        当前局面的完整 64 位 Zobrist 哈希值。
    """
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
