"""纯数据序列化：棋盘与 ``MoveEntry`` 链，供 GUI ↔ AI 子进程通信。

仅使用 ``list`` / ``dict`` / ``None`` / 标量等可安全经 ``multiprocessing`` 传递的类型，
不依赖 Qt 对象。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from engine.board import Board
from engine.piece import Piece
from engine import zobrist
from engine.rules import MoveEntry
from ai.ai_registry import build_ai_config_dict as _build_ai_config_dict

_SCHEMA_VERSION = 1


def serialize_board(board: Board) -> Dict[str, Any]:
    """将 ``Board`` 编码为可跨进程传递的 ``dict``。"""
    grid: List[List[Optional[List[str]]]] = []
    for r in range(board.rows):
        row: List[Optional[List[str]]] = []
        for c in range(board.cols):
            p = board.board[r][c]
            if p is None:
                row.append(None)
            else:
                row.append([p.color, p.piece_type])
        grid.append(row)
    counts: List[List[int]] = [[int(h), int(c)] for h, c in board.state_counts.items()]
    return {
        "v": _SCHEMA_VERSION,
        "grid": grid,
        "current_player": board.current_player,
        "state_counts": counts,
        "zobrist_hash": int(board.zobrist_hash),
    }


def deserialize_board(data: Dict[str, Any]) -> Board:
    """由 ``serialize_board`` 的 ``dict`` 还原 ``Board``。"""
    if int(data.get("v", 1)) != _SCHEMA_VERSION:
        raise ValueError(f"unsupported board schema v={data.get('v')!r}")
    b = Board.__new__(Board)
    b.rows = 10
    b.cols = 9
    b.board = [[None for _ in range(9)] for _ in range(10)]
    b.active_pieces = {"red": set(), "black": set()}
    b.red_king_pos = None
    b.black_king_pos = None
    grid = data["grid"]
    for r in range(10):
        for c in range(9):
            cell = grid[r][c]
            if cell is None:
                continue
            color, piece_type = str(cell[0]), str(cell[1])
            piece = Piece(color, piece_type)
            b.board[r][c] = piece
            b.active_pieces[color].add((r, c))
            if piece_type == "jiang":
                if color == "red":
                    b.red_king_pos = (r, c)
                else:
                    b.black_king_pos = (r, c)
    b.current_player = str(data["current_player"])
    b.zobrist_hash = zobrist.full_hash(b)
    pairs = data.get("state_counts") or []
    b.state_counts = {int(h): int(c) for h, c in pairs}
    return b


def serialize_move_history(history: Optional[List[MoveEntry]]) -> List[Dict[str, Any]]:
    """将 ``List[MoveEntry]`` 编码为可跨进程传递的 ``list``。"""
    if not history:
        return []
    out: List[Dict[str, Any]] = []
    for e in history:
        lm = e.last_move
        out.append(
            {
                "pos_hash": int(e.pos_hash),
                "mover": e.mover,
                "gave_check": e.gave_check,
                "last_move": [int(x) for x in lm] if lm is not None else None,
            }
        )
    return out


def deserialize_move_history(data: Optional[List[Dict[str, Any]]]) -> List[MoveEntry]:
    """由 ``serialize_move_history`` 的结果还原 ``List[MoveEntry]``。"""
    if not data:
        return []
    hist: List[MoveEntry] = []
    for item in data:
        lm = item.get("last_move")
        tup: Optional[Tuple[int, int, int, int]] = None
        if lm is not None:
            tup = (int(lm[0]), int(lm[1]), int(lm[2]), int(lm[3]))
        hist.append(
            MoveEntry(
                pos_hash=int(item["pos_hash"]),
                mover=item.get("mover"),
                gave_check=item.get("gave_check"),
                last_move=tup,
            )
        )
    return hist


def build_ai_config_dict(agent) -> Dict[str, Any]:
    """从 AI 实例提取可重建配置（仅数据）。

    兼容入口：历史上 GUI 通过本模块提取 AI 配置；现在配置协议统一由
    ``ai.ai_registry`` 维护，本函数仅做转发。
    """
    return _build_ai_config_dict(agent)
