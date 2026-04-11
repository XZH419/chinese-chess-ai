"""Board state container (Model).

Physical migration target from `chinese-chess/ai/board.py`, but *only* the
pure state management responsibilities remain here:
- board grid initialization and initial setup
- get_piece / set_piece
- apply_move (formerly make_move): execute move without legality checks
- undo_move
- copy

All rule logic (move validity, check detection, move generation, game end)
is moved to `chess/model/rules.py`.
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

from . import zobrist
from .piece import Piece


class Board:
    def __init__(self):
        # 标准中国象棋棋盘大小：10 行 9 列。
        self.rows = 10
        self.cols = 9
        self.board = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        # 本实现中红方先走。
        self.current_player = "red"
        # 活跃棋子坐标（与 grid 同步；供走法生成 / 将军判定避免 90 格扫描）
        self.active_pieces: Dict[str, Set[Tuple[int, int]]] = {"red": set(), "black": set()}
        # 将帅坐标由走子同步维护，供 Rules 免全盘扫描；被吃后为 None
        self.red_king_pos: Optional[Tuple[int, int]] = (9, 4)
        self.black_king_pos: Optional[Tuple[int, int]] = (0, 4)
        # Zobrist 局面键：由走子异或增量维护；置换表可直接用此值
        self.zobrist_hash: int = 0
        # 当前对局路径上各 Zobrist 键出现次数（apply 增、undo 减；O(1) 查询重复）
        self.state_counts: Dict[int, int] = {}
        self.init_board()

    def init_board(self):
        self.state_counts.clear()
        self.active_pieces["red"].clear()
        self.active_pieces["black"].clear()
        # 初始化黑方棋子在上方。
        self.board[0][0] = Piece("black", "che")
        self.board[0][1] = Piece("black", "ma")
        self.board[0][2] = Piece("black", "xiang")
        self.board[0][3] = Piece("black", "shi")
        self.board[0][4] = Piece("black", "jiang")
        self.board[0][5] = Piece("black", "shi")
        self.board[0][6] = Piece("black", "xiang")
        self.board[0][7] = Piece("black", "ma")
        self.board[0][8] = Piece("black", "che")

        self.board[2][1] = Piece("black", "pao")
        self.board[2][7] = Piece("black", "pao")

        self.board[3][0] = Piece("black", "bing")
        self.board[3][2] = Piece("black", "bing")
        self.board[3][4] = Piece("black", "bing")
        self.board[3][6] = Piece("black", "bing")
        self.board[3][8] = Piece("black", "bing")

        # 初始化红方棋子在下方。
        self.board[9][0] = Piece("red", "che")
        self.board[9][1] = Piece("red", "ma")
        self.board[9][2] = Piece("red", "xiang")
        self.board[9][3] = Piece("red", "shi")
        self.board[9][4] = Piece("red", "jiang")
        self.board[9][5] = Piece("red", "shi")
        self.board[9][6] = Piece("red", "xiang")
        self.board[9][7] = Piece("red", "ma")
        self.board[9][8] = Piece("red", "che")

        self.board[7][1] = Piece("red", "pao")
        self.board[7][7] = Piece("red", "pao")

        self.board[6][0] = Piece("red", "bing")
        self.board[6][2] = Piece("red", "bing")
        self.board[6][4] = Piece("red", "bing")
        self.board[6][6] = Piece("red", "bing")
        self.board[6][8] = Piece("red", "bing")

        self.red_king_pos = (9, 4)
        self.black_king_pos = (0, 4)
        for r in range(self.rows):
            for c in range(self.cols):
                p = self.board[r][c]
                if p is not None:
                    self.active_pieces[p.color].add((r, c))
        self.zobrist_hash = zobrist.full_hash(self)
        self.state_counts[self.zobrist_hash] = 1

    def get_repetition_count(self) -> int:
        return self.state_counts.get(self.zobrist_hash, 0)

    def get_piece(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.board[row][col]
        return None

    def set_piece(self, row: int, col: int, piece: Optional[Piece]) -> None:
        """Set a square to a specific piece (or None).

        This is a pure state operation; callers are responsible for ensuring
        consistency (primarily used by Rules during simulation and by tests).
        Does not update ``active_pieces`` / ``zobrist_hash`` / king positions.
        """

        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.board[row][col] = piece

    def toggle_player(self) -> None:
        """切换行棋方并同步更新 Zobrist 哈希（空步剪枝专用）。"""
        self.current_player = "black" if self.current_player == "red" else "red"
        self.zobrist_hash ^= zobrist.BLACK_TO_MOVE

    def apply_move(self, start_row, start_col, end_row, end_col):
        """Execute a move without legality checking.

        This is a renamed version of the old `make_move`.
        It only mutates the grid and switches `current_player`.
        """

        piece = self.board[start_row][start_col]
        if piece is None:
            return None
        captured = self.board[end_row][end_col]
        sq_s = start_row * 9 + start_col
        sq_e = end_row * 9 + end_col
        h = self.zobrist_hash
        h ^= zobrist.piece_key(sq_s, piece)
        if captured is not None:
            h ^= zobrist.piece_key(sq_e, captured)
        h ^= zobrist.piece_key(sq_e, piece)
        h ^= zobrist.BLACK_TO_MOVE
        self.zobrist_hash = h
        mover = piece.color
        opp = "black" if mover == "red" else "red"
        self.active_pieces[mover].discard((start_row, start_col))
        if captured is not None:
            self.active_pieces[opp].discard((end_row, end_col))
        self.active_pieces[mover].add((end_row, end_col))
        if captured is not None and captured.piece_type == "jiang":
            if captured.color == "red":
                self.red_king_pos = None
            else:
                self.black_king_pos = None
        self.board[end_row][end_col] = piece
        self.board[start_row][start_col] = None
        if piece.piece_type == "jiang":
            if piece.color == "red":
                self.red_king_pos = (end_row, end_col)
            else:
                self.black_king_pos = (end_row, end_col)
        self.current_player = "black" if self.current_player == "red" else "red"
        h = self.zobrist_hash
        self.state_counts[h] = self.state_counts.get(h, 0) + 1
        return captured

    def undo_move(self, start_row, start_col, end_row, end_col, captured):
        # 撤销刚才的走子，还原被吃棋子和当前执子方。
        h_leave = self.zobrist_hash
        c = self.state_counts.get(h_leave, 0)
        if c <= 1:
            self.state_counts.pop(h_leave, None)
        else:
            self.state_counts[h_leave] = c - 1
        piece = self.board[end_row][end_col]
        sq_s = start_row * 9 + start_col
        sq_e = end_row * 9 + end_col
        h = self.zobrist_hash
        h ^= zobrist.BLACK_TO_MOVE
        h ^= zobrist.piece_key(sq_e, piece)
        if captured is not None:
            h ^= zobrist.piece_key(sq_e, captured)
        h ^= zobrist.piece_key(sq_s, piece)
        self.zobrist_hash = h
        mover = piece.color
        opp = "black" if mover == "red" else "red"
        self.active_pieces[mover].discard((end_row, end_col))
        if captured is not None:
            self.active_pieces[opp].add((end_row, end_col))
        self.active_pieces[mover].add((start_row, start_col))
        self.board[start_row][start_col] = piece
        self.board[end_row][end_col] = captured
        if piece is not None and piece.piece_type == "jiang":
            if piece.color == "red":
                self.red_king_pos = (start_row, start_col)
            else:
                self.black_king_pos = (start_row, start_col)
        if captured is not None and captured.piece_type == "jiang":
            if captured.color == "red":
                self.red_king_pos = (end_row, end_col)
            else:
                self.black_king_pos = (end_row, end_col)
        self.current_player = "black" if self.current_player == "red" else "red"

    def copy(self):
        # 浅拷贝网格引用（Piece 视为不可变）；避免 deepcopy 在辅助逻辑里拖慢性能。
        new_board = Board()
        new_board.board = [row[:] for row in self.board]
        new_board.current_player = self.current_player
        new_board.red_king_pos = self.red_king_pos
        new_board.black_king_pos = self.black_king_pos
        new_board.zobrist_hash = self.zobrist_hash
        new_board.state_counts = dict(self.state_counts)
        new_board.active_pieces = {
            "red": set(self.active_pieces["red"]),
            "black": set(self.active_pieces["black"]),
        }
        return new_board

    def column_mirror_copy(self) -> "Board":
        """左右镜像棋盘（列 ``c`` → ``8-c``），子力不变；用于开局库 Zobrist 对称查表。"""
        nb = Board.__new__(Board)
        nb.rows = 10
        nb.cols = 9
        nb.board = [[None] * 9 for _ in range(10)]
        nb.active_pieces = {"red": set(), "black": set()}
        nb.red_king_pos = None
        nb.black_king_pos = None
        for r in range(10):
            for c in range(9):
                p = self.board[r][c]
                if p is None:
                    continue
                mc = 8 - c
                nb.board[r][mc] = p
                nb.active_pieces[p.color].add((r, mc))
                if p.piece_type == "jiang":
                    if p.color == "red":
                        nb.red_king_pos = (r, mc)
                    else:
                        nb.black_king_pos = (r, mc)
        nb.current_player = self.current_player
        nb.zobrist_hash = zobrist.full_hash(nb)
        nb.state_counts = {nb.zobrist_hash: 1}
        return nb

    def __str__(self):
        # 将棋盘渲染为简单文本表格，便于在控制台打印查看。
        result = ""
        for r in range(self.rows):
            result += f"{r} "
            for c in range(self.cols):
                piece = self.board[r][c]
                if piece:
                    result += f"{piece} "
                else:
                    result += "· "
            result += "\n"
        result += "  a b c d e f g h i\n"
        return result

