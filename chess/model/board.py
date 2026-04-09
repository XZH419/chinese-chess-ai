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

from typing import Optional

from .piece import Piece


class Board:
    def __init__(self):
        # 标准中国象棋棋盘大小：10 行 9 列。
        self.rows = 10
        self.cols = 9
        self.board = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        # 本实现中红方先走。
        self.current_player = "red"
        self.init_board()

    def init_board(self):
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

    def get_piece(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.board[row][col]
        return None

    def set_piece(self, row: int, col: int, piece: Optional[Piece]) -> None:
        """Set a square to a specific piece (or None).

        This is a pure state operation; callers are responsible for ensuring
        consistency (primarily used by Rules during simulation and by tests).
        """

        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.board[row][col] = piece

    def _inside_board(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def apply_move(self, start_row, start_col, end_row, end_col):
        """Execute a move without legality checking.

        This is a renamed version of the old `make_move`.
        It only mutates the grid and switches `current_player`.
        """

        piece = self.board[start_row][start_col]
        if piece is None:
            return None
        captured = self.board[end_row][end_col]
        self.board[end_row][end_col] = piece
        self.board[start_row][start_col] = None
        self.current_player = "black" if self.current_player == "red" else "red"
        return captured

    def undo_move(self, start_row, start_col, end_row, end_col, captured):
        # 撤销刚才的走子，还原被吃棋子和当前执子方。
        piece = self.board[end_row][end_col]
        self.board[start_row][start_col] = piece
        self.board[end_row][end_col] = captured
        self.current_player = "black" if self.current_player == "red" else "red"

    def copy(self):
        # 创建棋盘的深拷贝，用于搜索或模拟而不影响原棋盘状态。
        import copy

        new_board = Board()
        new_board.board = copy.deepcopy(self.board)
        new_board.current_player = self.current_player
        return new_board

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

