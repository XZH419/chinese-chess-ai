"""棋盘状态容器（模型层 Model）。

本文件仅保留**纯状态管理**职责：

- 棋盘网格的初始化与开局布局
- ``get_piece`` / ``set_piece``：单格读写
- ``apply_move``（原 ``make_move``）：执行走子，不做合法性校验
- ``undo_move``：撤销走子，还原棋盘
- ``copy`` / ``column_mirror_copy``：棋盘复制与镜像

"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

from engine import zobrist
from engine.piece import Piece


class Board:
    """中国象棋棋盘状态容器。

    负责维护 10×9 网格、当前行棋方、活跃棋子集合、将帅坐标、
    Zobrist 哈希及局面重复计数。所有规则判定不在此类中，
    而是委托给 ``Rules`` 静态方法。

    Attributes:
        rows: 棋盘行数（固定为 10）。
        cols: 棋盘列数（固定为 9）。
        board: 10×9 二维列表，每格为 ``Piece`` 或 ``None``。
        current_player: 当前行棋方，``"red"`` 或 ``"black"``。
        active_pieces: 各方活跃棋子坐标集合，键为颜色字符串。
            供走法生成和将军判定使用，避免遍历全部 90 格。
        red_king_pos: 红方将帅坐标，被吃后为 ``None``。
        black_king_pos: 黑方将帅坐标，被吃后为 ``None``。
        zobrist_hash: 当前局面的 Zobrist 哈希值，由走子异或增量维护。
        state_counts: 当前对局路径上各 Zobrist 键的出现次数，
            ``apply_move`` 时递增、``undo_move`` 时递减，用于 O(1) 查询局面重复。
    """

    def __init__(self):
        """初始化棋盘，创建标准开局布局。

        标准中国象棋棋盘大小为 10 行 9 列。
        红方先手，棋子位于棋盘下半部分（第 6–9 行），
        黑方棋子位于棋盘上半部分（第 0–3 行）。
        """
        # 标准中国象棋棋盘大小：10 行 9 列
        self.rows = 10
        self.cols = 9
        self.board = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        # 本实现中红方先走
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
        """将棋盘重置为标准中国象棋初始局面。

        清空所有状态，按照标准规则摆放 32 枚棋子：
        - 黑方（第 0–3 行）：车马象士将士象马车 + 炮×2 + 卒×5
        - 红方（第 6–9 行）：车马相仕帅仕相马车 + 炮×2 + 兵×5

        同时重新计算 Zobrist 哈希并初始化局面计数。
        """
        self.state_counts.clear()
        self.active_pieces["red"].clear()
        self.active_pieces["black"].clear()
        # 初始化黑方棋子在上方（第 0 行：主力，第 2 行：炮，第 3 行：卒）
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

        # 初始化红方棋子在下方（第 9 行：主力，第 7 行：炮，第 6 行：兵）
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

    def piece_count(self) -> int:
        """统计棋盘上当前存活棋子总数。

        Returns:
            红方与黑方活跃棋子数量之和。
        """
        return len(self.active_pieces["red"]) + len(self.active_pieces["black"])

    def get_repetition_count(self) -> int:
        """获取当前局面在对局路径中的重复出现次数。

        通过 Zobrist 哈希在 ``state_counts`` 中进行 O(1) 查找。

        Returns:
            当前局面哈希在对局历史中出现的次数；首次出现返回 1，未记录返回 0。
        """
        return self.state_counts.get(self.zobrist_hash, 0)

    def get_piece(self, row, col):
        """获取指定坐标处的棋子。

        Args:
            row: 行坐标（0–9）。
            col: 列坐标（0–8）。

        Returns:
            对应位置的 ``Piece`` 对象；坐标越界或该格为空时返回 ``None``。
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.board[row][col]
        return None

    def set_piece(self, row: int, col: int, piece: Optional[Piece]) -> None:
        """将指定格设置为特定棋子（或清空为 ``None``）。

        这是一个纯状态写操作，调用者需自行保证一致性
        （主要由 ``Rules`` 在模拟过程中和测试用例使用）。

        注意：本方法**不会**同步更新 ``active_pieces``、``zobrist_hash`` 及将帅坐标，
        这些附属状态的维护由调用者负责。

        Args:
            row: 行坐标（0–9）。
            col: 列坐标（0–8）。
            piece: 要放置的棋子对象，传入 ``None`` 表示清空该格。
        """

        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.board[row][col] = piece

    def toggle_player(self) -> None:
        """切换行棋方并同步更新 Zobrist 哈希。

        专为空步剪枝（Null Move Pruning）设计：仅翻转行棋方，
        不执行任何棋子移动。Zobrist 哈希通过异或 ``BLACK_TO_MOVE``
        常量来反映行棋方变化。
        """
        self.current_player = "black" if self.current_player == "red" else "red"
        self.zobrist_hash ^= zobrist.BLACK_TO_MOVE

    def apply_move(self, start_row, start_col, end_row, end_col):
        """执行一步走子（不做合法性校验）。

        这是旧 ``make_move`` 方法的重命名版本。仅修改棋盘网格状态
        并切换 ``current_player``，不进行任何规则判定。

        内部维护以下附属状态的一致性：
        1. Zobrist 哈希：异或移除起点棋子、异或移除终点被吃棋子（如有）、
           异或添加终点棋子、异或翻转行棋方
        2. ``active_pieces``：更新走子方与被吃方的坐标集合
        3. 将帅坐标：若移动的是将/帅则更新位置，若被吃的是将/帅则置 ``None``
        4. ``state_counts``：递增新局面哈希的出现次数

        Args:
            start_row: 起点行坐标。
            start_col: 起点列坐标。
            end_row: 终点行坐标。
            end_col: 终点列坐标。

        Returns:
            被吃掉的棋子（``Piece``），若终点为空则返回 ``None``。
            起点无棋子时也返回 ``None``（异常情况）。
        """

        piece = self.board[start_row][start_col]
        if piece is None:
            return None
        captured = self.board[end_row][end_col]
        # 将二维坐标映射为一维索引，用于 Zobrist 查表
        sq_s = start_row * 9 + start_col
        sq_e = end_row * 9 + end_col
        # 增量更新 Zobrist 哈希：异或掉起点棋子、终点被吃子，异或入终点新棋子和行棋方标志
        h = self.zobrist_hash
        h ^= zobrist.piece_key(sq_s, piece)
        if captured is not None:
            h ^= zobrist.piece_key(sq_e, captured)
        h ^= zobrist.piece_key(sq_e, piece)
        h ^= zobrist.BLACK_TO_MOVE
        self.zobrist_hash = h
        # 同步更新活跃棋子集合
        mover = piece.color
        opp = "black" if mover == "red" else "red"
        self.active_pieces[mover].discard((start_row, start_col))
        if captured is not None:
            self.active_pieces[opp].discard((end_row, end_col))
        self.active_pieces[mover].add((end_row, end_col))
        # 若被吃的是将/帅，将对应坐标置空（标记将/帅已被吃）
        if captured is not None and captured.piece_type == "jiang":
            if captured.color == "red":
                self.red_king_pos = None
            else:
                self.black_king_pos = None
        # 更新网格状态
        self.board[end_row][end_col] = piece
        self.board[start_row][start_col] = None
        # 若移动的是将/帅，同步更新将帅坐标
        if piece.piece_type == "jiang":
            if piece.color == "red":
                self.red_king_pos = (end_row, end_col)
            else:
                self.black_king_pos = (end_row, end_col)
        # 切换行棋方
        self.current_player = "black" if self.current_player == "red" else "red"
        # 递增新局面哈希的出现次数（用于重复局面判定）
        h = self.zobrist_hash
        self.state_counts[h] = self.state_counts.get(h, 0) + 1
        return captured

    def undo_move(self, start_row, start_col, end_row, end_col, captured):
        """撤销一步走子，将棋盘完整还原至走子前的状态。

        操作顺序为 ``apply_move`` 的严格逆序：先递减局面计数，
        再逆向恢复 Zobrist 哈希、活跃棋子集合、网格内容、
        将帅坐标以及行棋方。

        Args:
            start_row: 原始起点行坐标（走子前棋子所在行）。
            start_col: 原始起点列坐标（走子前棋子所在列）。
            end_row: 原始终点行坐标（走子后棋子所在行）。
            end_col: 原始终点列坐标（走子后棋子所在列）。
            captured: 被吃掉的棋子（由 ``apply_move`` 返回），
                若该步未吃子则传入 ``None``。
        """
        # 先递减当前局面哈希的出现次数；若降为 0 则从字典中移除以节省内存
        h_leave = self.zobrist_hash
        c = self.state_counts.get(h_leave, 0)
        if c <= 1:
            self.state_counts.pop(h_leave, None)
        else:
            self.state_counts[h_leave] = c - 1
        # 逆向恢复 Zobrist 哈希：异或顺序与 apply_move 相反
        piece = self.board[end_row][end_col]
        if piece is None:
            raise RuntimeError(
                "undo_move failed: destination square is empty. "
                f"move=({start_row},{start_col})->({end_row},{end_col}), "
                f"captured={'None' if captured is None else (captured.color + ':' + captured.piece_type)}, "
                f"current_player={self.current_player!r}, zobrist={self.zobrist_hash:#x}"
            )
        sq_s = start_row * 9 + start_col
        sq_e = end_row * 9 + end_col
        h = self.zobrist_hash
        h ^= zobrist.BLACK_TO_MOVE
        h ^= zobrist.piece_key(sq_e, piece)
        if captured is not None:
            h ^= zobrist.piece_key(sq_e, captured)
        h ^= zobrist.piece_key(sq_s, piece)
        self.zobrist_hash = h
        # 逆向恢复活跃棋子集合
        mover = piece.color
        opp = "black" if mover == "red" else "red"
        self.active_pieces[mover].discard((end_row, end_col))
        if captured is not None:
            self.active_pieces[opp].add((end_row, end_col))
        self.active_pieces[mover].add((start_row, start_col))
        # 还原网格：棋子回到起点，终点恢复被吃子（或空）
        self.board[start_row][start_col] = piece
        self.board[end_row][end_col] = captured
        # 若移动的是将/帅，还原其坐标到起点
        if piece is not None and piece.piece_type == "jiang":
            if piece.color == "red":
                self.red_king_pos = (start_row, start_col)
            else:
                self.black_king_pos = (start_row, start_col)
        # 若被吃的是将/帅，恢复其坐标到终点（即被吃前的位置）
        if captured is not None and captured.piece_type == "jiang":
            if captured.color == "red":
                self.red_king_pos = (end_row, end_col)
            else:
                self.black_king_pos = (end_row, end_col)
        # 切换回走子前的行棋方
        self.current_player = "black" if self.current_player == "red" else "red"

    def copy(self):
        """创建当前棋盘的浅拷贝副本。

        对网格采用行级浅拷贝（``Piece`` 对象视为不可变，无需深拷贝），
        避免 ``deepcopy`` 在辅助逻辑（如 MCTS 模拟）中造成的性能瓶颈。
        ``active_pieces`` 和 ``state_counts`` 均创建新的容器副本。

        Returns:
            一个独立的 ``Board`` 实例，与原棋盘共享 ``Piece`` 对象引用，
            但网格、集合、字典等可变容器均为独立副本。
        """
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
        """创建当前棋盘的左右镜像副本（列 ``c`` → ``8-c``），子力不变。

        主要用于开局库的 Zobrist 对称查表：对称开局只需存储一侧，
        查询时同时查原局面与镜像局面的哈希即可。

        使用 ``__new__`` 绕过 ``__init__`` 避免重复初始化开销，
        手动构建所有属性。

        Returns:
            左右镜像的新 ``Board`` 实例，拥有独立的网格和状态。
        """
        new_board = Board.__new__(Board)
        new_board.rows = 10
        new_board.cols = 9
        new_board.board = [[None] * 9 for _ in range(10)]
        new_board.active_pieces = {"red": set(), "black": set()}
        new_board.red_king_pos = None
        new_board.black_king_pos = None
        for r in range(10):
            for c in range(9):
                p = self.board[r][c]
                if p is None:
                    continue
                mc = 8 - c
                new_board.board[r][mc] = p
                new_board.active_pieces[p.color].add((r, mc))
                if p.piece_type == "jiang":
                    if p.color == "red":
                        new_board.red_king_pos = (r, mc)
                    else:
                        new_board.black_king_pos = (r, mc)
        new_board.current_player = self.current_player
        new_board.zobrist_hash = zobrist.full_hash(new_board)
        new_board.state_counts = {new_board.zobrist_hash: 1}
        return new_board

    def __str__(self):
        """将棋盘渲染为可读的文本表格，便于在控制台调试输出。

        格式说明：
        - 每行以行号开头，后跟 9 列棋子符号
        - 空格用 ``·`` 表示
        - 底部标注列标 a–i

        Returns:
            棋盘的多行字符串表示。
        """
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
