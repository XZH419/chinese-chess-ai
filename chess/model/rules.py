"""中国象棋规则引擎（Model / Rules）。

本文件从旧实现 `chinese-chess/ai/board.py` “物理搬运”而来，职责是：
- **所有规则判定**：每个棋子的走法合法性、将军判定、将帅对面（白脸将）判定
- **走法生成**：get_all_moves / get_legal_moves
- **终局判定**：将被吃掉、无子可走（困毙/将死）

注意：
- 迁移过程中尽量不改核心判断/数学逻辑；主要是把 `self.*` 改为 `board.*`
"""

from __future__ import annotations

from .board import Board


class Rules:
    """All rules as static methods to avoid stateful coupling."""

    @staticmethod
    def is_valid_move(
        board: Board,
        start_row,
        start_col,
        end_row,
        end_col,
        player=None,
        check_legality=True,
    ):
        """检查指定棋子走子是否合法（严格版）。

        如果 check_legality 为 True，还会验证走子后是否导致己方将帅被将军，
        或者是否出现对面将帅见面的非法局面（**白脸将拦截**）。
        """

        player = player or board.current_player
        if not board._inside_board(start_row, start_col) or not board._inside_board(
            end_row, end_col
        ):
            return False

        piece = board.get_piece(start_row, start_col)
        if not piece or piece.color != player:
            return False

        target = board.get_piece(end_row, end_col)
        if target and target.color == piece.color:
            return False

        if piece.piece_type == "jiang":
            valid = Rules._is_valid_jiang_move(
                board, start_row, start_col, end_row, end_col, player
            )
        elif piece.piece_type == "shi":
            valid = Rules._is_valid_shi_move(
                board, start_row, start_col, end_row, end_col, player
            )
        elif piece.piece_type == "xiang":
            valid = Rules._is_valid_xiang_move(
                board, start_row, start_col, end_row, end_col, player
            )
        elif piece.piece_type == "ma":
            valid = Rules._is_valid_ma_move(board, start_row, start_col, end_row, end_col)
        elif piece.piece_type == "che":
            valid = Rules._is_valid_che_move(board, start_row, start_col, end_row, end_col)
        elif piece.piece_type == "pao":
            valid = Rules._is_valid_pao_move(board, start_row, start_col, end_row, end_col)
        elif piece.piece_type == "bing":
            valid = Rules._is_valid_bing_move(
                board, start_row, start_col, end_row, end_col, player
            )
        else:
            valid = False

        if not valid:
            return False

        if not check_legality:
            return True

        # ===== 关键：自杀将军 + 白脸将拦截 =====
        # 通过“临时落子”模拟走完这一步后的局面：
        # - 若出现将帅对面（同列且无子遮挡） -> 非法
        # - 若走完后己方将/帅被将军 -> 非法
        captured = board.board[end_row][end_col]
        board.board[end_row][end_col] = piece
        board.board[start_row][start_col] = None
        kings_facing = Rules._jiang_face_to_face(board)
        in_check = Rules.is_check(board, player)
        board.board[start_row][start_col] = piece
        board.board[end_row][end_col] = captured
        if kings_facing or in_check:
            return False

        return True

    @staticmethod
    def _is_valid_jiang_move(board: Board, sr, sc, er, ec, player):
        # 将/帅：九宫格内直走一步（九宫格限制）
        if abs(sr - er) + abs(sc - ec) != 1:
            return False
        if player == "red":
            return 7 <= er <= 9 and 3 <= ec <= 5
        return 0 <= er <= 2 and 3 <= ec <= 5

    @staticmethod
    def _is_valid_shi_move(board: Board, sr, sc, er, ec, player):
        # 士：九宫格内斜走一步（九宫格限制）
        if abs(sr - er) != 1 or abs(sc - ec) != 1:
            return False
        if player == "red":
            return 7 <= er <= 9 and 3 <= ec <= 5
        return 0 <= er <= 2 and 3 <= ec <= 5

    @staticmethod
    def _is_valid_xiang_move(board: Board, sr, sc, er, ec, player):
        # 象：
        # - 走“田”字（行列各走 2）
        # - **塞象眼**：象眼（中点）被占则不能走
        # - **不能过河**：红象不可到 0-4 行，黑象不可到 5-9 行（按本实现坐标系）
        if abs(sr - er) != 2 or abs(sc - ec) != 2:
            return False
        eye_r = (sr + er) // 2
        eye_c = (sc + ec) // 2
        if board.get_piece(eye_r, eye_c):
            return False
        if player == "red":
            return er >= 5
        return er <= 4

    @staticmethod
    def _is_valid_ma_move(board: Board, sr, sc, er, ec):
        # 马：
        # - 走“日”字（2,1）
        # - **蹩马脚**：马腿（中间格）被占则不能走
        dr = abs(sr - er)
        dc = abs(sc - ec)
        if not ((dr == 2 and dc == 1) or (dr == 1 and dc == 2)):
            return False
        if dr == 2:
            leg_r = (sr + er) // 2
            leg_c = sc
        else:
            leg_r = sr
            leg_c = (sc + ec) // 2
        if board.get_piece(leg_r, leg_c):
            return False
        return True

    @staticmethod
    def _is_valid_che_move(board: Board, sr, sc, er, ec):
        # 车直线移动，路径上不能有阻挡棋子。
        if sr != er and sc != ec:
            return False
        if sr == er:
            step = 1 if ec > sc else -1
            for c in range(sc + step, ec, step):
                if board.get_piece(sr, c):
                    return False
        else:
            step = 1 if er > sr else -1
            for r in range(sr + step, er, step):
                if board.get_piece(r, sc):
                    return False
        return True

    @staticmethod
    def _is_valid_pao_move(board: Board, sr, sc, er, ec):
        # 炮：
        # - 不吃子：直线走，路径无子
        # - 吃子：直线走，且中间必须**恰好隔一个子（炮架）**
        if sr != er and sc != ec:
            return False
        target = board.get_piece(er, ec)
        count = 0
        if sr == er:
            step = 1 if ec > sc else -1
            for c in range(sc + step, ec, step):
                if board.get_piece(sr, c):
                    count += 1
        else:
            step = 1 if er > sr else -1
            for r in range(sr + step, er, step):
                if board.get_piece(r, sc):
                    count += 1
        if target:
            return count == 1
        return count == 0

    @staticmethod
    def _is_valid_bing_move(board: Board, sr, sc, er, ec, player):
        # 兵/卒向前一步，过河后可以横着走一步。
        if player == "red":
            if er > sr:
                return False
            if sr <= 4:  # 红兵过河后允许横走。
                return (er == sr - 1 and sc == ec) or (er == sr and abs(sc - ec) == 1)
            return er == sr - 1 and sc == ec
        else:
            if er < sr:
                return False
            if sr >= 5:  # 黑卒过河后允许横走。
                return (er == sr + 1 and sc == ec) or (er == sr and abs(sc - ec) == 1)
            return er == sr + 1 and sc == ec

    @staticmethod
    def _jiang_face_to_face(board: Board):
        # **将帅对面（白脸将）**判定：
        # 若红/黑将帅在同一列，且中间无任何棋子遮挡，则该局面非法
        jiang_pos = None
        shuai_pos = None
        for r in range(board.rows):
            for c in range(board.cols):
                piece = board.board[r][c]
                if piece and piece.piece_type == "jiang":
                    if piece.color == "red":
                        jiang_pos = (r, c)
                    else:
                        shuai_pos = (r, c)
        if not jiang_pos or not shuai_pos:
            return False
        if jiang_pos[1] != shuai_pos[1]:
            return False
        col = jiang_pos[1]
        start = min(jiang_pos[0], shuai_pos[0]) + 1
        end = max(jiang_pos[0], shuai_pos[0])
        for r in range(start, end):
            if board.board[r][col] is not None:
                return False
        return True

    @staticmethod
    def get_all_moves(board: Board, player, validate_self_check=True):
        # 生成指定方的所有合法走法。
        # 如果 validate_self_check 为 True，则返回的走法不会使己方被将军。
        moves = []
        for r in range(board.rows):
            for c in range(board.cols):
                piece = board.board[r][c]
                if piece and piece.color == player:
                    for er in range(board.rows):
                        for ec in range(board.cols):
                            if Rules.is_valid_move(
                                board,
                                r,
                                c,
                                er,
                                ec,
                                player=player,
                                check_legality=validate_self_check,
                            ):
                                moves.append((r, c, er, ec))
        return moves

    @staticmethod
    def get_legal_moves(board: Board, player):
        return Rules.get_all_moves(board, player, validate_self_check=True)

    @staticmethod
    def is_check(board: Board, player):
        # 判断 player 方是否被将军：
        # - 找到 player 的将/帅坐标
        # - 枚举对手所有“伪合法”攻击走法（validate_self_check=False）
        #   若有一步能落到将/帅位置，则为将军
        opponent = "black" if player == "red" else "red"
        jiang_pos = None
        for r in range(board.rows):
            for c in range(board.cols):
                piece = board.board[r][c]
                if piece and piece.color == player and piece.piece_type == "jiang":
                    jiang_pos = (r, c)
                    break
        if not jiang_pos:
            return True
        for move in Rules.get_all_moves(board, opponent, validate_self_check=False):
            if move[2] == jiang_pos[0] and move[3] == jiang_pos[1]:
                return True
        return False

    @staticmethod
    def has_legal_moves(board: Board, player):
        return len(Rules.get_legal_moves(board, player)) > 0

    @staticmethod
    def is_checkmate(board: Board, player):
        return Rules.is_check(board, player) and not Rules.has_legal_moves(board, player)

    @staticmethod
    def is_stalemate(board: Board, player):
        return not Rules.is_check(board, player) and not Rules.has_legal_moves(board, player)

    @staticmethod
    def winner(board: Board):
        """返回胜者（'red'/'black'）或 None（未结束）。

        规则（符合中国象棋常用判负方式）：
        - 将/帅被吃：对方获胜
        - **困毙（无子可走）判负**：轮到谁走但无任何合法走法 -> 对手获胜
          （这也覆盖将死与“困毙”两种情况；与国际象棋 stalemate=draw 不同）
        """

        # 1) 将/帅是否还在
        red_jiang = any(
            piece and piece.color == "red" and piece.piece_type == "jiang"
            for row in board.board
            for piece in row
        )
        black_jiang = any(
            piece and piece.color == "black" and piece.piece_type == "jiang"
            for row in board.board
            for piece in row
        )
        if not red_jiang and black_jiang:
            return "black"
        if not black_jiang and red_jiang:
            return "red"
        if not red_jiang and not black_jiang:
            # 理论上不应出现，保守返回 None
            return None

        # 2) 困毙/将死：轮到谁走但没有合法走法 -> 判负
        if not Rules.has_legal_moves(board, board.current_player):
            return "black" if board.current_player == "red" else "red"

        return None

    @staticmethod
    def is_game_over(board: Board):
        # 游戏结束条件：
        # - 任意一方将/帅被吃
        # - 轮到走子的一方无合法走法（困毙/将死）
        return Rules.winner(board) is not None

