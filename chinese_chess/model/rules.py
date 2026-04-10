"""中国象棋规则引擎（Model / Rules）。

本文件从旧实现 `chinese-chess/ai/board.py` “物理搬运”而来，职责是：
- **所有规则判定**：每个棋子的走法合法性、将军判定、将帅对面（白脸将）判定
- **走法生成**：get_all_moves / get_legal_moves / get_pseudo_legal_moves（AI 伪合法）
- **终局判定**：将被吃掉、无子可走（困毙/将死）

注意：
- 迁移过程中尽量不改核心判断/数学逻辑；主要是把 `self.*` 改为 `board.*`
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

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
        if not (
            0 <= start_row < 10
            and 0 <= start_col < 9
            and 0 <= end_row < 10
            and 0 <= end_col < 9
        ):
            return False

        b = board.board
        piece = b[start_row][start_col]
        if not piece or piece.color != player:
            return False

        target = b[end_row][end_col]
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

        # ===== 关键：利用核心 Board 状态机进行无损模拟 =====
        # 使用 apply_move 可以确保 grid, active_pieces, king_pos, zobrist_hash 全部精准同步
        captured = board.apply_move(start_row, start_col, end_row, end_col)

        # apply_move 会将 current_player 切换给对手
        # 但我们需要验证的是原本走子的 player（即传参进来的 player）是否处于被将军状态
        kings_facing = Rules._jiang_face_to_face(board)
        in_check = Rules.is_king_in_check(board, player)

        board.undo_move(start_row, start_col, end_row, end_col, captured)

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
        if board.board[eye_r][eye_c]:
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
        if board.board[leg_r][leg_c]:
            return False
        return True

    @staticmethod
    def _is_valid_che_move(board: Board, sr, sc, er, ec):
        # 车直线移动，路径上不能有阻挡棋子。
        if sr != er and sc != ec:
            return False
        b = board.board
        if sr == er:
            step = 1 if ec > sc else -1
            for c in range(sc + step, ec, step):
                if b[sr][c]:
                    return False
        else:
            step = 1 if er > sr else -1
            for r in range(sr + step, er, step):
                if b[r][sc]:
                    return False
        return True

    @staticmethod
    def _is_valid_pao_move(board: Board, sr, sc, er, ec):
        # 炮：
        # - 不吃子：直线走，路径无子
        # - 吃子：直线走，且中间必须**恰好隔一个子（炮架）**
        if sr != er and sc != ec:
            return False
        b = board.board
        target = b[er][ec]
        count = 0
        if sr == er:
            step = 1 if ec > sc else -1
            for c in range(sc + step, ec, step):
                if b[sr][c]:
                    count += 1
        else:
            step = 1 if er > sr else -1
            for r in range(sr + step, er, step):
                if b[r][sc]:
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
        # 使用 Board 维护的将帅坐标，避免全盘扫描。
        rk = board.red_king_pos
        bk = board.black_king_pos
        if rk is None or bk is None:
            return False
        if rk[1] != bk[1]:
            return False
        col = rk[1]
        min_r = min(rk[0], bk[0])
        max_r = max(rk[0], bk[0])
        for r in range(min_r + 1, max_r):
            if board.board[r][col] is not None:
                return False
        return True

    @staticmethod
    def get_all_moves(board: Board, player, validate_self_check=True):
        # 生成指定方的所有合法走法。
        # 如果 validate_self_check 为 True，则返回的走法不会使己方被将军。
        moves = []
        for r, c in board.active_pieces[player]:
            piece = board.board[r][c]
            if not piece or piece.color != player:
                continue
            candidates = Rules._candidate_targets(board, r, c, piece.piece_type, player)
            for er, ec in candidates:
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
    def _candidate_targets(board: Board, r: int, c: int, piece_type: str, player: str):
        """按棋子类型生成候选落点（不保证合法，后续用 is_valid_move 过滤）。

        目标：显著降低 get_all_moves 的复杂度，使 Minimax 深度 3 在 Python 中可用。
        """

        cand = []

        def add(rr, cc):
            if 0 <= rr < 10 and 0 <= cc < 9:
                cand.append((rr, cc))

        if piece_type == "jiang":
            add(r - 1, c)
            add(r + 1, c)
            add(r, c - 1)
            add(r, c + 1)
            # 将帅对面“照面吃”由 is_valid_move 的 legality 检查处理
            return cand

        if piece_type == "shi":
            add(r - 1, c - 1)
            add(r - 1, c + 1)
            add(r + 1, c - 1)
            add(r + 1, c + 1)
            return cand

        if piece_type == "xiang":
            add(r - 2, c - 2)
            add(r - 2, c + 2)
            add(r + 2, c - 2)
            add(r + 2, c + 2)
            return cand

        if piece_type == "ma":
            # 8 个日字落点，蹩马脚由 is_valid_move 内部处理
            for dr, dc in [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]:
                add(r + dr, c + dc)
            return cand

        if piece_type in ("che", "pao"):
            # 车/炮：四个方向一直走到边界；炮的“隔子吃”由 is_valid_move 内部处理
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr, cc = r + dr, c + dc
                while 0 <= rr < 10 and 0 <= cc < 9:
                    cand.append((rr, cc))
                    rr += dr
                    cc += dc
            return cand

        if piece_type == "bing":
            # 兵/卒：最多 3 个候选点（过河后可横走）
            if player == "red":
                add(r - 1, c)
                if r <= 4:
                    add(r, c - 1)
                    add(r, c + 1)
            else:
                add(r + 1, c)
                if r >= 5:
                    add(r, c - 1)
                    add(r, c + 1)
            return cand

        return cand

    @staticmethod
    def get_pseudo_legal_moves(board: Board, player: str) -> Iterator[Tuple[int, int, int, int]]:
        """伪合法走法生成（仅几何与吃子颜色），供 AI 搜索批量过滤。

        不调用 ``get_piece`` / ``_inside_board`` / ``is_valid_move``；不校验自将、白脸将。
        产出 ``(r, c, nr, nc)``，与 ``get_all_moves`` 元组格式一致。
        """
        b = board.board

        def dest_ok(cell) -> bool:
            return cell is None or cell.color != player

        for r, c in board.active_pieces[player]:
            p = b[r][c]
            if p is None or p.color != player:
                continue
            pt = p.piece_type

            if pt == "che":
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < 10 and 0 <= nc < 9:
                        cell = b[nr][nc]
                        if cell is None:
                            yield (r, c, nr, nc)
                        else:
                            if cell.color != player:
                                yield (r, c, nr, nc)
                            break
                        nr += dr
                        nc += dc

            elif pt == "pao":
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    seen_screen = False
                    while 0 <= nr < 10 and 0 <= nc < 9:
                        cell = b[nr][nc]
                        if cell is None:
                            if not seen_screen:
                                yield (r, c, nr, nc)
                        else:
                            if not seen_screen:
                                seen_screen = True
                            else:
                                if cell.color != player:
                                    yield (r, c, nr, nc)
                                break
                        nr += dr
                        nc += dc

            elif pt == "ma":
                for dr, dc in (
                    (2, 1),
                    (2, -1),
                    (-2, 1),
                    (-2, -1),
                    (1, 2),
                    (1, -2),
                    (-1, 2),
                    (-1, -2),
                ):
                    nr, nc = r + dr, c + dc
                    if abs(dr) == 2:
                        lr, lc = r + dr // 2, c
                    else:
                        lr, lc = r, c + dc // 2
                    if not (0 <= lr < 10 and 0 <= lc < 9):
                        continue
                    if b[lr][lc] is not None:
                        continue
                    if not (0 <= nr < 10 and 0 <= nc < 9):
                        continue
                    dest = b[nr][nc]
                    if dest_ok(dest):
                        yield (r, c, nr, nc)

            elif pt == "xiang":
                for dr, dc in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < 10 and 0 <= nc < 9):
                        continue
                    if player == "red":
                        if nr < 5:
                            continue
                    else:
                        if nr > 4:
                            continue
                    ir, ic = (r + nr) // 2, (c + nc) // 2
                    if b[ir][ic]:
                        continue
                    dest = b[nr][nc]
                    if dest_ok(dest):
                        yield (r, c, nr, nc)

            elif pt == "shi":
                for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < 10 and 0 <= nc < 9):
                        continue
                    if player == "red":
                        if not (7 <= nr <= 9 and 3 <= nc <= 5):
                            continue
                    elif not (0 <= nr <= 2 and 3 <= nc <= 5):
                        continue
                    dest = b[nr][nc]
                    if dest_ok(dest):
                        yield (r, c, nr, nc)

            elif pt == "jiang":
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < 10 and 0 <= nc < 9):
                        continue
                    if player == "red":
                        if not (7 <= nr <= 9 and 3 <= nc <= 5):
                            continue
                    elif not (0 <= nr <= 2 and 3 <= nc <= 5):
                        continue
                    dest = b[nr][nc]
                    if dest_ok(dest):
                        yield (r, c, nr, nc)

            elif pt == "bing":
                if player == "red":
                    cand = [(r - 1, c)]
                    if r <= 4:
                        cand.append((r, c - 1))
                        cand.append((r, c + 1))
                else:
                    cand = [(r + 1, c)]
                    if r >= 5:
                        cand.append((r, c - 1))
                        cand.append((r, c + 1))
                for nr, nc in cand:
                    if not (0 <= nr < 10 and 0 <= nc < 9):
                        continue
                    dest = b[nr][nc]
                    if dest_ok(dest):
                        yield (r, c, nr, nc)

    @staticmethod
    def get_legal_moves(board: Board, player):
        return Rules.get_all_moves(board, player, validate_self_check=True)

    @staticmethod
    def is_king_in_check(board: Board, player: str) -> bool:
        """从将/帅位置反向射线与定点检测是否被将军（不调用 is_valid_move、不扫 active_pieces）。"""
        if player == "red":
            jiang_pos = board.red_king_pos
        else:
            jiang_pos = board.black_king_pos
        if not jiang_pos:
            return True
        kr, kc = jiang_pos
        opponent = "black" if player == "red" else "red"
        b = board.board

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            obstacles = 0
            r, c = kr + dr, kc + dc
            while 0 <= r < 10 and 0 <= c < 9:
                p = b[r][c]
                if p is not None:
                    if obstacles == 0:
                        if p.color == opponent and p.piece_type == "che":
                            return True
                        obstacles = 1
                    else:
                        if p.color == opponent and p.piece_type == "pao":
                            return True
                        break
                r += dr
                c += dc

        for dr, dc, leg_dr, leg_dc in (
            (-2, -1, -1, 0),
            (-2, 1, -1, 0),
            (2, -1, 1, 0),
            (2, 1, 1, 0),
            (-1, -2, 0, -1),
            (1, -2, 0, -1),
            (-1, 2, 0, 1),
            (1, 2, 0, 1),
        ):
            hr, hc = kr + dr, kc + dc
            if not (0 <= hr < 10 and 0 <= hc < 9):
                continue
            lr, lc = kr + leg_dr, kc + leg_dc
            if not (0 <= lr < 10 and 0 <= lc < 9):
                continue
            if b[lr][lc] is not None:
                continue
            hp = b[hr][hc]
            if hp is not None and hp.color == opponent and hp.piece_type == "ma":
                return True

        if player == "red":
            for pr, pc in ((kr - 1, kc), (kr, kc - 1), (kr, kc + 1)):
                if 0 <= pr < 10 and 0 <= pc < 9:
                    pp = b[pr][pc]
                    if (
                        pp is not None
                        and pp.color == opponent
                        and pp.piece_type == "bing"
                    ):
                        return True
        else:
            for pr, pc in ((kr + 1, kc), (kr, kc - 1), (kr, kc + 1)):
                if 0 <= pr < 10 and 0 <= pc < 9:
                    pp = b[pr][pc]
                    if (
                        pp is not None
                        and pp.color == opponent
                        and pp.piece_type == "bing"
                    ):
                        return True

        return False

    @staticmethod
    def is_check(board: Board, player):
        """判断 player 方是否被将军（与 is_king_in_check 语义相同，保留旧名）。"""
        return Rules.is_king_in_check(board, player)

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
    def is_threefold_repetition_draw(board: Board, position_history: list) -> bool:
        """同一局面（Zobrist）在局面链中出现至少 3 次时判和（简易防循环）。

        注意：Minimax 在搜索路径上遇到重复局面时，用
        ``Evaluation.repetition_leaf_score`` 给分（大优厌战、大劣接受和棋），
        与终局层面的本布尔判定相互独立。
        """
        if not position_history:
            return False
        h = board.zobrist_hash
        return sum(1 for x in position_history if x == h) >= 3

    @staticmethod
    def is_game_over(board: Board, position_history: Optional[list] = None):
        # 游戏结束条件：
        # - 任意一方将/帅被吃
        # - 轮到走子的一方无合法走法（困毙/将死）
        # - 可选：三次重复局面判和
        if Rules.winner(board) is not None:
            return True
        if position_history is not None and Rules.is_threefold_repetition_draw(
            board, position_history
        ):
            return True
        return False

