"""中国象棋规则引擎（Model / Rules）。

本文件从旧实现 `chinese-chess/ai/board.py` “物理搬运”而来，职责是：
- **所有规则判定**：每个棋子的走法合法性、将军判定、将帅对面（白脸将）判定
- **走法生成**：get_all_moves / get_legal_moves / get_pseudo_legal_moves（AI 伪合法）
- **终局判定**：将被吃掉、无子可走（困毙/将死）

注意：
- 迁移过程中尽量不改核心判断/数学逻辑；主要是把 `self.*` 改为 `board.*`
"""

from __future__ import annotations

from typing import Iterator, List, NamedTuple, Optional, Tuple

from .board import Board


class MoveEntry(NamedTuple):
    """对局历史中的一条记录，与单步走子一一对应。

    ``history[0]`` 为初始局面（走子前），其 ``mover`` 和 ``gave_check`` 均为 ``None``。
    ``history[i]``（i >= 1）记录第 i 手走后的局面哈希、行棋方及是否将军。
    """
    pos_hash: int
    mover: Optional[str] = None
    gave_check: Optional[bool] = None

# 非法走法说明（供 GUI / Controller 展示）
_MSG_FRIENDLY_FIRE = "友军误伤"
_MSG_BING_BACK = "士卒不得后退"
_MSG_BING_NO_SIDE = "士卒过河前不得横移"
_MSG_MA_BLOCKED = "蹩马腿"
_MSG_XIANG_EYE = "塞象眼"
_MSG_PAO_BAD = "炮无炮架或非法翻山"
_MSG_BAD_GEOMETRY = "棋子走法不合规"
_MSG_NO_CAPTURE_KING = "老将不可被捕获，请通过走法形成将死"
_MSG_KINGS_FACE = "王不见王"
_MSG_IN_CHECK_STILL = "正在被将军，必须移动老将或吃掉将军棋子"
_MSG_SELF_CHECK = "由于飞将或未解将，行动后将处于被将军状态"
_MSG_LONG_CHECK = "长将违规，必须变招"


class Rules:
    """All rules as static methods to avoid stateful coupling."""

    # 马从 (sr,sc) 跃至 (er,ec) 时马腿格（与 ``_geometry_error`` / 伪合法生成一致）
    _MA_ATTACK_DELTAS: Tuple[Tuple[int, int], ...] = (
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
    )

    @staticmethod
    def _ma_leg_square(sr: int, sc: int, er: int, ec: int) -> Tuple[int, int]:
        """马腿坐标：沿「走两格」的那条线取中点（行列不可反）。"""
        if abs(sr - er) == 2:
            return (sr + er) // 2, sc
        return sr, (sc + ec) // 2

    @staticmethod
    def _long_check_violation(
        mover: str,
        history: List[MoveEntry],
        board_after_move: Board,
    ) -> Optional[str]:
        """若本步将导致第三次相同局面且循环内该方着法均为将军，返回 ``_MSG_LONG_CHECK``。

        ``history`` 为截止到走子前的完整历史（含初始局面占位项）；
        本函数自行计算走子后的哈希与将军状态，无需调用者预先存入。
        """
        h_new = board_after_move.zobrist_hash
        n_prior = sum(1 for e in history if e.pos_hash == h_new)
        if n_prior < 2:
            return None
        # 找到最后一次出现该哈希的索引，取其后续区间作为循环节
        last_idx = -1
        for i in range(len(history) - 1, -1, -1):
            if history[i].pos_hash == h_new:
                last_idx = i
                break
        # 循环节 = history[last_idx+1 :] + 本步（尚未入栈）
        sim_gave_check = Rules.is_king_in_check(
            board_after_move, board_after_move.current_player
        )
        found = False
        for entry in history[last_idx + 1:]:
            if entry.mover != mover:
                continue
            found = True
            if not entry.gave_check:
                return None
        # 本步（即将入栈的条目）也属于 mover
        found = True
        if not sim_gave_check:
            return None
        if not found:
            return None
        return _MSG_LONG_CHECK

    @staticmethod
    def is_valid_move(
        board: Board,
        start_row,
        start_col,
        end_row,
        end_col,
        player=None,
        check_legality=True,
        history: Optional[List[MoveEntry]] = None,
    ) -> Tuple[bool, str]:
        """检查指定棋子走子是否合法；返回 ``(是否合法, 错误原因)``。

        Args:
            board: 当前棋盘。
            start_row, start_col: 起点。
            end_row, end_col: 终点。
            player: 行棋方，缺省取 ``board.current_player``。
            check_legality: 为 ``False`` 时仅做几何与吃子规则，不做将军/飞将模拟。
            history: 完整的对局历史记录（``List[MoveEntry]``），含初始局面占位项。
                提供时额外检测「长将」违规。

        Returns:
            ``(True, "")`` 或 ``(False, 错误原因)``。
        """
        player = player or board.current_player
        if not (
            0 <= start_row < 10
            and 0 <= start_col < 9
            and 0 <= end_row < 10
            and 0 <= end_col < 9
        ):
            return False, _MSG_BAD_GEOMETRY

        b = board.board
        piece = b[start_row][start_col]
        if not piece or piece.color != player:
            return False, _MSG_BAD_GEOMETRY

        target = b[end_row][end_col]
        if target and target.color == piece.color:
            return False, _MSG_FRIENDLY_FIRE
        if target is not None and target.piece_type == "jiang":
            return False, _MSG_NO_CAPTURE_KING

        geom_err = Rules._geometry_error(board, piece, start_row, start_col, end_row, end_col, player)
        if geom_err is not None:
            return False, geom_err

        if not check_legality:
            return True, ""

        was_in_check = Rules.is_king_in_check(board, player)
        captured = board.apply_move(start_row, start_col, end_row, end_col)
        kings_facing = Rules._jiang_face_to_face(board)
        still_in_check = Rules.is_king_in_check(board, player)
        long_chk: Optional[str] = None
        if (
            history is not None
            and not kings_facing
            and not still_in_check
        ):
            long_chk = Rules._long_check_violation(player, history, board)
        board.undo_move(start_row, start_col, end_row, end_col, captured)

        if kings_facing:
            return False, _MSG_KINGS_FACE
        if still_in_check:
            if was_in_check:
                return False, _MSG_IN_CHECK_STILL
            return False, _MSG_SELF_CHECK
        if long_chk is not None:
            return False, long_chk
        return True, ""

    @staticmethod
    def _geometry_error(
        board: Board, piece, sr: int, sc: int, er: int, ec: int, player: str
    ) -> Optional[str]:
        """几何与路径规则；合法返回 ``None``。"""
        pt = piece.piece_type
        b = board.board

        if pt == "jiang":
            if abs(sr - er) + abs(sc - ec) != 1:
                return _MSG_BAD_GEOMETRY
            if player == "red":
                if not (7 <= er <= 9 and 3 <= ec <= 5):
                    return _MSG_BAD_GEOMETRY
            elif not (0 <= er <= 2 and 3 <= ec <= 5):
                return _MSG_BAD_GEOMETRY
            return None

        if pt == "shi":
            if abs(sr - er) != 1 or abs(sc - ec) != 1:
                return _MSG_BAD_GEOMETRY
            if player == "red":
                if not (7 <= er <= 9 and 3 <= ec <= 5):
                    return _MSG_BAD_GEOMETRY
            elif not (0 <= er <= 2 and 3 <= ec <= 5):
                return _MSG_BAD_GEOMETRY
            return None

        if pt == "xiang":
            if abs(sr - er) != 2 or abs(sc - ec) != 2:
                return _MSG_BAD_GEOMETRY
            eye_r = (sr + er) // 2
            eye_c = (sc + ec) // 2
            if b[eye_r][eye_c]:
                return _MSG_XIANG_EYE
            if player == "red":
                if er < 5:
                    return _MSG_BAD_GEOMETRY
            elif er > 4:
                return _MSG_BAD_GEOMETRY
            return None

        if pt == "ma":
            dr = abs(sr - er)
            dc = abs(sc - ec)
            if not ((dr == 2 and dc == 1) or (dr == 1 and dc == 2)):
                return _MSG_BAD_GEOMETRY
            leg_r, leg_c = Rules._ma_leg_square(sr, sc, er, ec)
            if b[leg_r][leg_c]:
                return _MSG_MA_BLOCKED
            return None

        if pt == "che":
            if sr != er and sc != ec:
                return _MSG_BAD_GEOMETRY
            if sr == er:
                step = 1 if ec > sc else -1
                for c in range(sc + step, ec, step):
                    if b[sr][c]:
                        return _MSG_BAD_GEOMETRY
            else:
                step = 1 if er > sr else -1
                for r in range(sr + step, er, step):
                    if b[r][sc]:
                        return _MSG_BAD_GEOMETRY
            return None

        if pt == "pao":
            if sr != er and sc != ec:
                return _MSG_BAD_GEOMETRY
            tgt = b[er][ec]
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
            if tgt is not None:
                if count != 1:
                    return _MSG_PAO_BAD
            elif count != 0:
                return _MSG_PAO_BAD
            return None

        if pt == "bing":
            return Rules._bing_geometry_error(sr, sc, er, ec, player)

        return _MSG_BAD_GEOMETRY

    @staticmethod
    def _bing_geometry_error(sr: int, sc: int, er: int, ec: int, player: str) -> Optional[str]:
        if player == "red":
            if er > sr:
                return _MSG_BING_BACK
            if er == sr and abs(ec - sc) == 1:
                if sr > 4:
                    return _MSG_BING_NO_SIDE
                return None
            if er == sr - 1 and ec == sc:
                return None
            return _MSG_BAD_GEOMETRY
        if er < sr:
            return _MSG_BING_BACK
        if er == sr and abs(ec - sc) == 1:
            if sr < 5:
                return _MSG_BING_NO_SIDE
            return None
        if er == sr + 1 and ec == sc:
            return None
        return _MSG_BAD_GEOMETRY

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
    def get_all_moves(
        board: Board,
        player,
        validate_self_check=True,
        history: Optional[List[MoveEntry]] = None,
    ):
        """生成指定方的所有走法；``validate_self_check=True`` 时过滤自将/长将。"""
        moves = []
        for r, c in board.active_pieces[player]:
            piece = board.board[r][c]
            if not piece or piece.color != player:
                continue
            candidates = Rules._candidate_targets(board, r, c, piece.piece_type, player)
            for er, ec in candidates:
                ok, _ = Rules.is_valid_move(
                    board,
                    r,
                    c,
                    er,
                    ec,
                    player=player,
                    check_legality=validate_self_check,
                    history=history,
                )
                if ok:
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
    def _pseudo_capture_ok(cell, player: str) -> bool:
        """伪合法生成用：可落子格为空，或为可吃的敌方子（禁止吃将）。"""
        if cell is None:
            return True
        if cell.color == player:
            return False
        return cell.piece_type != "jiang"

    @staticmethod
    def get_pseudo_legal_moves(board: Board, player: str) -> Iterator[Tuple[int, int, int, int]]:
        """伪合法走法生成（仅几何与吃子颜色），供 AI 搜索批量过滤。

        不调用 ``get_piece`` / ``is_valid_move``；目标格用 ``b[nr][nc]`` 直判；不吃将；
        不校验自将、白脸将。完整合法性仍由根节点 ``is_valid_move`` / 搜索内 ``is_king_in_check`` 保证。
        """
        b = board.board

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
                            if Rules._pseudo_capture_ok(cell, player):
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
                                if Rules._pseudo_capture_ok(cell, player):
                                    yield (r, c, nr, nc)
                                break
                        nr += dr
                        nc += dc

            elif pt == "ma":
                for dr, dc in Rules._MA_ATTACK_DELTAS:
                    nr, nc = r + dr, c + dc
                    lr, lc = Rules._ma_leg_square(r, c, nr, nc)
                    if not (0 <= lr < 10 and 0 <= lc < 9):
                        continue
                    if b[lr][lc] is not None:
                        continue
                    if not (0 <= nr < 10 and 0 <= nc < 9):
                        continue
                    dest = b[nr][nc]
                    if Rules._pseudo_capture_ok(dest, player):
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
                    if Rules._pseudo_capture_ok(dest, player):
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
                    if Rules._pseudo_capture_ok(dest, player):
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
                    if Rules._pseudo_capture_ok(dest, player):
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
                    if Rules._pseudo_capture_ok(dest, player):
                        yield (r, c, nr, nc)

    @staticmethod
    def get_legal_moves(
        board: Board,
        player,
        history: Optional[List[MoveEntry]] = None,
    ):
        """合法走法；被将军时仅返回能解除己方老将受攻的着法（经 ``is_valid_move`` 过滤）。"""
        return Rules.get_all_moves(
            board,
            player,
            validate_self_check=True,
            history=history,
        )

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

        for ddr, ddc in Rules._MA_ATTACK_DELTAS:
            hr, hc = kr + ddr, kc + ddc
            if not (0 <= hr < 10 and 0 <= hc < 9):
                continue
            hp = b[hr][hc]
            if hp is None or hp.color != opponent or hp.piece_type != "ma":
                continue
            leg_r, leg_c = Rules._ma_leg_square(hr, hc, kr, kc)
            if not (0 <= leg_r < 10 and 0 <= leg_c < 9):
                continue
            if b[leg_r][leg_c] is not None:
                continue
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

