"""中国象棋规则引擎（模型层 Model / Rules）。

本文件从旧实现 ``chinese-chess/ai/board.py`` 物理搬运而来，职责包括：

- **棋子走法合法性判定**：每种棋子的几何规则、路径障碍检测
- **将军检测**：反向射线法判断是否被将军，以及白脸将（将帅对面）判定
- **走法生成**：
  - ``get_all_moves`` / ``get_legal_moves``：完整合法走法（含将军/长将过滤）
  - ``get_pseudo_legal_moves``：伪合法走法（仅几何规则，供 AI 搜索用）
- **终局判定**：将被吃掉、无子可走（困毙/将死）、三次重复判和

注意：
- 迁移过程中核心判断与数学逻辑保持不变；主要将 ``self.*`` 改为 ``board.*``
- 所有方法均为静态方法，避免有状态耦合
"""

from __future__ import annotations

from typing import Iterator, List, NamedTuple, Optional, Tuple

from .board import Board


class MoveEntry(NamedTuple):
    """对局历史中的一条记录，与单步走子一一对应。

    用于追踪局面哈希、行棋方及是否将军，支持长将违规检测和局面重复判定。

    ``history[0]`` 为初始局面（走子前的占位项），其 ``mover`` 和
    ``gave_check`` 均为 ``None``。
    ``history[i]``（i >= 1）记录第 i 手走后的局面哈希、行棋方及是否将军。

    Attributes:
        pos_hash: 走子后的 Zobrist 局面哈希值。
        mover: 该手的行棋方（``"red"`` 或 ``"black"``），
            初始占位项为 ``None``。
        gave_check: 该手走后是否将军对方，初始占位项为 ``None``。
    """
    pos_hash: int
    mover: Optional[str] = None
    gave_check: Optional[bool] = None

# ──────────────────────────────────────────────
# 非法走法说明常量（供 GUI / Controller 向用户展示具体原因）
# ──────────────────────────────────────────────
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
    """中国象棋完整规则引擎，全部方法为静态方法以避免有状态耦合。

    本类封装了中国象棋的全部规则逻辑，包括：

    - 棋子几何走法验证（将、士、象、马、车、炮、兵各有专用方法）
    - 完整合法性校验（含将军检测、白脸将检测、长将违规检测）
    - 走法生成（完整合法走法与伪合法走法两种模式）
    - 终局判定（将死、困毙、三次重复判和）

    设计为纯静态类的原因：规则本身不持有状态，所有判定都基于传入的
    ``Board`` 对象进行，这使得规则逻辑可以在 Minimax、MCTS 等不同
    算法间无状态地复用。
    """

    # 四个正交方向偏移量（用于车 / 炮 / 将军射线扫描）
    _ORTH_DELTAS: Tuple[Tuple[int, int], ...] = (
        (1, 0), (-1, 0), (0, 1), (0, -1),
    )

    # 马的 8 个攻击偏移量（行差, 列差），与 ``_geometry_error`` / 伪合法生成一致
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
        """计算马从起点跳到终点时的马腿（别腿）坐标。

        马走"日"字，先沿行或列走两格，再拐一格。马腿位于"走两格"方向上的
        中间格。若该格有棋子则构成蹩马腿，走法无效。

        计算原理：沿位移较大的轴取中点。若行差为 2，马腿在同列的中间行；
        若列差为 2，马腿在同行的中间列。

        Args:
            sr: 起点行坐标。
            sc: 起点列坐标。
            er: 终点行坐标。
            ec: 终点列坐标。

        Returns:
            马腿所在格的 ``(行, 列)`` 坐标。
        """
        if abs(sr - er) == 2:
            return (sr + er) // 2, sc
        return sr, (sc + ec) // 2

    @staticmethod
    def _long_check_violation(
        mover: str,
        history: List[MoveEntry],
        board_after_move: Board,
    ) -> Optional[str]:
        """检测「长将」违规。

        长将规则：若本步将导致同一局面第三次出现，且从上一次相同局面到本步之间
        该方每一手都在将军对方，则判定为长将违规（必须变招）。

        检测流程：
        1. 统计历史中与走后局面哈希相同的条目数，不足 2 次则不可能三次重复
        2. 找到最后一次出现该哈希的索引，取其后续区间作为循环节
        3. 检查循环节内该方的每一手是否全部为将军

        本函数直接读取 ``MoveEntry.mover`` 字段判定行棋方，不依赖奇偶索引假设，
        因此支持任意一方先走的非标准局面。

        Args:
            mover: 当前行棋方（``"red"`` 或 ``"black"``）。
            history: 截止到走子前的完整对局历史（含初始局面占位项 ``history[0]``）。
            board_after_move: 模拟执行本步后的棋盘状态（用于读取哈希与将军判定）。

        Returns:
            违规时返回 ``_MSG_LONG_CHECK`` 错误提示字符串，否则返回 ``None``。
        """
        h_new = board_after_move.zobrist_hash
        n_prior = sum(1 for e in history if e.pos_hash == h_new)
        if n_prior < 2:
            return None
        # 逆序扫描找到最后一次出现该哈希的索引，确定循环节起点
        last_idx = -1
        for i in range(len(history) - 1, -1, -1):
            if history[i].pos_hash == h_new:
                last_idx = i
                break
        # 循环节 = history[last_idx+1 :] 加上本步（尚未入栈）
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
        # 本步（即将入栈的条目）也属于该方，需一并检查是否将军
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
        """检查指定走子是否合法，返回校验结果与错误原因。

        校验分为两个层次：
        1. **基础校验**（始终执行）：坐标边界、起点有己方棋子、不吃友军、
           不直接吃将、棋子几何走法合规
        2. **完整合法性校验**（``check_legality=True`` 时执行）：
           模拟走子后检查白脸将、自将、长将等高级规则

        两层分离的原因：AI 搜索的伪合法走法生成只需基础校验，
        完整校验留给关键节点以节省性能。

        Args:
            board: 当前棋盘状态。
            start_row: 起点行坐标。
            start_col: 起点列坐标。
            end_row: 终点行坐标。
            end_col: 终点列坐标。
            player: 行棋方，缺省取 ``board.current_player``。
            check_legality: 为 ``False`` 时仅做几何与吃子规则校验，
                不做将军/飞将模拟检测。
            history: 完整的对局历史记录（``List[MoveEntry]``），含初始局面占位项。
                提供时额外检测「长将」违规。

        Returns:
            二元组 ``(是否合法, 错误原因)``。合法时返回 ``(True, "")``，
            不合法时返回 ``(False, 具体错误说明字符串)``。
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

        # ── 完整合法性校验：模拟走子，检查走后局面是否合规 ──
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
        """检查棋子的几何走法与路径规则是否合规。

        对每种棋子类型分别校验其特有的移动规则：
        - 将/帅：九宫内上下左右一步
        - 士/仕：九宫内斜走一步
        - 象/相：田字斜走两步，不可过河，检查塞象眼
        - 马：日字走法，检查蹩马腿
        - 车：直线走，路径上不可有障碍物
        - 炮：直线走，不吃子时路径无障碍，吃子时恰好隔一子
        - 兵/卒：前进一步，过河后可横走，不可后退

        Args:
            board: 当前棋盘状态。
            piece: 要移动的棋子对象。
            sr: 起点行坐标。
            sc: 起点列坐标。
            er: 终点行坐标。
            ec: 终点列坐标。
            player: 行棋方（``"red"`` 或 ``"black"``）。

        Returns:
            走法合规时返回 ``None``，不合规时返回对应的错误说明字符串。
        """
        pt = piece.piece_type
        b = board.board

        if pt == "jiang":
            # 将/帅：只能在九宫格内上下左右走一步（曼哈顿距离恰好为 1）
            if abs(sr - er) + abs(sc - ec) != 1:
                return _MSG_BAD_GEOMETRY
            if player == "red":
                if not (7 <= er <= 9 and 3 <= ec <= 5):
                    return _MSG_BAD_GEOMETRY
            elif not (0 <= er <= 2 and 3 <= ec <= 5):
                return _MSG_BAD_GEOMETRY
            return None

        if pt == "shi":
            # 士/仕：只能在九宫格内斜走一步（行差和列差均为 1）
            if abs(sr - er) != 1 or abs(sc - ec) != 1:
                return _MSG_BAD_GEOMETRY
            if player == "red":
                if not (7 <= er <= 9 and 3 <= ec <= 5):
                    return _MSG_BAD_GEOMETRY
            elif not (0 <= er <= 2 and 3 <= ec <= 5):
                return _MSG_BAD_GEOMETRY
            return None

        if pt == "xiang":
            # 象/相：走田字（行差和列差均为 2），不可过河，需检查象眼
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
            # 马：走日字（一个方向 2 格、另一方向 1 格），需检查蹩马腿
            dr = abs(sr - er)
            dc = abs(sc - ec)
            if not ((dr == 2 and dc == 1) or (dr == 1 and dc == 2)):
                return _MSG_BAD_GEOMETRY
            leg_r, leg_c = Rules._ma_leg_square(sr, sc, er, ec)
            if b[leg_r][leg_c]:
                return _MSG_MA_BLOCKED
            return None

        if pt == "che":
            # 车：直线行走（同行或同列），路径上不可有任何障碍物
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
            # 炮：直线行走，不吃子时路径无障碍（count=0），
            # 吃子时恰好翻过一个炮架（count=1）
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
        """检查兵/卒的几何走法是否合规。

        兵/卒规则：
        - 不可后退
        - 未过河时只能直进一步，不可横移
        - 过河后可直进一步或横移一步

        红方从下往上走（行号递减为前进），过河线为第 4 行（含）以上。
        黑方从上往下走（行号递增为前进），过河线为第 5 行（含）以下。

        Args:
            sr: 起点行坐标。
            sc: 起点列坐标。
            er: 终点行坐标。
            ec: 终点列坐标。
            player: 行棋方（``"red"`` 或 ``"black"``）。

        Returns:
            走法合规时返回 ``None``，不合规时返回对应的错误说明字符串。
        """
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
        """判定当前局面是否存在将帅对面（白脸将）。

        中国象棋规则：将/帅不可在同一列上直接面对（中间无子遮挡），
        否则该局面非法。这条规则在走子合法性检查中用于过滤：
        走子后若导致将帅对面，则该走法不合法。

        利用 ``Board`` 维护的将帅坐标直接定位，避免全盘扫描。
        仅当两者在同一列时才需逐格检查中间是否有遮挡。

        Args:
            board: 当前棋盘状态。

        Returns:
            若将帅直接对面（中间无子）返回 ``True``，否则返回 ``False``。
        """
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
        """生成指定方的所有走法。

        先通过 ``_candidate_targets`` 生成各棋子的候选落点，
        再逐一调用 ``is_valid_move`` 进行合法性过滤。

        当 ``validate_self_check=True`` 时，会过滤掉导致自将、
        白脸将和长将违规的走法。返回列表已去重（``active_pieces``
        在极端情况下可能产生重复条目）。

        Args:
            board: 当前棋盘状态。
            player: 行棋方（``"red"`` 或 ``"black"``）。
            validate_self_check: 是否进行自将/长将校验，默认 ``True``。
                AI 搜索在内部节点可设为 ``False`` 以提升性能。
            history: 完整对局历史，提供时额外检测长将违规。

        Returns:
            合法走法列表，每个元素为 ``(起行, 起列, 终行, 终列)`` 四元组。
        """
        seen: set[Tuple[int, int, int, int]] = set()
        moves: list[Tuple[int, int, int, int]] = []
        for r, c in board.active_pieces[player]:
            piece = board.board[r][c]
            if not piece or piece.color != player:
                continue
            candidates = Rules._candidate_targets(board, r, c, piece.piece_type, player)
            for er, ec in candidates:
                m = (r, c, er, ec)
                if m in seen:
                    continue
                ok, _ = Rules.is_valid_move(
                    board,
                    r, c, er, ec,
                    player=player,
                    check_legality=validate_self_check,
                    history=history,
                )
                if ok:
                    seen.add(m)
                    moves.append(m)
        return moves

    @staticmethod
    def _candidate_targets(board: Board, r: int, c: int, piece_type: str, player: str):
        """根据棋子类型生成候选落点坐标列表。

        这些候选落点仅基于棋子的基本移动模式生成，不保证合法性。
        后续由 ``is_valid_move`` 进行完整过滤（包括友军吃子、蹩马腿、
        塞象眼、路径障碍、自将等检查）。

        设计目的：通过预先缩小候选集，显著降低 ``get_all_moves`` 的复杂度，
        使 Minimax 在纯 Python 实现中达到可用的搜索深度。

        Args:
            board: 当前棋盘状态。
            r: 棋子当前行坐标。
            c: 棋子当前列坐标。
            piece_type: 棋子类型标识（如 ``"jiang"``、``"che"``、``"ma"`` 等）。
            player: 行棋方（``"red"`` 或 ``"black"``）。

        Returns:
            候选落点坐标列表，每个元素为 ``(行, 列)`` 二元组。
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
            # 将帅对面"照面吃"由 is_valid_move 的 legality 检查处理
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
            for dr, dc in Rules._MA_ATTACK_DELTAS:
                add(r + dr, c + dc)
            return cand

        if piece_type in ("che", "pao"):
            # 车/炮沿四个方向延伸到边界；炮的"隔子吃"由 is_valid_move 内部处理
            for dr, dc in Rules._ORTH_DELTAS:
                rr, cc = r + dr, c + dc
                while 0 <= rr < 10 and 0 <= cc < 9:
                    cand.append((rr, cc))
                    rr += dr
                    cc += dc
            return cand

        if piece_type == "bing":
            # 兵/卒：最多 3 个候选点（前进 + 过河后左右横移）
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
        """判断目标格是否可作为伪合法走法的落点。

        伪合法走法生成专用辅助函数：目标格为空可以落子，
        为敌方非将棋子可以吃子，为己方棋子或敌方将则不可。
        禁止吃将的原因是将/帅被吃应通过将死机制判定，而非直接捕获。

        Args:
            cell: 目标格上的棋子对象，为空时传入 ``None``。
            player: 当前行棋方（``"red"`` 或 ``"black"``）。

        Returns:
            目标格可落子返回 ``True``，否则返回 ``False``。
        """
        if cell is None:
            return True
        if cell.color == player:
            return False
        return cell.piece_type != "jiang"

    @staticmethod
    def get_pseudo_legal_moves(board: Board, player: str) -> Iterator[Tuple[int, int, int, int]]:
        """生成伪合法走法（仅几何与吃子颜色校验），供 AI 搜索批量使用。

        与 ``get_all_moves`` 的区别：
        - 不调用 ``get_piece`` / ``is_valid_move``，直接用 ``b[nr][nc]`` 快速判断
        - 不吃将（由 ``_pseudo_capture_ok`` 过滤）
        - **不校验**自将、白脸将、长将等高级规则

        完整合法性仍由根节点的 ``is_valid_move`` 或搜索内部的
        ``is_king_in_check`` 保证。这种两级过滤策略在性能与正确性间取得平衡。

        使用生成器（``yield``）而非列表，避免一次性分配大量内存。

        Args:
            board: 当前棋盘状态。
            player: 行棋方（``"red"`` 或 ``"black"``）。

        Yields:
            伪合法走法四元组 ``(起行, 起列, 终行, 终列)``。
        """
        b = board.board

        for r, c in board.active_pieces[player]:
            p = b[r][c]
            if p is None or p.color != player:
                continue
            pt = p.piece_type

            if pt == "che":
                # 车：四方向直线扫描，遇子停（可吃敌、不可穿越）
                for dr, dc in Rules._ORTH_DELTAS:
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
                # 炮：直线扫描，遇到第一个子为炮架（seen_screen），
                # 炮架之后的第一个可吃敌子为合法吃子目标
                for dr, dc in Rules._ORTH_DELTAS:
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
                # 马：8 个日字跳跃点，需检查马腿是否被阻
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
                # 象/相：4 个田字斜跳点，需检查象眼和过河限制
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
                # 士/仕：4 个斜向一步点，需在九宫范围内
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
                # 将/帅：4 个正交一步点，需在九宫范围内
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
                # 兵/卒：前进一步 + 过河后左右横移
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
        """生成指定方的所有完整合法走法。

        被将军时仅返回能解除己方老将受攻的着法。
        内部委托给 ``get_all_moves``，始终开启自将校验。

        Args:
            board: 当前棋盘状态。
            player: 行棋方（``"red"`` 或 ``"black"``）。
            history: 完整对局历史，提供时额外检测长将违规。

        Returns:
            合法走法列表，每个元素为 ``(起行, 起列, 终行, 终列)`` 四元组。
        """
        return Rules.get_all_moves(
            board,
            player,
            validate_self_check=True,
            history=history,
        )

    @staticmethod
    def is_king_in_check(board: Board, player: str) -> bool:
        """判断指定方的将/帅是否正在被将军。

        采用反向检测法：从将/帅位置出发，向外扫描是否存在能攻击到此处的敌子。
        不调用 ``is_valid_move``、不遍历 ``active_pieces``，性能优于正向枚举。

        检测覆盖以下攻击来源：
        1. **车/炮**：从将/帅位置向四个正交方向发射射线。
           第一个遇到的棋子若为敌方车则被将军（obstacles=0）；
           翻过一个棋子后若遇到敌方炮则被将军（obstacles=1）。
        2. **马**：检查 8 个反向日字位置是否有敌方马，并验证马腿未被阻。
        3. **兵/卒**：检查将/帅的前方和左右（仅过河后的兵可横攻）。

        Args:
            board: 当前棋盘状态。
            player: 被检测方（``"red"`` 或 ``"black"``）。

        Returns:
            该方将/帅正在被将军返回 ``True``，否则返回 ``False``。
            若将/帅已被吃（坐标为 ``None``），也返回 ``True``。
        """
        if player == "red":
            jiang_pos = board.red_king_pos
        else:
            jiang_pos = board.black_king_pos
        if not jiang_pos:
            return True
        kr, kc = jiang_pos
        opponent = "black" if player == "red" else "red"
        b = board.board

        # ── 车/炮检测：四方向射线扫描 ──
        # obstacles 计数：0 时遇敌车即被将，1 时遇敌炮即被将（炮架已翻过）
        for dr, dc in Rules._ORTH_DELTAS:
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

        # ── 马检测：从将/帅位置检查 8 个反向日字位是否有未被阻的敌方马 ──
        for ddr, ddc in Rules._MA_ATTACK_DELTAS:
            hr, hc = kr + ddr, kc + ddc
            if not (0 <= hr < 10 and 0 <= hc < 9):
                continue
            hp = b[hr][hc]
            if hp is None or hp.color != opponent or hp.piece_type != "ma":
                continue
            # 注意：此处马腿方向是从马到将/帅，而非从将/帅到马
            leg_r, leg_c = Rules._ma_leg_square(hr, hc, kr, kc)
            if not (0 <= leg_r < 10 and 0 <= leg_c < 9):
                continue
            if b[leg_r][leg_c] is not None:
                continue
            return True

        # ── 兵/卒检测：检查将/帅前方和两侧是否有敌方兵/卒 ──
        if player == "red":
            # 红方将在下方，敌方兵从上方和两侧攻击
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
            # 黑方将在上方，敌方兵从下方和两侧攻击
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
        """判断指定方是否被将军（``is_king_in_check`` 的兼容别名）。

        保留此方法名以兼容旧代码调用，内部直接委托给 ``is_king_in_check``。

        Args:
            board: 当前棋盘状态。
            player: 被检测方（``"red"`` 或 ``"black"``）。

        Returns:
            该方正在被将军返回 ``True``，否则返回 ``False``。
        """
        return Rules.is_king_in_check(board, player)

    @staticmethod
    def has_legal_moves(board: Board, player):
        """判断指定方是否还有合法走法可走。

        Args:
            board: 当前棋盘状态。
            player: 被检测方（``"red"`` 或 ``"black"``）。

        Returns:
            存在至少一个合法走法返回 ``True``，否则返回 ``False``。
        """
        return len(Rules.get_legal_moves(board, player)) > 0

    @staticmethod
    def is_checkmate(board: Board, player):
        """判断指定方是否被将死。

        将死条件：正在被将军 **且** 没有任何合法走法可以解除将军。

        Args:
            board: 当前棋盘状态。
            player: 被检测方（``"red"`` 或 ``"black"``）。

        Returns:
            该方被将死返回 ``True``，否则返回 ``False``。
        """
        return Rules.is_check(board, player) and not Rules.has_legal_moves(board, player)

    @staticmethod
    def is_stalemate(board: Board, player):
        """判断指定方是否被困毙。

        困毙条件：未被将军 **但** 没有任何合法走法可走。
        注意：在中国象棋中困毙判负（与国际象棋 stalemate 判和不同）。

        Args:
            board: 当前棋盘状态。
            player: 被检测方（``"red"`` 或 ``"black"``）。

        Returns:
            该方被困毙返回 ``True``，否则返回 ``False``。
        """
        return not Rules.is_check(board, player) and not Rules.has_legal_moves(board, player)

    @staticmethod
    def winner(board: Board):
        """判断当前局面的胜者。

        判负规则（符合中国象棋常用判负方式）：
        1. **将/帅被吃**：将/帅不在棋盘上的一方判负
        2. **困毙/将死**：轮到走子但无任何合法走法的一方判负
           （同时覆盖将死与困毙两种情况；与国际象棋 stalemate=draw 不同）

        Args:
            board: 当前棋盘状态。

        Returns:
            胜者颜色字符串（``"red"`` 或 ``"black"``），
            未分胜负时返回 ``None``。
        """

        # 第一步：检查将/帅是否还在棋盘上
        # 利用 Board 维护的将帅坐标进行 O(1) 判定，替代 O(90) 全盘扫描。
        # red_king_pos / black_king_pos 由 apply_move / undo_move 增量维护，
        # 被吃后置为 None，与全盘扫描结果严格等价。
        is_red_king_alive = board.red_king_pos is not None
        is_black_king_alive = board.black_king_pos is not None
        if not is_red_king_alive and is_black_king_alive:
            return "black"
        if not is_black_king_alive and is_red_king_alive:
            return "red"
        if not is_red_king_alive and not is_black_king_alive:
            # 理论上不应出现双方将帅同时消失，保守返回 None
            return None

        # 第二步：检查困毙/将死——轮到走子但无合法走法的一方判负
        if not Rules.has_legal_moves(board, board.current_player):
            return "black" if board.current_player == "red" else "red"

        return None

    @staticmethod
    def is_threefold_repetition_draw(board: Board, position_history: list) -> bool:
        """判断当前局面是否因三次重复而判和。

        简易防循环机制：同一局面（Zobrist 哈希）在对局历史中出现至少 3 次时判和。

        注意：Minimax 在搜索路径上遇到重复局面时，使用
        ``Evaluation.repetition_leaf_score`` 进行评分（大优时厌战、大劣时接受和棋），
        与本终局层面的布尔判定相互独立。

        Args:
            board: 当前棋盘状态。
            position_history: 对局历史中各局面的 Zobrist 哈希值列表。

        Returns:
            当前局面已出现 3 次或以上返回 ``True``，否则返回 ``False``。
        """
        if not position_history:
            return False
        h = board.zobrist_hash
        return sum(1 for x in position_history if x == h) >= 3

    @staticmethod
    def is_game_over(board: Board, position_history: Optional[list] = None):
        """判断当前对局是否已结束。

        游戏结束条件（满足任一即可）：
        1. 任意一方将/帅被吃或无子可走（由 ``winner()`` 检测）
        2. 三次重复局面判和（可选，需提供 ``position_history``）

        Args:
            board: 当前棋盘状态。
            position_history: 对局历史中各局面的 Zobrist 哈希值列表，
                传入 ``None`` 时不检测三次重复。

        Returns:
            对局已结束返回 ``True``，否则返回 ``False``。
        """
        if Rules.winner(board) is not None:
            return True
        if position_history is not None and Rules.is_threefold_repetition_draw(
            board, position_history
        ):
            return True
        return False
