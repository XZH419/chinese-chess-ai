"""启发式评估函数（Evaluator）。

- **Tapered Evaluation**：MG/EG 双轨 + 阶段 ``phase`` 线性插值。
- 子力：``MG_VALUES`` / ``EG_VALUES``；位置：``PST_MG_MAP`` / ``PST_EG_MAP``。
- **位置**：马/车/兵的高价值格写入 PST（含卧槽/挂角、车深入、过河兵近九宫）。
- **战术**：``_tactical_synergy`` 车炮纵线、马腿、车马协同（无着法生成）。
- 兑子惩罚、马/炮机动与架炮、将军奖励。
"""

from __future__ import annotations

from typing import Optional

from chinese_chess.model.board import Board
from chinese_chess.model.rules import Rules


def _pst_ma_mg() -> list[list[int]]:
    """马 MG：强化 0–2 行 3/6 路（挂角、卧槽常见格）。"""
    t = [
        [0, 1, 4, 18, 6, 18, 4, 1, 0],
        [1, 4, 8, 22, 10, 22, 8, 4, 1],
        [2, 8, 12, 16, 13, 16, 12, 8, 2],
        [3, 9, 12, 14, 15, 14, 12, 9, 3],
        [3, 9, 13, 15, 16, 15, 13, 9, 3],
        [3, 8, 12, 14, 15, 14, 12, 8, 3],
        [2, 6, 10, 12, 13, 12, 10, 6, 2],
        [1, 4, 6, 8, 9, 8, 6, 4, 1],
        [0, 2, 3, 4, 5, 4, 3, 2, 0],
        [0, 0, 1, 2, 3, 2, 1, 0, 0],
    ]
    return t


def _pst_ma_eg() -> list[list[int]]:
    """马 EG：同结构略弱于 MG，残局仍奖励高位马。"""
    t = [
        [0, 1, 3, 14, 5, 14, 3, 1, 0],
        [1, 3, 6, 18, 8, 18, 6, 3, 1],
        [2, 6, 10, 13, 12, 13, 10, 6, 2],
        [3, 8, 11, 13, 14, 13, 11, 8, 3],
        [3, 8, 12, 14, 15, 14, 12, 8, 3],
        [3, 7, 11, 13, 14, 13, 11, 7, 3],
        [2, 5, 9, 11, 12, 11, 9, 5, 2],
        [1, 3, 5, 7, 8, 7, 5, 3, 1],
        [0, 2, 3, 4, 5, 4, 3, 2, 0],
        [0, 0, 1, 2, 3, 2, 1, 0, 0],
    ]
    return t


class Evaluation:
    # 静态评估置换表：``zobrist_hash`` → 分数（``MinimaxAI.get_best_move`` 根入口会清空）
    _eval_cache: dict[int, float] = {}

    # 可参与将杀/实质性进攻的子力（仅剩将、士、象视为无法将死 → 评估为和）
    ATTACKING_PIECE_TYPES = frozenset({"che", "ma", "pao", "bing"})
    MAJOR_TYPES = frozenset({"che", "ma", "pao"})

    # 游戏阶段权重（车=2, 马=1, 炮=1）。满子 phase 和为 16。
    PHASE_WEIGHTS: dict[str, int] = {"che": 2, "ma": 1, "pao": 1}
    TOTAL_PHASE: float = 16.0

    # 中局 / 残局子力价值（与 taper 插值配套）
    MG_VALUES: dict[str, int] = {
        "che": 900,
        "pao": 470,
        "ma": 400,
        "xiang": 200,
        "shi": 200,
        "bing": 90,
        "jiang": 0,
    }
    EG_VALUES: dict[str, int] = {
        "che": 920,
        "pao": 410,
        "ma": 430,
        "xiang": 200,
        "shi": 200,
        "bing": 140,
        "jiang": 0,
    }
    # 兼容旧代码路径（如 ``_raw_material``）：等价于中局子力表
    PIECE_VALUES: dict[str, int] = MG_VALUES

    # 开局子力总和约量级，用于兑子惩罚归一
    _REF_TOTAL_MATERIAL = 4400
    # 领先方（|diff|>50）在子力减少时受到的兑子惩罚系数
    _ANTI_TRADE_COEFF = 0.018

    # 过河兵近九宫（列 3–5）强化；原 evaluate 内纵向/顶线奖励并入表
    PST_BING_MG = [
        [18, 18, 22, 38, 48, 38, 22, 18, 18],
        [20, 20, 26, 40, 50, 40, 26, 20, 20],
        [16, 16, 24, 36, 46, 36, 24, 16, 16],
        [12, 12, 20, 32, 40, 32, 20, 12, 12],
        [8, 8, 14, 22, 28, 22, 14, 8, 8],
        [4, 4, 8, 10, 12, 10, 8, 4, 4],
        [2, 2, 4, 6, 6, 6, 4, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    PST_BING_EG = [
        [38, 38, 48, 68, 82, 68, 48, 38, 38],
        [32, 32, 42, 60, 72, 60, 42, 32, 32],
        [26, 26, 36, 52, 62, 52, 36, 26, 26],
        [20, 20, 28, 42, 50, 42, 28, 20, 20],
        [14, 14, 20, 32, 38, 32, 20, 14, 14],
        [8, 8, 10, 16, 18, 16, 10, 8, 8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    PST_MA_MG = _pst_ma_mg()
    PST_MA_EG = _pst_ma_eg()

    # 车深入（原 r<=1 / r>=8 的 +30）并入 0–2 行抬升
    PST_CHE = [
        [18, 19, 20, 21, 22, 21, 20, 19, 18],
        [16, 17, 18, 19, 20, 19, 18, 17, 16],
        [12, 13, 14, 15, 16, 15, 14, 13, 12],
        [7, 8, 9, 10, 11, 10, 9, 8, 7],
        [7, 8, 10, 11, 12, 11, 10, 8, 7],
        [7, 8, 10, 11, 12, 11, 10, 8, 7],
        [7, 8, 9, 10, 11, 10, 9, 8, 7],
        [6, 7, 8, 9, 10, 9, 8, 7, 6],
        [5, 6, 7, 8, 9, 8, 7, 6, 5],
        [4, 5, 6, 7, 8, 7, 6, 5, 4],
    ]

    # 红方视角（棋盘行 0=黑侧上方 … 9=红侧底线，列 0–8）
    PST_PAO = [
        [6, 4, 0, -10, -12, -10, 0, 4, 6],
        [4, 2, 0, -8, -14, -8, 0, 2, 4],
        [2, 0, 0, -6, -8, -6, 0, 0, 2],
        [0, 0, 0, -2, -4, -2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-2, 0, 4, 2, 6, 2, 4, 0, -2],
        [0, 0, 0, 2, 4, 2, 0, 0, 0],
        [2, 2, 0, 4, 6, 4, 0, 2, 2],
        [2, 2, 0, 4, 6, 4, 0, 2, 2],
        [0, 0, 0, 2, 4, 2, 0, 0, 0],
    ]

    PST_SHI = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 15, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    PST_XIANG = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 15, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 2, 0, 0],
    ]

    PST_JIANG = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -10, -10, -10, 0, 0, 0],
        [0, 0, 0, -5, 0, -5, 0, 0, 0],
        [0, 0, 0, 5, 15, 5, 0, 0, 0],
    ]

    PST_DEFAULT = [[0] * 9 for _ in range(10)]

    PST_MG_MAP: dict[str, list[list[int]]] = {
        "bing": PST_BING_MG,
        "ma": PST_MA_MG,
        "che": PST_CHE,
        "shi": PST_SHI,
        "xiang": PST_XIANG,
        "pao": PST_PAO,
        "jiang": PST_JIANG,
    }
    PST_EG_MAP: dict[str, list[list[int]]] = {
        "bing": PST_BING_EG,
        "ma": PST_MA_EG,
        "che": PST_CHE,
        "shi": PST_SHI,
        "xiang": PST_XIANG,
        "pao": PST_PAO,
        "jiang": PST_JIANG,
    }

    _MA_DELTAS = (
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
    )
    _ORTH = ((0, 1), (0, -1), (1, 0), (-1, 0))

    # 战术分（同时加在 MG/EG 轨道，与旧 palace_pressure 量级可比）
    _T_ROOK_FILE = 32.0
    _T_PAO_SCREEN = 26.0
    _T_MA_LEG_PREMIUM = 18.0
    _T_MA_CHE_SYNERGY = 24.0

    @staticmethod
    def repetition_leaf_score(board: Board) -> float:
        """搜索中遇到将重演路径时的叶子分：大优时厌战（负分），大劣时接受和棋 (0)。

        与 ``Rules.is_threefold_repetition_draw``（布尔终局）不同，本函数仅用于 Minimax 内部估值。
        """
        e = Evaluation.evaluate(board)
        if e < -200:
            return 0.0
        if e > 50:
            return min(0.0, 50.0 - 0.65 * e)
        return 0.0

    @staticmethod
    def _raw_material(board: Board) -> tuple[float, float, float, int]:
        b = board.board
        red_m = black_m = 0.0
        n = 0
        for color_key in ("red", "black"):
            for r, c in board.active_pieces.get(color_key, ()):
                p = b[r][c]
                if p is None or p.color != color_key:
                    continue
                v = float(Evaluation.MG_VALUES.get(p.piece_type, 0))
                n += 1
                if color_key == "red":
                    red_m += v
                else:
                    black_m += v
        return red_m, black_m, red_m + black_m, n

    @staticmethod
    def _ma_mobility(board: Board, sr: int, sc: int, color: str) -> float:
        """马腿判定 + 机动分；落点为己方马时计连环马协同 +15。"""
        b = board.board
        bonus = 0.0
        for dr, dc in Evaluation._MA_DELTAS:
            er, ec = sr + dr, sc + dc
            if not (0 <= er < 10 and 0 <= ec < 9):
                continue
            if abs(dr) == 2:
                leg_r, leg_c = sr + (1 if dr > 0 else -1), sc
            else:
                leg_r, leg_c = sr, sc + (1 if dc > 0 else -1)
            if b[leg_r][leg_c] is not None:
                continue
            tgt = b[er][ec]
            if tgt is not None and tgt.color == color:
                if tgt.piece_type == "ma":
                    bonus += 15.0
                continue
            bonus += 5.0
        return bonus

    @staticmethod
    def _pao_screen_bonus(board: Board, r: int, c: int, color: str) -> float:
        b = board.board
        bonus = 0.0
        for dr, dc in Evaluation._ORTH:
            tr, tc = r + dr, c + dc
            while 0 <= tr < 10 and 0 <= tc < 9:
                p = b[tr][tc]
                if p is None:
                    tr += dr
                    tc += dc
                    continue
                if p.color == color:
                    bonus += 18.0
                break
        return bonus

    @staticmethod
    def _col_pieces_between(b, r0: int, r1: int, c: int) -> int:
        """同一列上严格位于 r0、r1 之间的棋子数（不含端点）。"""
        if r0 > r1:
            r0, r1 = r1, r0
        n = 0
        for r in range(r0 + 1, r1):
            if b[r][c] is not None:
                n += 1
        return n

    @staticmethod
    def _ma_premium_square(player: str, r: int, c: int) -> bool:
        pst_r = r if player == "red" else 9 - r
        return pst_r <= 2 and c in (3, 6)

    @staticmethod
    def _ma_any_leg_clear(b, sr: int, sc: int) -> bool:
        for dr, dc in Evaluation._MA_DELTAS:
            er, ec = sr + dr, sc + dc
            if not (0 <= er < 10 and 0 <= ec < 9):
                continue
            if abs(dr) == 2:
                leg_r, leg_c = sr + (1 if dr > 0 else -1), sc
            else:
                leg_r, leg_c = sr, sc + (1 if dc > 0 else -1)
            if b[leg_r][leg_c] is None:
                return True
        return False

    @staticmethod
    def _tactical_synergy(
        board: Board,
        player: str,
        b,
        enemy_kr: Optional[int],
        enemy_kc: Optional[int],
    ) -> float:
        """轻量战术分：纵线车/炮、高位马马腿、车马协同。不着法生成。"""
        if enemy_kr is None or enemy_kc is None:
            return 0.0
        ekr, ekc = enemy_kr, enemy_kc
        score = 0.0
        ma_deep = False
        che_flank = False

        if player == "red":
            deep_lo, deep_hi = 0, 2
            che_deep_lo, che_deep_hi = 0, 2
        else:
            deep_lo, deep_hi = 7, 9
            che_deep_lo, che_deep_hi = 7, 9

        for r, c in board.active_pieces.get(player, ()):
            p = b[r][c]
            if p is None or p.color != player:
                continue
            pt = p.piece_type

            if pt == "che":
                if c == ekc:
                    between = Evaluation._col_pieces_between(b, r, ekr, c)
                    if between == 0:
                        score += Evaluation._T_ROOK_FILE
                if che_deep_lo <= r <= che_deep_hi:
                    che_flank = True
                elif c in (0, 8) and (
                    (player == "red" and r <= 4) or (player == "black" and r >= 5)
                ):
                    che_flank = True

            elif pt == "pao":
                if c == ekc:
                    between = Evaluation._col_pieces_between(b, r, ekr, c)
                    if between == 1:
                        score += Evaluation._T_PAO_SCREEN

            elif pt == "ma":
                if Evaluation._ma_premium_square(player, r, c) and Evaluation._ma_any_leg_clear(
                    b, r, c
                ):
                    score += Evaluation._T_MA_LEG_PREMIUM
                if deep_lo <= r <= deep_hi:
                    ma_deep = True

        if ma_deep and che_flank:
            score += Evaluation._T_MA_CHE_SYNERGY
        return score

    @staticmethod
    def evaluate(board: Board) -> float:
        """Tapered Evaluation：中局/残局双轨按 ``phase`` 线性插值；Negamax 视角。"""
        h = board.zobrist_hash
        if h in Evaluation._eval_cache:
            return Evaluation._eval_cache[h]

        b = board.board
        rk = board.red_king_pos
        bk = board.black_king_pos
        red_kr = rk[0] if rk else None
        red_kc = rk[1] if rk else None
        black_kr = bk[0] if bk else None
        black_kc = bk[1] if bk else None

        attacking = Evaluation.ATTACKING_PIECE_TYPES
        mg_map = Evaluation.PST_MG_MAP
        eg_map = Evaluation.PST_EG_MAP
        mgv = Evaluation.MG_VALUES
        egv = Evaluation.EG_VALUES
        pw = Evaluation.PHASE_WEIGHTS

        red_attack = black_attack = 0
        red_mg = red_eg = black_mg = black_eg = 0.0
        phase = 0.0
        total_mat = 0.0
        red_shi = red_xiang = red_che = red_pao = 0
        black_shi = black_xiang = black_che = black_pao = 0

        for color_key in ("red", "black"):
            for r, c in board.active_pieces.get(color_key, ()):
                p = b[r][c]
                if p is None or p.color != color_key:
                    continue
                pt = p.piece_type
                if pt in attacking:
                    if color_key == "red":
                        red_attack += 1
                    else:
                        black_attack += 1
                if pt == "shi":
                    if color_key == "red":
                        red_shi += 1
                    else:
                        black_shi += 1
                elif pt == "xiang":
                    if color_key == "red":
                        red_xiang += 1
                    else:
                        black_xiang += 1
                elif pt == "che":
                    if color_key == "red":
                        red_che += 1
                    else:
                        black_che += 1
                elif pt == "pao":
                    if color_key == "red":
                        red_pao += 1
                    else:
                        black_pao += 1

                phase += float(pw.get(pt, 0))
                total_mat += float(mgv.get(pt, 0))

                mg_base = float(mgv.get(pt, 0))
                eg_base = float(egv.get(pt, 0))
                pst_r = r if color_key == "red" else 9 - r
                mg_tbl = mg_map.get(pt, Evaluation.PST_DEFAULT)
                eg_tbl = eg_map.get(pt, Evaluation.PST_DEFAULT)
                mg_pst = float(mg_tbl[pst_r][c])
                eg_pst = float(eg_tbl[pst_r][c])

                bonus = 0.0
                if pt == "ma":
                    bonus = Evaluation._ma_mobility(board, r, c, color_key)
                elif pt == "pao":
                    bonus = Evaluation._pao_screen_bonus(board, r, c, color_key)

                acc_mg = mg_base + mg_pst + bonus
                acc_eg = eg_base + eg_pst + bonus
                if color_key == "red":
                    red_mg += acc_mg
                    red_eg += acc_eg
                else:
                    black_mg += acc_mg
                    black_eg += acc_eg

        if red_attack == 0 and black_attack == 0:
            Evaluation._eval_cache[h] = 0.0
            return 0.0

        tact_r = Evaluation._tactical_synergy(board, "red", b, black_kr, black_kc)
        tact_b = Evaluation._tactical_synergy(board, "black", b, red_kr, red_kc)
        red_mg += tact_r
        red_eg += tact_r
        black_mg += tact_b
        black_eg += tact_b

        if red_xiang < 2 and black_pao > 0:
            black_mg += 30.0
            black_eg += 30.0
        if black_xiang < 2 and red_pao > 0:
            red_mg += 30.0
            red_eg += 30.0
        if red_shi < 2 and black_che == 2:
            black_mg += 50.0
            black_eg += 50.0
        if black_shi < 2 and red_che == 2:
            red_mg += 50.0
            red_eg += 50.0

        phase = min(phase, Evaluation.TOTAL_PHASE)
        tp = Evaluation.TOTAL_PHASE
        red_score = (red_mg * phase + red_eg * (tp - phase)) / tp
        black_score = (black_mg * phase + black_eg * (tp - phase)) / tp

        diff = red_score - black_score
        deficit = max(0.0, Evaluation._REF_TOTAL_MATERIAL - total_mat)
        pen = Evaluation._ANTI_TRADE_COEFF * deficit
        if diff > 50:
            red_score -= pen
        elif diff < -50:
            black_score -= pen

        opp = "black" if board.current_player == "red" else "red"
        check_bonus = 15.0 if Rules.is_king_in_check(board, opp) else 0.0
        if board.current_player == "red":
            res = red_score - black_score + check_bonus
        else:
            res = black_score - red_score + check_bonus
        Evaluation._eval_cache[h] = res
        return res
