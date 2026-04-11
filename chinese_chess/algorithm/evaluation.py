"""启发式评估函数（Evaluator）。

- **Tapered Evaluation**：MG/EG 双轨 + 阶段 ``phase`` 线性插值。
- 子力：``MG_VALUES`` / ``EG_VALUES``；位置：``PST_MG_MAP`` / ``PST_EG_MAP``。
- 兑子惩罚、马/炮附加、九宫压力、过河兵纵深、兵种相克、将军奖励。
"""

from __future__ import annotations

from chinese_chess.model.board import Board
from chinese_chess.model.rules import Rules

# 黑方九宫（棋盘上方）、红方九宫（棋盘下方），列 3–5；用于与伪合法落点集合 O(1) 求交
BLACK_PALACE_SQUARES: frozenset[tuple[int, int]] = frozenset(
    (r, c) for r in range(0, 3) for c in range(3, 6)
)
RED_PALACE_SQUARES: frozenset[tuple[int, int]] = frozenset(
    (r, c) for r in range(7, 10) for c in range(3, 6)
)


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

    PST_BING_MG = [
        [14, 14, 16, 18, 20, 18, 16, 14, 14],
        [16, 16, 18, 22, 24, 22, 18, 16, 16],
        [12, 12, 16, 20, 22, 20, 16, 12, 12],
        [8, 8, 12, 16, 18, 16, 12, 8, 8],
        [4, 4, 8, 10, 12, 10, 8, 4, 4],
        [2, 2, 4, 6, 6, 6, 4, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    PST_BING_EG = [
        [30, 30, 40, 50, 60, 50, 40, 30, 30],
        [25, 25, 35, 45, 50, 45, 35, 25, 25],
        [20, 20, 30, 40, 45, 40, 30, 20, 20],
        [15, 15, 20, 30, 35, 30, 20, 15, 15],
        [10, 10, 15, 20, 25, 20, 15, 10, 10],
        [5, 5, 5, 10, 10, 10, 5, 5, 5],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    PST_MA = [
        [0, 1, 2, 4, 4, 4, 2, 1, 0],
        [1, 4, 7, 9, 10, 9, 7, 4, 1],
        [2, 7, 10, 12, 13, 12, 10, 7, 2],
        [3, 9, 12, 14, 15, 14, 12, 9, 3],
        [3, 9, 13, 15, 16, 15, 13, 9, 3],
        [3, 8, 12, 14, 15, 14, 12, 8, 3],
        [2, 6, 10, 12, 13, 12, 10, 6, 2],
        [1, 4, 6, 8, 9, 8, 6, 4, 1],
        [0, 2, 3, 4, 5, 4, 3, 2, 0],
        [0, 0, 1, 2, 3, 2, 1, 0, 0],
    ]

    PST_CHE = [
        [4, 5, 6, 7, 8, 7, 6, 5, 4],
        [5, 6, 7, 8, 9, 8, 7, 6, 5],
        [6, 7, 8, 9, 10, 9, 8, 7, 6],
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
        "ma": PST_MA,
        "che": PST_CHE,
        "shi": PST_SHI,
        "xiang": PST_XIANG,
        "pao": PST_PAO,
        "jiang": PST_JIANG,
    }
    PST_EG_MAP: dict[str, list[list[int]]] = {
        "bing": PST_BING_EG,
        "ma": PST_MA,
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
    def _palace_pressure(board: Board, player: str) -> float:
        """大子伪合法落点与敌方九宫的交集规模 × 分；每方一次全量 ``get_pseudo_legal_moves``。"""
        enemy_palace = BLACK_PALACE_SQUARES if player == "red" else RED_PALACE_SQUARES
        # 单次遍历：每起点落在九宫内的走法条数（无 per-start set；与「去重后的落点数」一致当且仅当
        # 同一 (起点, 终点) 在伪合法列表中至多出现一次）
        palace_hits_by_start: dict[tuple[int, int], int] = {}
        for sr, sc, er, ec in Rules.get_pseudo_legal_moves(board, player):
            if (er, ec) in enemy_palace:
                k = (sr, sc)
                palace_hits_by_start[k] = palace_hits_by_start.get(k, 0) + 1

        pressure = 0.0
        b = board.board
        for r, c in board.active_pieces.get(player, ()):
            p = b[r][c]
            if p is None or p.color != player:
                continue
            pt = p.piece_type
            if pt not in Evaluation.MAJOR_TYPES:
                continue
            if pt == "ma":
                if player == "red" and r >= 5:
                    continue
                if player == "black" and r <= 4:
                    continue
            hits = palace_hits_by_start.get((r, c), 0)
            pressure += hits * 20.0
        return pressure

    @staticmethod
    def _apply_anti_trading(red_score: float, black_score: float, total_mat: float) -> tuple[float, float]:
        diff = red_score - black_score
        deficit = max(0.0, Evaluation._REF_TOTAL_MATERIAL - total_mat)
        pen = Evaluation._ANTI_TRADE_COEFF * deficit
        if diff > 50:
            red_score -= pen
        elif diff < -50:
            black_score -= pen
        return red_score, black_score

    @staticmethod
    def evaluate(board: Board) -> float:
        """Tapered Evaluation：中局/残局双轨按 ``phase`` 线性插值；Negamax 视角。"""
        h = board.zobrist_hash
        if h in Evaluation._eval_cache:
            return Evaluation._eval_cache[h]

        b = board.board
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

                b_mg = 0.0
                b_eg = 0.0
                if pt == "bing":
                    if color_key == "red" and r <= 4:
                        vert = float((4 - r) * 20)
                        ctr = max(0.0, 15.0 - 5.0 * abs(c - 4))
                        extra = vert + ctr
                        if r == 0:
                            extra += 80.0
                        b_mg = b_eg = extra
                    elif color_key == "black" and r >= 5:
                        vert = float((r - 5) * 20)
                        ctr = max(0.0, 15.0 - 5.0 * abs(c - 4))
                        extra = vert + ctr
                        if r == 9:
                            extra += 80.0
                        b_mg = b_eg = extra
                elif pt == "che":
                    if color_key == "red" and r <= 1:
                        b_mg = b_eg = 30.0
                    elif color_key == "black" and r >= 8:
                        b_mg = b_eg = 30.0

                acc_mg = mg_base + mg_pst + bonus + b_mg
                acc_eg = eg_base + eg_pst + bonus + b_eg
                if color_key == "red":
                    red_mg += acc_mg
                    red_eg += acc_eg
                else:
                    black_mg += acc_mg
                    black_eg += acc_eg

        if red_attack == 0 and black_attack == 0:
            Evaluation._eval_cache[h] = 0.0
            return 0.0

        red_pressure = Evaluation._palace_pressure(board, "red")
        black_pressure = Evaluation._palace_pressure(board, "black")
        red_mg += red_pressure
        red_eg += red_pressure
        black_mg += black_pressure
        black_eg += black_pressure

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

        red_score, black_score = Evaluation._apply_anti_trading(red_score, black_score, total_mat)

        opp = "black" if board.current_player == "red" else "red"
        check_bonus = 15.0 if Rules.is_king_in_check(board, opp) else 0.0
        if board.current_player == "red":
            res = red_score - black_score + check_bonus
        else:
            res = black_score - red_score + check_bonus
        Evaluation._eval_cache[h] = res
        return res
