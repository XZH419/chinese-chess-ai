"""启发式评估函数（Evaluator）。

目标：
- 子力价值：车900，炮450，马400，兵100，士/相200
- 位置分：Piece-Square Tables（PST）
- 兑子惩罚、炮架/马腿机动、九宫压力、残局兵价值、将军奖励
- 返回：当前行棋方视角；双方均无车/马/炮/兵时视为物质和棋，直接 0 分
"""

from __future__ import annotations

from collections import defaultdict

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
    # 可参与将杀/实质性进攻的子力（仅剩将、士、象视为无法将死 → 评估为和）
    ATTACKING_PIECE_TYPES = frozenset({"che", "ma", "pao", "bing"})
    MAJOR_TYPES = frozenset({"che", "ma", "pao"})

    # 基础子力价值（单位分）
    PIECE_VALUES = {
        "che": 900,
        "pao": 450,
        "ma": 400,
        "xiang": 200,
        "shi": 200,
        "bing": 100,
        "jiang": 0,
    }

    # 开局子力总和约量级，用于兑子惩罚归一
    _REF_TOTAL_MATERIAL = 4400
    # 领先方（|diff|>50）在子力减少时受到的兑子惩罚系数
    _ANTI_TRADE_COEFF = 0.018
    # 残局：子力总和低于此或棋子数少时强化过河兵
    _ENDGAME_MATERIAL = 2800
    _ENDGAME_PIECE_COUNT = 15

    PST_BING = [
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

    PST_DEFAULT = [[0] * 9 for _ in range(10)]

    PST_MAP = {
        "bing": PST_BING,
        "ma": PST_MA,
        "che": PST_CHE,
        "shi": PST_DEFAULT,
        "xiang": PST_DEFAULT,
        "pao": PST_DEFAULT,
        "jiang": PST_DEFAULT,
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
                v = float(Evaluation.PIECE_VALUES.get(p.piece_type, 0))
                n += 1
                if color_key == "red":
                    red_m += v
                else:
                    black_m += v
        return red_m, black_m, red_m + black_m, n

    @staticmethod
    def _ma_mobility(board: Board, sr: int, sc: int, color: str) -> float:
        b = board.board
        cnt = 0
        for dr, dc in Evaluation._MA_DELTAS:
            er, ec = sr + dr, sc + dc
            if not (0 <= er < 10 and 0 <= ec < 9):
                continue
            tgt = b[er][ec]
            if tgt is not None and tgt.color == color:
                continue
            if Rules._is_valid_ma_move(board, sr, sc, er, ec):
                cnt += 1
        return float(cnt * 5)

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
        dest_by_start: defaultdict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
        for sr, sc, er, ec in Rules.get_pseudo_legal_moves(board, player):
            dest_by_start[(sr, sc)].add((er, ec))

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
            hits = dest_by_start.get((r, c), set()) & enemy_palace
            pressure += len(hits) * 20.0
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
        """评估函数（纯 Negamax 版）。

        永远返回“当前即将行棋方（board.current_player）”的视角分数。
        """

        b = board.board
        pv = Evaluation.PIECE_VALUES
        pst_map = Evaluation.PST_MAP
        attacking = Evaluation.ATTACKING_PIECE_TYPES

        red_attack = 0
        black_attack = 0
        for color_key in ("red", "black"):
            for r, c in board.active_pieces.get(color_key, ()):
                p = b[r][c]
                if p is None or p.color != color_key:
                    continue
                if p.piece_type in attacking:
                    if color_key == "red":
                        red_attack += 1
                    else:
                        black_attack += 1
        if red_attack == 0 and black_attack == 0:
            return 0.0

        _, _, total_mat, piece_count = Evaluation._raw_material(board)
        is_endgame = total_mat <= Evaluation._ENDGAME_MATERIAL or piece_count <= Evaluation._ENDGAME_PIECE_COUNT

        red_score = 0.0
        black_score = 0.0

        for r, c in board.active_pieces.get("red", ()):
            piece = b[r][c]
            if piece is None or piece.color != "red":
                continue
            pt = piece.piece_type
            base = float(pv.get(pt, 0))
            pst = pst_map.get(pt, Evaluation.PST_DEFAULT)
            red_score += base + float(pst[r][c])
            if pt == "bing" and r <= 4:
                step = float((4 - r) * 20)
                if is_endgame:
                    step *= 2.5
                red_score += step
                if r == 0:
                    red_score += 80.0
            if pt == "che" and r <= 1:
                red_score += 30.0
            if pt == "ma":
                red_score += Evaluation._ma_mobility(board, r, c, "red")
            if pt == "pao":
                red_score += Evaluation._pao_screen_bonus(board, r, c, "red")

        for r, c in board.active_pieces.get("black", ()):
            piece = b[r][c]
            if piece is None or piece.color != "black":
                continue
            pt = piece.piece_type
            base = float(pv.get(pt, 0))
            pst = pst_map.get(pt, Evaluation.PST_DEFAULT)
            black_score += base + float(pst[9 - r][c])
            if pt == "bing" and r >= 5:
                step = float((r - 5) * 20)
                if is_endgame:
                    step *= 2.5
                black_score += step
                if r == 9:
                    black_score += 80.0
            if pt == "che" and r >= 8:
                black_score += 30.0
            if pt == "ma":
                black_score += Evaluation._ma_mobility(board, r, c, "black")
            if pt == "pao":
                black_score += Evaluation._pao_screen_bonus(board, r, c, "black")

        red_score += Evaluation._palace_pressure(board, "red")
        black_score += Evaluation._palace_pressure(board, "black")

        red_score, black_score = Evaluation._apply_anti_trading(red_score, black_score, total_mat)

        opp = "black" if board.current_player == "red" else "red"
        check_bonus = 15.0 if Rules.is_king_in_check(board, opp) else 0.0
        if board.current_player == "red":
            return red_score - black_score + check_bonus
        return black_score - red_score + check_bonus
