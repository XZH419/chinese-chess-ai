"""启发式评估函数（Evaluator）。

目标：
- 子力价值：车900，马400，炮450，相/士200，卒100，将10000（可微调）
- 位置分：Piece-Square Tables（兵/马/炮等）
- 统一评分：score = 红(子力+位置) - 黑(子力+位置)
- 返回：基于调用方视角（is_maximizing_player）返回分数
"""

from __future__ import annotations

from typing import Optional

from chess.model.rules import Rules


class Evaluation:
    # 基础子力价值（单位分）
    PIECE_VALUES = {
        "che": 900,
        "ma": 400,
        "pao": 450,
        "xiang": 200,
        "shi": 200,
        "bing": 100,
        "jiang": 10000,
    }

    # 位置矩阵：以红方视角（row 从上到下 0..9，红方在下方）
    # 黑方使用“翻转行坐标”进行镜像使用。
    PST_BING = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 6, 8, 10, 12, 10, 8, 6, 4],     # 黑方河界附近（红方已过河）更有价值
        [6, 8, 10, 14, 16, 14, 10, 8, 6],
        [8, 10, 12, 16, 20, 16, 12, 10, 8],
        [10, 12, 14, 18, 22, 18, 14, 12, 10],
        [8, 8, 10, 12, 14, 12, 10, 8, 8],
        [4, 4, 6, 8, 10, 8, 6, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    PST_MA = [
        [0, 2, 4, 6, 8, 6, 4, 2, 0],
        [2, 6, 10, 12, 14, 12, 10, 6, 2],
        [4, 10, 14, 16, 18, 16, 14, 10, 4],
        [6, 12, 16, 18, 20, 18, 16, 12, 6],
        [6, 12, 16, 20, 22, 20, 16, 12, 6],  # 中路略加分
        [6, 10, 14, 18, 20, 18, 14, 10, 6],
        [4, 8, 12, 14, 16, 14, 12, 8, 4],
        [2, 4, 6, 8, 10, 8, 6, 4, 2],
        [0, 0, 2, 4, 6, 4, 2, 0, 0],
        [0, 0, 0, 2, 4, 2, 0, 0, 0],
    ]

    PST_PAO = [
        [0, 0, 2, 4, 6, 4, 2, 0, 0],
        [0, 2, 4, 6, 8, 6, 4, 2, 0],
        [2, 4, 6, 8, 10, 8, 6, 4, 2],
        [2, 4, 6, 10, 12, 10, 6, 4, 2],
        [2, 4, 8, 12, 14, 12, 8, 4, 2],
        [2, 4, 8, 12, 14, 12, 8, 4, 2],
        [2, 4, 6, 10, 12, 10, 6, 4, 2],
        [2, 4, 6, 8, 10, 8, 6, 4, 2],
        [0, 2, 4, 6, 8, 6, 4, 2, 0],
        [0, 0, 2, 4, 6, 4, 2, 0, 0],
    ]

    PST_CHE = [
        [6, 8, 10, 12, 14, 12, 10, 8, 6],
        [6, 8, 10, 12, 14, 12, 10, 8, 6],
        [4, 6, 8, 10, 12, 10, 8, 6, 4],
        [2, 4, 6, 8, 10, 8, 6, 4, 2],
        [0, 2, 4, 6, 8, 6, 4, 2, 0],
        [0, 2, 4, 6, 8, 6, 4, 2, 0],
        [2, 4, 6, 8, 10, 8, 6, 4, 2],
        [4, 6, 8, 10, 12, 10, 8, 6, 4],
        [6, 8, 10, 12, 14, 12, 10, 8, 6],
        [6, 8, 10, 12, 14, 12, 10, 8, 6],
    ]

    PST_DEFAULT = [[0] * 9 for _ in range(10)]

    PST_MAP = {
        "bing": PST_BING,
        "ma": PST_MA,
        "pao": PST_PAO,
        "che": PST_CHE,
        "shi": PST_DEFAULT,
        "xiang": PST_DEFAULT,
        "jiang": PST_DEFAULT,
    }

    @staticmethod
    def evaluate(board, is_maximizing_player: bool = True, maximizing_color: Optional[str] = None) -> float:
        """评估函数。

        - 基础分以“红方视角”计算：score = red - black
        - 若指定 maximizing_color，则返回其视角分数：
          - maximizing_color == 'red' -> 返回 score
          - maximizing_color == 'black' -> 返回 -score
        - 否则按 is_maximizing_player 返回：
          - True  -> 返回 score
          - False -> 返回 -score
        """

        score = 0.0
        for r in range(board.rows):
            for c in range(board.cols):
                piece = board.board[r][c]
                if not piece:
                    continue

                base = float(Evaluation.PIECE_VALUES.get(piece.piece_type, 0))
                pst = Evaluation.PST_MAP.get(piece.piece_type, Evaluation.PST_DEFAULT)

                if piece.color == "red":
                    score += base + float(pst[r][c])
                else:
                    # 黑方位置表按行镜像（保持“红方视角表”可复用）
                    score -= base + float(pst[9 - r][c])

        # 轻量机动性奖励（避免过强影响，保持可解释性）
        red_moves = len(Rules.get_all_moves(board, "red"))
        black_moves = len(Rules.get_all_moves(board, "black"))
        score += (red_moves - black_moves) * 0.05

        if maximizing_color is not None:
            return score if maximizing_color == "red" else -score
        return score if is_maximizing_player else -score

