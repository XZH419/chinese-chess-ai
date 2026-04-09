"""Evaluation functions for Xiangqi.

Physical migration from `chinese-chess/ai/evaluation.py` with only import-path
updates (no math/heuristics changes).
"""

from __future__ import annotations

from chess.model.rules import Rules


class Evaluation:
    # 各棋子类型的基础价值。
    PIECE_VALUES = {
        "jiang": 10000,
        "shi": 120,
        "xiang": 110,
        "ma": 300,
        "che": 600,
        "pao": 300,
        "bing": 70,
    }

    POSITION_VALUES = {
        "jiang": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 2, 2, 2, 0, 0, 0],
            [0, 0, 0, 11, 15, 11, 0, 0, 0],
        ],
        "bing": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        # 如果需要，可为其他棋子添加位置价值表。
    }

    @staticmethod
    def evaluate(board):
        """计算当前局面的启发式评分。

        正分表示红方优势，负分表示黑方优势。
        """

        score = 0
        for r in range(board.rows):
            for c in range(board.cols):
                piece = board.board[r][c]
                if piece:
                    value = Evaluation.PIECE_VALUES[piece.piece_type]
                    if piece.color == "red":
                        score += value
                        # 加上棋子的位置加成。
                        if piece.piece_type in Evaluation.POSITION_VALUES:
                            score += Evaluation.POSITION_VALUES[piece.piece_type][r][c]
                    else:
                        score -= value
                        # 黑方位置加成需要翻转行坐标。
                        if piece.piece_type in Evaluation.POSITION_VALUES:
                            score -= Evaluation.POSITION_VALUES[piece.piece_type][9 - r][c]

        # 加入机动性奖励，走法更多的一方略占优势。
        red_moves = len(Rules.get_all_moves(board, "red"))
        black_moves = len(Rules.get_all_moves(board, "black"))
        score += (red_moves - black_moves) * 0.1

        return score

