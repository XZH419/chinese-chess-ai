"""启发式评估函数（Evaluator）。

目标：
- 子力价值：车900，炮450，马400，兵100，士/相200
- 位置分：Piece-Square Tables（PST）
- 统一评分：score = 红(子力+位置) - 黑(子力+位置)
- 返回：基于调用方视角（is_maximizing_player）返回分数
"""

from __future__ import annotations


class Evaluation:
    # 基础子力价值（单位分）
    PIECE_VALUES = {
        "che": 900,
        "pao": 450,
        "ma": 400,
        "xiang": 200,
        "shi": 200,
        "bing": 100,
        # 将/帅在正常局面中恒存在，终局由搜索层 mate_score 处理；此处置 0 避免干扰 PST。
        "jiang": 0,
    }

    # 位置矩阵：以红方视角（row 从上到下 0..9，红方在下方）
    # 黑方使用“翻转行坐标”进行镜像使用。
    PST_BING = [
        # 红兵向“上”（r 递减）推进：过河前(>=5)很低；过河后越接近对方九宫越高；
        # 贴近底线(row==0)略回落，避免“老兵搜身”过度加分。
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
        # 中心位、卧槽位、挂角位更活跃；边线/死角弱
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
        # 车偏好开放线与过河后的进攻位；死角略低
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

    @staticmethod
    def evaluate(board) -> float:
        """评估函数（纯 Negamax 版）。

        永远返回“当前即将行棋方（board.current_player）”的视角分数：
        - 轮到红方走：return red_score - black_score
        - 轮到黑方走：return black_score - red_score
        """

        b = board.board
        pv = Evaluation.PIECE_VALUES
        pst_map = Evaluation.PST_MAP

        red_score = 0.0
        black_score = 0.0

        # 直接遍历 active_pieces，避免 90 格全盘扫描
        for r, c in board.active_pieces.get("red", ()):
            piece = b[r][c]
            if piece is None or piece.color != "red":
                continue
            base = float(pv.get(piece.piece_type, 0))
            pst = pst_map.get(piece.piece_type, Evaluation.PST_DEFAULT)
            red_score += base + float(pst[r][c])

        for r, c in board.active_pieces.get("black", ()):
            piece = b[r][c]
            if piece is None or piece.color != "black":
                continue
            base = float(pv.get(piece.piece_type, 0))
            pst = pst_map.get(piece.piece_type, Evaluation.PST_DEFAULT)
            # 黑方 PST 以红方视角镜像行
            black_score += base + float(pst[9 - r][c])

        # 不再在评估里调用 get_all_moves（原“机动性”项会对每个叶子节点做两次全量走法生成，
        # 往往占搜索总耗时的大部分；子力+PST 已足够支撑 Minimax 实验对比。）
        if board.current_player == "red":
            return red_score - black_score
        return black_score - red_score

