"""Minimax AI with alpha-beta pruning.

Physical migration from `chinese-chess/ai/ai_minimax.py` with only import-path
and API updates (Board.make_move -> Board.apply_move, Board.*rules -> Rules.*).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from chinese_chess.model.rules import Rules

from .evaluation import Evaluation

# 置换表边界类型（与当前 alpha/beta 窗口配合使用）
_TT_EXACT = 0  # 精确值：alpha < score < beta
_TT_LOWER = 1  # 下界（fail-high）：score >= beta
_TT_UPPER = 2  # 上界（fail-low）：score <= alpha

# 吃子走法与普通走法拉开差距，保证 MVV-LVA 只吃子排序仍优先于全部非吃子
_CAPTURE_SORT_BIAS = 10000
# 杀手走法排序权重（低于吃子基线 10000，高于普通走子 0）
_KILLER_PRIMARY_BONUS = 5000
_KILLER_SECONDARY_BONUS = 4000

# 杀手表按「当前 _alphabeta 的剩余深度 depth」索引；与常见深度上界对齐
MAX_KILLER_DEPTH = 10


class MinimaxAI:
    def __init__(self, depth=3):
        # Minimax搜索的深度限制。
        self.depth = depth
        self._nodes = 0
        self._tt_hits = 0
        # 供 GUI Dashboard/日志读取的最近一次搜索统计
        self.last_stats: Dict[str, Any] = {}
        # 每层深度最多保存 2 个杀手走法（位移覆盖：新杀手占 slot0，原 slot0 下沉到 slot1）
        self.killer_moves: List[List[Optional[Tuple[int, int, int, int]]]] = [
            [None, None] for _ in range(MAX_KILLER_DEPTH)
        ]
        # 普通 dict 置换表；值为 (stored_depth, score, flag)
        self.transposition_table: Dict[int, Tuple[int, float, int]] = {}

    def _tt_probe(self, key: int, depth: int, alpha: float, beta: float) -> Optional[float]:
        entry = self.transposition_table.get(key)
        if entry is None:
            return None
        stored_depth, score, flag = entry
        if stored_depth < depth:
            return None
        if flag == _TT_EXACT:
            return score
        if flag == _TT_LOWER and score >= beta:
            return score
        if flag == _TT_UPPER and score <= alpha:
            return score
        return None

    def _tt_store(self, key: int, depth: int, score: float, alpha_orig: float, beta_orig: float) -> None:
        # 避免 alpha=-inf / beta=+inf 时误分类（虽多数情况下有限分数不会踩坑）
        if score <= alpha_orig and alpha_orig != float("-inf"):
            flag = _TT_UPPER
        elif score >= beta_orig and beta_orig != float("inf"):
            flag = _TT_LOWER
        else:
            flag = _TT_EXACT
        self._tt_write_entry(key, depth, score, flag)

    def _tt_store_exact(self, key: int, depth: int, score: float) -> None:
        """叶节点/评估值与搜索窗口无关，固定为 EXACT。"""
        self._tt_write_entry(key, depth, score, _TT_EXACT)

    def _tt_write_entry(self, key: int, depth: int, score: float, flag: int) -> None:
        self.transposition_table[key] = (depth, score, flag)

    @staticmethod
    def _killer_index(depth: int) -> int:
        if depth < 0:
            return 0
        return min(depth, MAX_KILLER_DEPTH - 1)

    def _reset_killers(self) -> None:
        for slot in self.killer_moves:
            slot[0] = None
            slot[1] = None

    def _push_killer(self, depth: int, move: Tuple[int, int, int, int]) -> None:
        i = self._killer_index(depth)
        k0 = self.killer_moves[i][0]
        if move == k0 or move == self.killer_moves[i][1]:
            return
        self.killer_moves[i][1] = k0
        self.killer_moves[i][0] = move

    @staticmethod
    def _is_capture(board, move: Tuple[int, int, int, int]) -> bool:
        _, _, er, ec = move
        return board.get_piece(er, ec) is not None

    def order_moves(
        self, board, moves: List[Tuple[int, int, int, int]], depth: int
    ) -> None:
        """MVV-LVA + 杀手走法启发式，原地按优先级降序排列。

        吃子：``_CAPTURE_SORT_BIAS + victim - attacker``（同 Evaluation.PIECE_VALUES）。
        非吃子但若命中 ``killer_moves[depth]``：slot0 +5000，slot1 +4000。
        """
        pv = Evaluation.PIECE_VALUES
        ki = self._killer_index(depth)
        killers = self.killer_moves[ki]
        k0, k1 = killers[0], killers[1]

        def move_score(m: Tuple[int, int, int, int]) -> int:
            sr, sc, er, ec = m
            victim = board.get_piece(er, ec)
            if victim is None:
                score = 0
            else:
                attacker = board.get_piece(sr, sc)
                victim_value = int(pv.get(victim.piece_type, 0))
                attacker_value = int(pv.get(attacker.piece_type, 0)) if attacker else 0
                score = _CAPTURE_SORT_BIAS + victim_value - attacker_value
            if m == k0:
                score += _KILLER_PRIMARY_BONUS
            elif m == k1:
                score += _KILLER_SECONDARY_BONUS
            return score

        moves.sort(key=move_score, reverse=True)

    def choose_move(self, board, time_limit: Optional[float] = 10.0) -> Optional[Tuple[int, int, int, int]]:
        """Searcher 接口：为当前 board.current_player 选择一步。"""
        return self.get_best_move(board, time_limit=time_limit)

    def get_best_move(self, board, time_limit: Optional[float] = 10.0):
        """在允许时间内选择最优走法（Alpha-Beta Minimax）。"""
        start = time.time()
        self._nodes = 0
        self._tt_hits = 0
        # 每步根搜索清空 TT：跨回合复用同键曾导致子树全命中、nodes=0 与着法异常
        self.transposition_table.clear()
        self._reset_killers()

        player = board.current_player
        maximizing = (player == "red")  # 评估函数基础分为 red-black
        best_move = None
        best_value = float("-inf") if maximizing else float("inf")

        moves = Rules.get_all_moves(board, player, validate_self_check=False)
        self.order_moves(board, moves, self.depth)
        for move in moves:
            if time_limit is not None and (time.time() - start) > time_limit:
                break

            mover = board.current_player
            captured = board.apply_move(*move)
            if Rules.is_king_in_check(board, mover) or Rules._jiang_face_to_face(board):
                board.undo_move(*move, captured)
                continue
            value = self._alphabeta(
                board=board,
                depth=self.depth - 1,
                alpha=float("-inf"),
                beta=float("inf"),
                maximizing=not maximizing,
                start_time=start,
                time_limit=time_limit,
                maximizing_color=player,
            )
            board.undo_move(*move, captured)

            if maximizing:
                if value > best_value:
                    best_value = value
                    best_move = move
            else:
                if value < best_value:
                    best_value = value
                    best_move = move

        elapsed = time.time() - start
        self.last_stats = {
            "depth": int(self.depth),
            "time_taken": float(elapsed),
            "nodes_evaluated": int(self._nodes),
            "tt_hits": int(self._tt_hits),
        }
        print(f"本次搜索深度: {self.depth}")
        print(f"搜索耗时 (秒): {elapsed:.3f}")
        print(f"评估的节点总数: {self._nodes} (置换表命中: {self._tt_hits})")
        return best_move

    def _alphabeta(
        self,
        board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        start_time: float,
        time_limit: Optional[float],
        maximizing_color: str,
    ) -> float:
        """标准 Alpha-Beta 剪枝递归。"""
        if time_limit is not None and (time.time() - start_time) > time_limit:
            # 超时：返回当前评估（不再扩展）
            self._nodes += 1
            return Evaluation.evaluate(board, maximizing_color=maximizing_color)

        alpha_orig, beta_orig = alpha, beta
        pos_key = board.zobrist_hash
        tt_hit = self._tt_probe(pos_key, depth, alpha, beta)
        if tt_hit is not None:
            self._tt_hits += 1
            return tt_hit

        if depth == 0 or Rules.is_game_over(board):
            self._nodes += 1
            leaf = Evaluation.evaluate(board, maximizing_color=maximizing_color)
            self._tt_store_exact(pos_key, depth, leaf)
            return leaf

        player = board.current_player
        moves = Rules.get_all_moves(board, player, validate_self_check=False)
        self.order_moves(board, moves, depth)

        if maximizing:
            value = float("-inf")
            any_legal = False
            for move in moves:
                is_cap = self._is_capture(board, move)
                mover = board.current_player
                captured = board.apply_move(*move)
                if Rules.is_king_in_check(board, mover) or Rules._jiang_face_to_face(board):
                    board.undo_move(*move, captured)
                    continue
                any_legal = True
                child = self._alphabeta(
                    board=board,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    maximizing=False,
                    start_time=start_time,
                    time_limit=time_limit,
                    maximizing_color=maximizing_color,
                )
                board.undo_move(*move, captured)
                value = max(value, child)
                alpha = max(alpha, value)
                if beta <= alpha:
                    if not is_cap:
                        self._push_killer(depth, move)
                    break
            if not any_legal:
                self._nodes += 1
                value = Evaluation.evaluate(board, maximizing_color=maximizing_color)
            self._tt_store(pos_key, depth, value, alpha_orig, beta_orig)
            return value
        else:
            value = float("inf")
            any_legal = False
            for move in moves:
                is_cap = self._is_capture(board, move)
                mover = board.current_player
                captured = board.apply_move(*move)
                if Rules.is_king_in_check(board, mover) or Rules._jiang_face_to_face(board):
                    board.undo_move(*move, captured)
                    continue
                any_legal = True
                child = self._alphabeta(
                    board=board,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    maximizing=True,
                    start_time=start_time,
                    time_limit=time_limit,
                    maximizing_color=maximizing_color,
                )
                board.undo_move(*move, captured)
                value = min(value, child)
                beta = min(beta, value)
                if beta <= alpha:
                    if not is_cap:
                        self._push_killer(depth, move)
                    break
            if not any_legal:
                self._nodes += 1
                value = Evaluation.evaluate(board, maximizing_color=maximizing_color)
            self._tt_store(pos_key, depth, value, alpha_orig, beta_orig)
            return value

