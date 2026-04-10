"""Minimax AI with alpha-beta pruning.

Physical migration from `chinese-chess/ai/ai_minimax.py` with only import-path
and API updates (Board.make_move -> Board.apply_move, Board.*rules -> Rules.*).
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

from chinese_chess.model import zobrist
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
# 置换表着法排序分（须高于历史+吃子+杀手之和）
_TT_MOVE_SORT_SCORE = 1_000_000
# 排序键中历史分量上限，避免累计过大压过 TT
_HISTORY_SORT_CAP = 500_000

# 杀手表按「当前 _alphabeta 的剩余深度 depth」索引；与常见深度上界对齐
MAX_KILLER_DEPTH = 10


class SearchTimeoutException(Exception):
    """Abort the current iterative-deepening iteration on timeout."""


class MinimaxAI:
    def __init__(
        self,
        depth=3,
        stochastic: bool = False,
        top_k: int = 2,
        tolerance: int = 5,
        verbose: bool = True,
    ):
        # Minimax搜索的深度限制。
        self.depth = depth
        # 根节点：在「近最优」走法中随机化，打破纯确定性开局
        self.stochastic = stochastic
        self.top_k = top_k
        self.tolerance = tolerance
        self.verbose = verbose
        self._nodes = 0
        self._tt_hits = 0
        # 供 GUI Dashboard/日志读取的最近一次搜索统计
        self.last_stats: Dict[str, Any] = {}
        # 无头基准：跨多局累加本实例每次 get_best_move 的耗时与节点（由 reset_benchmark_stats 清零）
        self._bench_total_time: float = 0.0
        self._bench_total_nodes: int = 0
        self._bench_search_count: int = 0
        # 每层深度最多保存 2 个杀手走法（位移覆盖：新杀手占 slot0，原 slot0 下沉到 slot1）
        self.killer_moves: List[List[Optional[Tuple[int, int, int, int]]]] = [
            [None, None] for _ in range(MAX_KILLER_DEPTH)
        ]
        # 普通 dict 置换表；值为 (stored_depth, score, flag, best_move)
        self.transposition_table: Dict[
            int, Tuple[int, float, int, Optional[Tuple[int, int, int, int]]]
        ] = {}
        # 仅供搜索期间使用的上下文（避免每层层层传递）
        self.start_time: float = 0.0
        # 重复局面检测：保存“从根到当前节点”的 zobrist_hash 路径（可叠加外部 game_history）
        self.history_hashes: List[int] = []
        # 历史启发：beta 剪枝着法加权，跨着法积累（与 TT 不同，不在每步根搜索清空）
        self.history_table: Dict[Tuple[int, int, int, int], int] = {}

    def reset_benchmark_stats(self) -> None:
        """清零基准统计（每局对弈开始前调用）。"""
        self._bench_total_time = 0.0
        self._bench_total_nodes = 0
        self._bench_search_count = 0

    def _tt_probe(self, key: int, depth: int, alpha: float, beta: float) -> Optional[float]:
        entry = self.transposition_table.get(key)
        if entry is None:
            return None
        stored_depth, score, flag, _best_move = entry
        if stored_depth < depth:
            return None
        if flag == _TT_EXACT:
            return score
        if flag == _TT_LOWER and score >= beta:
            return score
        if flag == _TT_UPPER and score <= alpha:
            return score
        return None

    def _tt_store(
        self,
        key: int,
        depth: int,
        score: float,
        alpha_orig: float,
        beta_orig: float,
        best_move: Optional[Tuple[int, int, int, int]],
    ) -> None:
        # 避免 alpha=-inf / beta=+inf 时误分类（虽多数情况下有限分数不会踩坑）
        if score <= alpha_orig and alpha_orig != float("-inf"):
            flag = _TT_UPPER
        elif score >= beta_orig and beta_orig != float("inf"):
            flag = _TT_LOWER
        else:
            flag = _TT_EXACT
        self._tt_write_entry(key, depth, score, flag, best_move)

    def _tt_store_exact(
        self,
        key: int,
        depth: int,
        score: float,
        best_move: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """叶节点/评估值与搜索窗口无关，固定为 EXACT。"""
        self._tt_write_entry(key, depth, score, _TT_EXACT, best_move)

    def _tt_write_entry(
        self,
        key: int,
        depth: int,
        score: float,
        flag: int,
        best_move: Optional[Tuple[int, int, int, int]],
    ) -> None:
        self.transposition_table[key] = (depth, score, flag, best_move)

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

    _MAJOR_PIECE_TYPES = frozenset({"che", "ma", "pao"})

    def has_enough_material(self, board, player: str) -> bool:
        """己方车/马/炮大子数 >= 2 时才认为空步假设较安全，减轻残局 Zugzwang 下的过度剪枝。"""
        n = 0
        b = board.board
        for r, c in board.active_pieces.get(player, ()):
            p = b[r][c]
            if p is None or p.color != player:
                continue
            if p.piece_type in self._MAJOR_PIECE_TYPES:
                n += 1
                if n >= 2:
                    return True
        return False

    def order_moves(
        self, board, moves: List[Tuple[int, int, int, int]], depth: int
    ) -> None:
        """排序键（高优先）：TT 着法 > 历史启发 > MVV-LVA 吃子 > 杀手 > 其余。

        历史分量 capped，保证恒低于 ``_TT_MOVE_SORT_SCORE``。
        """
        pv = Evaluation.PIECE_VALUES
        ki = self._killer_index(depth)
        killers = self.killer_moves[ki]
        k0, k1 = killers[0], killers[1]
        entry = self.transposition_table.get(board.zobrist_hash)
        tt_move = entry[3] if entry is not None else None

        def move_score(m: Tuple[int, int, int, int]) -> int:
            if tt_move is not None and m == tt_move:
                return _TT_MOVE_SORT_SCORE
            hist = min(self.history_table.get(m, 0), _HISTORY_SORT_CAP)
            sr, sc, er, ec = m
            victim = board.get_piece(er, ec)
            if victim is None:
                cap = 0
            else:
                attacker = board.get_piece(sr, sc)
                victim_value = int(pv.get(victim.piece_type, 0))
                attacker_value = int(pv.get(attacker.piece_type, 0)) if attacker else 0
                cap = _CAPTURE_SORT_BIAS + victim_value - attacker_value
            score = hist + cap
            if m == k0:
                score += _KILLER_PRIMARY_BONUS
            elif m == k1:
                score += _KILLER_SECONDARY_BONUS
            return score

        moves.sort(key=move_score, reverse=True)

    def choose_move(
        self,
        board,
        time_limit: Optional[float] = 10.0,
        game_history: Optional[List[int]] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Searcher 接口：为当前 board.current_player 选择一步。"""
        return self.get_best_move(board, game_history=game_history, time_limit=time_limit)

    def get_best_move(
        self,
        board,
        game_history: Optional[List[int]] = None,
        time_limit: Optional[float] = 10.0,
    ):
        """在允许时间内选择最优走法（Negamax + Iterative Deepening）。"""
        self.start_time = time.time()
        self._nodes = 0
        self._tt_hits = 0
        # 每步根搜索清空 TT：跨回合复用同键曾导致子树全命中、nodes=0 与着法异常
        self.transposition_table.clear()
        self._reset_killers()

        # 外部可传入从开局到当前局面的完整哈希链；若末尾已是当前局面则不再重复追加
        self.history_hashes = list(game_history) if game_history else []
        if not self.history_hashes or self.history_hashes[-1] != board.zobrist_hash:
            self.history_hashes.append(board.zobrist_hash)

        global_best_move: Optional[Tuple[int, int, int, int]] = None

        try:
            for current_depth in range(1, self.depth + 1):
                moves = list(Rules.get_pseudo_legal_moves(board, board.current_player))
                self.order_moves(board, moves, current_depth)

                alpha = float("-inf")
                beta = float("inf")
                scored_moves: List[Tuple[float, Tuple[int, int, int, int]]] = []
                best_score_so_far = float("-inf")
                any_legal = False

                for move in moves:
                    if time_limit is not None and (time.time() - self.start_time) > time_limit:
                        raise SearchTimeoutException()

                    mover = board.current_player
                    captured = board.apply_move(*move)
                    self.history_hashes.append(board.zobrist_hash)

                    # 延迟合法性校验（防送将/白脸将）
                    if Rules.is_king_in_check(board, mover) or Rules._jiang_face_to_face(board):
                        self.history_hashes.pop()
                        board.undo_move(*move, captured)
                        continue

                    any_legal = True
                    try:
                        score = -self._alphabeta(
                            board,
                            current_depth - 1,
                            -beta,
                            -alpha,
                            self.start_time,
                            time_limit,
                        )
                    finally:
                        self.history_hashes.pop()
                        board.undo_move(*move, captured)

                    # Fail-Low / 窗口剪枝返回的界值：若已不优于当前 alpha，不可信为实分，禁止进随机池
                    actual_record_score = score
                    if score <= alpha:
                        actual_record_score = float("-inf")

                    scored_moves.append((actual_record_score, move))

                    if actual_record_score > best_score_so_far:
                        best_score_so_far = actual_record_score

                    if best_score_so_far > float("-inf") and abs(best_score_so_far) >= 300:
                        current_tol = 0
                    else:
                        current_tol = self.tolerance

                    new_alpha = best_score_so_far - current_tol
                    if new_alpha > alpha:
                        alpha = new_alpha

                if not any_legal:
                    break  # 根节点无路可走：被将死/困毙

                if best_score_so_far > float("-inf") and abs(best_score_so_far) >= 300:
                    current_tol = 0
                else:
                    current_tol = self.tolerance

                finite_scored = [(s, m) for s, m in scored_moves if s > float("-inf")]
                if not finite_scored:
                    break

                if not self.stochastic:
                    finite_scored.sort(key=lambda x: x[0], reverse=True)
                    max_score = finite_scored[0][0]
                    current_best_move = finite_scored[0][1]
                    root_tt_score = max_score
                else:
                    near_best = [
                        (s, m)
                        for s, m in scored_moves
                        if s > float("-inf") and s >= best_score_so_far - current_tol
                    ]
                    near_best.sort(key=lambda x: x[0], reverse=True)
                    pool = near_best[: self.top_k]
                    picked = random.choice(pool)
                    current_best_move = picked[1]
                    root_tt_score = picked[0]

                global_best_move = current_best_move
                # 根节点显式写入 TT：避免 depth=1 直接进入 QS 时超时回退丢失
                self._tt_store_exact(
                    board.zobrist_hash, current_depth, float(root_tt_score), current_best_move
                )
        except SearchTimeoutException:
            pass

        elapsed = time.time() - self.start_time
        self.last_stats = {
            "depth": int(self.depth),
            "time_taken": float(elapsed),
            "nodes_evaluated": int(self._nodes),
            "tt_hits": int(self._tt_hits),
        }
        self._bench_total_time += float(elapsed)
        self._bench_total_nodes += int(self._nodes)
        self._bench_search_count += 1
        if self.verbose:
            print(f"本次搜索深度: {self.depth}")
            print(f"搜索耗时 (秒): {elapsed:.3f}")
            print(f"评估的节点总数: {self._nodes} (置换表命中: {self._tt_hits})")
        return global_best_move

    def _quiescence_search(
        self,
        board,
        alpha: float,
        beta: float,
        depth_limit: int = 4,
    ) -> float:
        """静止搜索（Quiescence Search, QS）：仅扩展吃子走法，缓解水平线效应。

        返回值与 `_alphabeta` 一致：当前行棋方视角，分越高越好。
        """
        if self.history_hashes and board.zobrist_hash in self.history_hashes[:-1]:
            return 0.0

        if depth_limit <= 0:
            self._nodes += 1
            return Evaluation.evaluate(board)

        self._nodes += 1
        alpha_bound = alpha
        stand_pat = Evaluation.evaluate(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        moves = list(Rules.get_pseudo_legal_moves(board, board.current_player))
        captures = [m for m in moves if board.board[m[2]][m[3]] is not None]
        if not captures:
            return alpha

        self.order_moves(board, captures, depth_limit)

        DELTA_MARGIN = 900 + 200
        if stand_pat + DELTA_MARGIN < alpha_bound:
            return alpha_bound

        for move in captures:
            mover = board.current_player
            captured = board.apply_move(*move)
            self.history_hashes.append(board.zobrist_hash)
            if Rules.is_king_in_check(board, mover) or Rules._jiang_face_to_face(board):
                self.history_hashes.pop()
                board.undo_move(*move, captured)
                continue

            try:
                score = -self._quiescence_search(board, -beta, -alpha, depth_limit - 1)
            finally:
                self.history_hashes.pop()
                board.undo_move(*move, captured)

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def _alphabeta(
        self,
        board,
        depth: int,
        alpha: float,
        beta: float,
        start_time: float,
        time_limit: Optional[float],
        *,
        allow_null: bool = True,
        use_tt: bool = True,
    ) -> float:
        """Negamax Alpha-Beta（当前行棋方视角，分越高越好）。"""
        if self.history_hashes and board.zobrist_hash in self.history_hashes[:-1]:
            return 0.0
        if time_limit is not None and (time.time() - start_time) > time_limit:
            raise SearchTimeoutException()

        alpha_orig, beta_orig = alpha, beta
        pos_key = board.zobrist_hash
        if use_tt:
            tt_hit = self._tt_probe(pos_key, depth, alpha, beta)
            if tt_hit is not None:
                self._tt_hits += 1
                return tt_hit

        # Null Move Pruning：深水区 (depth>=5)、非将军、己方大子充足时才探测，减轻残局 Zugzwang 误剪。
        player = board.current_player
        if (
            allow_null
            and depth >= 5
            and not Rules.is_king_in_check(board, player)
            and self.has_enough_material(board, player)
        ):
            R = 2
            reduced_depth = depth - 1 - R
            if reduced_depth < 0:
                reduced_depth = 0
            saved_player = board.current_player
            saved_hash = board.zobrist_hash
            board.current_player = "black" if saved_player == "red" else "red"
            board.zobrist_hash = saved_hash ^ zobrist.BLACK_TO_MOVE
            try:
                null_score = -self._alphabeta(
                    board,
                    reduced_depth,
                    -beta,
                    -beta + 1,
                    start_time,
                    time_limit,
                    allow_null=False,
                    use_tt=False,
                )
            finally:
                board.current_player = saved_player
                board.zobrist_hash = saved_hash
            if null_score >= beta:
                return beta

        if depth == 0:
            return self._quiescence_search(board, alpha, beta)

        moves = list(Rules.get_pseudo_legal_moves(board, board.current_player))
        self.order_moves(board, moves, depth)

        best = float("-inf")
        best_move = None
        any_legal = False

        for move in moves:
            if time_limit is not None and (time.time() - start_time) > time_limit:
                raise SearchTimeoutException()
            is_cap = self._is_capture(board, move)
            mover = board.current_player
            captured = board.apply_move(*move)
            self.history_hashes.append(board.zobrist_hash)
            if Rules.is_king_in_check(board, mover) or Rules._jiang_face_to_face(board):
                self.history_hashes.pop()
                board.undo_move(*move, captured)
                continue
            any_legal = True

            try:
                score = -self._alphabeta(
                    board,
                    depth - 1,
                    -beta,
                    -alpha,
                    start_time,
                    time_limit,
                    allow_null=allow_null,
                    use_tt=use_tt,
                )
            finally:
                self.history_hashes.pop()
                board.undo_move(*move, captured)

            if score > best:
                best = score
                best_move = move
            if best > alpha:
                alpha = best
            if alpha >= beta:
                self.history_table[move] = self.history_table.get(move, 0) + depth * depth
                if not is_cap:
                    self._push_killer(depth, move)
                break

        if not any_legal:
            self._nodes += 1
            mate_score = float(-10000 + (self.depth - depth))
            if use_tt:
                self._tt_store_exact(pos_key, depth, mate_score, None)
            return mate_score

        if use_tt:
            self._tt_store(pos_key, depth, best, alpha_orig, beta_orig, best_move)
        return best

