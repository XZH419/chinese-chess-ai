"""高性能纯 MCTS（Monte Carlo Tree Search）引擎。
精简版：去除了冗余函数，合并了启发式偏置逻辑，优化了模拟路径。
"""

from __future__ import annotations
import concurrent.futures
import math
import multiprocessing
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from engine.board import Board
from engine.rules import MoveEntry, Rules
from ai.evaluation import Evaluation
from ai.search_move_helpers import (
    MoveGivesCheckCache,
    PostApplyFlagsCache,
    apply_pseudo_legal_with_rule_cache,
)
from ai.opening_book import OPENING_BOOK, mirror_move

# ── 核心常量 ──
_UCB_C = 1.414
_SCORE_SCALE = 600.0
_RAVE_CONST = 300
_POLICY_HVCAP_VALUE = 300
_ROOT_BIAS_SCALE = 75.0
_ROOT_VISITS_TIE_FRAC = 0.85
_FPU_PRIOR_SCALE = 0.20
_SELECTION_MAX_PLIES = 512

Move4 = Tuple[int, int, int, int]

def _parallel_workers_when_safe(requested: int) -> int:
    if requested <= 1: return requested
    if threading.current_thread() is not threading.main_thread(): return 1
    return requested

# ═══════════════════════════════════════════════════════════════
#  启发式辅助函数（合并与精简）
# ═══════════════════════════════════════════════════════════════

def _get_tactical_bias(board: Board, mover: str, m: Move4, mcts_gives: Dict) -> float:
    """统一的战术偏置计算：包含吃子分、将军分、推进分。"""
    sr, sc, er, ec = m
    b = board.board
    bias = 0.0
    
    # 1. 吃子偏置 (MVV)
    victim = b[er][ec]
    if victim:
        bias += min(1.0, float(Evaluation.PIECE_VALUES.get(victim.piece_type, 0)) / 900.0)
    
    # 2. 将军偏置
    if mcts_fast_move_gives_check(board, m, mover, mcts_gives):
        bias += 0.45
        
    # 3. 推进偏置 (过河兵/大子压境)
    piece = b[sr][sc]
    if not victim and piece:
        pt = piece.piece_type
        if pt == "bing":
            if (mover == "red" and er < sr) or (mover == "black" and er > sr): bias += 0.15
        elif (mover == "red" and er <= 4) or (mover == "black" and er >= 5):
            if pt in ("che", "ma", "pao"): bias += 0.1
    return bias

def mcts_fast_move_gives_check(board: Board, move: Move4, mover: str, cache: Dict) -> bool:
    """精简后的将军检测，带局部缓存。"""
    key = (board.zobrist_hash, move)
    if key in cache: return cache[key]
    
    captured = board.apply_move(*move)
    opp = "black" if mover == "red" else "red"
    is_check = Rules.is_king_in_check(board, opp)
    board.undo_move(*move, captured)
    
    if len(cache) < 100000: cache[key] = is_check
    return is_check

# ═══════════════════════════════════════════════════════════════
#  MCTS 核心组件
# ═══════════════════════════════════════════════════════════════

class MCTSNode:
    __slots__ = ["state_hash", "children", "visits", "wins", "untried_moves", "player_just_moved", "rave_visits", "rave_wins"]

    def __init__(self, state_hash: int, player_just_moved: str, prior: float = 0.0):
        self.state_hash = state_hash
        self.children: Dict[Move4, MCTSNode] = {}
        self.visits = 1 if prior > 0 else 0
        self.wins = prior
        self.untried_moves = None
        self.player_just_moved = player_just_moved
        self.rave_visits = 0
        self.rave_wins = 0.0

    def best_child_ucb(self, log_parent: float) -> Tuple[Move4, MCTSNode]:
        best_score = -1.0
        res = (None, None)
        for move, ch in self.children.items():
            v, rv = ch.visits, ch.rave_visits
            if v + rv == 0: return move, ch
            beta = rv / (rv + v + _RAVE_CONST + 1e-5)
            exploit = (1.0 - beta) * (ch.wins / v if v > 0 else 0) + beta * (ch.rave_wins / rv if rv > 0 else 0)
            score = exploit + _UCB_C * math.sqrt(log_parent / (v + 1e-5))
            if score > best_score:
                best_score, res = score, (move, ch)
        return res

# ═══════════════════════════════════════════════════════════════
#  搜索流程
# ═══════════════════════════════════════════════════════════════

def _simulate(board: Board, root_player: str) -> float:
    """极致精简的 Rollout，去除了复杂的逃避逻辑以换取 SPS。"""
    sim_board = board.copy()
    limit = 30 if sim_board.piece_count() > 20 else 50
    for _ in range(limit):
        cp = sim_board.current_player
        moves = list(Rules.get_pseudo_legal_moves(sim_board, cp))
        if not moves: return 0.0 if cp == root_player else 1.0
        
        # 贪婪吃子偏好
        caps = [m for m in moves if sim_board.board[m[2]][m[3]]]
        move = random.choice(caps) if caps and random.random() < 0.8 else random.choice(moves)
        
        sim_board.apply_move(*move)
        if sim_board.red_king_pos is None: return 0.0 if root_player == "red" else 1.0
        if sim_board.black_king_pos is None: return 1.0 if root_player == "red" else 0.0
        
    # 截断评估
    raw = Evaluation.evaluate(sim_board)
    if sim_board.current_player != root_player: raw = -raw
    return 1.0 / (1.0 + math.exp(-raw / _SCORE_SCALE))

def _run_single_mcts_tree(board: Board, max_sims: int, tl: float, seed: int = 0) -> Dict:
    random.seed(time.time_ns() + seed)
    t0, root_board = time.time(), board
    root = MCTSNode(board.zobrist_hash, "black" if board.current_player == "red" else "red")
    tt, mcts_cache = {root.state_hash: root}, {}
    
    for _ in range(max_sims):
        if time.time() - t0 >= tl: break
        
        # 1. Selection & 2. Expansion
        sim_board, node, path = root_board.copy(), root, [root]
        while node.untried_moves is not None and len(node.untried_moves) == 0 and node.children:
            move, node = node.best_child_ucb(math.log(node.visits))
            sim_board.apply_move(*move)
            path.append(node)
            
        if node.untried_moves is None:
            node.untried_moves = list(Rules.get_pseudo_legal_moves(sim_board, sim_board.current_player))
            random.shuffle(node.untried_moves)

        if node.untried_moves:
            move = node.untried_moves.pop()
            sim_board.apply_move(*move)
            h = sim_board.zobrist_hash
            if h in tt:
                child = tt[h]
            else:
                bias = _get_tactical_bias(sim_board, sim_board.current_player, move, mcts_cache)
                child = tt[h] = MCTSNode(h, sim_board.current_player, 0.1 + bias * _FPU_PRIOR_SCALE)
            node.children[move] = child
            path.append(child)

        # 3. Simulation & 4. Backprop
        res = _simulate(sim_board, root_board.current_player)
        for i in range(len(path)-1, -1, -1):
            path[i].visits += 1
            if path[i].player_just_moved != root_board.current_player: path[i].wins += res
            res = 1.0 - res
            
    return {m: {"v": c.visits, "w": c.wins} for m, c in root.children.items()}

# ═══════════════════════════════════════════════════════════════
#  AI 接口
# ═══════════════════════════════════════════════════════════════

class MCTSAI:
    def __init__(self, max_simulations: int = 5000, time_limit: float = 10.0, workers: int = None, verbose: bool = True):
        self.max_simulations = max_simulations
        self.time_limit = time_limit
        self.verbose = verbose
        self.workers = workers or min(8, multiprocessing.cpu_count())
        self.last_stats: Optional[Dict[str, Any]] = None

    def get_best_move(
        self,
        board: Board,
        time_limit: Optional[float] = None,
        game_history: List[int] = None,
        **kwargs,
    ) -> Optional[Move4]:
        # 1. 开局库拦截
        if len(game_history or []) < 20:
            res = self._probe_book(board)
            if res:
                self.last_stats = {
                    "opening_book": True,
                    "time_taken": 0.0,
                    "time_limit": time_limit,
                    "workers": 1,
                }
                return res

        # 2. 搜索
        tl = self.time_limit if time_limit is None else float(time_limit)
        workers = _parallel_workers_when_safe(self.workers)
        t0 = time.time()
        if workers <= 1:
            merged = _run_single_mcts_tree(board, self.max_simulations, tl)
        else:
            # Windows 下子进程并行（spawn + pickle + 递归导入）在 GUI/交互环境里更容易“卡死”；
            # 这里优先使用线程池保证对局流程可终止，并给所有 future 加硬超时保护。
            Executor = (
                concurrent.futures.ThreadPoolExecutor
                if os.name == "nt"
                else concurrent.futures.ProcessPoolExecutor
            )
            try:
                with Executor(max_workers=workers) as pool:
                    sims = max(1, self.max_simulations // workers)
                    tasks = [
                        pool.submit(_run_single_mcts_tree, board.copy(), sims, tl, i * 100)
                        for i in range(workers)
                    ]
                    # 允许少量调度/回收开销，避免 future 永久阻塞
                    hard_timeout = max(0.1, tl + 1.5)
                    results = [t.result(timeout=hard_timeout) for t in tasks]
            except Exception:
                # 并行失败时立即降级为单线程，保证不会把对局“挂死”
                merged = _run_single_mcts_tree(board, self.max_simulations, tl)
            else:
                merged = {}
                for r in results:
                    for m, st in r.items():
                        if m not in merged: merged[m] = {"v": 0, "w": 0}
                        merged[m]["v"] += st["v"]; merged[m]["w"] += st["w"]

        # 3. 决策：综合 Visits 和 战术偏置
        v_max = max((s["v"] for s in merged.values()), default=0)
        best_move, max_score = None, -1.0
        mcts_cache = {}
        for m, st in merged.items():
            wr = st["w"] / st["v"] if st["v"] > 0 else 0
            bias = _get_tactical_bias(board, board.current_player, m, mcts_cache)
            # 核心评分公式：访问量决定主方向，战术偏置用于平局决胜
            score = st["v"] + bias * _ROOT_BIAS_SCALE * (st["v"] / (v_max + 1e-6))
            if score > max_score: max_score, best_move = score, m

        elapsed = time.time() - t0
        sims_done = int(sum(s.get("v", 0) for s in merged.values()))
        # 提供给 GUI/AI worker 的统计信息（ui/qt/main_window.py 会读取 simulations/time_taken/workers）
        self.last_stats = {
            "time_taken": float(elapsed),
            "time_limit": float(tl),
            "simulations": sims_done,
            "workers": int(workers),
        }
        if sims_done > 0:
            # 简单胜率估计：以访问最多的根走法的均值作为展示
            try:
                top_m, top_st = max(merged.items(), key=lambda kv: kv[1].get("v", 0))
                v = float(top_st.get("v", 0) or 0)
                w = float(top_st.get("w", 0) or 0)
                if v > 0:
                    self.last_stats["win_rate"] = f"{(w / v) * 100.0:.1f}%"
            except Exception:
                pass

        if self.verbose:
            print(
                f"MCTS完成: {sims_done} sims, time_limit={tl:.2f}s, 耗时 {elapsed:.2f}s"
            )
        return best_move

    def _probe_book(self, board: Board) -> Optional[Move4]:
        move = OPENING_BOOK.get(board.zobrist_hash)
        if not move:
            mirrored_board = board.column_mirror_copy()
            alt = OPENING_BOOK.get(mirrored_board.zobrist_hash)
            if alt: move = [mirror_move(m) for m in alt]
        if move: return random.choice([m for m in move if Rules.is_valid_move(board, *m)[0]])
        return None

    def choose_move(self, *args, **kwargs):
        return self.get_best_move(*args, **kwargs)