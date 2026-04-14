""" MCTS（Monte Carlo Tree Search）引擎。

"""

from __future__ import annotations
import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

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


def _deadline_reached(t0: float, tl: float) -> bool:
    return time.perf_counter() - t0 >= tl


Move4 = Tuple[int, int, int, int]

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
            if v + rv == 0:
                return move, ch
            beta = rv / (rv + v + _RAVE_CONST + 1e-5)
            exploit = (1.0 - beta) * (ch.wins / v if v > 0 else 0) + beta * (ch.rave_wins / rv if rv > 0 else 0)
            if log_parent > 0:
                exploration = _UCB_C * math.sqrt(log_parent / (v + 1e-5))
            else:
                exploration = 0.0
            score = exploit + exploration
            if math.isnan(score):
                continue
            if score > best_score:
                best_score, res = score, (move, ch)
        if res[0] is None and self.children:
            return next(iter(self.children.items()))
        return res

# ═══════════════════════════════════════════════════════════════
#  搜索流程
# ═══════════════════════════════════════════════════════════════

def _simulate(board: Board, root_player: str, t0: float, tl: float) -> float:
    """极致精简的 Rollout，去除了复杂的逃避逻辑以换取 SPS。"""
    sim_board = board.copy()
    limit = 30 if sim_board.piece_count() > 20 else 50
    for _ in range(limit):
        if _deadline_reached(t0, tl):
            return 0.5
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
    t0, root_board = time.perf_counter(), board
    root = MCTSNode(board.zobrist_hash, "black" if board.current_player == "red" else "red")
    # 不在扩展阶段复用「局面哈希 → 节点」：否则子边可能指向根/祖先，形成 DAG 回边，选择阶段会死循环。
    mcts_cache: Dict = {}
    
    max_sims = min(int(max_sims), 1_000_000)
    for _ in range(max_sims):
        if _deadline_reached(t0, tl):
            break

        # 1. Selection & 2. Expansion
        sim_board, node, path = root_board.copy(), root, [root]
        path_node_ids = {id(root)}
        selection_depth = 0
        while (
            node.untried_moves is not None
            and len(node.untried_moves) == 0
            and node.children
            and selection_depth < _SELECTION_MAX_PLIES
        ):
            if _deadline_reached(t0, tl):
                break
            logp = math.log(max(1, node.visits))
            move, next_node = node.best_child_ucb(logp)
            if move is None or next_node is None:
                break
            if id(next_node) in path_node_ids:
                break
            sim_board.apply_move(*move)
            path_node_ids.add(id(next_node))
            node = next_node
            path.append(node)
            selection_depth += 1
            
        if node.untried_moves is None:
            node.untried_moves = list(Rules.get_pseudo_legal_moves(sim_board, sim_board.current_player))
            random.shuffle(node.untried_moves)

        if node.untried_moves:
            move = node.untried_moves.pop()
            sim_board.apply_move(*move)
            h = sim_board.zobrist_hash
            bias = _get_tactical_bias(sim_board, sim_board.current_player, move, mcts_cache)
            child = MCTSNode(h, sim_board.current_player, 0.1 + bias * _FPU_PRIOR_SCALE)
            node.children[move] = child
            path.append(child)

        # 3. Simulation & 4. Backprop
        res = _simulate(sim_board, root_board.current_player, t0, tl)
        for i in range(len(path)-1, -1, -1):
            path[i].visits += 1
            if path[i].player_just_moved != root_board.current_player: path[i].wins += res
            res = 1.0 - res
            
    return {m: {"v": c.visits, "w": c.wins} for m, c in root.children.items()}

# ═══════════════════════════════════════════════════════════════
#  AI 接口
# ═══════════════════════════════════════════════════════════════

class MCTSAI:
    def __init__(self, max_simulations: int = 5000, time_limit: float = 10.0, verbose: bool = True):
        self.max_simulations = max_simulations
        self.time_limit = time_limit
        self.verbose = verbose
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
                }
                return res

        # 2. 搜索
        tl = self.time_limit if time_limit is None else float(time_limit)
        if not math.isfinite(tl) or tl <= 0:
            tl = float(self.time_limit)
        t0 = time.perf_counter()
        merged = _run_single_mcts_tree(board, self.max_simulations, tl)

        # 3. 决策：综合 Visits 和 战术偏置
        if not merged:
            legal = list(Rules.get_legal_moves(board, board.current_player))
            self.last_stats = {
                "time_taken": float(time.perf_counter() - t0),
                "time_limit": float(tl),
                "simulations": 0,
            }
            return random.choice(legal) if legal else None

        v_max = max((s["v"] for s in merged.values()), default=0)
        best_move, max_score = None, -1.0
        mcts_cache = {}
        for m, st in merged.items():
            wr = st["w"] / st["v"] if st["v"] > 0 else 0
            bias = _get_tactical_bias(board, board.current_player, m, mcts_cache)
            # 核心评分公式：访问量决定主方向，战术偏置用于平局决胜
            score = st["v"] + bias * _ROOT_BIAS_SCALE * (st["v"] / (v_max + 1e-6))
            if score > max_score: max_score, best_move = score, m

        elapsed = time.perf_counter() - t0
        sims_done = int(sum(s.get("v", 0) for s in merged.values()))
        # 提供给 GUI/AI worker 的统计信息（ui/qt/main_window.py 会读取 simulations/time_taken）
        self.last_stats = {
            "time_taken": float(elapsed),
            "time_limit": float(tl),
            "simulations": sims_done,
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
        if move:
            legal = [m for m in move if Rules.is_valid_move(board, *m)[0]]
            return random.choice(legal) if legal else None
        return None

    def choose_move(self, *args, **kwargs):
        return self.get_best_move(*args, **kwargs)