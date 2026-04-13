"""MCTS-Minimax 混合搜索引擎（Hybrid AI）。

本模块将 MCTS（全局探索 + RAVE + DAG 复用 + 多进程并行）与
Minimax（局部战术精算 + 静止搜索 + 启发排序）融合为一个独立可运行的
混合搜索器。

**核心架构**：

    MCTS 仍然是主干搜索框架，负责全局探索、UCB1-RAVE 选择、
    DAG 置换合并和根节点并行。Minimax 仅作为轻量级战术精算器，
    在 rollout 截断点按需启用——当局面存在战术紧张度（被将军、
    残局、大量吃子机会等）时，用 shallow negamax + alpha-beta +
    静止搜索替代纯静态评估，获得更精准的叶节点估值。

**关键设计决策**：

1. **走法为边属性**（DAG 安全）：
   与 ``mcts.py`` 一致，走法存储在 ``children: Dict[Move4, HybridNode]``
   的字典键中，Selection 阶段通过 ``(edge_move, child_node)`` 读取，
   防止 DAG 结构下的"幽灵棋"Bug。

2. **apply/undo 零拷贝树遍历**：
   Selection + Expansion 在真实棋盘上操作，仅 Simulation 阶段做一次
   ``board.copy()``。probe 在该副本上 apply/undo，零额外拷贝。

3. **选择性 probe 触发**：
   并非每次模拟都启动 minimax。仅当截断局面满足战术条件时才 probe，
   大部分中局模拟走"纯 MCTS + 静态评估"快速路径，保持吞吐量。

4. **RAVE 保留**：
   即使 probe 替代了部分 rollout 的终点评估，rollout 过程中收集的
   着法集合仍用于 RAVE/AMAF 反向传播，保持搜索早期的快速收敛。

5. **per-tree probe 状态**：
   probe TT / killer / history 在 ``_run_single_hybrid_tree()`` 内部创建，
   同一棵树的所有模拟共享这些表以积累经验；多进程 worker 之间
   天然隔离，无需加锁。
"""

from __future__ import annotations

import concurrent.futures
import math
import multiprocessing
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from chinese_chess.model.board import Board
from chinese_chess.model.rules import Rules

from .evaluation import Evaluation
from .opening_book import OPENING_BOOK, mirror_move

# ═══════════════════════════════════════════════════════════════
#  MCTS 常量（沿用 mcts.py）
# ═══════════════════════════════════════════════════════════════

_CAPTURE_PROB = 0.80    # 模拟阶段"吃子优先"概率
_UCB_C = 1.414          # UCB1 探索常数 c ≈ √2
_SCORE_SCALE = 600.0    # 评估分 → sigmoid 胜率的缩放因子
_RAVE_CONST = 300       # RAVE 等效常数 k：β 衰减速度

# ═══════════════════════════════════════════════════════════════
#  Probe 常量（精简自 minimax.py）
# ═══════════════════════════════════════════════════════════════

_PROBE_DEFAULT_DEPTH = 2     # 默认 probe 搜索深度
_PROBE_TACTICAL_DEPTH = 3    # 战术紧张局面 probe 深度
_PROBE_ENDGAME_DEPTH = 4     # 残局 probe 深度
_PROBE_QS_DEPTH_LIMIT = 4    # 静止搜索递归深度上限
_PROBE_DELTA_MARGIN = 1100   # QS Delta Pruning 阈值（≈ 一车价值 + 余量）
_PROBE_TT_MAX = 50_000       # probe TT 最大条目数
_PROBE_CAPTURE_SORT_BIAS = 10000  # 吃子排序基线偏移
_PROBE_KILLER_SLOTS = 8      # killer 表深度槽位数
_PROBE_MAX_CHECK_EXT = 1     # probe 中每条分支最多延伸 1 次

# TT 边界标志（与 minimax.py 语义一致）
_TT_EXACT = 0
_TT_LOWER = 1
_TT_UPPER = 2

# ═══════════════════════════════════════════════════════════════
#  Probe 触发条件常量
# ═══════════════════════════════════════════════════════════════

_ENDGAME_PIECE_THRESHOLD = 14  # 子力 ≤ 此值视为残局，强制 probe

# ═══════════════════════════════════════════════════════════════
#  类型别名
# ═══════════════════════════════════════════════════════════════

Move4 = Tuple[int, int, int, int]


# ═══════════════════════════════════════════════════════════════
#  动态截断步数（沿用 mcts.py）
# ═══════════════════════════════════════════════════════════════


def _dynamic_rollout_limit(piece_count: int) -> int:
    """根据场上子力数动态决定 rollout 截断步数。"""
    if piece_count > 24:
        return 20
    if piece_count >= 10:
        return 35
    return 50


# ═══════════════════════════════════════════════════════════════
#  HybridNode —— 搜索节点（与 MCTSNode __slots__ 完全相同）
# ═══════════════════════════════════════════════════════════════


class HybridNode:
    """MCTS-RAVE DAG 搜索节点。

    与 ``mcts.py`` 中的 ``MCTSNode`` 完全一致：走法存储在父节点
    ``children`` 字典的键中（边属性），确保 DAG 安全。
    """

    __slots__ = [
        "state_hash",
        "children",
        "visits",
        "wins",
        "untried_moves",
        "player_just_moved",
        "rave_visits",
        "rave_wins",
    ]

    def __init__(self, state_hash: int, player_just_moved: str):
        self.state_hash = state_hash
        self.children: Dict[Move4, HybridNode] = {}
        self.visits: int = 0
        self.wins: float = 0.0
        self.untried_moves: Optional[List[Move4]] = None
        self.player_just_moved = player_just_moved
        self.rave_visits: int = 0
        self.rave_wins: float = 0.0

    def ensure_moves(self, board: Board) -> None:
        """惰性初始化 untried_moves。"""
        if self.untried_moves is None:
            self.untried_moves = list(
                Rules.get_pseudo_legal_moves(board, board.current_player)
            )
            random.shuffle(self.untried_moves)

    def is_fully_expanded(self) -> bool:
        return self.untried_moves is not None and len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return (self.untried_moves is not None
                and len(self.untried_moves) == 0
                and len(self.children) == 0)

    def best_child_ucb(self, log_parent: float) -> Tuple[Move4, 'HybridNode']:
        """UCB1-RAVE 混合选择。"""
        best_move: Optional[Move4] = None
        best_node: Optional[HybridNode] = None
        best_score = -1.0
        for move, ch in self.children.items():
            rv = ch.rave_visits
            v = ch.visits
            if v + rv == 0:
                return move, ch
            mcts_val = ch.wins / v if v > 0 else 0.0
            rave_val = ch.rave_wins / rv if rv > 0 else 0.0
            beta = rv / (rv + v + _RAVE_CONST + 1e-5)
            exploit = (1.0 - beta) * mcts_val + beta * rave_val
            explore = _UCB_C * math.sqrt(log_parent / (v + 1e-5))
            s = exploit + explore
            if s > best_score:
                best_score = s
                best_move = move
                best_node = ch
        assert best_move is not None and best_node is not None
        return best_move, best_node


# ═══════════════════════════════════════════════════════════════
#  Probe TT 操作
# ═══════════════════════════════════════════════════════════════


def _probe_tt_lookup(
    tt: dict, key: int, depth: int, alpha: float, beta: float,
) -> Tuple[Optional[float], Optional[Move4]]:
    """查询 probe TT。返回 (score_or_None, best_move_or_None)。"""
    entry = tt.get(key)
    if entry is None:
        return None, None
    stored_depth, score, flag, best_move = entry
    if stored_depth < depth:
        return None, best_move
    if flag == _TT_EXACT:
        return score, best_move
    if flag == _TT_LOWER and score >= beta:
        return score, best_move
    if flag == _TT_UPPER and score <= alpha:
        return score, best_move
    return None, best_move


def _probe_tt_store(
    tt: dict, key: int, depth: int, score: float,
    alpha_orig: float, beta_orig: float,
    best_move: Optional[Move4],
) -> None:
    """写入 probe TT，自动推断边界标志。"""
    if len(tt) > _PROBE_TT_MAX:
        tt.clear()
    if score <= alpha_orig and alpha_orig != float("-inf"):
        flag = _TT_UPPER
    elif score >= beta_orig and beta_orig != float("inf"):
        flag = _TT_LOWER
    else:
        flag = _TT_EXACT
    tt[key] = (depth, score, flag, best_move)


# ═══════════════════════════════════════════════════════════════
#  Probe 走法排序
# ═══════════════════════════════════════════════════════════════


def _probe_order_moves(
    board: Board,
    moves: List[Move4],
    probe_state: dict,
    depth: int,
) -> None:
    """按 TT best_move > MVV-LVA > killer > history 就地降序排列。"""
    pv = Evaluation.PIECE_VALUES
    tt = probe_state["tt"]
    killers = probe_state["killers"]
    history = probe_state["history"]

    ki = max(0, min(depth, len(killers) - 1))
    k0, k1 = killers[ki][0], killers[ki][1]

    entry = tt.get(board.zobrist_hash)
    tt_best_move = entry[3] if entry is not None else None

    b = board.board

    def score(m: Move4) -> int:
        if tt_best_move is not None and m == tt_best_move:
            return 1_000_000
        hist = min(history.get(m, 0), 500_000)
        sr, sc, er, ec = m
        victim = b[er][ec]
        if victim is not None:
            attacker = b[sr][sc]
            vv = int(pv.get(victim.piece_type, 0))
            av = int(pv.get(attacker.piece_type, 0)) if attacker else 0
            cap = _PROBE_CAPTURE_SORT_BIAS + vv - av
        else:
            cap = 0
        s = hist + cap
        if m == k0:
            s += 5000
        elif m == k1:
            s += 4000
        return s

    moves.sort(key=score, reverse=True)


# ═══════════════════════════════════════════════════════════════
#  Probe 静止搜索（Quiescence Search）
# ═══════════════════════════════════════════════════════════════


def _probe_qs(
    board: Board,
    alpha: float,
    beta: float,
    qs_depth: int,
    probe_state: dict,
) -> float:
    """静止搜索：仅扩展吃子走法，缓解水平线效应。

    含 stand-pat 评估和 Delta Pruning，与 minimax.py 的 QS 逻辑一致。
    """
    if qs_depth <= 0:
        probe_state["nodes"] += 1
        return Evaluation.evaluate(board)

    probe_state["nodes"] += 1
    stand_pat = Evaluation.evaluate(board)
    if stand_pat >= beta:
        return beta
    if stand_pat + _PROBE_DELTA_MARGIN <= alpha:
        return alpha
    if stand_pat > alpha:
        alpha = stand_pat

    b = board.board
    moves = list(Rules.get_pseudo_legal_moves(board, board.current_player))
    captures = [m for m in moves if b[m[2]][m[3]] is not None]
    if not captures:
        return alpha

    # MVV-LVA 排序吃子走法
    pv = Evaluation.PIECE_VALUES

    def cap_score(m: Move4) -> int:
        victim = b[m[2]][m[3]]
        attacker = b[m[0]][m[1]]
        vv = int(pv.get(victim.piece_type, 0)) if victim else 0
        av = int(pv.get(attacker.piece_type, 0)) if attacker else 0
        return vv - av

    captures.sort(key=cap_score, reverse=True)

    for move in captures:
        mover = board.current_player
        captured = board.apply_move(*move)
        if Rules.is_king_in_check(board, mover) or Rules._jiang_face_to_face(board):
            board.undo_move(*move, captured)
            continue
        score = -_probe_qs(board, -beta, -alpha, qs_depth - 1, probe_state)
        board.undo_move(*move, captured)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


# ═══════════════════════════════════════════════════════════════
#  Probe Negamax + Alpha-Beta 核心
# ═══════════════════════════════════════════════════════════════


def _probe_negamax(
    board: Board,
    depth: int,
    alpha: float,
    beta: float,
    probe_state: dict,
    *,
    check_ext_left: int = _PROBE_MAX_CHECK_EXT,
) -> float:
    """轻量级 negamax + alpha-beta 搜索。

    含 TT 查询/写入、走法排序、killer/history 更新和将军延伸，
    不含 PVS / 空步剪枝 / 迭代加深 / 期望窗口。
    """
    alpha_orig = alpha
    key = board.zobrist_hash
    tt = probe_state["tt"]

    # TT 查询
    tt_score, _tt_bm = _probe_tt_lookup(tt, key, depth, alpha, beta)
    if tt_score is not None:
        return tt_score

    # 深度耗尽 → 静止搜索
    if depth <= 0:
        return _probe_qs(board, alpha, beta, _PROBE_QS_DEPTH_LIMIT, probe_state)

    moves = list(Rules.get_pseudo_legal_moves(board, board.current_player))
    _probe_order_moves(board, moves, probe_state, depth)

    best = float("-inf")
    best_move: Optional[Move4] = None
    has_legal = False

    for move in moves:
        mover = board.current_player
        captured = board.apply_move(*move)
        if Rules.is_king_in_check(board, mover) or Rules._jiang_face_to_face(board):
            board.undo_move(*move, captured)
            continue
        has_legal = True

        # 将军延伸：走后对手被将时不消耗深度
        child_player = board.current_player
        next_depth = depth - 1
        next_ext = check_ext_left
        if Rules.is_king_in_check(board, child_player) and check_ext_left > 0:
            next_depth = depth
            next_ext = check_ext_left - 1

        score = -_probe_negamax(
            board, next_depth, -beta, -alpha, probe_state,
            check_ext_left=next_ext,
        )
        board.undo_move(*move, captured)

        if score > best:
            best = score
            best_move = move
        if best > alpha:
            alpha = best
        if alpha >= beta:
            # Beta 截断：记录 killer + history
            is_cap = board.board[move[2]][move[3]] is not None
            if not is_cap:
                ki = max(0, min(depth, len(probe_state["killers"]) - 1))
                ks = probe_state["killers"][ki]
                if move != ks[0] and move != ks[1]:
                    ks[1] = ks[0]
                    ks[0] = move
            probe_state["history"][move] = (
                probe_state["history"].get(move, 0) + depth * depth
            )
            break

    if not has_legal:
        # 被将死：返回负向极大值 + 深度偏移（越早将死得分越高）
        return -10000.0 + (10 - depth)

    _probe_tt_store(tt, key, depth, best, alpha_orig, beta, best_move)
    return best


# ═══════════════════════════════════════════════════════════════
#  Probe 触发判定
# ═══════════════════════════════════════════════════════════════


def _should_use_probe(board: Board, piece_count: int) -> bool:
    """判定 rollout 截断点是否应使用 minimax probe。

    触发条件（任一满足即触发）：
    1. 残局（子力 ≤ 阈值）：静态评估在残局中误差更大，probe 价值高。
    2. 当前行棋方被将军：战术紧张，纯静态评估可能严重失真。
    3. 对方处于被将军状态：说明 rollout 末步是非法送将（伪合法 rollout
       不校验自将），此类局面需要更深评估。
    """
    if piece_count <= _ENDGAME_PIECE_THRESHOLD:
        return True
    cp = board.current_player
    if Rules.is_king_in_check(board, cp):
        return True
    opp = "black" if cp == "red" else "red"
    if Rules.is_king_in_check(board, opp):
        return True
    return False


def _probe_depth_for_position(piece_count: int, is_in_check: bool) -> int:
    """根据局面特征决定 probe 深度。"""
    if piece_count <= 10:
        return _PROBE_ENDGAME_DEPTH
    if is_in_check or piece_count <= _ENDGAME_PIECE_THRESHOLD:
        return _PROBE_TACTICAL_DEPTH
    return _PROBE_DEFAULT_DEPTH


# ═══════════════════════════════════════════════════════════════
#  MCTS 辅助函数（模块级，pickle 安全）
# ═══════════════════════════════════════════════════════════════


def _order_root_moves(board: Board, moves: List[Move4]) -> None:
    """根节点 untried_moves 启发排序：MVV-LVA 高分排末尾（pop 优先展开）。"""
    if not moves:
        return
    b = board.board
    pv = Evaluation.PIECE_VALUES

    def score(m: Move4) -> int:
        sr, sc, er, ec = m
        victim = b[er][ec]
        if victim is not None:
            attacker = b[sr][sc]
            vv = int(pv.get(victim.piece_type, 0))
            av = int(pv.get(attacker.piece_type, 0)) if attacker else 0
            return 10000 + vv - av
        return 0

    moves.sort(key=score)  # 升序：最差在前，最佳在尾（pop 弹出）


def _find_king_capture(
    board: Board, attacker: str, b_grid, tkr: int, tkc: int,
) -> Optional[Move4]:
    """单目标可达性检测：是否有一步可直接吃到对方老将。

    仅做几何 + 蹩腿判定，模拟阶段专用。
    """
    for r, c in board.active_pieces[attacker]:
        p = b_grid[r][c]
        if p is None:
            continue
        pt = p.piece_type

        if pt == "che":
            if r == tkr and c != tkc:
                lo, hi = (c + 1, tkc) if c < tkc else (tkc + 1, c)
                if not any(b_grid[r][x] is not None for x in range(lo, hi)):
                    return (r, c, tkr, tkc)
            elif c == tkc and r != tkr:
                lo, hi = (r + 1, tkr) if r < tkr else (tkr + 1, r)
                if not any(b_grid[x][c] is not None for x in range(lo, hi)):
                    return (r, c, tkr, tkc)

        elif pt == "pao":
            if r == tkr and c != tkc:
                lo, hi = (c + 1, tkc) if c < tkc else (tkc + 1, c)
                if sum(1 for x in range(lo, hi) if b_grid[r][x] is not None) == 1:
                    return (r, c, tkr, tkc)
            elif c == tkc and r != tkr:
                lo, hi = (r + 1, tkr) if r < tkr else (tkr + 1, r)
                if sum(1 for x in range(lo, hi) if b_grid[x][c] is not None) == 1:
                    return (r, c, tkr, tkc)

        elif pt == "ma":
            dr, dc = tkr - r, tkc - c
            if (abs(dr), abs(dc)) in ((1, 2), (2, 1)):
                lr, lc = Rules._ma_leg_square(r, c, tkr, tkc)
                if 0 <= lr < 10 and 0 <= lc < 9 and b_grid[lr][lc] is None:
                    return (r, c, tkr, tkc)

        elif pt == "bing":
            if attacker == "red":
                if (tkr == r - 1 and tkc == c) or (r <= 4 and tkr == r and abs(tkc - c) == 1):
                    return (r, c, tkr, tkc)
            else:
                if (tkr == r + 1 and tkc == c) or (r >= 5 and tkr == r and abs(tkc - c) == 1):
                    return (r, c, tkr, tkc)

    return None


def _pick_rollout_move_fast(
    sim_board: Board, moves: List[Move4], b_grid,
) -> Move4:
    """零校验启发式走子：80% 概率吃子优先。"""
    captures = [m for m in moves if b_grid[m[2]][m[3]] is not None]
    if captures and random.random() < _CAPTURE_PROB:
        m = random.choice(captures)
    else:
        m = random.choice(moves)
    sim_board.apply_move(*m)
    return m


def _eval_to_winrate(board: Board, root_player: str) -> float:
    """静态评估 → sigmoid 胜率（快速路径，无 probe）。"""
    raw = Evaluation.evaluate(board)
    if board.current_player != root_player:
        raw = -raw
    return 1.0 / (1.0 + math.exp(-raw / _SCORE_SCALE))


def _terminal_score(board: Board, root_player: str) -> float:
    """终局局面评估：胜 1.0 / 负 0.0 / 和 0.5。"""
    w = Rules.winner(board)
    if w == root_player:
        return 1.0
    if w is not None:
        return 0.0
    return 0.5


def _backpropagate(
    path: List[HybridNode],
    result: float,
    red_moves: Set[Move4],
    black_moves: Set[Move4],
) -> None:
    """反向传播 + RAVE/AMAF 更新。"""
    score = result
    for i in range(len(path) - 1, -1, -1):
        node = path[i]
        node.visits += 1
        if node.player_just_moved is not None:
            node.wins += score
        for move, child in node.children.items():
            pjm = child.player_just_moved
            if pjm == "red" and move in red_moves:
                child.rave_visits += 1
                child.rave_wins += score
            elif pjm == "black" and move in black_moves:
                child.rave_visits += 1
                child.rave_wins += score
        score = 1.0 - score


# ═══════════════════════════════════════════════════════════════
#  核心搜索函数
# ═══════════════════════════════════════════════════════════════


def _simulate_or_probe(
    board: Board,
    root_player: str,
    probe_state: dict,
) -> Tuple[float, Set[Move4], Set[Move4]]:
    """轻量级伪合法 rollout + 选择性 minimax probe。

    与 ``mcts.py`` 的 ``_simulate()`` 流程一致，区别在于 rollout 截断后：
    若局面满足战术条件，用 ``_probe_negamax()`` 替代纯静态评估。
    """
    sim_board = board.copy()
    b_grid = sim_board.board
    rollout_limit = _dynamic_rollout_limit(sim_board.piece_count())
    red_moves: Set[Move4] = set()
    black_moves: Set[Move4] = set()

    for _ in range(rollout_limit):
        cp = sim_board.current_player
        opp = "black" if cp == "red" else "red"

        # O(1) 终局检测：对方老将被将军 → 当前方可"吃将"获胜
        if Rules.is_king_in_check(sim_board, opp):
            return (1.0 if cp == root_player else 0.0), red_moves, black_moves

        moves = list(Rules.get_pseudo_legal_moves(sim_board, cp))
        if not moves:
            return (0.0 if cp == root_player else 1.0), red_moves, black_moves

        # 一击必杀检测
        opp_king = sim_board.black_king_pos if cp == "red" else sim_board.red_king_pos
        if opp_king is not None:
            okr, okc = opp_king
            king_cap = _find_king_capture(sim_board, cp, b_grid, okr, okc)
            if king_cap is not None:
                sim_board.apply_move(*king_cap)
                if cp == "red":
                    red_moves.add(king_cap)
                else:
                    black_moves.add(king_cap)
                return (1.0 if cp == root_player else 0.0), red_moves, black_moves

        chosen = _pick_rollout_move_fast(sim_board, moves, b_grid)
        if cp == "red":
            red_moves.add(chosen)
        else:
            black_moves.add(chosen)

    # ── Rollout 截断：选择性 probe 或快速静态评估 ──
    current_pc = sim_board.piece_count()
    if _should_use_probe(sim_board, current_pc):
        is_in_check = Rules.is_king_in_check(sim_board, sim_board.current_player)
        depth = _probe_depth_for_position(current_pc, is_in_check)
        probe_state["probes"] += 1
        raw = _probe_negamax(
            sim_board, depth, float("-inf"), float("inf"), probe_state,
        )
        if sim_board.current_player != root_player:
            raw = -raw
        result = 1.0 / (1.0 + math.exp(-raw / _SCORE_SCALE))
    else:
        result = _eval_to_winrate(sim_board, root_player)

    return result, red_moves, black_moves


def _expand_one(
    board: Board,
    node: HybridNode,
    move_stack: List[Tuple[Move4, Any]],
    tt: Dict[int, HybridNode],
) -> Optional[Tuple[Move4, HybridNode]]:
    """从 untried_moves 弹出一个合法走法并建立父子边（含 DAG 合并）。"""
    mover = board.current_player
    while node.untried_moves:
        move = node.untried_moves.pop()
        captured = board.apply_move(*move)
        if Rules.is_king_in_check(board, mover) or Rules._jiang_face_to_face(board):
            board.undo_move(*move, captured)
            continue
        move_stack.append((move, captured))
        child_hash = board.zobrist_hash
        existing = tt.get(child_hash)
        if existing is not None:
            node.children[move] = existing
            return move, existing
        child = HybridNode(state_hash=child_hash, player_just_moved=mover)
        tt[child_hash] = child
        node.children[move] = child
        return move, child
    return None


def _merge_child_stats(
    results: list,
) -> Tuple[Dict[Move4, Dict[str, float]], Dict[str, int]]:
    """合并多棵树的根子节点统计 + probe 统计。"""
    merged: Dict[Move4, Dict[str, float]] = {}
    total_probes = 0
    total_probe_nodes = 0
    for child_stats, p_stats in results:
        total_probes += p_stats.get("probes", 0)
        total_probe_nodes += p_stats.get("probe_nodes", 0)
        for mv, st in child_stats.items():
            if mv in merged:
                merged[mv]["visits"] += st["visits"]
                merged[mv]["wins"] += st["wins"]
                merged[mv]["rave_visits"] += st["rave_visits"]
                merged[mv]["rave_wins"] += st["rave_wins"]
            else:
                merged[mv] = {
                    "visits": st["visits"],
                    "wins": st["wins"],
                    "rave_visits": st["rave_visits"],
                    "rave_wins": st["rave_wins"],
                }
    return merged, {"probes": total_probes, "probe_nodes": total_probe_nodes}


# ═══════════════════════════════════════════════════════════════
#  单棵混合树搜索（worker 主函数，模块级以支持 pickle）
# ═══════════════════════════════════════════════════════════════


def _run_single_hybrid_tree(
    board: Board,
    max_simulations: int,
    time_limit: float,
    seed_offset: int = 0,
) -> Tuple[Dict[Move4, Dict[str, float]], Dict[str, int]]:
    """在独立进程中执行一棵完整的 MCTS-Minimax 混合搜索树。

    四阶段流程与 ``mcts.py`` 一致：Selection → Expansion → Simulation/Probe
    → Backpropagation。区别在于 Simulation 阶段会选择性地启用 minimax probe。

    Returns:
        ``(child_stats, probe_stats)`` 二元组。
    """
    random.seed(time.time_ns() + seed_offset)
    t0 = time.time()

    # 初始化 per-tree probe 状态（所有模拟共享，多进程天然隔离）
    probe_state: dict = {
        "tt": {},
        "killers": [[None, None] for _ in range(_PROBE_KILLER_SLOTS)],
        "history": {},
        "nodes": 0,
        "probes": 0,
    }

    root_player = board.current_player
    opp_of_root = "black" if root_player == "red" else "red"
    root = HybridNode(state_hash=board.zobrist_hash, player_just_moved=opp_of_root)
    root.ensure_moves(board)

    # 根着法启发排序：吃子优先展开（最佳排末尾供 pop 弹出）
    _order_root_moves(board, root.untried_moves)

    tt: Dict[int, HybridNode] = {root.state_hash: root}
    move_stack: List[Tuple[Move4, Any]] = []
    sims_done = 0

    while sims_done < max_simulations:
        if time.time() - t0 >= time_limit:
            break

        node = root
        path: List[HybridNode] = [root]

        # ── Selection ──
        while node.is_fully_expanded() and node.children:
            log_n = math.log(node.visits) if node.visits > 0 else 0.0
            edge_move, next_node = node.best_child_ucb(log_n)
            captured = board.apply_move(*edge_move)
            move_stack.append((edge_move, captured))
            path.append(next_node)
            node = next_node

        # ── Expansion ──
        node.ensure_moves(board)
        expanded = False
        if node.untried_moves:
            result = _expand_one(board, node, move_stack, tt)
            if result is not None:
                _exp_move, child = result
                path.append(child)
                node = child
                expanded = True

        if not expanded and not node.children and node.visits > 0:
            sim_result = _terminal_score(board, root_player)
            red_moves: Set[Move4] = set()
            black_moves: Set[Move4] = set()
        else:
            # ── Simulation / Probe ──
            sim_result, red_moves, black_moves = _simulate_or_probe(
                board, root_player, probe_state,
            )

        # ── Backpropagation ──
        _backpropagate(path, sim_result, red_moves, black_moves)
        sims_done += 1

        # 还原棋盘
        while move_stack:
            mv, cap = move_stack.pop()
            board.undo_move(*mv, cap)

    # 收集根子节点统计
    child_stats: Dict[Move4, Dict[str, float]] = {}
    for move, ch in root.children.items():
        child_stats[move] = {
            "visits": ch.visits,
            "wins": ch.wins,
            "rave_visits": ch.rave_visits,
            "rave_wins": ch.rave_wins,
        }
    probe_stats = {
        "probes": probe_state["probes"],
        "probe_nodes": probe_state["nodes"],
    }
    return child_stats, probe_stats


# ═══════════════════════════════════════════════════════════════
#  MCTSMinimaxAI —— 公开接口
# ═══════════════════════════════════════════════════════════════


class MCTSMinimaxAI:
    """MCTS-Minimax 混合搜索 AI（对外接口与 MinimaxAI / MCTSAI 一致）。

    MCTS 负责全局探索与 RAVE 收敛，Minimax 负责叶节点战术精算。
    支持多进程根节点并行 + 开局库查询 + DAG 置换合并。

    Args:
        max_simulations: 所有 worker 合计的最大模拟次数。
        time_limit: 搜索时间上限（秒）。
        workers: 并行进程数（0/1 = 单进程）。
        probe_depth: minimax probe 默认深度。
        verbose: 搜索完成后是否输出统计信息。
    """

    def __init__(
        self,
        max_simulations: int = 5000,
        time_limit: float = 10.0,
        workers: Optional[int] = None,
        probe_depth: int = _PROBE_DEFAULT_DEPTH,
        verbose: bool = True,
    ):
        self.max_simulations = max_simulations
        self.time_limit = time_limit
        self.verbose = verbose
        self.probe_depth = probe_depth
        if workers is None:
            try:
                self.workers = min(4, multiprocessing.cpu_count())
            except NotImplementedError:
                self.workers = 1
        else:
            self.workers = max(1, workers)
        self.simulations_run: int = 0
        self.last_stats: Dict[str, Any] = {}

    def choose_move(
        self,
        board: Board,
        time_limit: Optional[float] = None,
        game_history: Optional[List[int]] = None,
    ) -> Optional[Move4]:
        """Searcher 统一接口。"""
        return self.get_best_move(board, time_limit=time_limit, game_history=game_history)

    def get_best_move(
        self,
        board: Board,
        game_history: Optional[List[int]] = None,
        time_limit: Optional[float] = None,
        max_simulations: Optional[int] = None,
    ) -> Optional[Move4]:
        """执行 MCTS-Minimax 混合搜索并返回最佳走法。"""
        Evaluation._eval_cache.clear()
        move_history: List[int] = [] if game_history is None else list(game_history)

        # ── 开局库查询 ──
        if len(move_history) < 30:
            book_move = self._probe_opening_book(board, move_history)
            if book_move is not None:
                self.simulations_run = 0
                self.last_stats = {
                    "time_taken": 0.0,
                    "simulations": 0,
                    "workers": self.workers,
                    "win_rate": "Book Move",
                    "opening_book": True,
                }
                if self.verbose:
                    print(f"[Hybrid] 命中开局库: {book_move}")
                return book_move

        # ── MCTS-Minimax 混合搜索 ──
        tl = time_limit if time_limit is not None else self.time_limit
        ms = max_simulations if max_simulations is not None else self.max_simulations
        t0 = time.time()

        effective_workers = self.workers
        if effective_workers <= 1 or ms < effective_workers * 10:
            child_stats, probe_stats = _run_single_hybrid_tree(
                board, ms, tl, seed_offset=0,
            )
            total_sims = sum(int(s["visits"]) for s in child_stats.values())
        else:
            child_stats, probe_stats = self._parallel_search(
                board, tl, ms, effective_workers,
            )
            total_sims = sum(int(s["visits"]) for s in child_stats.values())

        elapsed = time.time() - t0
        self.simulations_run = total_sims

        # 选择访问次数最多的走法
        best_move: Optional[Move4] = None
        best_visits = -1
        best_wr = 0.0
        for mv, st in child_stats.items():
            v = int(st["visits"])
            if v > best_visits:
                best_visits = v
                best_move = mv
                best_wr = st["wins"] / v if v > 0 else 0.0

        self.last_stats = {
            "time_taken": elapsed,
            "simulations": total_sims,
            "workers": effective_workers,
            "win_rate": f"{best_wr * 100:.1f}%",
            "probe_count": probe_stats.get("probes", 0),
            "probe_nodes": probe_stats.get("probe_nodes", 0),
        }
        if self.verbose:
            print(
                f"[Hybrid] 搜索完成: {total_sims} sims, "
                f"{effective_workers} workers, {elapsed:.3f}s"
            )
            print(
                f"[Hybrid] Probe 触发: {probe_stats.get('probes', 0)} 次, "
                f"Probe 节点: {probe_stats.get('probe_nodes', 0)}"
            )
            if best_move is not None:
                print(f"[Hybrid] 最佳走法: {best_move}  胜率: {best_wr:.1%}")
        return best_move

    @staticmethod
    def _probe_opening_book(
        board: Board, move_history: List[int],
    ) -> Optional[Move4]:
        """查询开局库，含列镜像回退机制（与 MCTSAI 一致）。"""
        zkey = board.zobrist_hash
        book_moves = OPENING_BOOK.get(zkey)
        if book_moves is None:
            zm = board.column_mirror_copy().zobrist_hash
            alt = OPENING_BOOK.get(zm)
            if alt is not None:
                book_moves = [mirror_move(m) for m in alt]
        if book_moves is None and len(move_history) == 0:
            keys = list(OPENING_BOOK.keys())
            disp = keys if len(keys) <= 48 else keys[:24] + ["..."] + keys[-16:]
            print(
                f"[Hybrid 开局库] 根局面未命中（zkey={zkey:#x}）；"
                f"OPENING_BOOK 共 {len(keys)} 个键: {disp}"
            )
        if book_moves:
            valid = [
                m for m in book_moves
                if Rules.is_valid_move(board, m[0], m[1], m[2], m[3])[0]
            ]
            if valid:
                return random.choice(valid)
        return None

    def _parallel_search(
        self,
        board: Board,
        time_limit: float,
        total_sims: int,
        num_workers: int,
    ) -> Tuple[Dict[Move4, Dict[str, float]], Dict[str, int]]:
        """多进程根节点并行搜索 + 合并。"""
        sims_per_worker = total_sims // num_workers
        remainder = total_sims % num_workers
        futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
            for i in range(num_workers):
                worker_sims = sims_per_worker + (1 if i < remainder else 0)
                board_copy = board.copy()
                futures.append(
                    pool.submit(
                        _run_single_hybrid_tree,
                        board_copy,
                        worker_sims,
                        time_limit,
                        seed_offset=i * 1000,
                    )
                )
            results = [f.result() for f in futures]
        return _merge_child_stats(results)
