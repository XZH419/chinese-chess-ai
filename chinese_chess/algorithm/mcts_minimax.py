"""MCTS-Minimax 组合搜索引擎。

本模块将 MCTS（全局探索 + RAVE + DAG 复用 + 多进程并行）与
Minimax（局部战术精算 + 静止搜索 + 启发排序）融合为一个独立可运行的
MCTS-Minimax 搜索引擎。

**核心架构**：

    MCTS 仍然是主干搜索框架，负责全局探索、UCB1-RAVE 选择、
    DAG 置换合并和根节点并行。Minimax 仅作为轻量级战术精算器，
    在 rollout 截断点按需启用——当局面存在战术紧张度（被将军、
    残局、大量吃子机会等）时，用 shallow negamax + alpha-beta +
    静止搜索替代纯静态评估，获得更精准的叶节点估值。

**关键设计决策**：

1. **走法为边属性**（DAG 安全）：
   与 ``mcts.py`` 一致，走法存储在 ``children: Dict[Move4, MCTSMinimaxNode]``
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
   probe TT / killer / history 在 ``_run_single_mcts_minimax_tree()`` 内部创建，
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
from chinese_chess.model.rules import MoveEntry, Rules

from .evaluation import Evaluation
from .search_move_helpers import (
    MoveGivesCheckCache,
    PostApplyFlagsCache,
    apply_pseudo_legal_with_rule_cache,
)
from .opening_book import OPENING_BOOK, mirror_move
from .mcts import (
    _append_path_move_entry,
    _policy_attack_bias,
    _order_untried_moves_policy,
    _pick_rollout_move_fast,
    _move_gives_check,
    _is_aggressive_push,
    _POLICY_HVCAP_VALUE,
    _ROOT_BIAS_SCALE,
    _ROOT_VISITS_TIE_FRAC,
)

# ═══════════════════════════════════════════════════════════════
#  MCTS 常量（沿用 mcts.py）
# ═══════════════════════════════════════════════════════════════

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
#  Probe 多因子触发 · 评分阈值 · 概率 · 预算 · 冷却
# ═══════════════════════════════════════════════════════════════

# ── 残局阶段子力阈值 ──
_ENDGAME_PIECE_THRESHOLD = 14  # 子力 ≤ 此值视为残局阶段

# ── 评分因子权重 ──
# 各因子叠加后与阈值比较，决定 probe level（0/1/2）
_SCORE_DEEP_ENDGAME = 4    # piece_count <= 10（恰好达到 THRESHOLD_LIGHT，
                           # 使非安静深残局以 40% 概率获得 Level 1 轻量 probe）
_SCORE_EARLY_ENDGAME = 2   # 10 < piece_count <= 14
_SCORE_LATE_MIDGAME = 1    # 14 < piece_count <= 20
_SCORE_IN_CHECK = 5         # 当前行棋方被将军（最强战术信号）
_SCORE_OPP_IN_CHECK = 4     # 对方被将军（rollout 末步给将或伪合法送将）
_SCORE_LAST_CAPTURE = 2     # rollout 末步是吃子（战术转换点）
_SCORE_HIGH_VISITS = 2      # 展开节点访问 ≥ 此阈值 → 重要分支
_HIGH_VISIT_THRESHOLD = 8   # 用于上条因子的节点访问数门槛

# ── 安静局面惩罚 ──
_QUIET_SCORE_PENALTY = 3    # 安静残局扣分（无将军 + 无攻击线）

# ── 预算/冷却惩罚 ──
_BUDGET_EXHAUSTED_PENALTY = 6  # 预算耗尽时的评分惩罚（几乎必然降到 Level 0）
_COOLDOWN_PENALTY = 3          # 冷却期内的评分惩罚

# ── 分级阈值 ──
_PROBE_SCORE_THRESHOLD_LIGHT = 4   # 评分 ≥ 此值 → 候选 Level 1 (轻量 probe)
_PROBE_SCORE_THRESHOLD_FULL = 7    # 评分 ≥ 此值 → 候选 Level 2 (标准 probe)

# ── 概率触发 ──
# 达到阈值后仍通过概率门控，避免锁死在 100% 触发
_PROBE_PROB_LIGHT = 0.40   # Level 1 触发概率
_PROBE_PROB_FULL = 0.80    # Level 2 触发概率

# ── 冷却 ──
_PROBE_COOLDOWN_SIMS = 5   # 两次 probe 之间最少间隔模拟数

# ── Policy：probe 触发 / 排序 轻量进攻偏置（不加到 Evaluation 分值上）──
_SCORE_POLICY_PRESSURE = 2   # 存在将军着法或高价值吃子机会时加分
_PROBE_ATTACK_BONUS_CHECK = 130
_PROBE_ATTACK_BONUS_AGGRESSIVE = 48

# ── per-tree 预算 ──
_MAX_PROBE_RATIO = 0.25           # probe 次数上限 = max_sims × ratio
_MAX_PROBE_NODES_PER_TREE = 25_000  # 每棵树 probe 节点数上限

# ── 分级 probe 深度配置 ──
_LIGHT_PROBE_DEPTH = 1       # Level 1 轻量 probe 主搜索深度
_LIGHT_PROBE_QS_DEPTH = 2    # Level 1 轻量 probe 静止搜索深度

# ═══════════════════════════════════════════════════════════════
#  类型别名
# ═══════════════════════════════════════════════════════════════

Move4 = Tuple[int, int, int, int]


# ═══════════════════════════════════════════════════════════════
#  ProbeBudget —— per-tree probe 预算 / 冷却追踪器
# ═══════════════════════════════════════════════════════════════


class ProbeBudget:
    """单棵搜索树的 probe 预算与冷却状态。

    在 ``_run_single_mcts_minimax_tree`` 开始时创建，伴随整棵树的生命周期。
    多进程 worker 各自持有独立实例，无需加锁。

    Attributes:
        probe_calls: 本棵树已消耗的 probe 调用次数。
        probe_nodes: 本棵树所有 probe 累计访问的搜索节点数。
        max_probe_calls: 允许的最大 probe 调用次数。
        max_probe_nodes: 允许的最大 probe 累计节点数。
        last_probe_sim: 上次触发 probe 时的模拟序号（用于冷却计算）。
    """

    __slots__ = (
        "probe_calls",
        "probe_nodes",
        "max_probe_calls",
        "max_probe_nodes",
        "last_probe_sim",
    )

    def __init__(self, max_simulations: int):
        self.probe_calls: int = 0
        self.probe_nodes: int = 0
        self.max_probe_calls: int = max(10, int(max_simulations * _MAX_PROBE_RATIO))
        self.max_probe_nodes: int = _MAX_PROBE_NODES_PER_TREE
        # 初始设为负值，使首次 probe 不受冷却限制
        self.last_probe_sim: int = -_PROBE_COOLDOWN_SIMS

    @property
    def is_calls_exhausted(self) -> bool:
        return self.probe_calls >= self.max_probe_calls

    @property
    def is_nodes_exhausted(self) -> bool:
        return self.probe_nodes >= self.max_probe_nodes

    @property
    def is_any_exhausted(self) -> bool:
        return self.is_calls_exhausted or self.is_nodes_exhausted

    def is_cooldown_active(self, current_sim: int) -> bool:
        """当前模拟是否处于冷却期（距上次 probe 间隔不足）。"""
        return (current_sim - self.last_probe_sim) < _PROBE_COOLDOWN_SIMS

    def record_probe(self, current_sim: int, nodes_used: int) -> None:
        """记录一次 probe 消耗。"""
        self.probe_calls += 1
        self.probe_nodes += nodes_used
        self.last_probe_sim = current_sim


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
#  MCTSMinimaxNode —— 搜索节点（与 MCTSNode __slots__ 完全相同）
# ═══════════════════════════════════════════════════════════════


class MCTSMinimaxNode:
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
        self.children: Dict[Move4, MCTSMinimaxNode] = {}
        self.visits: int = 0
        self.wins: float = 0.0
        self.untried_moves: Optional[List[Move4]] = None
        self.player_just_moved = player_just_moved
        self.rave_visits: int = 0
        self.rave_wins: float = 0.0

    def ensure_moves(
        self,
        board: Board,
        gives_check_cache: Optional[MoveGivesCheckCache] = None,
    ) -> None:
        """惰性初始化 untried_moves。"""
        if self.untried_moves is None:
            self.untried_moves = list(
                Rules.get_pseudo_legal_moves(board, board.current_player)
            )
            _order_untried_moves_policy(board, self.untried_moves, gives_check_cache)

    def is_fully_expanded(self) -> bool:
        return self.untried_moves is not None and len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return (self.untried_moves is not None
                and len(self.untried_moves) == 0
                and len(self.children) == 0)

    def best_child_ucb(self, log_parent: float) -> Tuple[Move4, 'MCTSMinimaxNode']:
        """UCB1-RAVE 选择。"""
        best_move: Optional[Move4] = None
        best_node: Optional[MCTSMinimaxNode] = None
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

    mover = board.current_player

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
        if tt_best_move is None or m != tt_best_move:
            if _move_gives_check(board, m, mover):
                s += _PROBE_ATTACK_BONUS_CHECK
            elif victim is None and _is_aggressive_push(board, mover, m, b):
                s += _PROBE_ATTACK_BONUS_AGGRESSIVE
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
    当 per-tree 节点预算耗尽时，立即回退到静态评估（安全阀）。
    """
    # 节点预算安全阀：避免单次高开销 probe 拖垮整棵树的吞吐量
    budget: Optional[ProbeBudget] = probe_state.get("budget")
    if budget is not None and probe_state["nodes"] >= budget.max_probe_nodes:
        probe_state["nodes"] += 1
        return Evaluation.evaluate(board)

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
        applied = apply_pseudo_legal_with_rule_cache(
            board,
            move,
            mover,
            pre_move_cache=probe_state["pre_move_cache"],
            post_apply_cache=probe_state["post_apply_cache"],
        )
        if applied is None:
            continue
        captured, _ = applied
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
    qs_depth_limit: int = _PROBE_QS_DEPTH_LIMIT,
) -> float:
    """轻量级 negamax + alpha-beta 搜索。

    含 TT 查询/写入、走法排序、killer/history 更新和将军延伸，
    不含 PVS / 空步剪枝 / 迭代加深 / 期望窗口。

    ``qs_depth_limit`` 控制叶节点 QS 递归深度——Level 1 轻量 probe
    使用更浅的 QS（``_LIGHT_PROBE_QS_DEPTH``）以降低单次 probe 成本。
    """
    # 节点预算安全阀：全树累计节点超限时回退到静态评估
    budget: Optional[ProbeBudget] = probe_state.get("budget")
    if budget is not None and probe_state["nodes"] >= budget.max_probe_nodes:
        probe_state["nodes"] += 1
        return Evaluation.evaluate(board)

    alpha_orig = alpha
    key = board.zobrist_hash
    tt = probe_state["tt"]

    # TT 查询
    tt_score, _tt_bm = _probe_tt_lookup(tt, key, depth, alpha, beta)
    if tt_score is not None:
        return tt_score

    # 深度耗尽 → 静止搜索
    if depth <= 0:
        return _probe_qs(board, alpha, beta, qs_depth_limit, probe_state)

    moves = list(Rules.get_pseudo_legal_moves(board, board.current_player))
    _probe_order_moves(board, moves, probe_state, depth)

    best = float("-inf")
    best_move: Optional[Move4] = None
    has_legal = False

    for move in moves:
        mover = board.current_player
        applied = apply_pseudo_legal_with_rule_cache(
            board,
            move,
            mover,
            pre_move_cache=probe_state["pre_move_cache"],
            post_apply_cache=probe_state["post_apply_cache"],
        )
        if applied is None:
            continue
        captured, opp_in_check = applied
        has_legal = True

        # 将军延伸：走后对手被将时不消耗深度
        next_depth = depth - 1
        next_ext = check_ext_left
        if opp_in_check and check_ext_left > 0:
            next_depth = depth
            next_ext = check_ext_left - 1

        score = -_probe_negamax(
            board, next_depth, -beta, -alpha, probe_state,
            check_ext_left=next_ext, qs_depth_limit=qs_depth_limit,
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
#  Probe 智能触发判定（多因子评分 + 概率 + 预算 + 冷却）
# ═══════════════════════════════════════════════════════════════


def _is_quiet_endgame(board: Board) -> bool:
    """O(n) 轻量安静残局检测：双方主要攻击子是否与对方将帅有纵横/马步接触。

    遍历双方活跃棋子，若任意车/炮与敌方将同行或同列、或马与敌方将呈
    日字距离，则认为存在即时威胁线，局面不安静。否则为安静残局——
    双方暂时无法直接威胁将帅，静态评估的误差较小。

    开销：O(n) 其中 n = 双方活跃棋子总数（残局通常 ≤ 14）。
    """
    b = board.board
    rkp = board.red_king_pos
    bkp = board.black_king_pos
    if rkp is None or bkp is None:
        return False
    for color, (ekr, ekc) in [("red", bkp), ("black", rkp)]:
        for r, c in board.active_pieces.get(color, ()):
            p = b[r][c]
            if p is None:
                continue
            pt = p.piece_type
            if pt in ("che", "pao") and (r == ekr or c == ekc):
                return False
            if pt == "ma":
                dr, dc = abs(r - ekr), abs(c - ekc)
                if (dr, dc) in ((1, 2), (2, 1)):
                    return False
    return True


def _probe_depth_for_position(piece_count: int, is_in_check: bool) -> int:
    """根据局面特征决定 Level 2 标准 probe 的搜索深度。"""
    if piece_count <= 10:
        return _PROBE_ENDGAME_DEPTH
    if is_in_check or piece_count <= _ENDGAME_PIECE_THRESHOLD:
        return _PROBE_TACTICAL_DEPTH
    return _PROBE_DEFAULT_DEPTH


def _compute_probe_level(
    board: Board,
    piece_count: int,
    budget: ProbeBudget,
    sims_done: int,
    *,
    is_last_capture: bool = False,
    node_visits: int = 0,
) -> int:
    """多因子评分式 probe 触发判定，返回 probe 等级 0 / 1 / 2。

    评分流程：
    1. 按子力阶段、战术信号、上下文特征累加分数。
    2. 扣除安静局面惩罚、预算耗尽惩罚、冷却惩罚。
    3. 将总分与分级阈值比较，再通过概率门控最终决定等级。

    Level 含义：
        0 — 不 probe，直接使用静态评估（快速路径）。
        1 — 轻量 probe（depth=1, QS depth=2, 无将军延伸）。
        2 — 标准 probe（depth=2~4, QS depth=4, 含将军延伸）。

    Returns:
        0, 1, 或 2。
    """
    cp = board.current_player
    opp = "black" if cp == "red" else "red"
    is_cp_in_check = Rules.is_king_in_check(board, cp)
    is_opp_in_check = Rules.is_king_in_check(board, opp)

    # ── 1. 子力阶段基础分 ──
    score = 0
    if piece_count <= 10:
        score += _SCORE_DEEP_ENDGAME
    elif piece_count <= _ENDGAME_PIECE_THRESHOLD:
        score += _SCORE_EARLY_ENDGAME
    elif piece_count <= 20:
        score += _SCORE_LATE_MIDGAME

    # ── 2. 战术信号加分 ──
    if is_cp_in_check:
        score += _SCORE_IN_CHECK
    if is_opp_in_check:
        score += _SCORE_OPP_IN_CHECK
    if is_last_capture:
        score += _SCORE_LAST_CAPTURE
    if node_visits >= _HIGH_VISIT_THRESHOLD:
        score += _SCORE_HIGH_VISITS

    # ── 3. 安静局面惩罚（仅残局阶段计算，避免中局白耗开销） ──
    if not is_cp_in_check and not is_opp_in_check:
        if piece_count <= _ENDGAME_PIECE_THRESHOLD and _is_quiet_endgame(board):
            score -= _QUIET_SCORE_PENALTY

    # ── 4. 预算惩罚（硬约束：耗尽后几乎必定 Level 0） ──
    if budget.is_any_exhausted:
        score -= _BUDGET_EXHAUSTED_PENALTY
    # ── 5. 冷却惩罚（软约束：间隔不足时降低触发倾向） ──
    elif budget.is_cooldown_active(sims_done):
        score -= _COOLDOWN_PENALTY

    # ── 5b. 战术压力（将军机会 / 高价值吃子机会）→ 提高 probe 倾向，不改静态评估 ──
    pressure = _tactical_pressure_bonus(board)
    if pressure >= 2:
        score += _SCORE_POLICY_PRESSURE
    elif pressure == 1:
        score += max(1, _SCORE_POLICY_PRESSURE // 2)

    # ── 6. 分级 + 概率门控 ──
    if score >= _PROBE_SCORE_THRESHOLD_FULL:
        if random.random() < _PROBE_PROB_FULL:
            return 2
        return 0
    if score >= _PROBE_SCORE_THRESHOLD_LIGHT:
        if random.random() < _PROBE_PROB_LIGHT:
            return 1
        return 0
    return 0


# ═══════════════════════════════════════════════════════════════
#  MCTS 辅助函数（模块级，pickle 安全）
# ═══════════════════════════════════════════════════════════════


def _tactical_pressure_bonus(board: Board) -> int:
    """O(伪合法着法数)：是否存在将军着法或高价值吃子（仅用于 probe 触发评分）。"""
    cp = board.current_player
    b = board.board
    pv = Evaluation.PIECE_VALUES
    hi_cap = False
    for m in Rules.get_pseudo_legal_moves(board, cp):
        if _move_gives_check(board, m, cp):
            return 2
        vic = b[m[2]][m[3]]
        if vic is not None and pv.get(vic.piece_type, 0) >= _POLICY_HVCAP_VALUE:
            hi_cap = True
    return 1 if hi_cap else 0


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


def _eval_to_winrate(board: Board, root_player: str) -> float:
    """静态评估 → sigmoid 胜率（快速路径，无 probe）。"""
    raw = Evaluation.evaluate(board)
    if board.current_player != root_player:
        raw = -raw
    return 1.0 / (1.0 + math.exp(-raw / _SCORE_SCALE))


def _terminal_score(
    board: Board,
    root_player: str,
    *,
    move_history: Optional[List[MoveEntry]] = None,
) -> float:
    """终局局面评估：胜 1.0 / 负 0.0 / 未分 0.5（与 ``Rules.winner(..., move_history)`` 一致）。"""
    w = Rules.winner(board, move_history)
    if w == root_player:
        return 1.0
    if w is not None:
        return 0.0
    return 0.5


def _backpropagate(
    path: List[MCTSMinimaxNode],
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
    sims_done: int,
    node_visits: int = 0,
    *,
    gives_check_cache: Optional[MoveGivesCheckCache] = None,
    path_history: Optional[List[MoveEntry]] = None,
) -> Tuple[float, Set[Move4], Set[Move4]]:
    """轻量级伪合法 rollout + 分级选择性 minimax probe。

    与 ``mcts.py`` 的 ``_simulate()`` 流程一致，区别在于 rollout 截断后：
    调用 ``_compute_probe_level()`` 做多因子评分判定，返回三级策略：

    - **Level 0** — 不 probe，直接静态评估（快速路径）。
    - **Level 1** — 轻量 probe（depth 1, QS depth 2, 无将军延伸）。
    - **Level 2** — 标准 probe（depth 2~4, QS depth 4, 含将军延伸）。

    probe 触发受 ``ProbeBudget`` 的调用次数上限、节点上限和冷却间隔约束。

    Args:
        gives_check_cache: 与本棵搜索树共用，加速 rollout 内 ``_move_gives_check``。
        path_history: 到达当前节点前的 ``MoveEntry`` 链。
    """
    sim_board = board.copy()
    b_grid = sim_board.board
    rollout_limit = _dynamic_rollout_limit(sim_board.piece_count())
    red_moves: Set[Move4] = set()
    black_moves: Set[Move4] = set()
    is_last_capture = False
    hist: List[MoveEntry] = (
        list(path_history)
        if path_history is not None
        else [MoveEntry(pos_hash=sim_board.zobrist_hash)]
    )
    if not hist or hist[-1].pos_hash != sim_board.zobrist_hash:
        hist = [MoveEntry(pos_hash=sim_board.zobrist_hash)]

    for _ in range(rollout_limit):
        cp = sim_board.current_player
        opp = "black" if cp == "red" else "red"

        if Rules.is_move_limit_draw(hist):
            return 0.5, red_moves, black_moves

        st_pf, _off_pf = Rules.perpetual_check_status(sim_board, hist)
        if st_pf == "forfeit" and _off_pf is not None:
            w_pf = "black" if _off_pf == "red" else "red"
            return (1.0 if w_pf == root_player else 0.0), red_moves, black_moves

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
                _append_path_move_entry(hist, sim_board, king_cap, cp)
                if cp == "red":
                    red_moves.add(king_cap)
                else:
                    black_moves.add(king_cap)
                return (1.0 if cp == root_player else 0.0), red_moves, black_moves

        chosen, is_last_capture = _pick_rollout_move_fast(
            sim_board, moves, b_grid, gives_check_cache=gives_check_cache,
        )
        _append_path_move_entry(hist, sim_board, chosen, cp)
        if cp == "red":
            red_moves.add(chosen)
        else:
            black_moves.add(chosen)

    # ── Rollout 截断：多因子评分 → 分级 probe 或快速静态评估 ──
    current_pc = sim_board.piece_count()
    budget: ProbeBudget = probe_state["budget"]

    probe_level = _compute_probe_level(
        sim_board, current_pc, budget, sims_done,
        is_last_capture=is_last_capture,
        node_visits=node_visits,
    )

    if probe_level == 2:
        # ── Level 2: 标准 probe ──
        is_in_check = Rules.is_king_in_check(sim_board, sim_board.current_player)
        depth = _probe_depth_for_position(current_pc, is_in_check)
        nodes_before = probe_state["nodes"]
        probe_state["probes"] += 1
        raw = _probe_negamax(
            sim_board, depth, float("-inf"), float("inf"), probe_state,
            check_ext_left=_PROBE_MAX_CHECK_EXT,
            qs_depth_limit=_PROBE_QS_DEPTH_LIMIT,
        )
        budget.record_probe(sims_done, probe_state["nodes"] - nodes_before)
        if sim_board.current_player != root_player:
            raw = -raw
        result = 1.0 / (1.0 + math.exp(-raw / _SCORE_SCALE))

    elif probe_level == 1:
        # ── Level 1: 轻量 probe（浅搜索 + 浅 QS + 无将军延伸） ──
        nodes_before = probe_state["nodes"]
        probe_state["probes"] += 1
        raw = _probe_negamax(
            sim_board, _LIGHT_PROBE_DEPTH, float("-inf"), float("inf"),
            probe_state,
            check_ext_left=0,
            qs_depth_limit=_LIGHT_PROBE_QS_DEPTH,
        )
        budget.record_probe(sims_done, probe_state["nodes"] - nodes_before)
        if sim_board.current_player != root_player:
            raw = -raw
        result = 1.0 / (1.0 + math.exp(-raw / _SCORE_SCALE))

    else:
        # ── Level 0: 快速路径——纯静态评估 ──
        result = _eval_to_winrate(sim_board, root_player)

    return result, red_moves, black_moves


def _expand_one(
    board: Board,
    node: MCTSMinimaxNode,
    move_stack: List[Tuple[Move4, Any]],
    entry_stack: List[MoveEntry],
    tt: Dict[int, MCTSMinimaxNode],
    probe_state: dict,
) -> Optional[Tuple[Move4, MCTSMinimaxNode]]:
    """从 untried_moves 弹出一个合法走法并建立父子边（含 DAG 合并）。"""
    mover = board.current_player
    while node.untried_moves:
        move = node.untried_moves.pop()
        applied = apply_pseudo_legal_with_rule_cache(
            board,
            move,
            mover,
            pre_move_cache=probe_state["pre_move_cache"],
            post_apply_cache=probe_state["post_apply_cache"],
        )
        if applied is None:
            continue
        captured, _ = applied
        move_stack.append((move, captured))
        _append_path_move_entry(entry_stack, board, move, mover)
        child_hash = board.zobrist_hash
        existing = tt.get(child_hash)
        if existing is not None:
            node.children[move] = existing
            return move, existing
        child = MCTSMinimaxNode(state_hash=child_hash, player_just_moved=mover)
        tt[child_hash] = child
        node.children[move] = child
        return move, child
    return None


def _merge_child_stats(
    results: list,
) -> Tuple[Dict[Move4, Dict[str, float]], Dict[str, int]]:
    """合并多棵树的根子节点统计 + probe 统计（含预算使用量）。"""
    merged: Dict[Move4, Dict[str, float]] = {}
    total_probes = 0
    total_probe_nodes = 0
    total_budget_calls = 0
    total_budget_calls_max = 0
    total_budget_nodes = 0
    total_budget_nodes_max = 0
    for child_stats, p_stats in results:
        total_probes += p_stats.get("probes", 0)
        total_probe_nodes += p_stats.get("probe_nodes", 0)
        total_budget_calls += p_stats.get("budget_calls_used", 0)
        total_budget_calls_max += p_stats.get("budget_calls_max", 0)
        total_budget_nodes += p_stats.get("budget_nodes_used", 0)
        total_budget_nodes_max += p_stats.get("budget_nodes_max", 0)
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
    return merged, {
        "probes": total_probes,
        "probe_nodes": total_probe_nodes,
        "budget_calls_used": total_budget_calls,
        "budget_calls_max": total_budget_calls_max,
        "budget_nodes_used": total_budget_nodes,
        "budget_nodes_max": total_budget_nodes_max,
    }


# ═══════════════════════════════════════════════════════════════
#  单棵 MCTS-Minimax 树搜索（worker 主函数，模块级以支持 pickle）
# ═══════════════════════════════════════════════════════════════


def _run_single_mcts_minimax_tree(
    board: Board,
    max_simulations: int,
    time_limit: float,
    seed_offset: int = 0,
    root_move_history: Optional[List[MoveEntry]] = None,
) -> Tuple[Dict[Move4, Dict[str, float]], Dict[str, int]]:
    """在独立进程中执行一棵完整的 MCTS-Minimax 搜索树。

    四阶段流程与 ``mcts.py`` 一致：Selection → Expansion → Simulation/Probe
    → Backpropagation。Simulation 阶段可选择性启用 Minimax probe。

    Returns:
        ``(child_stats, probe_stats)`` 二元组。
    """
    random.seed(time.time_ns() + seed_offset)
    t0 = time.time()

    # 初始化 per-tree probe 预算（调用次数 + 节点数双重上限）
    budget = ProbeBudget(max_simulations)

    post_apply_cache = PostApplyFlagsCache(65536)
    gives_check_cache = MoveGivesCheckCache(131072, post_apply_cache=post_apply_cache)
    # 初始化 per-tree probe 状态（所有模拟共享，多进程天然隔离）
    probe_state: dict = {
        "tt": {},
        "killers": [[None, None] for _ in range(_PROBE_KILLER_SLOTS)],
        "history": {},
        "nodes": 0,
        "probes": 0,
        "budget": budget,
        "post_apply_cache": post_apply_cache,
        "pre_move_cache": gives_check_cache,
    }

    root_player = board.current_player
    opp_of_root = "black" if root_player == "red" else "red"
    root = MCTSMinimaxNode(state_hash=board.zobrist_hash, player_just_moved=opp_of_root)
    root.ensure_moves(board, gives_check_cache)

    tt: Dict[int, MCTSMinimaxNode] = {root.state_hash: root}
    move_stack: List[Tuple[Move4, Any]] = []
    sims_done = 0

    while sims_done < max_simulations:
        if time.time() - t0 >= time_limit:
            break

        if root_move_history is not None and (
            len(root_move_history) > 0
            and root_move_history[-1].pos_hash == board.zobrist_hash
        ):
            entry_stack: List[MoveEntry] = list(root_move_history)
        else:
            entry_stack = [MoveEntry(pos_hash=board.zobrist_hash)]

        node = root
        path: List[MCTSMinimaxNode] = [root]

        # ── Selection ──
        while node.is_fully_expanded() and node.children:
            log_n = math.log(node.visits) if node.visits > 0 else 0.0
            edge_move, next_node = node.best_child_ucb(log_n)
            mover_sel = board.current_player
            captured = board.apply_move(*edge_move)
            _append_path_move_entry(entry_stack, board, edge_move, mover_sel)
            move_stack.append((edge_move, captured))
            path.append(next_node)
            node = next_node

        # ── Expansion ──
        node.ensure_moves(board, gives_check_cache)
        expanded = False
        if node.untried_moves:
            result = _expand_one(
                board, node, move_stack, entry_stack, tt, probe_state
            )
            if result is not None:
                _exp_move, child = result
                path.append(child)
                node = child
                expanded = True

        if not expanded and not node.children and node.visits > 0:
            sim_result = _terminal_score(
                board, root_player, move_history=entry_stack
            )
            red_moves: Set[Move4] = set()
            black_moves: Set[Move4] = set()
        else:
            # ── Simulation / Probe ──
            sim_result, red_moves, black_moves = _simulate_or_probe(
                board, root_player, probe_state,
                sims_done=sims_done,
                node_visits=node.visits,
                gives_check_cache=gives_check_cache,
                path_history=entry_stack,
            )

        # ── Backpropagation ──
        _backpropagate(path, sim_result, red_moves, black_moves)
        sims_done += 1

        # 还原棋盘
        while move_stack:
            mv, cap = move_stack.pop()
            entry_stack.pop()
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
        "budget_calls_used": budget.probe_calls,
        "budget_calls_max": budget.max_probe_calls,
        "budget_nodes_used": budget.probe_nodes,
        "budget_nodes_max": budget.max_probe_nodes,
    }
    return child_stats, probe_stats


# ═══════════════════════════════════════════════════════════════
#  MCTSMinimaxAI —— 公开接口
# ═══════════════════════════════════════════════════════════════


class MCTSMinimaxAI:
    """MCTS-Minimax 搜索 AI（对外接口与 MinimaxAI / MCTSAI 一致）。

    MCTS 负责全局探索与 RAVE 收敛，Minimax 负责叶节点战术精算。
    支持多进程根节点并行 + 开局库查询 + DAG 置换合并。

    Args:
        max_simulations: 所有 worker 合计的最大模拟次数（默认 4000）。
        time_limit: 搜索时间上限（秒）。
        workers: 并行进程数（0/1 = 单进程）。默认 ``None`` 表示 ``min(8, cpu_count)``。
        probe_depth: minimax probe 默认深度。
        verbose: 搜索完成后是否输出统计信息。
    """

    def __init__(
        self,
        max_simulations: int = 4000,
        time_limit: float = 10.0,
        workers: Optional[int] = None,
        probe_depth: int = _PROBE_DEFAULT_DEPTH,
        verbose: bool = True,
        **kwargs: Any,
    ):
        # 忽略工厂/CLI 透传的无关参数（如 stochastic、verbose 重复键等），避免 TypeError
        if "depth" in kwargs and probe_depth == _PROBE_DEFAULT_DEPTH:
            try:
                probe_depth = int(kwargs["depth"])  # 兼容统一工厂把 depth 当作浅层 probe 深度
            except (TypeError, ValueError):
                pass
        self.max_simulations = max_simulations
        self.time_limit = time_limit
        self.verbose = verbose
        self.probe_depth = probe_depth
        if workers is None:
            try:
                self.workers = min(8, multiprocessing.cpu_count())
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
        move_history: Optional[List[MoveEntry]] = None,
    ) -> Optional[Move4]:
        """Searcher 统一接口。"""
        return self.get_best_move(
            board,
            time_limit=time_limit,
            game_history=game_history,
            move_history=move_history,
        )

    def get_best_move(
        self,
        board: Board,
        game_history: Optional[List[int]] = None,
        time_limit: Optional[float] = None,
        max_simulations: Optional[int] = None,
        move_history: Optional[List[MoveEntry]] = None,
    ) -> Optional[Move4]:
        """执行 MCTS-Minimax 搜索并返回最佳走法。"""
        Evaluation._eval_cache.clear()
        hash_history: List[int] = [] if game_history is None else list(game_history)
        root_mh = list(move_history) if move_history is not None else None

        # ── 开局库查询 ──
        if len(hash_history) < 30:
            book_move = self._probe_opening_book(board, hash_history)
            if book_move is not None:
                self.simulations_run = 0
                self.last_stats = {
                    "time_taken": 0.0,
                    "simulations": 0,
                    "workers": self.workers,
                    "win_rate": "开局库",
                    "opening_book": True,
                }
                if self.verbose:
                    print(f"[MCTS-Minimax] 命中开局库！瞬间出棋: {book_move}")
                return book_move

        # ── MCTS-Minimax 搜索 ──
        tl = time_limit if time_limit is not None else self.time_limit
        ms = max_simulations if max_simulations is not None else self.max_simulations
        t0 = time.time()

        effective_workers = self.workers
        if effective_workers <= 1 or ms < effective_workers * 10:
            child_stats, probe_stats = _run_single_mcts_minimax_tree(
                board, ms, tl, seed_offset=0, root_move_history=root_mh,
            )
            total_sims = sum(int(s["visits"]) for s in child_stats.values())
        else:
            child_stats, probe_stats = self._parallel_search(
                board, tl, ms, effective_workers, root_move_history=root_mh,
            )
            total_sims = sum(int(s["visits"]) for s in child_stats.values())

        elapsed = time.time() - t0
        self.simulations_run = total_sims

        root_player = board.current_player
        v_max = max((int(st["visits"]) for st in child_stats.values()), default=0)
        best_move: Optional[Move4] = None
        best_key = -1.0
        best_visits = -1
        best_wr = 0.0
        for mv, st in child_stats.items():
            v = int(st["visits"])
            w = float(st["wins"])
            wr = w / v if v > 0 else 0.0
            bias = _policy_attack_bias(board, root_player, mv)
            close = min(1.0, v / max(v_max * _ROOT_VISITS_TIE_FRAC, 1e-6))
            key = v * 1_000_000.0 + w * 1_000.0 + bias * _ROOT_BIAS_SCALE * close
            if key > best_key:
                best_key = key
                best_move = mv
                best_visits = v
                best_wr = wr

        probe_n = int(probe_stats.get("probe_nodes", 0))
        self.last_stats = {
            "time_taken": elapsed,
            "simulations": total_sims,
            "workers": effective_workers,
            "win_rate": f"{best_wr * 100:.1f}%",
            "probe_count": probe_stats.get("probes", 0),
            "probe_nodes": probe_n,
            "budget_calls_used": probe_stats.get("budget_calls_used", 0),
            "budget_calls_max": probe_stats.get("budget_calls_max", 0),
            "budget_nodes_used": probe_stats.get("budget_nodes_used", 0),
            "budget_nodes_max": probe_stats.get("budget_nodes_max", 0),
            # 与 Minimax 统计字段对齐，供基准脚本 / 通用日志读取
            "nodes_evaluated": probe_n + total_sims,
        }
        if self.verbose:
            print(
                f"[MCTS-Minimax] 搜索完成: 模拟次数 {total_sims}，"
                f"并行进程数 {effective_workers}，耗时 {elapsed:.3f} 秒"
            )
            pc = probe_stats.get("probes", 0)
            pn = probe_stats.get("probe_nodes", 0)
            bc = probe_stats.get("budget_calls_used", 0)
            bm = probe_stats.get("budget_calls_max", 0)
            print(
                f"[MCTS-Minimax] 局面探查: {pc} 次（节点数 {pn}），"
                f"子搜索预算（调用次数）: {bc}/{bm}"
            )
            if best_move is not None:
                print(f"[MCTS-Minimax] 最佳走法: {best_move}  胜率: {best_wr:.1%}")
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
                f"[MCTS-Minimax 开局库] 根局面未命中（局面键 zkey={zkey:#x}）；"
                f"开局库共 {len(keys)} 个局面键，示例: {disp}"
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
        root_move_history: Optional[List[MoveEntry]] = None,
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
                        _run_single_mcts_minimax_tree,
                        board_copy,
                        worker_sims,
                        time_limit,
                        i * 1000,
                        root_move_history,
                    )
                )
            results = [f.result() for f in futures]
        return _merge_child_stats(results)
