"""蒙特卡洛树搜索（MCTS）AI 引擎。

本模块实现了完整的中国象棋 MCTS 搜索框架，集成以下核心技术：

- **UCB1-RAVE 混合选择**：在传统 UCB1 置信上限公式的基础上，融合 RAVE
  (Rapid Action Value Estimation) / AMAF (All-Moves-As-First) 启发统计，
  使低访问量的着法也能借助模拟阶段的全局经验快速收敛。
- **惰性走法展开**：``untried_moves`` 仅在节点首次需要展开时才调用走法生成器
  （延迟求值），后续通过 ``pop()`` 逐个消费，避免在非叶节点上浪费走法生成开销。
- **apply / undo 零拷贝树遍历**：Selection 与 Expansion 阶段直接在真实棋盘上
  ``apply_move``，结束后通过 ``undo_move`` 栈严格还原——O(1) 回溯，无需深拷贝。
- **动态截断轻量模拟**：Simulation 阶段根据场上子力数（开局 / 中局 / 残局）
  自适应调整截断步数，配合 ``Evaluation.evaluate`` 兜底估分。
- **__slots__ 节点内存优化**：海量节点使用 ``__slots__`` 声明，避免 Python 默认
  ``__dict__`` 带来的额外内存开销。
- **多进程根节点并行 (Root Parallelism)**：利用 ``ProcessPoolExecutor`` 将总模拟次数
  分配给多个独立 worker，每个 worker 建一棵独立搜索树，搜索结束后在主进程合并
  各棵树根子节点的 visits / wins / rave 统计。
- **Zobrist 置换表 DAG 合并**：同一进程内的搜索树使用局部 Zobrist 置换表 ``tt``，
  将不同着法序列到达的相同局面（哈希碰撞概率极低）合并为同一节点实例，
  使搜索树退化为有向无环图 (DAG)，共享 visits / wins 统计，加速收敛。

**DAG 边属性设计（核心架构决策）**：

    注意：在 DAG 结构中，一个节点可能拥有多个父节点。如果将走法 (move)
    绑定在节点自身属性上，当不同的父节点通过**不同的走法**到达同一个子节点时，
    Selection 阶段从子节点读取 ``node.move`` 会取到第一次创建时存储的走法，
    而非当前路径上的走法——这就是"幽灵棋 (Ghost Move)"Bug。

    解决方案：走法 (move) 存储于父节点 ``children`` 字典的**键**中
    （即"边属性"），而非子节点自身的属性。Selection 阶段始终从
    ``best_child_ucb()`` 返回的 ``(edge_move, child_node)`` 元组中
    读取走法来执行 ``board.apply_move``，确保走法与当前路径严格一致。
"""

from __future__ import annotations

import concurrent.futures
import math
import multiprocessing
import random
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from chinese_chess.model.board import Board
from chinese_chess.model.rules import MoveEntry, Rules

from .evaluation import Evaluation
from .search_move_helpers import (
    MoveGivesCheckCache,
    PostApplyFlagsCache,
    apply_pseudo_legal_with_rule_cache,
    fast_move_gives_check,
    pseudo_move_post_apply_flags,
    try_fast_move_legality_and_opponent_check,
)
from .opening_book import OPENING_BOOK, mirror_move

# ── 模块级常量 ──

_UCB_C = 1.414          # UCB1 探索常数 c ≈ √2，平衡利用与探索
_SCORE_SCALE = 600.0    # Evaluation 原始分 → sigmoid 胜率的缩放因子
_RAVE_CONST = 300       # RAVE 等效常数 k：β = rave_visits / (rave_visits + visits + k)，
                        # k 越大 → RAVE 权重衰减越快 → 越快回归纯 MCTS

# ── Policy 层进攻偏置（不调用 Evaluation.evaluate，不重算静态分值）──
_POLICY_HVCAP_VALUE = 300   # 高价值吃子阈值（马/炮/车，沿用 MG 子力刻度）
_ROOT_BIAS_SCALE = 72.0     # 根决策 tie-break：略提高，更倾向吃子/将军类根着法
_ROOT_VISITS_TIE_FRAC = 0.84  # 略放宽，使更多近 visit 根子吃满 attack bias
_ROLL_PREFER_CHECK = 0.64     # rollout 更常优先选将军着法
_ROLL_PREFER_HVCAP = 0.66     # 更常优先高价值吃子
_ROLL_PREFER_ANY_CAPTURE = 0.80
_ROLL_PREFER_AGGRESSIVE = 0.50  # 更常选过河/压境类推进
# 根 tie-break：_policy_attack_bias 内对将军 / 攻击性推进的加分上限分量
_POLICY_BIAS_CHECK_COMPONENT = 0.42
_POLICY_BIAS_AGGRESSIVE_PUSH_COMPONENT = 0.16

# 选择阶段沿 DAG 下降的最大步数（防止局面环 + 全展开节点导致死循环）
_SELECTION_MAX_PLIES = 512

# MCTS 专用：轻量 dict 缓存 (zobrist, move) → (legal, opp_in_check)，满则 clear
_MCTS_GIVES_CHECK_CAP = 200_000


def _parallel_workers_when_safe(requested: int) -> int:
    """在非主线程（如 PyQt AI 后台线程）中禁用多进程根并行。

    Windows 上于子线程创建 ``ProcessPoolExecutor`` 常与 spawn 导入链死锁，
    表现为 MCTS 长时间无响应。主线程（CLI / benchmark）仍可使用多进程。
    """
    if requested <= 1:
        return requested
    try:
        if threading.current_thread() is not threading.main_thread():
            return 1
    except Exception:
        pass
    return requested


# 走法四元组：(起始行, 起始列, 目标行, 目标列)
Move4 = Tuple[int, int, int, int]


def mcts_fast_move_gives_check(
    board: Board,
    move: Move4,
    mover: str,
    mcts_cache: Dict[Tuple[int, Move4], Tuple[bool, bool]],
) -> bool:
    """MCTS 专用：是否对对方将军（非法则 False）。

    先查 ``mcts_cache``，再 ``try_fast_move_legality_and_opponent_check``（无 apply），
    失败则 ``apply`` + ``pseudo_move_post_apply_flags`` 后 undo，并写回缓存。
    满 ``_MCTS_GIVES_CHECK_CAP`` 时整表 ``clear()``（非 LRU，仅服务单棵树搜索）。
    """
    key = (board.zobrist_hash, move)
    hit = mcts_cache.get(key)
    if hit is not None:
        legal, gc = hit
        return gc if legal else False
    res = try_fast_move_legality_and_opponent_check(board, move, mover)
    if res is not None:
        legal, gc = res
        if len(mcts_cache) >= _MCTS_GIVES_CHECK_CAP:
            mcts_cache.clear()
        mcts_cache[key] = (legal, gc)
        return gc if legal else False
    captured = board.apply_move(*move)
    try:
        legal, gc = pseudo_move_post_apply_flags(board, mover)
    finally:
        board.undo_move(*move, captured)
    if len(mcts_cache) >= _MCTS_GIVES_CHECK_CAP:
        mcts_cache.clear()
    mcts_cache[key] = (legal, gc)
    return gc if legal else False


def _append_path_move_entry(
    entry_stack: List[MoveEntry], board: Board, move: Move4, mover: str
) -> None:
    """在 ``board`` 已执行 ``move`` 之后，追加对应的 ``MoveEntry``。"""
    opp = board.current_player
    entry_stack.append(
        MoveEntry(
            pos_hash=board.zobrist_hash,
            mover=mover,
            gave_check=Rules.is_king_in_check(board, opp),
            last_move=move,
        )
    )


def _move_gives_check(
    board: Board,
    move: Move4,
    mover: str,
    gives_check_cache: Optional[MoveGivesCheckCache] = None,
    *,
    mcts_gives: Optional[Dict[Tuple[int, Move4], Tuple[bool, bool]]] = None,
) -> bool:
    """走法是否对对方将军（非法则 False）。

    若传入 ``mcts_gives``（仅 MCTS / MCTS-Minimax 树内使用），走 MCTS 轻量缓存 +
    ``try_fast`` 路径，不经过 ``MoveGivesCheckCache`` LRU。未传入时行为与原先一致
    （``fast_move_gives_check``），Minimax 等调用方不受影响。
    """
    if mcts_gives is not None:
        return mcts_fast_move_gives_check(board, move, mover, mcts_gives)
    return fast_move_gives_check(board, move, mover, gives_check_cache)


def _is_aggressive_push(board: Board, mover: str, m: Move4, b_grid) -> bool:
    """轻量「攻击性推进」：过河兵/卒纵向前进，或大子深入敌阵（非吃子）。"""
    sr, sc, er, ec = m
    piece = b_grid[sr][sc]
    if piece is None or b_grid[er][ec] is not None:
        return False
    pt = piece.piece_type
    if pt == "bing":
        if mover == "red":
            return er < sr or (er == sr and abs(ec - sc) == 1 and sr <= 4)
        return er > sr or (er == sr and abs(ec - sc) == 1 and sr >= 5)
    if mover == "red" and er <= 4:
        return pt in ("che", "ma", "pao")
    if mover == "black" and er >= 5:
        return pt in ("che", "ma", "pao")
    return False


def _is_forcing_move(
    board: Board,
    mover: str,
    m: Move4,
    b_grid,
    gives_check_cache: Optional[MoveGivesCheckCache] = None,
    *,
    mcts_gives: Optional[Dict[Tuple[int, Move4], Tuple[bool, bool]]] = None,
) -> bool:
    """将军或吃子（不含完整局面评估）。"""
    if b_grid[m[2]][m[3]] is not None:
        return True
    return _move_gives_check(board, m, mover, gives_check_cache, mcts_gives=mcts_gives)


def _policy_attack_bias(
    board: Board,
    mover: str,
    m: Move4,
    gives_check_cache: Optional[MoveGivesCheckCache] = None,
    *,
    mcts_gives: Optional[Dict[Tuple[int, Move4], Tuple[bool, bool]]] = None,
) -> float:
    """根节点 tie-break 用轻量偏置，约 0~1.5，仅供 policy，不写入静态值。"""
    b = board.board
    pv = Evaluation.PIECE_VALUES
    bias = 0.0
    victim = b[m[2]][m[3]]
    if victim is not None and victim.piece_type != "jiang":
        bias += min(1.0, float(pv.get(victim.piece_type, 0)) / 900.0)
    if victim is None or victim.piece_type != "jiang":
        if _move_gives_check(board, m, mover, gives_check_cache, mcts_gives=mcts_gives):
            bias += _POLICY_BIAS_CHECK_COMPONENT
    if _is_aggressive_push(board, mover, m, b):
        bias += _POLICY_BIAS_AGGRESSIVE_PUSH_COMPONENT
    return bias


def _order_untried_moves_policy(
    board: Board,
    moves: List[Move4],
    gives_check_cache: Optional[MoveGivesCheckCache] = None,
    *,
    mcts_gives: Optional[Dict[Tuple[int, Move4], Tuple[bool, bool]]] = None,
) -> None:
    """展开顺序：低优先在前、高优先在尾（pop 先展高优先）；组内随机。"""
    if not moves:
        return
    mover = board.current_player
    b = board.board
    pv = Evaluation.PIECE_VALUES
    tier_rest: List[Move4] = []
    tier_agg: List[Move4] = []
    tier_cap: List[Move4] = []
    tier_hvcap: List[Move4] = []
    tier_check: List[Move4] = []
    for m in moves:
        victim = b[m[2]][m[3]]
        if victim is not None and victim.piece_type == "jiang":
            tier_check.append(m)
            continue
        if victim is not None:
            if pv.get(victim.piece_type, 0) >= _POLICY_HVCAP_VALUE:
                tier_hvcap.append(m)
            else:
                tier_cap.append(m)
            continue
        if _move_gives_check(board, m, mover, gives_check_cache, mcts_gives=mcts_gives):
            tier_check.append(m)
            continue
        if _is_aggressive_push(board, mover, m, b):
            tier_agg.append(m)
        else:
            tier_rest.append(m)
    for bucket in (tier_rest, tier_agg, tier_cap, tier_hvcap, tier_check):
        random.shuffle(bucket)
    moves[:] = tier_rest + tier_agg + tier_cap + tier_hvcap + tier_check


def _dynamic_rollout_limit(piece_count: int) -> int:
    """根据当前场上子力数量，动态决定模拟阶段的截断步数。

    子力越多（开局），局面越复杂，但随机模拟的信噪比也越低，
    因此用较少的步数快速截断；子力越少（残局），随机模拟更容易
    走出决定性结果，因此放宽步数上限。

    Args:
        piece_count: 当前棋盘上的总子力数（红 + 黑）。

    Returns:
        本次模拟允许的最大步数。
    """
    if piece_count > 24:
        return 20   # 开局：子力繁杂，20 步快速截断
    if piece_count >= 10:
        return 35   # 中局
    return 50       # 残局：子力稀少，50 步充分展开


# ═══════════════════════════════════════════════════════════════
#  MCTSNode —— 搜索节点（DAG 安全）
# ═══════════════════════════════════════════════════════════════


class MCTSNode:
    """MCTS-RAVE 有向无环图 (DAG) 搜索节点。

    **边属性设计**：走法 (move) 存储在父节点 ``children: Dict[Move4, MCTSNode]``
    的字典键中，而非节点自身的属性。这使得同一节点可以安全地被多个父节点共享
    （DAG 结构），Selection 阶段通过字典键读取对应路径上的正确走法，
    不会出现"幽灵棋"冲突。

    **RAVE 统计**：``rave_visits`` / ``rave_wins`` 实现 AMAF (All-Moves-As-First)
    快速估值——即使某子节点自身访问次数很少，也可通过模拟阶段出现的同一着法
    来累积统计量，在搜索早期显著加速收敛。

    Attributes:
        state_hash: 当前局面的 Zobrist 哈希值（用于置换表查重）。
        children: 已展开的子节点字典，键为到达该子节点的走法（边属性）。
        visits: 该节点被访问（经过 / 模拟）的总次数 N。
        wins: 累计胜利分数 W（胜=1, 和=0.5, 负=0；视角随回溯层交替翻转）。
        untried_moves: 尚未尝试展开的走法列表（惰性初始化，首次展开时生成）。
        player_just_moved: 刚走完一步到达本节点的玩家（"red" 或 "black"）。
        rave_visits: RAVE/AMAF 统计的访问计数。
        rave_wins: RAVE/AMAF 统计的累计胜利分。
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

    def __init__(
        self,
        state_hash: int,
        player_just_moved: str,
    ):
        """初始化一个空白搜索节点。

        Args:
            state_hash: 该节点对应局面的 Zobrist 哈希。
            player_just_moved: 刚走完一步到达此节点的玩家颜色。
        """
        self.state_hash = state_hash
        self.children: Dict[Move4, MCTSNode] = {}
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
        *,
        mcts_gives: Optional[Dict[Tuple[int, Move4], Tuple[bool, bool]]] = None,
    ) -> None:
        """惰性初始化 ``untried_moves``（仅首次调用时生成走法列表）。

        调用 ``Rules.get_pseudo_legal_moves`` 获取当前局面所有几何合法走法，
        随机打乱顺序以保证探索均匀性。后续通过 ``pop()`` 逐个消费。

        Args:
            board: 当前棋盘状态（必须与本节点对应的局面一致）。
            gives_check_cache: 可选；供 ``apply_pseudo_legal_with_rule_cache`` 的 LRU。
            mcts_gives: 可选；MCTS 专用将军判定 dict，见 ``mcts_fast_move_gives_check``。
        """
        if self.untried_moves is None:
            self.untried_moves = list(
                Rules.get_pseudo_legal_moves(board, board.current_player)
            )
            _order_untried_moves_policy(
                board, self.untried_moves, gives_check_cache, mcts_gives=mcts_gives,
            )

    def is_fully_expanded(self) -> bool:
        """判断该节点是否已完全展开（所有候选走法都已尝试过）。

        Returns:
            当 ``untried_moves`` 已初始化且为空列表时返回 True。
        """
        return self.untried_moves is not None and len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        """判断该节点是否为终局节点（完全展开且无任何子节点）。

        Returns:
            所有走法都已尝试但均不合法（例如被将死）时返回 True。
        """
        return (self.untried_moves is not None
                and len(self.untried_moves) == 0
                and len(self.children) == 0)

    def best_child_ucb(self, log_parent: float) -> Tuple[Move4, 'MCTSNode']:
        """UCB1-RAVE 混合选择：从已展开子节点中挑选"最值得探索"的一个。

        混合公式::

            score = (1 - β) × MCTS胜率 + β × RAVE胜率 + 探索项

        其中 β = rave_visits / (rave_visits + visits + RAVE_CONST)，
        随着节点真实访问量增大，β 自然衰减至 0，最终回归纯 MCTS 策略。

        **关键**：走法从字典键（边属性）读取，而非子节点自身属性，
        确保在多父节点 DAG 结构下不会取到属于其他路径的"幽灵棋"。

        Args:
            log_parent: 父节点 ``ln(N)`` 的预计算值，避免在循环内重复调用 log。

        Returns:
            ``(edge_move, child_node)`` 元组——边上的走法与对应的子节点。
        """
        best_move: Optional[Move4] = None
        best_node: Optional[MCTSNode] = None
        best_score = -1.0
        for move, ch in self.children.items():
            rv = ch.rave_visits
            v = ch.visits
            # 尚未被任何路径访问过的子节点，优先探索（相当于无穷大分数）
            if v + rv == 0:
                return move, ch

            # 纯 MCTS 胜率（利用项）
            mcts_val = ch.wins / v if v > 0 else 0.0
            # RAVE 胜率（AMAF 全局经验）
            rave_val = ch.rave_wins / rv if rv > 0 else 0.0
            # β 权重：RAVE 贡献随真实访问量增加而衰减
            beta = rv / (rv + v + _RAVE_CONST + 1e-5)
            # 混合利用值
            exploit = (1.0 - beta) * mcts_val + beta * rave_val
            # UCB 探索项
            explore = _UCB_C * math.sqrt(log_parent / (v + 1e-5))
            s = exploit + explore
            if s > best_score:
                best_score = s
                best_move = move
                best_node = ch
        assert best_move is not None and best_node is not None
        return best_move, best_node


# ═══════════════════════════════════════════════════════════════
#  单棵树搜索（供主进程直接调用 / 子进程 worker 调用）
# ═══════════════════════════════════════════════════════════════


def _run_single_mcts_tree(
    board: Board,
    max_simulations: int,
    time_limit: float,
    seed_offset: int = 0,
    root_move_history: Optional[List[MoveEntry]] = None,
) -> Dict[Move4, Dict[str, float]]:
    """在独立进程/线程中执行一棵完整的 MCTS-RAVE-DAG 搜索树。

    这是整个 MCTS 引擎的核心循环，执行标准的四阶段流程：
    选择 (Selection) → 展开 (Expansion) → 模拟 (Simulation) → 反向传播 (Backpropagation)。

    **置换表 DAG**：函数内部维护一个局部 Zobrist 置换表 ``tt``。当不同的走法序列
    到达相同的棋盘状态（Zobrist Hash 相同）时，直接复用已存在的节点实例，
    将搜索树转化为有向无环图 (DAG)，大幅节省内存并让不同路径共享统计信息。

    **边属性**：走法 (move) 通过 ``best_child_ucb`` 返回的元组 ``(edge_move, child_node)``
    读取（字典键），确保 ``board.apply_move`` 使用的走法与当前路径严格一致。

    Args:
        board: 棋盘副本（调用方必须保证独立拥有，函数结束后状态不变）。
        max_simulations: 该 worker 分配到的最大模拟次数。
        time_limit: 搜索时间上限（秒）。
        seed_offset: 随机种子偏移量，保证各 worker 的走法洗牌不同。

    Returns:
        根节点下每个子节点的统计字典：
        ``{move: {"visits": V, "wins": W, "rave_visits": RV, "rave_wins": RW}}``
    """
    random.seed(time.time_ns() + seed_offset)
    t0 = time.time()

    root_player = board.current_player
    opp_of_root = "black" if root_player == "red" else "red"
    root = MCTSNode(state_hash=board.zobrist_hash, player_just_moved=opp_of_root)
    post_apply_cache = PostApplyFlagsCache()
    gives_check_cache = MoveGivesCheckCache(post_apply_cache=post_apply_cache)
    mcts_gives: Dict[Tuple[int, Move4], Tuple[bool, bool]] = {}
    root.ensure_moves(board, gives_check_cache, mcts_gives=mcts_gives)

    # 局部置换表：Zobrist Hash → MCTSNode，用于 DAG 合并
    tt: Dict[int, MCTSNode] = {root.state_hash: root}
    # 走法栈：记录 Selection / Expansion 阶段 apply_move 的走法与被吃棋子，
    # 每轮迭代结束后按 LIFO 顺序 undo_move 还原棋盘
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
        path: List[MCTSNode] = [root]

        # ── 选择 (Selection) ──
        # 从根节点开始，沿 UCB1-RAVE 最优路径一直向下，
        # 直到遇到一个未完全展开的节点（还有 untried_moves）或终局节点。
        _sel_guard = 0
        while node.is_fully_expanded() and node.children:
            _sel_guard += 1
            if _sel_guard > _SELECTION_MAX_PLIES:
                break
            log_n = math.log(node.visits) if node.visits > 0 else 0.0
            # 从边（字典键）读取走法，而非子节点属性——DAG 安全
            edge_move, next_node = node.best_child_ucb(log_n)
            mover_sel = board.current_player
            captured = board.apply_move(*edge_move)
            _append_path_move_entry(entry_stack, board, edge_move, mover_sel)
            move_stack.append((edge_move, captured))
            path.append(next_node)
            node = next_node

        # ── 展开 (Expansion) ──
        # 从当前节点的 untried_moves 中弹出一个合法走法，创建（或复用）子节点。
        node.ensure_moves(board, gives_check_cache, mcts_gives=mcts_gives)
        expanded = False
        if node.untried_moves:
            result = _expand_one(
                board, node, move_stack, entry_stack, tt, gives_check_cache
            )
            if result is not None:
                _exp_move, child = result
                path.append(child)
                node = child
                expanded = True

        if not expanded and not node.children and node.visits > 0:
            # 所有走法均不合法（被将死），直接评估终局分数
            sim_result = _terminal_score(
                board, root_player, move_history=entry_stack
            )
            red_moves = set()
            black_moves = set()
        else:
            # ── 模拟 (Simulation / Rollout) ──
            # 从当前节点开始，进行一次轻量级随机对弈（伪合法走法 + 截断 + 兜底估分）。
            sim_result, red_moves, black_moves = _simulate(
                board,
                root_player,
                gives_check_cache=gives_check_cache,
                path_history=entry_stack,
                mcts_gives=mcts_gives,
            )

        # ── 反向传播 (Backpropagation) ──
        # 将模拟结果沿 path 从叶到根回传，同时更新 RAVE 统计。
        _backpropagate(path, sim_result, red_moves, black_moves)
        sims_done += 1

        # 还原棋盘：按 LIFO 顺序逐步 undo_move，恢复到根节点状态
        while move_stack:
            mv, cap = move_stack.pop()
            entry_stack.pop()
            board.undo_move(*mv, cap)

    # 收集根节点下所有子节点的统计数据，返回给调用方
    child_stats: Dict[Move4, Dict[str, float]] = {}
    for move, ch in root.children.items():
        child_stats[move] = {
            "visits": ch.visits,
            "wins": ch.wins,
            "rave_visits": ch.rave_visits,
            "rave_wins": ch.rave_wins,
        }
    return child_stats


# ── 搜索辅助函数（模块级，便于 pickle 序列化供多进程调用）──


def _expand_one(
    board: Board,
    node: MCTSNode,
    move_stack: List[Tuple[Move4, Any]],
    entry_stack: List[MoveEntry],
    tt: Dict[int, MCTSNode],
    gives_check_cache: MoveGivesCheckCache,
) -> Optional[Tuple[Move4, MCTSNode]]:
    """从当前节点的 ``untried_moves`` 中弹出一个合法走法，建立父子"边"。

    展开逻辑：
    1. 从 ``untried_moves`` 尾部 pop 出一个候选走法并 apply_move。
    2. 校验合法性（是否自将、是否王面对面），不合法则 undo 后继续尝试下一个。
    3. 合法后，计算子局面的 Zobrist Hash 并查询置换表 ``tt``：
       - **命中**：说明不同走法序列到达了相同的棋盘状态（Zobrist Hash 一致），
         直接复用已有节点，将搜索树转为 DAG，共享该节点的 visits / wins 统计。
       - **未命中**：实例化新节点并存入置换表。
    4. 通过 ``node.children[move] = child`` 建立父子边——走法是边的属性（字典键），
       而非子节点自身的属性，这在 DAG 中是必要的，否则会引发"幽灵棋"。

    Args:
        board: 当前棋盘（会被 apply_move 修改，调用方通过 move_stack 还原）。
        node: 待展开的父节点。
        move_stack: 走法栈，用于记录 apply_move 以便后续 undo。
        tt: 局部 Zobrist 置换表（同一棵搜索树内共享）。

    Returns:
        ``(edge_move, child_node)`` 或 ``None``（所有 untried_moves 均不合法时）。
    """
    mover = board.current_player
    while node.untried_moves:
        move = node.untried_moves.pop()
        applied = apply_pseudo_legal_with_rule_cache(
            board,
            move,
            mover,
            pre_move_cache=gives_check_cache,
            post_apply_cache=gives_check_cache.post_apply_cache,
        )
        if applied is None:
            continue
        captured, _ = applied
        move_stack.append((move, captured))
        _append_path_move_entry(entry_stack, board, move, mover)
        child_hash = board.zobrist_hash
        # 查置换表：不同走法序列可能到达相同的棋盘状态（Zobrist Hash 相同），
        # 此时直接复用已有节点，将搜索树转为有向无环图 (DAG)，
        # 大幅节省残局阶段的内存与计算开销。
        existing = tt.get(child_hash)
        if existing is not None:
            node.children[move] = existing
            return move, existing
        child = MCTSNode(
            state_hash=child_hash,
            player_just_moved=mover,
        )
        tt[child_hash] = child
        node.children[move] = child
        return move, child
    return None


def _simulate(
    board: Board,
    root_player: str,
    *,
    gives_check_cache: Optional[MoveGivesCheckCache] = None,
    path_history: Optional[List[MoveEntry]] = None,
    mcts_gives: Optional[Dict[Tuple[int, Move4], Tuple[bool, bool]]] = None,
) -> Tuple[float, Set[Move4], Set[Move4]]:
    """轻量级伪合法模拟（Lightweight Pseudo-legal Rollout）。

    为了榨取极致性能，模拟阶段跳过昂贵的完整合法性校验和终局检测
    （``Rules.winner()`` 内部需要生成全量合法走法，占 profile 90% 耗时），
    仅生成几何伪合法走法 (``get_pseudo_legal_moves``)，并用以下 O(1) 级别的
    快速检查作为兜底：

    1. **被吃将即判负**：每步开头检测对方老将是否处于被将军状态——
       若是，说明上一步行棋方未解将或主动送将，当前行棋方可"吃将"获胜。
    2. **一击必杀**：生成伪合法走法后，先扫描对方老将坐标是否可直达；
       若有可直接吃将的着法，立即执行并判胜。
    3. **启发式走子**：分级 forcing-first（将杀 / 将军 / 高价值吃子 / …），
       不做自将 / 白脸将检查。非法状态由下一步的"被吃将"检测兜底。

    同时记录双方尝试过的着法集合，供 RAVE 反向传播使用。

    Args:
        board: 当前棋盘（会被 copy 后在副本上模拟，不影响原盘）。
        root_player: MCTS 搜索的根节点行棋方（用于确定胜负视角）。
        gives_check_cache: 可选；与当前搜索树共用（apply 路径 LRU）。
        mcts_gives: 可选；rollout 内将军判定专用 dict。
        path_history: 到达当前节点前的 ``MoveEntry`` 链（与 ``board`` 一致）。

    Returns:
        ``(result, red_moves, black_moves)``——
        result 为 [0, 1] 的胜率分数（1.0 = root_player 获胜），
        red_moves / black_moves 为双方在模拟中尝试过的着法集合。
    """
    sim_board = board.copy()
    b_grid = sim_board.board
    rollout_limit = _dynamic_rollout_limit(sim_board.piece_count())
    red_moves: Set[Move4] = set()
    black_moves: Set[Move4] = set()
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

        # O(1) 终局检测：前一步行棋方若送将 / 未解将，
        # 对方老将此刻处于被将军状态 → 当前行棋方可"吃将"获胜
        if Rules.is_king_in_check(sim_board, opp):
            return (1.0 if cp == root_player else 0.0), red_moves, black_moves

        moves = list(Rules.get_pseudo_legal_moves(sim_board, cp))
        if not moves:
            return (0.0 if cp == root_player else 1.0), red_moves, black_moves

        # 一击必杀：扫描对方老将坐标，若有可直达的着法立即执行并判胜。
        # 因为 get_pseudo_legal_moves 不生成吃将着法，所以需要专门检测。
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

        # 启发式走子：policy 层 forcing-first，非法由下一步 O(1) 终局检测兜底
        chosen, _ = _pick_rollout_move_fast(
            sim_board,
            moves,
            b_grid,
            gives_check_cache=gives_check_cache,
            mcts_gives=mcts_gives,
        )
        # ``_pick_rollout_move_fast`` 内部已 ``apply_move``
        _append_path_move_entry(hist, sim_board, chosen, cp)
        if cp == "red":
            red_moves.add(chosen)
        else:
            black_moves.add(chosen)

    # 超过截断步数仍未分胜负，调用静态评估函数兜底
    return _eval_to_winrate(sim_board, root_player), red_moves, black_moves


def _find_king_capture(
    board: Board,
    attacker: str,
    b_grid,
    tkr: int,
    tkc: int,
) -> Optional[Move4]:
    """单目标可达性检测：扫描 ``attacker`` 方的所有棋子，判断是否有一步可直接吃到
    位于 ``(tkr, tkc)`` 的对方老将。

    仅做几何 + 蹩腿判定，不做完整合法性校验（模拟阶段专用）。
    之所以需要此函数，是因为 ``get_pseudo_legal_moves`` 出于安全考虑
    会过滤掉吃将着法，而模拟阶段需要快速检测"一击必杀"机会。

    Args:
        board: 当前棋盘。
        attacker: 攻击方颜色（"red" 或 "black"）。
        b_grid: ``board.board`` 二维数组的引用（避免重复属性访问）。
        tkr: 目标老将的行坐标。
        tkc: 目标老将的列坐标。

    Returns:
        可吃将的走法四元组，或 ``None``（无法一步吃到老将）。
    """
    for r, c in board.active_pieces[attacker]:
        p = b_grid[r][c]
        if p is None:
            continue
        pt = p.piece_type

        # 车：同行或同列，中间无阻隔即可直吃
        if pt == "che":
            if r == tkr and c != tkc:
                lo, hi = (c + 1, tkc) if c < tkc else (tkc + 1, c)
                if not any(b_grid[r][x] is not None for x in range(lo, hi)):
                    return (r, c, tkr, tkc)
            elif c == tkc and r != tkr:
                lo, hi = (r + 1, tkr) if r < tkr else (tkr + 1, r)
                if not any(b_grid[x][c] is not None for x in range(lo, hi)):
                    return (r, c, tkr, tkc)

        # 炮：同行或同列，中间恰好有一个炮架即可翻山吃
        elif pt == "pao":
            if r == tkr and c != tkc:
                lo, hi = (c + 1, tkc) if c < tkc else (tkc + 1, c)
                if sum(1 for x in range(lo, hi) if b_grid[r][x] is not None) == 1:
                    return (r, c, tkr, tkc)
            elif c == tkc and r != tkr:
                lo, hi = (r + 1, tkr) if r < tkr else (tkr + 1, r)
                if sum(1 for x in range(lo, hi) if b_grid[x][c] is not None) == 1:
                    return (r, c, tkr, tkc)

        # 马：日字跳跃，需检查蹩马腿
        elif pt == "ma":
            dr, dc = tkr - r, tkc - c
            if (abs(dr), abs(dc)) in ((1, 2), (2, 1)):
                lr, lc = Rules._ma_leg_square(r, c, tkr, tkc)
                if 0 <= lr < 10 and 0 <= lc < 9 and b_grid[lr][lc] is None:
                    return (r, c, tkr, tkc)

        # 兵/卒：前进一步或过河后横移一步
        elif pt == "bing":
            if attacker == "red":
                if (tkr == r - 1 and tkc == c) or (r <= 4 and tkr == r and abs(tkc - c) == 1):
                    return (r, c, tkr, tkc)
            else:
                if (tkr == r + 1 and tkc == c) or (r >= 5 and tkr == r and abs(tkc - c) == 1):
                    return (r, c, tkr, tkc)

    return None


def _pick_rollout_move_fast(
    sim_board: Board,
    moves: List[Move4],
    b_grid,
    *,
    gives_check_cache: Optional[MoveGivesCheckCache] = None,
    mcts_gives: Optional[Dict[Tuple[int, Move4], Tuple[bool, bool]]] = None,
) -> Tuple[Move4, bool]:
    """分级 forcing-first rollout：将 / 将军 / 高价值吃子 / 吃子 / 推进 / 随机。

    吃子按子力分档不调用将军判定；仅对**非吃子**着法调用 ``mcts_fast`` / LRU，
    减少 rollout 内 ``_move_gives_check`` 次数。不调用 ``Evaluation.evaluate``。
    返回 ``(move, 是否为吃子)``，后者供 ``mcts_minimax`` 模块中 probe 触发等逻辑使用。
    """
    cp = sim_board.current_player
    pv = Evaluation.PIECE_VALUES
    checking: List[Move4] = []
    hv_caps: List[Move4] = []
    lv_caps: List[Move4] = []
    aggressive: List[Move4] = []

    for m in moves:
        victim = b_grid[m[2]][m[3]]
        if victim is not None and victim.piece_type == "jiang":
            was_cap = True
            sim_board.apply_move(*m)
            return m, was_cap
        if victim is not None:
            if pv.get(victim.piece_type, 0) >= _POLICY_HVCAP_VALUE:
                hv_caps.append(m)
            else:
                lv_caps.append(m)
            continue
        if mcts_gives is not None:
            if mcts_fast_move_gives_check(sim_board, m, cp, mcts_gives):
                checking.append(m)
            elif _is_aggressive_push(sim_board, cp, m, b_grid):
                aggressive.append(m)
        else:
            if _move_gives_check(sim_board, m, cp, gives_check_cache):
                checking.append(m)
            elif _is_aggressive_push(sim_board, cp, m, b_grid):
                aggressive.append(m)

    m_pick: Optional[Move4] = None
    if checking and random.random() < _ROLL_PREFER_CHECK:
        m_pick = random.choice(checking)
    elif hv_caps and random.random() < _ROLL_PREFER_HVCAP:
        m_pick = random.choice(hv_caps)
    elif (hv_caps or lv_caps) and random.random() < _ROLL_PREFER_ANY_CAPTURE:
        m_pick = random.choice(hv_caps + lv_caps)
    elif aggressive and random.random() < _ROLL_PREFER_AGGRESSIVE:
        m_pick = random.choice(aggressive)
    if m_pick is None:
        m_pick = random.choice(moves)
    was_capture = b_grid[m_pick[2]][m_pick[3]] is not None
    sim_board.apply_move(*m_pick)
    return m_pick, was_capture


def _eval_to_winrate(board: Board, root_player: str) -> float:
    """将静态评估分数通过 Sigmoid 函数映射到 [0, 1] 胜率区间。

    当模拟阶段达到截断步数仍未分出胜负时，调用此函数作为兜底估分。
    使用 ``1 / (1 + exp(-score / SCALE))`` 将 Evaluation 的线性分数
    平滑压缩到概率空间。

    Args:
        board: 当前棋盘状态。
        root_player: MCTS 根节点的行棋方（决定分数正负方向）。

    Returns:
        [0, 1] 的胜率估计值（1.0 = root_player 完全获胜）。
    """
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
    """评估已确认的终局局面（将死 / 困毙 / 长将判负等）的胜率分。"""
    w = Rules.winner(board, move_history)
    if w == root_player:
        return 1.0
    if w is not None:
        return 0.0
    return 0.5


def _backpropagate(
    path: List[MCTSNode],
    result: float,
    red_moves: Set[Move4],
    black_moves: Set[Move4],
) -> None:
    """反向传播：将模拟结果沿访问路径从叶到根回传，同时更新 RAVE 统计。

    **常规更新**：路径上每个节点的 visits +1，wins 根据当前视角累加 score。
    每经过一层，score 翻转为 ``1 - score``（因为父子节点的行棋方互为对手）。

    **RAVE/AMAF 更新**：对路径上每个节点的所有已展开子节点，如果该子节点的
    边走法 (字典键) 出现在模拟阶段的对应颜色着法集合中，就给它累加
    rave_visits / rave_wins——即使该子节点本身在本次迭代中没有被选中和访问。
    这就是 AMAF (All-Moves-As-First) 的核心思想：模拟中出现的好着法，
    会"提前"积累到对应的树节点上，加速搜索早期的收敛。

    走法匹配使用字典键（边属性），确保 DAG 结构下 RAVE 更新正确。

    Args:
        path: 本次迭代的访问路径（从根到叶的节点列表）。
        result: 模拟结果（0.0 ~ 1.0，从 root_player 视角）。
        red_moves: 模拟阶段红方尝试过的着法集合。
        black_moves: 模拟阶段黑方尝试过的着法集合。
    """
    score = result
    for i in range(len(path) - 1, -1, -1):
        node = path[i]
        node.visits += 1
        if node.player_just_moved is not None:
            node.wins += score

        # RAVE 更新：遍历当前节点的所有已展开子节点（通过边/字典键），
        # 如果边走法在模拟着法集合中出现过，就给子节点累加 RAVE 统计
        for move, child in node.children.items():
            pjm = child.player_just_moved
            if pjm == "red" and move in red_moves:
                child.rave_visits += 1
                child.rave_wins += score
            elif pjm == "black" and move in black_moves:
                child.rave_visits += 1
                child.rave_wins += score

        # 每经过一层，分数翻转（父子节点行棋方互为对手）
        score = 1.0 - score


# ═══════════════════════════════════════════════════════════════
#  MCTSAI —— 公开接口（对外与 MinimaxAI 一致）
# ═══════════════════════════════════════════════════════════════


class MCTSAI:
    """中国象棋蒙特卡洛树搜索 AI（支持多进程根节点并行 + RAVE + DAG）。

    对外接口与 ``MinimaxAI`` 完全一致：``get_best_move`` / ``choose_move``，
    可无缝接入现有的 Controller 和 GUI 框架。

    **搜索流程**：
    1. 先查询开局库（Zobrist 哈希 + 镜像回退），命中则瞬间返回。
    2. 未命中时启动正式 MCTS 搜索：若 workers > 1 则多进程并行，
       每个 worker 独立建树，搜索结束后在主进程合并统计。
    3. 最终选择访问次数最多的根子节点对应的走法（最稳健策略）。

    Args:
        max_simulations: 所有 worker 合计的最大模拟次数。
        time_limit: 搜索时间上限（秒），与 ``max_simulations`` 取先到者。
        workers: 并行进程数（0 或 1 = 单进程，>1 = 多进程根并行）。
            默认 ``None`` 表示自动检测 ``min(8, cpu_count)``。
        verbose: 搜索结束后是否打印统计信息到控制台。
    """

    def __init__(
        self,
        max_simulations: int = 5000,
        time_limit: float = 10.0,
        workers: Optional[int] = None,
        verbose: bool = True,
    ):
        """初始化 MCTS AI 实例。

        Args:
            max_simulations: 每次搜索的总模拟次数上限。
            time_limit: 搜索时间上限（秒）。
            workers: 并行 worker 数。None 表示自动取 ``min(8, CPU 逻辑核心数)``。
            verbose: 是否打印搜索统计。
        """
        self.max_simulations = max_simulations
        self.time_limit = time_limit
        self.verbose = verbose
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
        """Searcher 统一接口：为当前行棋方选择一步最佳走法。

        Args:
            board: 当前棋盘状态。
            time_limit: 搜索时间上限覆盖（可选）。
            game_history: 从开局到当前的 Zobrist 哈希历史列表（用于开局库查询）。

        Returns:
            最佳走法四元组，或 ``None``（无合法走法）。
        """
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
        """执行 MCTS-RAVE 搜索并返回最佳走法。

        搜索前先查询开局库；命中时瞬间返回，``last_stats`` 中标注
        ``opening_book=True`` 与 ``win_rate="开局库"``。

        Args:
            board: 当前棋盘状态。
            game_history: Zobrist 哈希历史（用于开局库查询与长将检测）。
            time_limit: 搜索时间上限覆盖。
            max_simulations: 模拟次数上限覆盖。

        Returns:
            最佳走法四元组，或 ``None``（无合法走法）。
        """
        hash_history: List[int] = [] if game_history is None else list(game_history)
        root_mh = list(move_history) if move_history is not None else None

        # ── 开局库拦截 ──
        if len(hash_history) < 30:
            book_move = self._probe_opening_book(board, hash_history)
            if book_move is not None:
                self.simulations_run = 0
                self.last_stats = {
                    "time_taken": 0.0,
                    "simulations": 0,
                    "workers": _parallel_workers_when_safe(self.workers),
                    "win_rate": "开局库",
                    "opening_book": True,
                }
                if self.verbose:
                    print(f"命中开局库！瞬间出棋: {book_move}")
                return book_move

        # ── MCTS-RAVE 正式搜索 ──
        tl = time_limit if time_limit is not None else self.time_limit
        ms = max_simulations if max_simulations is not None else self.max_simulations
        t0 = time.time()

        effective_workers = _parallel_workers_when_safe(self.workers)
        # 模拟次数太少时，进程启动开销大于收益，退化为单进程
        if effective_workers <= 1 or ms < effective_workers * 10:
            merged = _run_single_mcts_tree(
                board, ms, tl, seed_offset=0, root_move_history=root_mh
            )
            total_sims = sum(int(s["visits"]) for s in merged.values())
        else:
            sims_per_worker = ms // effective_workers
            remainder = ms % effective_workers
            merged = self._parallel_search(
                board,
                tl,
                sims_per_worker,
                remainder,
                effective_workers,
                root_move_history=root_mh,
            )
            total_sims = sum(int(s["visits"]) for s in merged.values())

        elapsed = time.time() - t0
        self.simulations_run = total_sims

        # 主键：visits / wins；近优候选上叠加极小 policy_attack_bias（不压过统计）
        root_player = board.current_player
        v_max = max((int(st["visits"]) for st in merged.values()), default=0)
        best_move: Optional[Move4] = None
        best_key = -1.0
        best_visits = -1
        best_wr = 0.0
        root_bias_cache: Dict[Tuple[int, Move4], Tuple[bool, bool]] = {}
        for mv, st in merged.items():
            v = int(st["visits"])
            w = float(st["wins"])
            wr = w / v if v > 0 else 0.0
            bias = _policy_attack_bias(
                board, root_player, mv, mcts_gives=root_bias_cache,
            )
            close = min(
                1.0,
                v / max(v_max * _ROOT_VISITS_TIE_FRAC, 1e-6),
            )
            key = v * 1_000_000.0 + w * 1_000.0 + bias * _ROOT_BIAS_SCALE * close
            if key > best_key:
                best_key = key
                best_move = mv
                best_visits = v
                best_wr = wr

        self.last_stats = {
            "time_taken": elapsed,
            "simulations": total_sims,
            "workers": effective_workers,
            "win_rate": f"{best_wr * 100:.1f}%",
        }
        if self.verbose:
            print(
                f"MCTS 搜索完成，总模拟次数: {total_sims}  （并行进程数: {effective_workers}）"
            )
            print(f"搜索耗时 (秒): {elapsed:.3f}")
            if best_move is not None:
                print(f"最佳走法: {best_move}  胜率: {best_wr:.1%}  访问: {best_visits}")
        return best_move

    @staticmethod
    def _probe_opening_book(
        board: Board, move_history: List[int],
    ) -> Optional[Move4]:
        """查询开局库，含列镜像回退机制。

        处理对称棋局时，先用当前局面的 Zobrist 哈希直接查表；
        若未命中，则将棋盘沿中轴线（第 4 列）左右镜像后再查一次，
        确保对称开局（如左马 vs 右马）都能命中开局库。
        找到的镜像着法会通过 ``mirror_move`` 还原到实际坐标。

        Args:
            board: 当前棋盘。
            move_history: Zobrist 哈希历史列表（长度用于判断是否处于开局阶段）。

        Returns:
            开局库推荐的走法，或 ``None``（未命中）。
        """
        zkey = board.zobrist_hash
        book_moves = OPENING_BOOK.get(zkey)
        if book_moves is None:
            # 镜像回退：将棋盘沿中轴线翻转后再查表，
            # 确保对侧的坐标（行与列）能够正确映射，
            # 例如右马对应同侧卒的制约关系
            zm = board.column_mirror_copy().zobrist_hash
            alt = OPENING_BOOK.get(zm)
            if alt is not None:
                book_moves = [mirror_move(m) for m in alt]
        if book_moves is None and len(move_history) == 0:
            keys = list(OPENING_BOOK.keys())
            disp = keys if len(keys) <= 48 else keys[:24] + ["..."] + keys[-16:]
            print(
                f"[MCTS 开局库] 根局面未命中（局面键 zkey={zkey:#x}）；"
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
        sims_per_worker: int,
        remainder: int,
        num_workers: int,
        root_move_history: Optional[List[MoveEntry]] = None,
    ) -> Dict[Move4, Dict[str, float]]:
        """多进程根节点并行搜索并合并结果。

        每个 worker 进程获得棋盘的独立副本（``board.copy()``），在其上独立
        建树搜索。搜索结束后，主进程将相同走法的 visits / wins / rave 统计
        逐项相加，等效于一棵更大的搜索树。

        Args:
            board: 当前棋盘（会被 copy 后传给各 worker，自身不被修改）。
            time_limit: 每个 worker 的时间上限。
            sims_per_worker: 每个 worker 的基础模拟次数。
            remainder: 余数模拟次数（前 remainder 个 worker 各多分 1 次）。
            num_workers: worker 进程数。

        Returns:
            合并后的根子节点统计字典（含 RAVE 数据）。
        """
        futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
            for i in range(num_workers):
                worker_sims = sims_per_worker + (1 if i < remainder else 0)
                board_copy = board.copy()
                futures.append(
                    pool.submit(
                        _run_single_mcts_tree,
                        board_copy,
                        worker_sims,
                        time_limit,
                        i * 1000,
                        root_move_history,
                    )
                )
            results = [f.result() for f in futures]

        # 合并各棵树的子节点统计：相同走法的 visits / wins / rave 逐项相加
        merged: Dict[Move4, Dict[str, float]] = {}
        for child_stats in results:
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
        return merged
