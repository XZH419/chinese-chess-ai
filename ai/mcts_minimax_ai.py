"""MCTS-Minimax 混合搜索引擎。

本模块在 MCTS 框架中引入 Minimax 的部分能力，目标是在保持 MCTS 全局探索能力的
同时，提高对“明显战术着法”（将军/吃子/推进等）的利用效率。

实现要点（与代码行为一致的口径）：

- **共享走法偏置与排序策略**：复用 `mcts.py` 中的轻量 policy（吃子/将军/推进偏置），
  作为根节点/展开阶段的倾向性排序信号。
- **评估函数参与**：在部分阶段会调用 `Evaluation.evaluate()` 作为局面强弱的静态信号，
  用于在模拟截断或预算受限时提供更稳定的回传结果（具体使用点以实现为准）。
- **DAG + RAVE**：节点以 `zobrist_hash` 去重合并为 DAG，并维护 RAVE/AMAF 统计，
  用于加速早期收敛；走法作为父节点 `children` 的字典键（边属性）存储以保证 DAG 安全。

注意：本引擎仍可能在 simulation/rollout 阶段使用伪合法走法生成以提高吞吐量，
完整合法性由规则引擎在关键路径上兜底。
"""
from __future__ import annotations

import concurrent.futures
import math
import multiprocessing
import os
import random
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
from ai.mcts_common import (
    Move4,
    _append_path_move_entry,
    _policy_attack_bias,
    _order_untried_moves_policy,
    _pick_rollout_move_fast,
    _move_gives_check,
    _is_aggressive_push,
    _POLICY_HVCAP_VALUE,
    _ROOT_BIAS_SCALE,
    _ROOT_VISITS_TIE_FRAC,
    _SELECTION_MAX_PLIES,
    _parallel_workers_when_safe,
)

# ═══════════════════════════════════════════════════════════════
#  MCTS 常量（沿用 mcts.py）
# ═══════════════════════════════════════════════════════════════

_UCB_C = 1.414          # UCB1 探索常数 c ≈ √2
_SCORE_SCALE = 600.0    # 评估分 → sigmoid 胜率的缩放因子
_RAVE_CONST = 300       # RAVE 等效常数 k：β 衰减速度

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
        *,
        mcts_gives: Optional[Dict[Tuple[int, Move4], Tuple[bool, bool]]] = None,
    ) -> None:
        """惰性初始化 untried_moves。"""
        if self.untried_moves is None:
            self.untried_moves = list(
                Rules.get_pseudo_legal_moves(board, board.current_player)
            )
            _order_untried_moves_policy(
                board, self.untried_moves, gives_check_cache, mcts_gives=mcts_gives,
            )

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


def _mvv_lva_pick(board: Board, moves: List[Move4]) -> Move4:
    """极速 MVV-LVA：仅在伪合法 moves 中挑“划算”的吃子，否则随机。

    注意：该函数仅用于 MCTS-Minimax 的模拟阶段（rollout），不做合法性/自将校验。
    """
    # 项目内部 piece_type 采用：jiang/che/pao/ma/xiang/shi/bing
    pv = {
        "king": 10000,  # 兼容提示词
        "jiang": 10000,
        "che": 900,
        "pao": 450,
        "ma": 400,
        "xiang": 200,
        "shi": 200,
        "bing": 100,
    }
    b = board.board
    best: Optional[Move4] = None
    best_score = 0
    for m in moves:
        victim = b[m[2]][m[3]]
        if victim is None:
            continue
        attacker = b[m[0]][m[1]]
        if attacker is None:
            continue
        score = int(pv.get(victim.piece_type, 0)) * 10 - int(pv.get(attacker.piece_type, 0))
        if score > best_score:
            best_score = score
            best = m
    if best is not None and best_score > 0:
        return best
    return random.choice(moves)


def _rollout_pick_instinct_or_random(board: Board, moves: List[Move4]) -> Tuple[Move4, bool]:
    """模拟阶段走子：50% MVV-LVA 战术本能，50% 纯随机；并执行 apply_move。"""
    b = board.board
    if random.random() < 0.5:
        m = _mvv_lva_pick(board, moves)
    else:
        m = random.choice(moves)
    was_capture = b[m[2]][m[3]] is not None
    board.apply_move(*m)
    return m, was_capture


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
    *,
    path_history: Optional[List[MoveEntry]] = None,
) -> Tuple[float, Set[Move4], Set[Move4]]:
    """极致轻量 rollout：MVV-LVA 战术本能 + 截断后静态评估（O(1)）。"""
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

        chosen, _is_last_capture = _rollout_pick_instinct_or_random(sim_board, moves)
        _append_path_move_entry(hist, sim_board, chosen, cp)
        if cp == "red":
            red_moves.add(chosen)
        else:
            black_moves.add(chosen)

    # ── Rollout 截断：仅一次静态评估（O(1) 兜底），不再 probe ──
    result = _eval_to_winrate(sim_board, root_player)
    return result, red_moves, black_moves


def _expand_one(
    board: Board,
    node: MCTSMinimaxNode,
    move_stack: List[Tuple[Move4, Any]],
    entry_stack: List[MoveEntry],
    tt: Dict[int, MCTSMinimaxNode],
    root_player: str,
    *,
    pre_move_cache: MoveGivesCheckCache,
    post_apply_cache: PostApplyFlagsCache,
) -> Optional[Tuple[Move4, MCTSMinimaxNode]]:
    """从 untried_moves 弹出一个合法走法并建立父子边（含 DAG 合并）。"""
    mover = board.current_player
    while node.untried_moves:
        move = node.untried_moves.pop()
        applied = apply_pseudo_legal_with_rule_cache(
            board,
            move,
            mover,
            pre_move_cache=pre_move_cache,
            post_apply_cache=post_apply_cache,
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
        # Prior Score Injection：扩展即注入静态先验，让 UCB 初期不再盲扫
        child.visits = 1
        child.wins = _eval_to_winrate(board, root_player)
        tt[child_hash] = child
        node.children[move] = child
        return move, child
    return None


def _merge_child_stats(
    results: list,
) -> Dict[Move4, Dict[str, float]]:
    """合并多棵树的根子节点统计。"""
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


# ═══════════════════════════════════════════════════════════════
#  单棵 MCTS-Minimax 树搜索（worker 主函数，模块级以支持 pickle）
# ═══════════════════════════════════════════════════════════════


def _run_single_mcts_minimax_tree(
    board: Board,
    max_simulations: int,
    time_limit: float,
    seed_offset: int = 0,
    root_move_history: Optional[List[MoveEntry]] = None,
) -> Dict[Move4, Dict[str, float]]:
    """在独立进程中执行一棵完整的 MCTS-Minimax 搜索树。

    四阶段流程与 ``mcts.py`` 一致：Selection → Expansion → Simulation/Probe
    → Backpropagation。Simulation 阶段为 MVV-LVA 战术 rollout + O(1) 静态评估截断。

    Returns:
        根子节点统计 ``child_stats``。
    """
    # 临时观测日志：用于确认 Windows 下并行 worker 确实启动。
    if os.environ.get("CHESSAI_PARALLEL_LOG") == "1":
        print(
            f"[PID {os.getpid()}] Parallel worker started for MCTS-Minimax "
            f"(sims={max_simulations}, time_limit={time_limit})"
        )
    random.seed(time.time_ns() + seed_offset)
    t0 = time.time()

    post_apply_cache = PostApplyFlagsCache(65536)
    gives_check_cache = MoveGivesCheckCache(131072, post_apply_cache=post_apply_cache)
    mcts_gives: Dict[Tuple[int, Move4], Tuple[bool, bool]] = {}

    # 同 `ai.mcts_ai`：为避免 apply/undo 栈在极端伪合法路径下破坏可逆性，
    # 每次迭代使用独立棋盘副本。
    root_board = board

    root_player = root_board.current_player
    opp_of_root = "black" if root_player == "red" else "red"
    root = MCTSMinimaxNode(state_hash=root_board.zobrist_hash, player_just_moved=opp_of_root)
    root.ensure_moves(root_board, gives_check_cache, mcts_gives=mcts_gives)

    tt: Dict[int, MCTSMinimaxNode] = {root.state_hash: root}
    sims_done = 0

    while sims_done < max_simulations:
        if time.time() - t0 >= time_limit:
            break
        board = root_board.copy()
        move_stack: List[Tuple[Move4, Any]] = []

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
        _sel_guard = 0
        while node.is_fully_expanded() and node.children:
            _sel_guard += 1
            if _sel_guard > _SELECTION_MAX_PLIES:
                break
            log_n = math.log(node.visits) if node.visits > 0 else 0.0
            edge_move, next_node = node.best_child_ucb(log_n)
            mover_sel = board.current_player
            captured = board.apply_move(*edge_move)
            _append_path_move_entry(entry_stack, board, edge_move, mover_sel)
            move_stack.append((edge_move, captured))
            path.append(next_node)
            node = next_node

        # ── Expansion ──
        node.ensure_moves(board, gives_check_cache, mcts_gives=mcts_gives)
        expanded = False
        if node.untried_moves:
            result = _expand_one(
                board,
                node,
                move_stack,
                entry_stack,
                tt,
                root_player,
                pre_move_cache=gives_check_cache,
                post_apply_cache=post_apply_cache,
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
                path_history=entry_stack,
                board=board,
                root_player=root_player,
            )

        # ── Backpropagation ──
        _backpropagate(path, sim_result, red_moves, black_moves)
        sims_done += 1
        # 本轮使用的 board 为独立副本，无需 undo 回滚。

    # 收集根子节点统计
    child_stats: Dict[Move4, Dict[str, float]] = {}
    for move, ch in root.children.items():
        child_stats[move] = {
            "visits": ch.visits,
            "wins": ch.wins,
            "rave_visits": ch.rave_visits,
            "rave_wins": ch.rave_wins,
        }
    return child_stats


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
        verbose: 搜索完成后是否输出统计信息。
    """

    def __init__(
        self,
        max_simulations: int = 4000,
        time_limit: float = 10.0,
        workers: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ):
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
                    "workers": _parallel_workers_when_safe(self.workers),
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

        effective_workers = _parallel_workers_when_safe(self.workers)
        if effective_workers <= 1 or ms < effective_workers * 10:
            child_stats = _run_single_mcts_minimax_tree(
                board, ms, tl, seed_offset=0, root_move_history=root_mh,
            )
            total_sims = sum(int(s["visits"]) for s in child_stats.values())
        else:
            child_stats = self._parallel_search(
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
        root_bias_cache: Dict[Tuple[int, Move4], Tuple[bool, bool]] = {}
        for mv, st in child_stats.items():
            v = int(st["visits"])
            w = float(st["wins"])
            wr = w / v if v > 0 else 0.0
            bias = _policy_attack_bias(
                board, root_player, mv, mcts_gives=root_bias_cache,
            )
            close = min(1.0, v / max(v_max * _ROOT_VISITS_TIE_FRAC, 1e-6))
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
            # 与 Minimax 统计字段对齐，供基准脚本 / 通用日志读取
            "nodes_evaluated": total_sims,
        }
        if self.verbose:
            print(
                f"[MCTS-Minimax] 搜索完成: 模拟次数 {total_sims}，"
                f"并行进程数 {effective_workers}，耗时 {elapsed:.3f} 秒"
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
    ) -> Dict[Move4, Dict[str, float]]:
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
