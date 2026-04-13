"""蒙特卡洛树搜索（MCTS）AI 引擎。

集成技术：UCB1 选择、惰性走法展开、apply/undo 状态回溯（零拷贝树遍历）、
动态截断启发模拟（Heavy Playout + Evaluation 兜底）、__slots__ 节点内存优化、
多进程根节点并行（Root Parallelism）。
"""

from __future__ import annotations

import concurrent.futures
import math
import multiprocessing
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from chinese_chess.model.board import Board
from chinese_chess.model.rules import Rules

from .evaluation import Evaluation

_CAPTURE_PROB = 0.80
_UCB_C = 1.414
_SCORE_SCALE = 600.0


def _dynamic_rollout_limit(piece_count: int) -> int:
    """根据当前子力数量动态决定模拟截断步数。"""
    if piece_count > 24:
        return 20
    if piece_count >= 10:
        return 35
    return 50


# ═══════════════════════════════════════════════════════════════
#  MCTSNode
# ═══════════════════════════════════════════════════════════════


class MCTSNode:
    """MCTS 搜索树节点。

    使用 ``__slots__`` 降低海量节点的内存开销。``untried_moves`` 惰性初始化：
    仅在首次展开时调用走法生成器，后续通过 ``pop()`` 逐个消费。
    """

    __slots__ = [
        "state_hash",
        "parent",
        "move",
        "children",
        "visits",
        "wins",
        "untried_moves",
        "player_just_moved",
    ]

    def __init__(
        self,
        state_hash: int,
        player_just_moved: str,
        parent: Optional[MCTSNode] = None,
        move: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.state_hash = state_hash
        self.parent = parent
        self.move = move
        self.children: List[MCTSNode] = []
        self.visits: int = 0
        self.wins: float = 0.0
        self.untried_moves: Optional[List[Tuple[int, int, int, int]]] = None
        self.player_just_moved = player_just_moved

    def ensure_moves(self, board: Board) -> None:
        """惰性初始化 ``untried_moves``（仅首次调用时生成走法列表）。"""
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

    def best_child_ucb(self, log_parent: float) -> MCTSNode:
        """UCB1 选择：``log_parent`` 为父节点 ``ln(N)`` 的预计算值。"""
        best: Optional[MCTSNode] = None
        best_score = -1.0
        for ch in self.children:
            if ch.visits == 0:
                return ch
            exploit = ch.wins / ch.visits
            explore = _UCB_C * math.sqrt(log_parent / ch.visits)
            s = exploit + explore
            if s > best_score:
                best_score = s
                best = ch
        assert best is not None
        return best

    def best_child_robust(self) -> Optional[MCTSNode]:
        """搜索结束后选择访问次数最多的子节点（最稳健策略）。"""
        if not self.children:
            return None
        return max(self.children, key=lambda ch: ch.visits)


# ═══════════════════════════════════════════════════════════════
#  单棵树搜索（供主进程直接调用 / 子进程 worker 调用）
# ═══════════════════════════════════════════════════════════════


def _run_single_mcts_tree(
    board: Board,
    max_simulations: int,
    time_limit: float,
    seed_offset: int = 0,
) -> Dict[Tuple[int, int, int, int], Dict[str, float]]:
    """在独立进程/线程中执行一棵完整的 MCTS 搜索树。

    Args:
        board: 棋盘副本（调用方必须保证独立拥有，函数结束后状态不变）。
        max_simulations: 该 worker 分配到的最大模拟次数。
        time_limit: 搜索时间上限（秒）。
        seed_offset: 随机种子偏移，保证各 worker 的走法洗牌不同。

    Returns:
        ``{move: {"visits": V, "wins": W}}``——根节点下每个子节点的统计。
    """
    random.seed(time.time_ns() + seed_offset)
    t0 = time.time()

    root_player = board.current_player
    opp_of_root = "black" if root_player == "red" else "red"
    root = MCTSNode(state_hash=board.zobrist_hash, player_just_moved=opp_of_root)
    root.ensure_moves(board)

    move_stack: List[Tuple[Tuple[int, int, int, int], Any]] = []
    sims_done = 0

    while sims_done < max_simulations:
        if time.time() - t0 >= time_limit:
            break

        node = root

        # ── Selection ──
        while node.is_fully_expanded() and node.children:
            log_n = math.log(node.visits) if node.visits > 0 else 0.0
            node = node.best_child_ucb(log_n)
            captured = board.apply_move(*node.move)
            move_stack.append((node.move, captured))

        # ── Expansion ──
        node.ensure_moves(board)
        expanded = False
        if node.untried_moves:
            child = _expand_one(board, node, move_stack)
            if child is not None:
                node = child
                expanded = True

        if not expanded and not node.children and node.visits > 0:
            result = _terminal_score(board, root_player)
        else:
            # ── Simulation ──
            result = _simulate(board, root_player)

        # ── Backpropagation ──
        _backpropagate(node, result)
        sims_done += 1

        while move_stack:
            mv, cap = move_stack.pop()
            board.undo_move(*mv, cap)

    # 收集根节点子节点统计
    child_stats: Dict[Tuple[int, int, int, int], Dict[str, float]] = {}
    for ch in root.children:
        if ch.move is not None:
            child_stats[ch.move] = {"visits": ch.visits, "wins": ch.wins}
    return child_stats


# ── 搜索辅助函数（模块级，便于 pickle 序列化）──


def _expand_one(
    board: Board,
    node: MCTSNode,
    move_stack: List[Tuple[Tuple[int, int, int, int], Any]],
) -> Optional[MCTSNode]:
    """从 ``node.untried_moves`` 弹出一个合法走法并创建子节点。"""
    mover = board.current_player
    while node.untried_moves:
        move = node.untried_moves.pop()
        captured = board.apply_move(*move)
        if Rules.is_king_in_check(board, mover) or Rules._jiang_face_to_face(board):
            board.undo_move(*move, captured)
            continue
        move_stack.append((move, captured))
        child = MCTSNode(
            state_hash=board.zobrist_hash,
            player_just_moved=mover,
            parent=node,
            move=move,
        )
        node.children.append(child)
        return child
    return None


def _simulate(board: Board, root_player: str) -> float:
    """动态截断启发模拟（Heavy Playout）。

    截断步数根据子力数量自适应：繁杂局面截断更早，残局允许更深模拟。
    """
    sim_board = board.copy()
    b_grid = sim_board.board
    rollout_limit = _dynamic_rollout_limit(sim_board.piece_count())

    for _ in range(rollout_limit):
        cp = sim_board.current_player
        w = Rules.winner(sim_board)
        if w is not None:
            return 1.0 if w == root_player else 0.0

        moves = list(Rules.get_pseudo_legal_moves(sim_board, cp))
        if not moves:
            opp = "black" if cp == "red" else "red"
            return 1.0 if opp == root_player else 0.0

        if not _pick_rollout_move(sim_board, moves, cp, b_grid):
            break

    return _eval_to_winrate(sim_board, root_player)


def _pick_rollout_move(
    sim_board: Board,
    moves: List[Tuple[int, int, int, int]],
    mover: str,
    b_grid,
) -> bool:
    """启发式走子：优先吃子，跳过自将。"""
    captures = [m for m in moves if b_grid[m[2]][m[3]] is not None]
    use_captures = captures and random.random() < _CAPTURE_PROB
    pool = captures if use_captures else moves
    random.shuffle(pool)
    for m in pool:
        cap = sim_board.apply_move(*m)
        if Rules.is_king_in_check(sim_board, mover) or Rules._jiang_face_to_face(sim_board):
            sim_board.undo_move(*m, cap)
            continue
        return True
    if use_captures:
        random.shuffle(moves)
        for m in moves:
            if b_grid[m[2]][m[3]] is not None:
                continue
            cap = sim_board.apply_move(*m)
            if Rules.is_king_in_check(sim_board, mover) or Rules._jiang_face_to_face(sim_board):
                sim_board.undo_move(*m, cap)
                continue
            return True
    return False


def _eval_to_winrate(board: Board, root_player: str) -> float:
    """Evaluation 分数 → [0, 1] 胜率。"""
    raw = Evaluation.evaluate(board)
    if board.current_player != root_player:
        raw = -raw
    return 1.0 / (1.0 + math.exp(-raw / _SCORE_SCALE))


def _terminal_score(board: Board, root_player: str) -> float:
    w = Rules.winner(board)
    if w == root_player:
        return 1.0
    if w is not None:
        return 0.0
    return 0.5


def _backpropagate(node: MCTSNode, result: float) -> None:
    while node is not None:
        node.visits += 1
        if node.player_just_moved is not None:
            node.wins += result
        node = node.parent
        result = 1.0 - result


# ═══════════════════════════════════════════════════════════════
#  MCTSAI（公开接口）
# ═══════════════════════════════════════════════════════════════


class MCTSAI:
    """中国象棋蒙特卡洛树搜索 AI（支持多进程根节点并行）。

    对外接口与 ``MinimaxAI`` 一致：``get_best_move`` / ``choose_move``。

    Args:
        max_simulations: 所有 worker 合计的最大模拟次数。
        time_limit: 搜索时间上限（秒），与 ``max_simulations`` 取先到者。
        workers: 并行进程数（0 或 1 = 单进程，>1 = 多进程根并行）。
            默认 ``None`` 表示自动检测 ``min(4, cpu_count)``。
        verbose: 搜索结束后是否打印统计信息。
    """

    def __init__(
        self,
        max_simulations: int = 5000,
        time_limit: float = 10.0,
        workers: Optional[int] = None,
        verbose: bool = True,
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
    ) -> Optional[Tuple[int, int, int, int]]:
        """Searcher 统一接口。"""
        return self.get_best_move(board, time_limit=time_limit, game_history=game_history)

    def get_best_move(
        self,
        board: Board,
        game_history: Optional[List[int]] = None,
        time_limit: Optional[float] = None,
        max_simulations: Optional[int] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """执行 MCTS 搜索并返回最佳走法。

        当 ``workers > 1`` 时采用多进程根节点并行：每个 worker 独立建树，
        搜索完成后在主进程合并各棵树根节点的子节点统计（visits / wins）。
        """
        tl = time_limit if time_limit is not None else self.time_limit
        ms = max_simulations if max_simulations is not None else self.max_simulations
        t0 = time.time()

        effective_workers = self.workers
        if effective_workers <= 1 or ms < effective_workers * 10:
            # 单进程路径（避免极小模拟量下的进程启动开销）
            merged = _run_single_mcts_tree(board, ms, tl, seed_offset=0)
            total_sims = sum(int(s["visits"]) for s in merged.values())
        else:
            sims_per_worker = ms // effective_workers
            remainder = ms % effective_workers
            merged = self._parallel_search(board, tl, sims_per_worker,
                                           remainder, effective_workers)
            total_sims = sum(int(s["visits"]) for s in merged.values())

        elapsed = time.time() - t0
        self.simulations_run = total_sims

        best_move: Optional[Tuple[int, int, int, int]] = None
        best_visits = -1
        best_wr = 0.0
        for mv, st in merged.items():
            v = int(st["visits"])
            if v > best_visits:
                best_visits = v
                best_move = mv
                best_wr = st["wins"] / v if v > 0 else 0.0

        self.last_stats = {
            "time_taken": elapsed,
            "simulations": total_sims,
            "workers": effective_workers,
        }
        if self.verbose:
            print(f"MCTS 搜索完成，总模拟次数: {total_sims}  ({effective_workers} workers)")
            print(f"搜索耗时 (秒): {elapsed:.3f}")
            if best_move is not None:
                print(f"最佳走法: {best_move}  胜率: {best_wr:.1%}  访问: {best_visits}")
        return best_move

    def _parallel_search(
        self,
        board: Board,
        time_limit: float,
        sims_per_worker: int,
        remainder: int,
        num_workers: int,
    ) -> Dict[Tuple[int, int, int, int], Dict[str, float]]:
        """多进程根节点并行搜索并合并结果。"""
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
                        seed_offset=i * 1000,
                    )
                )
            results = [f.result() for f in futures]

        # 合并各棵树的子节点统计
        merged: Dict[Tuple[int, int, int, int], Dict[str, float]] = {}
        for child_stats in results:
            for mv, st in child_stats.items():
                if mv in merged:
                    merged[mv]["visits"] += st["visits"]
                    merged[mv]["wins"] += st["wins"]
                else:
                    merged[mv] = {"visits": st["visits"], "wins": st["wins"]}
        return merged
