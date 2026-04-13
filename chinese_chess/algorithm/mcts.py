"""蒙特卡洛树搜索（MCTS）AI 引擎。

集成技术：UCB1-RAVE 混合选择、惰性走法展开、apply/undo 状态回溯（零拷贝树遍历）、
动态截断启发模拟（Lightweight Rollout + Evaluation 兜底）、__slots__ 节点内存优化、
多进程根节点并行（Root Parallelism）、RAVE/AMAF 快速着法价值估计、
Zobrist 置换表 DAG 合并（同局面共享统计）。
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

_CAPTURE_PROB = 0.80
_UCB_C = 1.414
_SCORE_SCALE = 600.0
_RAVE_CONST = 300

Move4 = Tuple[int, int, int, int]


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
    """MCTS-RAVE 搜索树节点。

    ``rave_visits`` / ``rave_wins`` 用于 AMAF (All-Moves-As-First) 快速估值：
    即使某子节点自身访问次数很少，也可通过模拟阶段出现的同一着法来累积统计，
    加速收敛。
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
        "rave_visits",
        "rave_wins",
    ]

    def __init__(
        self,
        state_hash: int,
        player_just_moved: str,
        parent: Optional[MCTSNode] = None,
        move: Optional[Move4] = None,
    ):
        self.state_hash = state_hash
        self.parent = parent
        self.move = move
        self.children: List[MCTSNode] = []
        self.visits: int = 0
        self.wins: float = 0.0
        self.untried_moves: Optional[List[Move4]] = None
        self.player_just_moved = player_just_moved
        self.rave_visits: int = 0
        self.rave_wins: float = 0.0

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
        """UCB1-RAVE 混合选择。

        混合公式：score = (1 - β) * mcts_val + β * rave_val + exploration
        其中 β = rave_visits / (rave_visits + visits + RAVE_CONST)，
        随节点访问量增大而衰减至 0（回归纯 MCTS）。
        """
        best: Optional[MCTSNode] = None
        best_score = -1.0
        for ch in self.children:
            rv = ch.rave_visits
            v = ch.visits
            total = v + rv
            if total == 0:
                return ch

            mcts_val = ch.wins / v if v > 0 else 0.0
            rave_val = ch.rave_wins / rv if rv > 0 else 0.0
            beta = rv / (rv + v + _RAVE_CONST + 1e-5)
            exploit = (1.0 - beta) * mcts_val + beta * rave_val
            explore = _UCB_C * math.sqrt(log_parent / (v + 1e-5))
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
) -> Dict[Move4, Dict[str, float]]:
    """在独立进程/线程中执行一棵完整的 MCTS-RAVE-DAG 搜索树。

    局部置换表 ``tt`` 将相同 Zobrist 哈希的节点合并为同一实例（DAG），
    使不同着法序列到达的同一局面共享 visits / wins 统计，加速收敛。
    因为 ``_backpropagate`` 已改为接收显式 ``path`` 列表，
    DAG 中"一个节点有多个父节点无法回溯"的经典难题不复存在。

    Returns:
        ``{move: {"visits": V, "wins": W, "rave_visits": RV, "rave_wins": RW}}``
    """
    random.seed(time.time_ns() + seed_offset)
    t0 = time.time()

    root_player = board.current_player
    opp_of_root = "black" if root_player == "red" else "red"
    root = MCTSNode(state_hash=board.zobrist_hash, player_just_moved=opp_of_root)
    root.ensure_moves(board)

    tt: Dict[int, MCTSNode] = {root.state_hash: root}
    move_stack: List[Tuple[Move4, Any]] = []
    sims_done = 0

    while sims_done < max_simulations:
        if time.time() - t0 >= time_limit:
            break

        node = root
        path: List[MCTSNode] = [root]

        # ── Selection ──
        while node.is_fully_expanded() and node.children:
            log_n = math.log(node.visits) if node.visits > 0 else 0.0
            node = node.best_child_ucb(log_n)
            captured = board.apply_move(*node.move)
            move_stack.append((node.move, captured))
            path.append(node)

        # ── Expansion ──
        node.ensure_moves(board)
        expanded = False
        if node.untried_moves:
            child = _expand_one(board, node, move_stack, tt)
            if child is not None:
                node = child
                path.append(node)
                expanded = True

        if not expanded and not node.children and node.visits > 0:
            result, red_moves, black_moves = _terminal_score(board, root_player), set(), set()
        else:
            # ── Simulation ──
            result, red_moves, black_moves = _simulate(board, root_player)

        # ── Backpropagation (with RAVE/AMAF) ──
        _backpropagate(path, result, red_moves, black_moves)
        sims_done += 1

        while move_stack:
            mv, cap = move_stack.pop()
            board.undo_move(*mv, cap)

    child_stats: Dict[Move4, Dict[str, float]] = {}
    for ch in root.children:
        if ch.move is not None:
            child_stats[ch.move] = {
                "visits": ch.visits,
                "wins": ch.wins,
                "rave_visits": ch.rave_visits,
                "rave_wins": ch.rave_wins,
            }
    return child_stats


# ── 搜索辅助函数（模块级，便于 pickle 序列化）──


def _expand_one(
    board: Board,
    node: MCTSNode,
    move_stack: List[Tuple[Move4, Any]],
    tt: Dict[int, MCTSNode],
) -> Optional[MCTSNode]:
    """从 ``node.untried_moves`` 弹出一个合法走法并创建或复用子节点。

    置换表命中时直接复用已有节点（DAG 合并），使不同路径到达的同一局面
    共享 visits / wins / rave 统计。
    """
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
            node.children.append(existing)
            return existing
        child = MCTSNode(
            state_hash=child_hash,
            player_just_moved=mover,
            parent=node,
            move=move,
        )
        tt[child_hash] = child
        node.children.append(child)
        return child
    return None


def _simulate(
    board: Board, root_player: str,
) -> Tuple[float, Set[Move4], Set[Move4]]:
    """轻量级模拟（Lightweight Rollout），同时收集双方着法集合供 RAVE 使用。

    Returns:
        ``(result, red_moves, black_moves)``——胜率分数及双方在模拟中尝试的着法集合。
    """
    sim_board = board.copy()
    b_grid = sim_board.board
    rollout_limit = _dynamic_rollout_limit(sim_board.piece_count())
    red_moves: Set[Move4] = set()
    black_moves: Set[Move4] = set()

    for _ in range(rollout_limit):
        cp = sim_board.current_player
        opp = "black" if cp == "red" else "red"

        if Rules.is_king_in_check(sim_board, opp):
            return (1.0 if cp == root_player else 0.0), red_moves, black_moves

        moves = list(Rules.get_pseudo_legal_moves(sim_board, cp))
        if not moves:
            return (0.0 if cp == root_player else 1.0), red_moves, black_moves

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

    return _eval_to_winrate(sim_board, root_player), red_moves, black_moves


def _find_king_capture(
    board: Board,
    attacker: str,
    b_grid,
    tkr: int,
    tkc: int,
) -> Optional[Move4]:
    """单目标可达性：检查 ``attacker`` 方是否有棋子可直接吃到 ``(tkr, tkc)`` 处的老将。"""
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
    sim_board: Board,
    moves: List[Move4],
    b_grid,
) -> Move4:
    """零校验启发式走子：吃子优先，直接 apply_move，返回所选着法供 RAVE 记录。"""
    captures = [m for m in moves if b_grid[m[2]][m[3]] is not None]
    if captures and random.random() < _CAPTURE_PROB:
        m = random.choice(captures)
    else:
        m = random.choice(moves)
    sim_board.apply_move(*m)
    return m


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


def _backpropagate(
    path: List[MCTSNode],
    result: float,
    red_moves: Set[Move4],
    black_moves: Set[Move4],
) -> None:
    """反向传播 + AMAF/RAVE 更新。

    沿 path（从叶到根）回传 result（每层翻转视角），同时对每个节点的所有已展开
    子节点做 RAVE 更新：若子节点的 move 出现在对应颜色的模拟着法集合中，
    则累加该子节点的 rave_visits / rave_wins。
    """
    score = result
    for i in range(len(path) - 1, -1, -1):
        node = path[i]
        node.visits += 1
        if node.player_just_moved is not None:
            node.wins += score

        for child in node.children:
            if child.move is None:
                continue
            pjm = child.player_just_moved
            if pjm == "red" and child.move in red_moves:
                child.rave_visits += 1
                child.rave_wins += score
            elif pjm == "black" and child.move in black_moves:
                child.rave_visits += 1
                child.rave_wins += score

        score = 1.0 - score


# ═══════════════════════════════════════════════════════════════
#  MCTSAI（公开接口）
# ═══════════════════════════════════════════════════════════════


class MCTSAI:
    """中国象棋蒙特卡洛树搜索 AI（支持多进程根节点并行 + RAVE）。

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
        """执行 MCTS-RAVE 搜索并返回最佳走法。

        搜索前先查询开局库；命中时瞬间返回，``last_stats`` 中标注
        ``opening_book=True`` 与 ``win_rate="Book Move"``。
        """
        move_history: List[int] = [] if game_history is None else list(game_history)

        # ── 开局库拦截 ──
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
                    print(f"命中开局库！瞬间出棋: {book_move}")
                return book_move

        # ── MCTS-RAVE 正式搜索 ──
        tl = time_limit if time_limit is not None else self.time_limit
        ms = max_simulations if max_simulations is not None else self.max_simulations
        t0 = time.time()

        effective_workers = self.workers
        if effective_workers <= 1 or ms < effective_workers * 10:
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

        best_move: Optional[Move4] = None
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
            "win_rate": f"{best_wr * 100:.1f}%",
        }
        if self.verbose:
            print(f"MCTS 搜索完成，总模拟次数: {total_sims}  ({effective_workers} workers)")
            print(f"搜索耗时 (秒): {elapsed:.3f}")
            if best_move is not None:
                print(f"最佳走法: {best_move}  胜率: {best_wr:.1%}  访问: {best_visits}")
        return best_move

    @staticmethod
    def _probe_opening_book(
        board: Board, move_history: List[int],
    ) -> Optional[Move4]:
        """查询开局库，含镜像回退；未命中返回 None。"""
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
                f"[MCTS opening book] 根局面未命中（zkey={zkey:#x}）；"
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
        sims_per_worker: int,
        remainder: int,
        num_workers: int,
    ) -> Dict[Move4, Dict[str, float]]:
        """多进程根节点并行搜索并合并结果（含 RAVE 统计）。"""
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
