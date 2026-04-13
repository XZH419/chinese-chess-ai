"""蒙特卡洛树搜索（MCTS）AI 引擎。

集成技术：UCB1 选择、惰性走法展开、apply/undo 状态回溯（零拷贝树遍历）、
截断式启发模拟（Heavy Playout + Evaluation 兜底）、__slots__ 节点内存优化。
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from chinese_chess.model.board import Board
from chinese_chess.model.rules import Rules

from .evaluation import Evaluation

# 模拟阶段吃子优先概率（剩余概率走随机非吃子着法以保持探索性）
_CAPTURE_PROB = 0.80
# UCB1 探索常数
_UCB_C = 1.414
# 评估分数到 [0, 1] 胜率的缩放系数：sigmoid(score / _SCORE_SCALE)
_SCORE_SCALE = 600.0


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
        return self.untried_moves is not None and len(self.untried_moves) == 0 and len(self.children) == 0

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


class MCTSAI:
    """中国象棋蒙特卡洛树搜索 AI。

    对外接口与 ``MinimaxAI`` 一致：``get_best_move`` / ``choose_move``。

    Args:
        max_simulations: 单次搜索的最大模拟次数。
        time_limit: 搜索时间上限（秒），与 ``max_simulations`` 取先到者。
        rollout_limit: 单次模拟最大步数（截断后以 ``Evaluation`` 兜底）。
        verbose: 搜索结束后是否打印统计信息。
    """

    def __init__(
        self,
        max_simulations: int = 5000,
        time_limit: float = 10.0,
        rollout_limit: int = 50,
        verbose: bool = True,
    ):
        self.max_simulations = max_simulations
        self.time_limit = time_limit
        self.rollout_limit = rollout_limit
        self.verbose = verbose
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

        Args:
            board: 当前棋盘（搜索过程中会原地 apply/undo，搜索结束后状态不变）。
            game_history: 开局至今的 Zobrist 哈希链（兼容 MinimaxAI 接口，本实现暂未使用）。
            time_limit: 覆盖实例默认的搜索时限（秒）。
            max_simulations: 覆盖实例默认的最大模拟次数。

        Returns:
            最佳走法元组 ``(sr, sc, er, ec)``，无合法走法时返回 ``None``。
        """
        tl = time_limit if time_limit is not None else self.time_limit
        ms = max_simulations if max_simulations is not None else self.max_simulations
        t0 = time.time()

        root_player = board.current_player
        opp_of_root = "black" if root_player == "red" else "red"
        root = MCTSNode(
            state_hash=board.zobrist_hash,
            player_just_moved=opp_of_root,
        )
        root.ensure_moves(board)

        self.simulations_run = 0
        move_stack: List[Tuple[Tuple[int, int, int, int], Any]] = []

        while self.simulations_run < ms:
            if time.time() - t0 >= tl:
                break

            node = root
            # ── 选择（Selection）──
            # 沿 UCB1 最优路径下行，直到遇到可扩展节点或终端节点
            while node.is_fully_expanded() and node.children:
                log_n = math.log(node.visits) if node.visits > 0 else 0.0
                node = node.best_child_ucb(log_n)
                captured = board.apply_move(*node.move)
                move_stack.append((node.move, captured))

            # ── 扩展（Expansion）──
            node.ensure_moves(board)
            expanded = False
            if node.untried_moves:
                child_node = self._expand_one(board, node, move_stack)
                if child_node is not None:
                    node = child_node
                    expanded = True

            if not expanded and not node.children and node.visits > 0:
                # 终端节点：走法耗尽
                result = self._terminal_score(board, root_player)
            else:
                # ── 模拟（Simulation）──
                result = self._simulate(board, root_player)

            # ── 反向传播（Backpropagation）──
            self._backpropagate(node, result)
            self.simulations_run += 1

            # 回溯树遍历期间的所有 apply_move
            while move_stack:
                mv, cap = move_stack.pop()
                board.undo_move(*mv, cap)

        elapsed = time.time() - t0
        best_child = root.best_child_robust()
        best_move = best_child.move if best_child else None

        self.last_stats = {
            "time_taken": elapsed,
            "simulations": self.simulations_run,
        }
        if self.verbose:
            print(f"MCTS 搜索完成，总模拟次数: {self.simulations_run}")
            print(f"搜索耗时 (秒): {elapsed:.3f}")
            if best_child and best_child.visits > 0:
                wr = best_child.wins / best_child.visits
                print(f"最佳走法: {best_move}  胜率: {wr:.1%}  访问: {best_child.visits}")
        return best_move

    # ── 内部方法 ──

    def _expand_one(
        self,
        board: Board,
        node: MCTSNode,
        move_stack: List[Tuple[Tuple[int, int, int, int], Any]],
    ) -> Optional[MCTSNode]:
        """从 ``node.untried_moves`` 弹出一个走法并创建子节点。

        跳过非法走法（伪合法生成可能包含自将），在真实棋盘上 apply_move 并压入 move_stack。
        """
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

    def _simulate(self, board: Board, root_player: str) -> float:
        """从当前局面进行截断式启发模拟（Heavy Playout）。

        在 ``board.copy()`` 上执行最多 ``rollout_limit`` 步随机对弈。
        吃子走法以 ``_CAPTURE_PROB`` 概率优先选择。
        截断后以 ``Evaluation.evaluate`` 将分数映射为 ``[0, 1]`` 区间胜率。
        """
        sim_board = board.copy()
        b_grid = sim_board.board

        for _ in range(self.rollout_limit):
            cp = sim_board.current_player
            w = Rules.winner(sim_board)
            if w is not None:
                return 1.0 if w == root_player else 0.0

            moves = list(Rules.get_pseudo_legal_moves(sim_board, cp))
            if not moves:
                opp = "black" if cp == "red" else "red"
                return 1.0 if opp == root_player else 0.0

            if not self._pick_rollout_move(sim_board, moves, cp, b_grid):
                break

        return self._eval_to_winrate(sim_board, root_player)

    @staticmethod
    def _pick_rollout_move(
        sim_board: Board,
        moves: List[Tuple[int, int, int, int]],
        mover: str,
        b_grid,
    ) -> bool:
        """启发式走子：优先吃子，跳过自将。已在 ``sim_board`` 上执行合法走法。

        Returns:
            是否成功走出一步（``False`` 表示无合法走法，应终止模拟）。
        """
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

    @staticmethod
    def _eval_to_winrate(board: Board, root_player: str) -> float:
        """将 ``Evaluation.evaluate`` 的 Negamax 分数转换为根玩家视角的 ``[0, 1]`` 胜率。"""
        raw = Evaluation.evaluate(board)
        if board.current_player != root_player:
            raw = -raw
        return 1.0 / (1.0 + math.exp(-raw / _SCORE_SCALE))

    def _terminal_score(self, board: Board, root_player: str) -> float:
        """终端节点（无合法走法）的得分。"""
        w = Rules.winner(board)
        if w == root_player:
            return 1.0
        if w is not None:
            return 0.0
        return 0.5

    @staticmethod
    def _backpropagate(node: MCTSNode, result: float) -> None:
        """将模拟结果从叶节点回传至根：每层翻转视角。"""
        while node is not None:
            node.visits += 1
            if node.player_just_moved is not None:
                node.wins += result
            node = node.parent
            result = 1.0 - result
