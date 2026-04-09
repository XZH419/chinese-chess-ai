"""GameController（控制器层）。

目标：把“谁走、是否合法、是否终局、AI 走子”这些流程统一收拢到 Controller，
让 CLI/GUI 变成薄 View（只负责输入/渲染），从而对齐参考项目的 MVC 结构。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any

from chinese_chess.model.board import Board
from chinese_chess.model.rules import Rules
from chinese_chess.algorithm.minimax import MinimaxAI


class AgentProtocol:
    """AI 接口协议（无需继承，仅做鸭子类型约定）。

    支持两种方法之一：
    - choose_move(board, time_limit=...) -> move | None
    - get_best_move(board, time_limit=...) -> move | None
    """

    # 这里只用于类型提示，不做强制运行时检查
    def choose_move(self, board: Board, time_limit: Optional[float] = None):  # pragma: no cover
        raise NotImplementedError

    def get_best_move(self, board: Board, time_limit: Optional[float] = None):  # pragma: no cover
        raise NotImplementedError


@dataclass(slots=True)
class MoveOutcome:
    """Result of attempting/applying a move."""

    ok: bool
    message: str = ""
    captured: object = None


class GameController:
    """Central orchestrator for moves and game status."""

    def __init__(
        self,
        board: Optional[Board] = None,
        red_agent: Optional[Any] = None,
        black_agent: Optional[Any] = None,
    ):
        self.board: Board = board or Board()
        # agent 为 None 表示该方是人类（由 GUI 点击驱动）
        self.red_agent = red_agent
        self.black_agent = black_agent

        # 默认行为：若未指定 black_agent，则用 Minimax(depth=3) 作为黑方 AI
        if self.black_agent is None:
            self.black_agent = MinimaxAI(depth=3)

    def agent_for(self, color: str):
        """返回指定方的 agent；None 表示人类。"""
        return self.red_agent if color == "red" else self.black_agent

    def can_move(self, move: Tuple[int, int, int, int], player: Optional[str] = None) -> bool:
        """Check if a move is legal for player on current board."""
        sr, sc, er, ec = move
        return Rules.is_valid_move(self.board, sr, sc, er, ec, player=player)

    # --- Stable interface for Views (GUI/CLI) ---
    def try_apply_player_move(
        self, move: Tuple[int, int, int, int], player: Optional[str] = None
    ) -> MoveOutcome:
        """尝试让玩家走一步（失败不会改变棋盘状态）。"""
        return self.apply_move(move, player=player)

    def apply_move(self, move: Tuple[int, int, int, int], player: Optional[str] = None) -> MoveOutcome:
        """Apply a move if legal, otherwise return a failure outcome."""
        sr, sc, er, ec = move
        if not Rules.is_valid_move(self.board, sr, sc, er, ec, player=player):
            return MoveOutcome(ok=False, message="illegal move")
        captured = self.board.apply_move(sr, sc, er, ec)
        return MoveOutcome(ok=True, captured=captured)

    def undo_move(self, move: Tuple[int, int, int, int], captured) -> None:
        sr, sc, er, ec = move
        self.board.undo_move(sr, sc, er, ec, captured)

    def is_game_over(self) -> bool:
        return Rules.is_game_over(self.board)

    def winner(self) -> Optional[str]:
        """返回胜者颜色字符串：'red'/'black'/None。"""
        return Rules.winner(self.board)

    def current_result(self) -> dict:
        """返回当前对局状态摘要（供 GUI 显示）。"""
        return {
            "game_over": self.is_game_over(),
            "winner": self.winner(),
            "current_player": self.board.current_player,
            "red_is_human": self.red_agent is None,
            "black_is_human": self.black_agent is None,
        }

    def reset_game(self) -> None:
        """重置棋盘到初始局面，但**保留**当前 red_agent / black_agent 配置。

        用途：
        - GUI/Benchmark 反复开新局时不丢失命令行传入的对局模式
        - AI vs AI / 人机模式都能保持一致
        """

        self.board = Board()

    def maybe_play_ai_turn(self, time_limit: float = 5.0) -> MoveOutcome:
        """若轮到 AI，则让对应 agent 走一步；否则返回 ok=False。

        注意：
        - 本方法是“同步”走一步，适用于 CLI/无头对弈。
        - GUI 场景下，为避免卡 UI，应使用 QThread 在后台计算 move，
          再由主线程调用 `apply_move`。
        """
        if self.is_game_over():
            return MoveOutcome(ok=False, message="game over")
        agent = self.agent_for(self.board.current_player)
        if agent is None:
            return MoveOutcome(ok=False, message="human turn")

        if hasattr(agent, "choose_move"):
            move = agent.choose_move(self.board, time_limit=time_limit)
        else:
            move = agent.get_best_move(self.board, time_limit=time_limit)
        if move is None:
            # 无合法走法：困毙/将死，胜负由 Rules.winner 判定
            return MoveOutcome(ok=False, message="ai has no legal moves")

        return self.apply_move(move, player=self.board.current_player)

