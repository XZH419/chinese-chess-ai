"""GameController（控制器层）。

目标：把“谁走、是否合法、是否终局、AI 走子”这些流程统一收拢到 Controller，
让 CLI/GUI 变成薄 View（只负责输入/渲染），从而对齐参考项目的 MVC 结构。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

from chess.model.board import Board
from chess.model.rules import Rules
from chess.algorithm.random_ai import RandomAI


@dataclass(slots=True)
class MoveOutcome:
    """Result of attempting/applying a move."""

    ok: bool
    message: str = ""
    captured: object = None


class GameController:
    """Central orchestrator for moves and game status."""

    def __init__(self, board: Optional[Board] = None):
        self.board: Board = board or Board()
        # 默认对手（黑方）使用 RandomAI，便于快速验证链路可用性
        self.ai = RandomAI()
        self.ai_color = "black"

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
        }

    def maybe_play_ai_turn(self, time_limit: float = 5.0) -> MoveOutcome:
        """如果轮到 AI（默认黑方），则让 AI 走一步。否则返回 ok=False。"""
        if self.is_game_over():
            return MoveOutcome(ok=False, message="game over")
        if self.board.current_player != self.ai_color:
            return MoveOutcome(ok=False, message="not ai turn")

        move = self.ai.choose_move(self.board, time_limit=time_limit)
        if move is None:
            # 无合法走法：困毙/将死，胜负由 Rules.winner 判定
            return MoveOutcome(ok=False, message="ai has no legal moves")

        return self.apply_move(move, player=self.ai_color)

