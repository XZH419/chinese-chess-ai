"""GameController（控制器层）。

目标：把“谁走、是否合法、是否终局、AI 走子”这些流程统一收拢到 Controller，
让 CLI/GUI 变成薄 View（只负责输入/渲染），从而对齐参考项目的 MVC 结构。

重要：不在此模块内为任何一方隐式挂载 AI；``red_agent`` / ``black_agent`` 为 ``None`` 即人类，
须在入口（CLI/GUI）显式构造后传入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from chinese_chess.model.board import Board
from chinese_chess.model.rules import MoveEntry, Rules


def describe_player_agent(agent: Optional[Any]) -> str:
    """人类 / Minimax / Random / MCTS 等单侧展示名（不含「红方」前缀）。"""
    if agent is None:
        return "Human"
    cls = type(agent).__name__
    if cls == "MinimaxAI":
        d = getattr(agent, "depth", None)
        return f"Minimax, Depth={d}" if isinstance(d, int) else "Minimax"
    if cls == "RandomAI":
        return "Random"
    if cls == "MCTSAI":
        sims = getattr(agent, "max_simulations", None)
        return f"MCTS, Sims={sims}" if sims is not None else "MCTS"
    return cls


def format_matchup_line(red_agent: Optional[Any], black_agent: Optional[Any]) -> str:
    """例如：红方 (Human) vs 黑方 (Minimax, Depth=3)"""
    return (
        f"红方 ({describe_player_agent(red_agent)}) "
        f"vs 黑方 ({describe_player_agent(black_agent)})"
    )


class AgentProtocol:
    """AI 接口协议（无需继承，仅做鸭子类型约定）。

    支持两种方法之一（可选传入 game_history：开局至当前的 zobrist_hash 列表）：
    - choose_move(board, time_limit=..., game_history=...) -> move | None
    - get_best_move(board, time_limit=..., game_history=...) -> move | None
    """

    # 这里只用于类型提示，不做强制运行时检查
    def choose_move(self, board: Board, time_limit: Optional[float] = None, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def get_best_move(self, board: Board, time_limit: Optional[float] = None, **kwargs):  # pragma: no cover
        raise NotImplementedError


@dataclass(slots=True)
class MoveOutcome:
    """Result of attempting/applying a move."""

    ok: bool
    message: str = ""
    captured: object = None
    # 仅当 ok=True 且本步执行后局面已终局时有意义；winner=None 表示和棋（如三次重复）
    game_over: bool = False
    winner: Optional[str] = None


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

        # 完整对局历史：history[0] 为初始局面（mover/gave_check 为 None），
        # history[i>=1] 记录第 i 手走后的局面哈希、行棋方及是否将军。
        self.history: List[MoveEntry] = [
            MoveEntry(pos_hash=self.board.zobrist_hash)
        ]

    def agent_for(self, color: str):
        """返回指定方的 agent；None 表示人类。必须与 board.current_player 的 'red'/'black' 一致使用。"""
        if color == "red":
            return self.red_agent
        if color == "black":
            return self.black_agent
        raise ValueError(f"unknown side: {color!r}")

    def can_move(self, move: Tuple[int, int, int, int], player: Optional[str] = None) -> bool:
        """Check if a move is legal for player on current board."""
        sr, sc, er, ec = move
        ok, _ = Rules.is_valid_move(
            self.board, sr, sc, er, ec,
            player=player,
            history=self.history,
        )
        return ok

    # --- Stable interface for Views (GUI/CLI) ---
    def try_apply_player_move(
        self, move: Tuple[int, int, int, int], player: Optional[str] = None
    ) -> MoveOutcome:
        """尝试让玩家走一步（失败不会改变棋盘状态）。"""
        return self.apply_move(move, player=player)

    @property
    def game_history_hashes(self) -> List[int]:
        """向后兼容属性：从 ``history`` 中提取纯 Zobrist 哈希链（供 AI / 重复判和）。"""
        return [e.pos_hash for e in self.history]

    def apply_move(self, move: Tuple[int, int, int, int], player: Optional[str] = None) -> MoveOutcome:
        """Apply a move if legal, otherwise return a failure outcome."""
        sr, sc, er, ec = move
        ok, reason = Rules.is_valid_move(
            self.board, sr, sc, er, ec,
            player=player,
            history=self.history,
        )
        if not ok:
            detail = reason or "未知原因"
            return MoveOutcome(ok=False, message=f"非法走法：{detail}")
        mover = self.board.current_player
        captured = self.board.apply_move(sr, sc, er, ec)
        opp = self.board.current_player
        gave_check = Rules.is_king_in_check(self.board, opp)
        self.history.append(MoveEntry(
            pos_hash=self.board.zobrist_hash,
            mover=mover,
            gave_check=gave_check,
        ))
        pos_hashes = self.game_history_hashes
        over = Rules.is_game_over(self.board, position_history=pos_hashes)
        win: Optional[str] = Rules.winner(self.board) if over else None
        return MoveOutcome(ok=True, captured=captured, game_over=over, winner=win)

    def undo_move(self, move: Tuple[int, int, int, int], captured) -> None:
        sr, sc, er, ec = move
        self.board.undo_move(sr, sc, er, ec, captured)
        if len(self.history) > 1:
            self.history.pop()

    def is_game_over(self) -> bool:
        return Rules.is_game_over(self.board, position_history=self.game_history_hashes)


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
        self.history = [MoveEntry(pos_hash=self.board.zobrist_hash)]

    def matchup_line(self) -> str:
        """当前控制器绑定的红黑对阵说明（供 CLI/GUI 打印）。"""
        return format_matchup_line(self.red_agent, self.black_agent)

    def maybe_play_ai_turn(self, time_limit: float = 5.0) -> MoveOutcome:
        """若轮到 AI，则让对应 agent 走一步；否则返回 ok=False。

        注意：
        - 本方法是“同步”走一步，适用于 CLI/无头对弈。
        - GUI 场景下，为避免卡 UI，应使用 QThread 在后台计算 move，
          再由主线程调用 `apply_move`。
        """
        if self.is_game_over():
            return MoveOutcome(ok=False, message="game over")
        cp = self.board.current_player
        agent = self.red_agent if cp == "red" else self.black_agent
        if agent is None:
            return MoveOutcome(ok=False, message="human turn")

        gh = list(self.game_history_hashes)
        if hasattr(agent, "choose_move"):
            move = agent.choose_move(self.board, time_limit=time_limit, game_history=gh)
        else:
            move = agent.get_best_move(self.board, time_limit=time_limit, game_history=gh)
        if move is None:
            # 无合法走法：困毙/将死，胜负由 Rules.winner 判定
            return MoveOutcome(ok=False, message="ai has no legal moves")

        return self.apply_move(move, player=self.board.current_player)

