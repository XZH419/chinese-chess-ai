"""游戏控制器模块（MVC 架构中的控制器层）。

本模块负责将走子流程、合法性校验、终局判定以及 AI 行棋等核心逻辑
统一收拢到 GameController 中，使 CLI / GUI 只需承担薄 View 职责
（仅负责用户输入与界面渲染），从而严格对齐参考项目的 MVC 分层结构。

设计约定:
    - ``red_agent`` / ``black_agent`` 为 ``None`` 时代表该方由人类操控。
    - 本模块不会隐式地为任何一方挂载 AI，必须由入口层（CLI / GUI）
      显式构造 agent 后传入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from engine.board import Board
from engine.rules import MoveEntry, Rules


def describe_player_agent(agent: Optional[Any]) -> str:
    """生成单侧玩家的展示名称（不含"红方"/"黑方"前缀）。

    根据 agent 的类型和关键参数，返回一段简短的可读描述，
    例如 ``"Minimax AI，深度 5"`` 或 ``"MCTS AI，模拟上限 5000"``。

    Args:
        agent: AI 实例，若为 ``None`` 则视为人类玩家。

    Returns:
        str: 玩家类型的简短描述字符串。
    """
    if agent is None:
        return "玩家"
    cls = type(agent).__name__
    if cls == "MinimaxAI":
        d = getattr(agent, "depth", None)
        return f"Minimax AI，深度 {d}" if isinstance(d, int) else "Minimax AI"
    if cls == "RandomAI":
        return "随机 AI"
    if cls == "MCTSAI":
        sims = getattr(agent, "max_simulations", None)
        return (
            f"MCTS AI，模拟上限 {sims}"
            if sims is not None
            else "MCTS AI"
        )
    return cls


def format_matchup_line(red_agent: Optional[Any], black_agent: Optional[Any]) -> str:
    """格式化红黑双方的对阵描述行。

    生成形如 ``"红方（玩家）对阵 黑方（Minimax AI，深度 3）"`` 的完整对阵说明，
    供 CLI / GUI 显示当前对局配置。

    Args:
        red_agent: 红方 AI 实例，``None`` 表示人类。
        black_agent: 黑方 AI 实例，``None`` 表示人类。

    Returns:
        str: 完整的红黑对阵描述字符串。
    """
    return (
        f"红方（{describe_player_agent(red_agent)}）"
        f"对阵 黑方（{describe_player_agent(black_agent)}）"
    )


class AgentProtocol:
    """AI 代理接口协议（鸭子类型约定，无需强制继承）。

    所有 AI 实现应至少提供以下两个方法之一：
        - ``choose_move(board, time_limit=..., game_history=..., move_history=...)``
        - ``get_best_move(board, time_limit=..., game_history=..., move_history=...)``

    ``game_history``：Zobrist 哈希列表（开局库、路径重复启发等）。
    ``move_history``：``MoveEntry`` 列表，与 ``Rules.perpetual_check_status`` 一致。
    """

    def choose_move(self, board: Board, time_limit: Optional[float] = None, **kwargs):  # pragma: no cover
        """选择最佳走法（接口方法，子类须实现）。

        Args:
            board: 当前棋盘状态。
            time_limit: 思考时间上限（秒），``None`` 表示不限。
            **kwargs: 扩展参数，如 ``game_history``。

        Returns:
            走法元组 ``(sr, sc, er, ec)``，若无合法走法则返回 ``None``。

        Raises:
            NotImplementedError: 未被子类覆写时抛出。
        """
        raise NotImplementedError

    def get_best_move(self, board: Board, time_limit: Optional[float] = None, **kwargs):  # pragma: no cover
        """获取最佳走法（``choose_move`` 的别名接口）。

        Args:
            board: 当前棋盘状态。
            time_limit: 思考时间上限（秒），``None`` 表示不限。
            **kwargs: 扩展参数，如 ``game_history``。

        Returns:
            走法元组 ``(sr, sc, er, ec)``，若无合法走法则返回 ``None``。

        Raises:
            NotImplementedError: 未被子类覆写时抛出。
        """
        raise NotImplementedError


@dataclass(slots=True)
class MoveOutcome:
    """走子操作的结果封装。

    用于统一表达一步走子尝试后的所有可能结果，包括是否成功、
    失败原因、被吃棋子、是否导致终局以及胜者信息。

    Attributes:
        ok: 走子是否成功执行。
        message: 失败时的原因说明，成功时通常为空。
        captured: 本步吃掉的棋子对象，未吃子则为 ``None``。
        game_over: 仅当 ``ok=True`` 时有意义——本步执行后局面是否已终局。
        winner: 终局时的胜者颜色（``'red'`` / ``'black'``），和棋时为 ``None``。
        perpetual_warning: 本步后规则层为长将「第二次同形」警告（须为长将方走出同形，
            且未超本局 ``GameController.MAX_PERPETUAL_WARNINGS_PER_GAME`` 次数上限）。
        perpetual_forfeit: 本步后长将第三次判负导致终局。
        perpetual_offender: 长将方颜色（``warning`` / ``forfeit`` 时有效）。
        move_limit_draw: 本步后达到限着手数（``Rules.MAX_PLIES_AUTODRAW``）和棋。
    """

    ok: bool
    message: str = ""
    captured: object = None
    game_over: bool = False
    winner: Optional[str] = None
    perpetual_warning: bool = False
    perpetual_forfeit: bool = False
    perpetual_offender: Optional[str] = None
    move_limit_draw: bool = False


class GameController:
    """中国象棋游戏控制器——走子流程与对局状态的核心调度器。

    负责管理棋盘状态、走子合法性校验、对局历史记录、终局判定
    以及 AI 行棋调度等全部核心游戏逻辑。CLI / GUI 通过本控制器
    的公开接口驱动整个对局流程。

    Attributes:
        board: 当前棋盘实例。
        red_agent: 红方 AI 代理，``None`` 表示人类操控。
        black_agent: 黑方 AI 代理，``None`` 表示人类操控。
        history: 完整对局历史记录列表。

    Note:
        「第二次同形」是否构成 ``warning`` 由 ``Rules.perpetual_check_status`` 决定（须为
        长将方走出）。本控制器对界面「长将警告」另设 ``MAX_PERPETUAL_WARNINGS_PER_GAME``
        按局累计上限；判负逻辑不受影响。
    """

    #: 每局长将警告（含延后到长将方回合的那次）最多触发次数
    MAX_PERPETUAL_WARNINGS_PER_GAME = 3

    def __init__(
        self,
        board: Optional[Board] = None,
        red_agent: Optional[Any] = None,
        black_agent: Optional[Any] = None,
    ):
        """初始化游戏控制器。

        Args:
            board: 初始棋盘实例，``None`` 则自动创建标准开局棋盘。
            red_agent: 红方 AI 实例，``None`` 表示人类。
            black_agent: 黑方 AI 实例，``None`` 表示人类。
        """
        self.board: Board = board or Board()
        # agent 为 None 表示该方是人类（由 GUI 点击驱动）
        self.red_agent = red_agent
        self.black_agent = black_agent

        # 完整对局历史：history[0] 为初始局面（mover/gave_check 为 None），
        # history[i>=1] 记录第 i 手走后的局面哈希、行棋方及是否将军。
        self.history: List[MoveEntry] = [
            MoveEntry(pos_hash=self.board.zobrist_hash)
        ]
        self._perpetual_warning_shown_count: int = 0

    def agent_for(self, color: str):
        """获取指定颜色方的 AI 代理。

        Args:
            color: 阵营颜色，``'red'`` 或 ``'black'``。

        Returns:
            对应方的 AI 实例；若该方为人类则返回 ``None``。

        Raises:
            ValueError: 当 ``color`` 不是 ``'red'`` 或 ``'black'`` 时抛出。
        """
        if color == "red":
            return self.red_agent
        if color == "black":
            return self.black_agent
        raise ValueError(f"unknown side: {color!r}")

    def can_move(self, move: Tuple[int, int, int, int], player: Optional[str] = None) -> bool:
        """检查指定走法在当前局面下是否合法。

        Args:
            move: 走法四元组 ``(起始行, 起始列, 目标行, 目标列)``。
            player: 行棋方颜色，``None`` 时使用当前轮次玩家。

        Returns:
            bool: 走法合法返回 ``True``，否则返回 ``False``。
        """
        sr, sc, er, ec = move
        ok, _ = Rules.is_valid_move(
            self.board, sr, sc, er, ec,
            player=player,
            history=self.history,
        )
        return ok

    # --- 面向 View 层（GUI / CLI）的稳定接口 ---
    def try_apply_player_move(
        self, move: Tuple[int, int, int, int], player: Optional[str] = None
    ) -> MoveOutcome:
        """尝试执行玩家走子（失败时不会改变棋盘状态）。

        本方法是面向 View 层的安全包装，内部委托给 ``apply_move``。

        Args:
            move: 走法四元组 ``(起始行, 起始列, 目标行, 目标列)``。
            player: 行棋方颜色，``None`` 时使用当前轮次玩家。

        Returns:
            MoveOutcome: 走子结果，包含成功/失败状态及附带信息。
        """
        return self.apply_move(move, player=player)

    @property
    def game_history_hashes(self) -> List[int]:
        """从对局历史中提取纯 Zobrist 哈希链（向后兼容属性）。

        供 AI 搜索和重复局面判定使用。

        Returns:
            List[int]: 从开局到当前局面的 Zobrist 哈希值列表。
        """
        return [e.pos_hash for e in self.history]

    def apply_move(self, move: Tuple[int, int, int, int], player: Optional[str] = None) -> MoveOutcome:
        """执行走子：合法则应用到棋盘，否则返回失败结果。

        完整流程：合法性校验 → 执行走子 → 记录历史 → 检测终局。

        Args:
            move: 走法四元组 ``(起始行, 起始列, 目标行, 目标列)``。
            player: 行棋方颜色，``None`` 时使用当前轮次玩家。

        Returns:
            MoveOutcome: 走子结果，包含吃子信息、终局状态和胜者。
        """
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
        self.history.append(
            MoveEntry(
                pos_hash=self.board.zobrist_hash,
                mover=mover,
                gave_check=gave_check,
                last_move=(sr, sc, er, ec),
            )
        )
        pst, poff = Rules.perpetual_check_status(self.board, self.history)
        cap = self.MAX_PERPETUAL_WARNINGS_PER_GAME
        if pst == "warning" and self._perpetual_warning_shown_count < cap:
            p_warn = True
            self._perpetual_warning_shown_count += 1
        else:
            p_warn = False
        p_ff = pst == "forfeit"
        mld = Rules.is_move_limit_draw(self.history)
        over = Rules.is_game_over(self.board, move_history=self.history)
        win: Optional[str] = Rules.winner(self.board, self.history) if over else None
        return MoveOutcome(
            ok=True,
            captured=captured,
            game_over=over,
            winner=win,
            perpetual_warning=p_warn,
            perpetual_forfeit=p_ff,
            perpetual_offender=poff if pst != "none" else None,
            move_limit_draw=mld,
        )

    def undo_move(self, move: Tuple[int, int, int, int], captured) -> None:
        """撤销一步走子，恢复棋盘和历史到上一状态。

        Args:
            move: 需要撤销的走法四元组 ``(起始行, 起始列, 目标行, 目标列)``。
            captured: 该步走子时被吃掉的棋子对象（用于恢复）。
        """
        sr, sc, er, ec = move
        self.board.undo_move(sr, sc, er, ec, captured)
        if len(self.history) > 1:
            self.history.pop()

    def is_game_over(self) -> bool:
        """判断当前局面是否已终局。

        Returns:
            bool: 终局返回 ``True``（含将死、困毙、和棋等情况）。
        """
        return Rules.is_game_over(self.board, move_history=self.history)


    def winner(self) -> Optional[str]:
        """获取当前对局的胜者。

        Returns:
            Optional[str]: 胜者颜色字符串 ``'red'`` / ``'black'``，
            和棋或未终局时返回 ``None``。
        """
        return Rules.winner(self.board, move_history=self.history)

    def current_result(self) -> dict:
        """获取当前对局状态摘要（供 GUI 状态栏显示）。

        Returns:
            dict: 包含以下键值的状态字典：
                - ``game_over`` (bool): 是否终局
                - ``winner`` (Optional[str]): 胜者颜色
                - ``current_player`` (str): 当前行棋方
                - ``red_is_human`` (bool): 红方是否为人类
                - ``black_is_human`` (bool): 黑方是否为人类
        """
        return {
            "game_over": self.is_game_over(),
            "winner": self.winner(),
            "current_player": self.board.current_player,
            "red_is_human": self.red_agent is None,
            "black_is_human": self.black_agent is None,
        }

    def reset_game(self) -> None:
        """重置棋盘到初始局面，但保留当前的 AI 代理配置。

        用途：
            - GUI / Benchmark 反复开新局时不丢失命令行传入的对局模式。
            - AI vs AI / 人机模式均能保持一致的代理绑定。
        """

        self.board = Board()
        self.history = [MoveEntry(pos_hash=self.board.zobrist_hash)]
        self._perpetual_warning_shown_count = 0

    def matchup_line(self) -> str:
        """生成当前控制器绑定的红黑对阵说明字符串。

        供 CLI / GUI 标题栏或日志输出使用。

        Returns:
            str: 形如 ``"红方（玩家）对阵 黑方（Minimax AI，深度 3）"`` 的描述。
        """
        return format_matchup_line(self.red_agent, self.black_agent)

    def maybe_play_ai_turn(self, time_limit: float = 5.0) -> MoveOutcome:
        """若当前轮到 AI 行棋，则让对应 agent 走一步；否则返回失败。

        本方法是同步阻塞调用，适用于 CLI / 无头对弈场景。
        GUI 场景下应使用 QThread 在后台计算走法，再由主线程调用 ``apply_move``。

        Args:
            time_limit: AI 思考时间上限（秒），默认 5.0。

        Returns:
            MoveOutcome: 走子结果。若轮到人类或对局已结束，返回 ``ok=False``。
        """
        if self.is_game_over():
            return MoveOutcome(ok=False, message="game over")
        cp = self.board.current_player
        agent = self.red_agent if cp == "red" else self.black_agent
        if agent is None:
            return MoveOutcome(ok=False, message="human turn")

        gh = list(self.game_history_hashes)
        mh = list(self.history)
        if hasattr(agent, "choose_move"):
            move = agent.choose_move(
                self.board,
                time_limit=time_limit,
                game_history=gh,
                move_history=mh,
            )
        else:
            move = agent.get_best_move(
                self.board,
                time_limit=time_limit,
                game_history=gh,
                move_history=mh,
            )
        if move is None:
            # 无合法走法：困毙/将死，胜负由 Rules.winner 判定
            return MoveOutcome(ok=False, message="ai has no legal moves")

        return self.apply_move(move, player=self.board.current_player)
