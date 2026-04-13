"""搜索器共享：伪合法走子后的局面判定（不自成规则引擎）。

仅封装 ``Board`` + ``Rules`` 的常见调用组合，供 Minimax / MCTS / MCTS-Minimax
在热点路径复用。内部不实现车炮马兵几何；一律委托 ``Rules``。

缓存策略：
- **走子前键** ``(zobrist_hash, move)``：命中时可跳过整段 ``apply → Rules → undo``
  （用于 ``move_gives_check_with_undo`` 与 Minimax 主循环等）。
- **走子后键** ``zobrist_hash``：命中时可跳过 ``pseudo_move_post_apply_flags`` 内
  对 ``Rules`` 的重复调用（置换/重复展开同一走后局面时）。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional, Tuple

from chinese_chess.model.board import Board
from chinese_chess.model.rules import Rules

# 与三处搜索器一致的走法四元组
Move4 = Tuple[int, int, int, int]


def pseudo_move_illegal_after_apply(board: Board, mover: str) -> bool:
    """已执行 ``apply_move`` 后：是否因自将或飞将而不合法。

    先测飞将（常为数格扫描；不同列时 O(1) 否定），再测己方是否仍被将军。
    """
    if Rules._jiang_face_to_face(board):
        return True
    return Rules.is_king_in_check(board, mover)


def pseudo_move_post_apply_flags(board: Board, mover: str) -> Tuple[bool, bool]:
    """已 ``apply_move`` 后：是否合法、以及对方是否被将军（用于将军延伸等）。

    Returns:
        ``(legal, opp_in_check)``。不合法时 ``opp_in_check`` 恒为 ``False``，
        且不对对方调用 ``is_king_in_check``。
    """
    if Rules._jiang_face_to_face(board):
        return False, False
    if Rules.is_king_in_check(board, mover):
        return False, False
    opp = board.current_player
    return True, Rules.is_king_in_check(board, opp)


def pseudo_move_post_apply_flags_cached(
    board: Board,
    mover: str,
    cache: Optional["PostApplyFlagsCache"],
) -> Tuple[bool, bool]:
    """同 ``pseudo_move_post_apply_flags``，但以走后 ``board.zobrist_hash`` 做 LRU。"""
    if cache is None:
        return pseudo_move_post_apply_flags(board, mover)
    k = board.zobrist_hash
    hit = cache.lookup(k)
    if hit is not None:
        return hit
    res = pseudo_move_post_apply_flags(board, mover)
    cache.remember(k, res)
    return res


class PostApplyFlagsCache:
    """走后局面 ``zobrist_hash`` → ``(legal, opp_in_check)`` LRU。"""

    __slots__ = ("_maxlen", "_data")

    def __init__(self, maxlen: int = 65536) -> None:
        self._maxlen = max(256, int(maxlen))
        self._data: "OrderedDict[int, Tuple[bool, bool]]" = OrderedDict()

    def lookup(self, zobrist: int) -> Optional[Tuple[bool, bool]]:
        hit = self._data.get(zobrist)
        if hit is not None:
            self._data.move_to_end(zobrist)
        return hit

    def remember(self, zobrist: int, value: Tuple[bool, bool]) -> None:
        self._data[zobrist] = value
        self._data.move_to_end(zobrist)
        while len(self._data) > self._maxlen:
            self._data.popitem(last=False)


class MoveGivesCheckCache:
    """走子前 ``(zobrist_hash, move)`` → ``(legal, opp_in_check)`` LRU。

    - ``probe_move_gives_check``：命中时跳过 ``apply / Rules / undo``。
    - ``lookup_before_apply`` / ``remember_before_apply``：供已显式 ``apply`` 的路径
      在 Minimax 等与 ``PostApplyFlagsCache`` 组合使用。
    """

    __slots__ = ("_maxlen", "_data", "_post_apply")

    def __init__(
        self,
        maxlen: int = 65536,
        *,
        post_apply_cache: Optional[PostApplyFlagsCache] = None,
    ) -> None:
        self._maxlen = max(256, int(maxlen))
        self._data: "OrderedDict[Tuple[int, Move4], Tuple[bool, bool]]" = OrderedDict()
        self._post_apply = post_apply_cache

    @property
    def post_apply_cache(self) -> Optional[PostApplyFlagsCache]:
        return self._post_apply

    def lookup_before_apply(self, board: Board, move: Move4) -> Optional[Tuple[bool, bool]]:
        key = (board.zobrist_hash, move)
        hit = self._data.get(key)
        if hit is not None:
            self._data.move_to_end(key)
        return hit

    def remember_before_apply(
        self,
        pre_zobrist: int,
        move: Move4,
        legal: bool,
        opp_in_check: bool,
    ) -> None:
        key = (pre_zobrist, move)
        self._data[key] = (legal, opp_in_check)
        self._data.move_to_end(key)
        while len(self._data) > self._maxlen:
            self._data.popitem(last=False)

    def _trim(self) -> None:
        while len(self._data) > self._maxlen:
            self._data.popitem(last=False)

    def probe_move_gives_check(self, board: Board, move: Move4, mover: str) -> bool:
        key = (board.zobrist_hash, move)
        hit = self._data.get(key)
        if hit is not None:
            self._data.move_to_end(key)
            legal, gc = hit
            return gc if legal else False
        captured = board.apply_move(*move)
        try:
            legal, gc = pseudo_move_post_apply_flags_cached(board, mover, self._post_apply)
            self._data[key] = (legal, gc)
            self._data.move_to_end(key)
            self._trim()
            return gc if legal else False
        finally:
            board.undo_move(*move, captured)


def move_gives_check_with_undo(
    board: Board,
    move: Move4,
    mover: str,
    gives_check_cache: Optional[MoveGivesCheckCache] = None,
) -> bool:
    """``apply`` → 判定是否给对方将军 → ``undo``；非法着法则返回 ``False``。"""
    if gives_check_cache is not None:
        return gives_check_cache.probe_move_gives_check(board, move, mover)
    captured = board.apply_move(*move)
    try:
        legal, opp_in_check = pseudo_move_post_apply_flags(board, mover)
        return opp_in_check if legal else False
    finally:
        board.undo_move(*move, captured)


def apply_pseudo_legal_with_rule_cache(
    board: Board,
    move: Move4,
    mover: str,
    *,
    pre_move_cache: Optional[MoveGivesCheckCache] = None,
    post_apply_cache: Optional[PostApplyFlagsCache] = None,
) -> Optional[Tuple[Any, bool]]:
    """在 ``board`` 上尝试 ``move``：合法则保持已 apply，返回 ``(captured, opp_in_check)``；非法则盘面不变，返回 ``None``。

    ``pre_move_cache`` 命中且合法时：只做 ``apply_move``，不再调用 ``Rules``。
    未命中或需写入时：``apply`` 后用 ``pseudo_move_post_apply_flags_cached`` 判定，
    非法则 ``undo`` 并写入走子前缓存。
    """
    pre_z = board.zobrist_hash
    if pre_move_cache is not None:
        hit = pre_move_cache.lookup_before_apply(board, move)
        if hit is not None:
            legal, opp_ic = hit
            if not legal:
                return None
            captured = board.apply_move(*move)
            return captured, opp_ic

    captured = board.apply_move(*move)
    legal, opp_ic = pseudo_move_post_apply_flags_cached(board, mover, post_apply_cache)
    if not legal:
        board.undo_move(*move, captured)
        if pre_move_cache is not None:
            pre_move_cache.remember_before_apply(pre_z, move, False, False)
        return None
    if pre_move_cache is not None:
        pre_move_cache.remember_before_apply(pre_z, move, True, opp_ic)
    return captured, opp_ic
