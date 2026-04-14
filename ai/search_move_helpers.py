"""搜索器共享：伪合法走子判定 + 「是否将军」虚拟几何快速路径。

语义对齐 ``Rules``（不修改 ``rules.py``）。车/炮/马/兵照将与揭线用虚拟 ``cell``；
无法安全推断时 fallback ``apply → Rules → undo``。LRU 见类 ``MoveGivesCheckCache`` 等。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Optional, Tuple

from engine.board import Board
from engine.rules import Rules

# 与三处搜索器一致的走法四元组
Move4 = Tuple[int, int, int, int]

_VCell = Callable[[int, int], Any]


def _virtual_cell_factory(
    board: Board, sr: int, sc: int, er: int, ec: int, moving
) -> _VCell:
    b = board.board

    def cell(r: int, c: int):
        if r == sr and c == sc:
            return None
        if r == er and c == ec:
            return moving
        return b[r][c]

    return cell


def _virtual_king_positions(
    board: Board,
    piece,
    er: int,
    ec: int,
    captured,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    rk = board.red_king_pos
    bk = board.black_king_pos
    if piece.piece_type == "jiang":
        if piece.color == "red":
            rk = (er, ec)
        else:
            bk = (er, ec)
    if captured is not None and captured.piece_type == "jiang":
        if captured.color == "red":
            rk = None
        else:
            bk = None
    return rk, bk


def _virtual_jiang_face(
    rk: Optional[Tuple[int, int]],
    bk: Optional[Tuple[int, int]],
    cell: _VCell,
) -> bool:
    if rk is None or bk is None:
        return False
    if rk[1] != bk[1]:
        return False
    col = rk[1]
    lo, hi = min(rk[0], bk[0]), max(rk[0], bk[0])
    for r in range(lo + 1, hi):
        if cell(r, col) is not None:
            return False
    return True


def _virtual_is_king_in_check_at(
    kr: int,
    kc: int,
    defender: str,
    cell: _VCell,
) -> bool:
    """镜像 ``Rules.is_king_in_check`` 的车/炮/马/兵反向检测，读虚拟 ``cell``。"""
    opponent = "black" if defender == "red" else "red"

    for dr, dc in Rules._ORTH_DELTAS:
        obstacles = 0
        r, c = kr + dr, kc + dc
        while 0 <= r < 10 and 0 <= c < 9:
            p = cell(r, c)
            if p is not None:
                if obstacles == 0:
                    if p.color == opponent and p.piece_type == "che":
                        return True
                    obstacles = 1
                else:
                    if p.color == opponent and p.piece_type == "pao":
                        return True
                    break
            r += dr
            c += dc

    for ddr, ddc in Rules._MA_ATTACK_DELTAS:
        hr, hc = kr + ddr, kc + ddc
        if not (0 <= hr < 10 and 0 <= hc < 9):
            continue
        hp = cell(hr, hc)
        if hp is None or hp.color != opponent or hp.piece_type != "ma":
            continue
        leg_r, leg_c = Rules._ma_leg_square(hr, hc, kr, kc)
        if not (0 <= leg_r < 10 and 0 <= leg_c < 9):
            continue
        if cell(leg_r, leg_c) is not None:
            continue
        return True

    if defender == "red":
        for pr, pc in ((kr - 1, kc), (kr, kc - 1), (kr, kc + 1)):
            if 0 <= pr < 10 and 0 <= pc < 9:
                pp = cell(pr, pc)
                if (
                    pp is not None
                    and pp.color == opponent
                    and pp.piece_type == "bing"
                ):
                    return True
    else:
        for pr, pc in ((kr + 1, kc), (kr, kc - 1), (kr, kc + 1)):
            if 0 <= pr < 10 and 0 <= pc < 9:
                pp = cell(pr, pc)
                if (
                    pp is not None
                    and pp.color == opponent
                    and pp.piece_type == "bing"
                ):
                    return True

    return False


def try_fast_move_legality_and_opponent_check(
    board: Board,
    move: Move4,
    mover: str,
) -> Optional[Tuple[bool, bool]]:
    """虚拟走子后 ``(legal, opp_in_check)``；无法安全推断时 ``None`` → fallback。"""
    sr, sc, er, ec = move
    b = board.board
    if not (0 <= sr < 10 and 0 <= sc < 9 and 0 <= er < 10 and 0 <= ec < 9):
        return None
    piece = b[sr][sc]
    if piece is None or piece.color != mover:
        return None
    captured = b[er][ec]
    if captured is not None and captured.color == mover:
        return None

    cell = _virtual_cell_factory(board, sr, sc, er, ec, piece)
    rk, bk = _virtual_king_positions(board, piece, er, ec, captured)

    if _virtual_jiang_face(rk, bk, cell):
        return False, False

    mk = rk if mover == "red" else bk
    if mk is None:
        return False, False
    if _virtual_is_king_in_check_at(mk[0], mk[1], mover, cell):
        return False, False

    opp = "black" if mover == "red" else "red"
    ok = bk if mover == "red" else rk
    if ok is None:
        return True, False
    gives = _virtual_is_king_in_check_at(ok[0], ok[1], opp, cell)
    return True, gives


def _slow_move_gives_check_apply(board: Board, move: Move4, mover: str) -> bool:
    captured = board.apply_move(*move)
    try:
        legal, opp_in_check = pseudo_move_post_apply_flags(board, mover)
        return opp_in_check if legal else False
    finally:
        board.undo_move(*move, captured)


def fast_move_gives_check(
    board: Board,
    move: Move4,
    mover: str,
    gives_check_cache: Optional["MoveGivesCheckCache"] = None,
) -> bool:
    """走法是否给对方将军（非法则 ``False``）。优先虚拟几何 + LRU，必要时 fallback。"""
    if gives_check_cache is not None:
        return gives_check_cache.probe_move_gives_check(board, move, mover)
    res = try_fast_move_legality_and_opponent_check(board, move, mover)
    if res is not None:
        legal, gc = res
        return gc if legal else False
    return _slow_move_gives_check_apply(board, move, mover)


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
        res = try_fast_move_legality_and_opponent_check(board, move, mover)
        if res is not None:
            legal, gc = res
            self._data[key] = (legal, gc)
            self._data.move_to_end(key)
            self._trim()
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
    """判定是否给对方将军（非法则 ``False``）；等同 ``fast_move_gives_check``。"""
    return fast_move_gives_check(board, move, mover, gives_check_cache)


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
    # 防御式检查：在 MCTS/Minimax 的 apply/undo 高频路径中，
    # 一旦有“从空格走子”或“吃友军/吃将”等异常走法混入，就会破坏 move_stack 的可逆性，
    # 最终在 undo_move 时触发难以定位的状态崩溃。
    sr, sc, er, ec = move
    b = board.board
    piece = b[sr][sc]
    if piece is None or piece.color != mover:
        return None
    target = b[er][ec]
    if target is not None:
        if target.color == mover:
            return None
        if target.piece_type == "jiang":
            return None

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
