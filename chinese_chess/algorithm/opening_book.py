"""开局库：手写 ``BASE_BOOK`` + 左右镜像自动生成对称变例。

- ``OPENING_SEQUENCE_BOOK``：键为从开局起的走子序列 ``Tuple[Move4, ...]``，值为推荐着法列表（含镜像合并）。
- ``OPENING_BOOK``：由序列书回放盘面得到的 **Zobrist → 推荐着法**，供 ``MinimaxAI`` 等仅持有哈希链的调用方 O(1) 查表。

走法格式：``(src_r, src_c, dst_r, dst_c)``；列镜像关于中轴第 4 列：``c' = 8 - c``。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from chinese_chess.model.board import Board
from chinese_chess.model.rules import Rules

Move4 = Tuple[int, int, int, int]


def mirror_move(move: Move4) -> Move4:
    """将走法沿棋盘中轴线（第 4 列）进行左右镜像翻转。"""
    r1, c1, r2, c2 = move
    return (r1, 8 - c1, r2, 8 - c2)


def _merge_moves(book: Dict[Tuple[Move4, ...], List[Move4]], seq: Tuple[Move4, ...], moves: List[Move4]) -> None:
    bucket = book.setdefault(seq, [])
    for m in moves:
        if m not in bucket:
            bucket.append(m)


def _sequences_to_zobrist(seq_book: Dict[Tuple[Move4, ...], List[Move4]]) -> Dict[int, List[Move4]]:
    """将「历史走子序列 → 推荐着」投影为「当前局 Zobrist → 推荐着」。"""
    out: Dict[int, List[Move4]] = {}
    for seq, moves in seq_book.items():
        b = Board()
        ok = True
        for m in seq:
            if not Rules.is_valid_move(b, m[0], m[1], m[2], m[3])[0]:
                ok = False
                break
            b.apply_move(*m)
        if not ok:
            continue
        h = b.zobrist_hash
        bucket = out.setdefault(h, [])
        for mv in moves:
            if mv in bucket:
                continue
            if Rules.is_valid_move(b, mv[0], mv[1], mv[2], mv[3])[0]:
                bucket.append(mv)
    return out


# ---------------------------------------------------------------------------
# 手写开局路线（仅维护一侧；镜像由程序生成）
# ---------------------------------------------------------------------------
BASE_BOOK: Dict[Tuple[Move4, ...], List[Move4]] = {
    # 0 回合：初始局面，红方起手
    (): [
        (7, 7, 7, 4),  # 当头炮（炮二平五）
        (6, 6, 5, 6),  # 仙人指路（兵七进一）
        (9, 7, 7, 6),  # 起马局（马八进七）
        (9, 2, 7, 4),  # 飞相局（相三进五）
    ],
    # 路线一：中炮对屏风马
    ((7, 7, 7, 4),): [
        (0, 7, 2, 6),
        (2, 7, 2, 4),
    ],
    ((7, 7, 7, 4), (0, 7, 2, 6)): [
        (9, 7, 7, 6),
    ],
    ((7, 7, 7, 4), (0, 7, 2, 6), (9, 7, 7, 6)): [
        (0, 1, 2, 2),
    ],
    ((7, 7, 7, 4), (0, 7, 2, 6), (9, 7, 7, 6), (0, 1, 2, 2)): [
        (6, 6, 5, 6),
    ],
    # 路线二：仙人指路
    ((6, 6, 5, 6),): [
        (2, 7, 2, 6),
        (3, 6, 4, 6),
    ],
    ((6, 6, 5, 6), (2, 7, 2, 6)): [
        (9, 2, 7, 4),
    ],
    ((6, 6, 5, 6), (2, 7, 2, 6), (9, 2, 7, 4)): [
        (0, 7, 2, 6),
    ],
    ((6, 6, 5, 6), (3, 6, 4, 6)): [
        (9, 7, 7, 6),
    ],
    ((6, 6, 5, 6), (3, 6, 4, 6), (9, 7, 7, 6)): [
        (0, 7, 2, 6),
    ],
    # 路线三：起马局
    ((9, 7, 7, 6),): [
        (3, 2, 4, 2),
    ],
    ((9, 7, 7, 6), (3, 2, 4, 2)): [
        (7, 1, 7, 3),
        (9, 2, 7, 4),
    ],
    ((9, 7, 7, 6), (3, 2, 4, 2), (7, 1, 7, 3)): [
        (0, 1, 2, 2),
    ],
    # 路线四：飞相局
    ((9, 2, 7, 4),): [
        (2, 7, 2, 4),
        (0, 7, 2, 6),
    ],
    ((9, 2, 7, 4), (2, 7, 2, 4)): [
        (9, 7, 7, 6),
    ],
    ((9, 2, 7, 4), (2, 7, 2, 4), (9, 7, 7, 6)): [
        (0, 7, 2, 6),
    ],
}

# 合并原序列与镜像序列（对称翼、左右炮/马等变例）
OPENING_SEQUENCE_BOOK: Dict[Tuple[Move4, ...], List[Move4]] = {}
for _seq, _moves in BASE_BOOK.items():
    _merge_moves(OPENING_SEQUENCE_BOOK, _seq, _moves)
    _mirrored_seq = tuple(mirror_move(m) for m in _seq)
    _mirrored_moves = [mirror_move(m) for m in _moves]
    _merge_moves(OPENING_SEQUENCE_BOOK, _mirrored_seq, _mirrored_moves)

# 引擎查表：Zobrist → 库着（``MinimaxAI`` 仍 ``from .opening_book import OPENING_BOOK``）
OPENING_BOOK: Dict[int, List[Move4]] = _sequences_to_zobrist(OPENING_SEQUENCE_BOOK)


def run_sanity_check() -> bool:
    """对 ``BASE_BOOK`` 做冗余自检；在包内导入完成后调用。"""
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    ok = True

    # 1. 同一 Key 下 Value 列表内重复坐标
    for seq, moves in BASE_BOOK.items():
        seen: set[Move4] = set()
        dups: List[Move4] = []
        for m in moves:
            if m in seen:
                dups.append(m)
            seen.add(m)
        if dups:
            ok = False
            print(f"{RED}[ERROR]{RESET} 重复应对走法 Key={seq!r} 重复坐标: {dups!r}")

    # 2. 跨分支：某序列的完全镜像已作为另一 Key 存在（键级对称重复手写）
    base_keys = set(BASE_BOOK.keys())
    reported_mirror_key_pairs: set[Tuple[Tuple[Move4, ...], Tuple[Move4, ...]]] = set()
    for seq_a in list(BASE_BOOK.keys()):
        mirrored_seq_a = tuple(mirror_move(m) for m in seq_a)
        if mirrored_seq_a in base_keys and seq_a != mirrored_seq_a:
            pair = (seq_a, mirrored_seq_a) if seq_a < mirrored_seq_a else (mirrored_seq_a, seq_a)
            if pair in reported_mirror_key_pairs:
                continue
            reported_mirror_key_pairs.add(pair)
            ok = False
            print(
                f"{RED}[ERROR]{RESET} 跨分支镜像键冗余（请只保留一侧，另一侧依赖镜像生成）:\n"
                f"       seq_A          = {seq_a!r}\n"
                f"       mirrored_seq_A = {mirrored_seq_a!r}"
            )

    # 3. 同一 Key 下 Value 中同时出现某走法与其镜像（着法级对称重复手写）
    for seq, moves in BASE_BOOK.items():
        move_set = set(moves)
        for m in moves:
            mm = mirror_move(m)
            if m != mm and mm in move_set:
                ok = False
                print(
                    f"{RED}[ERROR]{RESET} 同局面镜像走法冗余 Key={seq!r}\n"
                    f"       m = {m!r} 与 mirror_move(m) = {mm!r} 同时出现在列表中"
                )
                break

    if ok:
        print(f"{GREEN}[SUCCESS] BASE_BOOK 自检通过，无任何数据冗余！{RESET}")
        print(f"  BASE_BOOK 键数量: {len(BASE_BOOK)}")
        print(f"  合并镜像后 OPENING_SEQUENCE_BOOK 键数量: {len(OPENING_SEQUENCE_BOOK)}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run_sanity_check() else 1)
