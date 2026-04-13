"""开局库：手写 ``BASE_BOOK`` + 左右镜像自动生成对称变例。

本模块维护两级数据结构：

- ``OPENING_SEQUENCE_BOOK``：键为从开局起的走子序列 ``Tuple[Move4, ...]``，
  值为该局面下的推荐着法列表（含镜像合并）。适合已知完整路径的调用方。
- ``OPENING_BOOK``：由序列书回放棋盘得到的 **Zobrist Hash → 推荐着法**，
  供 ``MinimaxAI`` / ``MCTSAI`` 等仅持有当前局面哈希的调用方 O(1) 查表。

**走法格式**：``(src_row, src_col, dst_row, dst_col)``。

**镜像机制**：棋盘沿中轴线（第 4 列）左右对称，镜像公式 ``col' = 8 - col``。
手写 ``BASE_BOOK`` 仅维护一侧的变例，镜像变例由程序自动生成并合并，
确保不会遗漏对称局面，同时避免人工维护两侧导致的数据不一致。

**开局知识举例**：
- 起马局：红右马进七 ``(9,7,7,6)``，黑应挺同侧卒 ``(3,6,4,6)`` 制约马腿。
  处理对称棋局时，镜像映射确保对侧的坐标（行与列）能够正确映射，
  例如右马对应同侧卒的制约关系在镜像后依然成立。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from chinese_chess.model.board import Board
from chinese_chess.model.rules import Rules

# 走法四元组：(起始行, 起始列, 目标行, 目标列)
Move4 = Tuple[int, int, int, int]


def mirror_move(move: Move4) -> Move4:
    """将走法沿棋盘中轴线（第 4 列）进行左右镜像翻转。

    镜像公式：列坐标 ``col' = 8 - col``，行坐标不变。
    用于从手写的单侧开局库自动生成对称变例。

    Args:
        move: 原始走法四元组 ``(src_row, src_col, dst_row, dst_col)``。

    Returns:
        镜像后的走法四元组。
    """
    r1, c1, r2, c2 = move
    return (r1, 8 - c1, r2, 8 - c2)


def _merge_moves(
    book: Dict[Tuple[Move4, ...], List[Move4]],
    seq: Tuple[Move4, ...],
    moves: List[Move4],
) -> None:
    """将推荐着法列表合并到开局库中，自动去重。

    若 ``seq`` 键不存在则创建新条目；若已存在则逐一追加未收录的着法。

    Args:
        book: 目标开局库字典（会被原地修改）。
        seq: 走子序列键。
        moves: 待合并的推荐着法列表。
    """
    bucket = book.setdefault(seq, [])
    for m in moves:
        if m not in bucket:
            bucket.append(m)


def _sequences_to_zobrist(
    seq_book: Dict[Tuple[Move4, ...], List[Move4]],
) -> Dict[int, List[Move4]]:
    """将"走子序列 → 推荐着法"映射投影为"Zobrist Hash → 推荐着法"。

    逐一回放每条走子序列到一个新棋盘上，记录终态的 Zobrist Hash；
    如果回放过程中某步不合法，则跳过该条目（防止手写数据错误导致崩溃）。

    Args:
        seq_book: 以走子序列为键的开局库。

    Returns:
        以 Zobrist Hash 为键、合法推荐着法列表为值的查询表。
    """
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
# 手写开局路线（仅维护一侧；镜像变例由程序自动生成）
# ---------------------------------------------------------------------------
BASE_BOOK: Dict[Tuple[Move4, ...], List[Move4]] = {
    # ── 0 回合：初始局面，红方起手 ──
    (): [
        (7, 7, 7, 4),  # 当头炮（炮二平五）：中炮开局，控制中路
        (6, 6, 5, 6),  # 仙人指路（兵七进一）：探路兵，试探黑方应手
        (9, 7, 7, 6),  # 起马局（马八进七）：右马活跃，准备巡河
        (9, 2, 7, 4),  # 飞相局（相三进五）：中相稳固，防守型开局
    ],

    # ── 路线一：中炮对屏风马 ──
    ((7, 7, 7, 4),): [
        (0, 7, 2, 6),  # 黑应：马八进七（屏风马）
        (2, 7, 2, 4),  # 黑应：炮八平五（顺手炮 / 列手炮）
    ],
    ((7, 7, 7, 4), (0, 7, 2, 6)): [
        (9, 7, 7, 6),  # 红续：马八进七，双马齐出
    ],
    ((7, 7, 7, 4), (0, 7, 2, 6), (9, 7, 7, 6)): [
        (0, 1, 2, 2),  # 黑续：马二进三
    ],
    ((7, 7, 7, 4), (0, 7, 2, 6), (9, 7, 7, 6), (0, 1, 2, 2)): [
        (6, 6, 5, 6),  # 红续：兵七进一，仙人指路
    ],

    # ── 路线二：仙人指路 ──
    ((6, 6, 5, 6),): [
        (2, 7, 2, 6),  # 黑应：炮八平七（对角炮）
        (3, 6, 4, 6),  # 黑应：卒七进一（挺卒应对）
    ],
    ((6, 6, 5, 6), (2, 7, 2, 6)): [
        (9, 2, 7, 4),  # 红续：相三进五（飞相）
    ],
    ((6, 6, 5, 6), (2, 7, 2, 6), (9, 2, 7, 4)): [
        (0, 7, 2, 6),  # 黑续：马八进七
    ],
    ((6, 6, 5, 6), (3, 6, 4, 6)): [
        (9, 7, 7, 6),  # 红续：马八进七
    ],
    ((6, 6, 5, 6), (3, 6, 4, 6), (9, 7, 7, 6)): [
        (0, 7, 2, 6),  # 黑续：马八进七
    ],

    # ── 路线三：起马局（马二进三 / 镜像自动生成马八进七） ──
    # 黑方应挺同侧卒制约马腿：col 6 卒进 1（卒七进一），
    # 确保挡住红马下一步进窝（7,6 → 5,5 的蹩马腿位于 6,6）。
    # 镜像后：红左马 (9,1,7,2) 对应黑挺 col 2 卒 (3,2,4,2)，
    # 同侧制约关系在对称映射下依然成立。
    ((9, 7, 7, 6),): [
        (3, 6, 4, 6),  # 黑应：挺同侧卒制约马腿
    ],
    ((9, 7, 7, 6), (3, 6, 4, 6)): [
        (7, 1, 7, 3),  # 红续：炮二平四（偏炮）
        (9, 2, 7, 4),  # 红续：相三进五（飞相）
    ],
    ((9, 7, 7, 6), (3, 6, 4, 6), (7, 1, 7, 3)): [
        (0, 1, 2, 2),  # 黑续：马二进三
    ],

    # ── 路线四：飞相局 ──
    ((9, 2, 7, 4),): [
        (2, 7, 2, 4),  # 黑应：炮八平五（中炮对飞相）
        (0, 7, 2, 6),  # 黑应：马八进七
    ],
    ((9, 2, 7, 4), (2, 7, 2, 4)): [
        (9, 7, 7, 6),  # 红续：马八进七
    ],
    ((9, 2, 7, 4), (2, 7, 2, 4), (9, 7, 7, 6)): [
        (0, 7, 2, 6),  # 黑续：马八进七
    ],
}

# ---------------------------------------------------------------------------
# 合并原序列与镜像序列（自动为每条路线生成对称翼的变例）
# ---------------------------------------------------------------------------
OPENING_SEQUENCE_BOOK: Dict[Tuple[Move4, ...], List[Move4]] = {}
for _seq, _moves in BASE_BOOK.items():
    _merge_moves(OPENING_SEQUENCE_BOOK, _seq, _moves)
    _mirrored_seq = tuple(mirror_move(m) for m in _seq)
    _mirrored_moves = [mirror_move(m) for m in _moves]
    _merge_moves(OPENING_SEQUENCE_BOOK, _mirrored_seq, _mirrored_moves)

# ---------------------------------------------------------------------------
# 引擎查表：Zobrist Hash → 推荐着法列表
# MinimaxAI / MCTSAI 通过 ``from .opening_book import OPENING_BOOK`` 使用
# ---------------------------------------------------------------------------
OPENING_BOOK: Dict[int, List[Move4]] = _sequences_to_zobrist(OPENING_SEQUENCE_BOOK)


def run_sanity_check() -> bool:
    """对 ``BASE_BOOK`` 进行数据冗余自检。

    检查三类常见错误：
    1. 同一键下的推荐着法列表中是否存在重复坐标。
    2. 是否有走子序列与其镜像序列同时作为键出现（应仅保留一侧）。
    3. 同一键下是否同时出现某走法与其镜像（着法级冗余）。

    Returns:
        自检通过返回 True，存在任何冗余则打印错误信息并返回 False。
    """
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    is_valid = True

    # 检查一：同一键下推荐着法是否有重复坐标
    for seq, moves in BASE_BOOK.items():
        seen: set[Move4] = set()
        dups: List[Move4] = []
        for m in moves:
            if m in seen:
                dups.append(m)
            seen.add(m)
        if dups:
            is_valid = False
            print(f"{RED}[ERROR]{RESET} 重复应对走法 Key={seq!r} 重复坐标: {dups!r}")

    # 检查二：某序列的完全镜像是否已作为另一个键存在（键级对称冗余）
    base_keys = set(BASE_BOOK.keys())
    reported_mirror_key_pairs: set[Tuple[Tuple[Move4, ...], Tuple[Move4, ...]]] = set()
    for seq_a in list(BASE_BOOK.keys()):
        mirrored_seq_a = tuple(mirror_move(m) for m in seq_a)
        if mirrored_seq_a in base_keys and seq_a != mirrored_seq_a:
            pair = (seq_a, mirrored_seq_a) if seq_a < mirrored_seq_a else (mirrored_seq_a, seq_a)
            if pair in reported_mirror_key_pairs:
                continue
            reported_mirror_key_pairs.add(pair)
            is_valid = False
            print(
                f"{RED}[ERROR]{RESET} 跨分支镜像键冗余（请只保留一侧，另一侧依赖镜像生成）:\n"
                f"       seq_A          = {seq_a!r}\n"
                f"       mirrored_seq_A = {mirrored_seq_a!r}"
            )

    # 检查三：同一键下是否同时出现某走法与其镜像（着法级对称冗余）
    for seq, moves in BASE_BOOK.items():
        move_set = set(moves)
        for m in moves:
            mm = mirror_move(m)
            if m != mm and mm in move_set:
                is_valid = False
                print(
                    f"{RED}[ERROR]{RESET} 同局面镜像走法冗余 Key={seq!r}\n"
                    f"       m = {m!r} 与 mirror_move(m) = {mm!r} 同时出现在列表中"
                )
                break

    if is_valid:
        print(f"{GREEN}[SUCCESS] BASE_BOOK 自检通过，无任何数据冗余！{RESET}")
        print(f"  BASE_BOOK 键数量: {len(BASE_BOOK)}")
        print(f"  合并镜像后 OPENING_SEQUENCE_BOOK 键数量: {len(OPENING_SEQUENCE_BOOK)}")
    return is_valid


if __name__ == "__main__":
    raise SystemExit(0 if run_sanity_check() else 1)
