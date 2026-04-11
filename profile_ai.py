"""AI 搜索性能分析：cProfile + pstats，观测 Minimax 与 Tapered Evaluation。

- 绕过开局库：``game_history`` 长度 ≥ 30 时不再查 ``OPENING_BOOK``（占位键与真实 Zobrist 碰撞可忽略）。
- 压力点：中局（自开局起若干合法步）+ 子力较少的残局样例，便于不同 ``phase`` 下的评估开销。
- 固定搜索深度 5，保证 ncalls 有统计意义。

用法（仓库根目录）:
  python profile_ai.py
  conda run -n chessai python profile_ai.py
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 与 ``MinimaxAI.get_best_move`` 中 ``len(gh_book) < 30`` 对齐；负整数占位，避免误入重复局面检测
OPENING_BOOK_BYPASS_HISTORY: List[int] = [-(i + 1) for i in range(30)]


def _play_legal_plies(board, n: int) -> int:
    """自当前局面起执行至多 ``n`` 步**完全合法**走法（确定性：按坐标排序取首步）。返回实际执行的步数。"""
    from chinese_chess.model.rules import Rules

    played = 0
    for _ in range(n):
        moves = list(Rules.get_pseudo_legal_moves(board, board.current_player))
        moves.sort()
        progressed = False
        for move in moves:
            sr, sc, er, ec = move
            mover = board.current_player
            captured = board.apply_move(sr, sc, er, ec)
            if Rules.is_king_in_check(board, mover) or Rules._jiang_face_to_face(board):
                board.undo_move(sr, sc, er, ec, captured)
                continue
            progressed = True
            played += 1
            break
        if not progressed:
            break
    return played


def build_midgame_board(plies: int = 10):
    """标准开局上叠 ``plies`` 步合法走子，得到较复杂的中局。"""
    from chinese_chess.model.board import Board

    board = Board()
    _play_legal_plies(board, plies)
    return board


def build_sparse_endgame_board():
    """手动残局：双方保留将 + 车/马/炮，子力少、仍有战术空间；``phase`` 明显低于满编。"""
    from chinese_chess.model import zobrist
    from chinese_chess.model.board import Board
    from chinese_chess.model.piece import Piece

    b = Board()
    for r in range(b.rows):
        for c in range(b.cols):
            b.board[r][c] = None
    b.active_pieces["red"].clear()
    b.active_pieces["black"].clear()

    # 黑方九宫与过河子力
    b.board[0][4] = Piece("black", "jiang")
    b.board[0][0] = Piece("black", "che")
    b.board[2][4] = Piece("black", "pao")
    b.board[3][1] = Piece("black", "ma")
    # 红方
    b.board[9][4] = Piece("red", "jiang")
    b.board[9][8] = Piece("red", "che")
    b.board[7][4] = Piece("red", "pao")
    b.board[6][7] = Piece("red", "ma")

    b.red_king_pos = (9, 4)
    b.black_king_pos = (0, 4)
    b.current_player = "red"
    for r in range(b.rows):
        for c in range(b.cols):
            p = b.board[r][c]
            if p is not None:
                b.active_pieces[p.color].add((r, c))
    b.zobrist_hash = zobrist.full_hash(b)
    b.state_counts.clear()
    b.state_counts[b.zobrist_hash] = 1
    return b


def _print_top(pr: cProfile.Profile, title: str, sort_key: pstats.SortKey) -> None:
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats(sort_key).print_stats(30)
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(buf.getvalue())


def _run_scenario(name: str, board, depth: int) -> None:
    from chinese_chess.algorithm.minimax import MinimaxAI

    ai = MinimaxAI(depth=depth, verbose=False)
    pr = cProfile.Profile()
    pr.enable()
    ai.get_best_move(board, game_history=OPENING_BOOK_BYPASS_HISTORY, time_limit=None)
    pr.disable()

    print("\n" + "#" * 72)
    print(f"# 场景: {name}  |  depth={depth}  |  opening book 已绕过")
    print("#" * 72)
    for title, key in (
        ("tottime（函数自身时间，降序 Top 30）", pstats.SortKey.TIME),
        ("cumtime（含子调用累积时间，降序 Top 30）", pstats.SortKey.CUMULATIVE),
    ):
        _print_top(pr, title, key)


def main() -> None:
    from chinese_chess.model.board import Board
    from chinese_chess.model.rules import Rules

    # 预热 import，避免误计入剖析
    Rules.is_game_over(Board())

    profile_depth = 5

    mid = build_midgame_board(plies=10)
    _run_scenario("中局（开局 + 10 步合法走法）", mid, profile_depth)

    end = build_sparse_endgame_board()
    _run_scenario("残局样例（将车马炮 vs 将车马炮）", end, profile_depth)


if __name__ == "__main__":
    main()
