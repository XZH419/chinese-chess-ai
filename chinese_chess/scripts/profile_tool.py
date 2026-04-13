"""统一性能分析工具：Minimax / MCTS 引擎的 cProfile 热点剖析。

提供两个子命令：

- ``minimax``：Minimax 搜索引擎性能分析（中局 + 残局双场景，绕过开局库）。
- ``mcts``：MCTS 单核极限性能分析（直接调用底层搜索函数）。

用法（仓库根目录）::

    # Minimax 性能分析（depth=5，中局 + 残局，绕过开局库）
    python -m chinese_chess.scripts.profile_tool minimax
    python -m chinese_chess.scripts.profile_tool minimax --depth 4 --plies 15 --top 20

    # MCTS 单核极限热点（2000 次模拟，初始局面）
    python -m chinese_chess.scripts.profile_tool mcts
    python -m chinese_chess.scripts.profile_tool mcts --simulations 5000 --top 20
"""

from __future__ import annotations

import sys
import os

# ── 路径修复 ──
# 本文件位于 chinese_chess/scripts/ 下，需要将项目根目录（上两级）加入
# sys.path，才能正确 import chinese_chess.* 包。
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse
import cProfile
import io
import pstats
from typing import List


# ═══════════════════════════════════════════════════════════════
#  公共工具函数
# ═══════════════════════════════════════════════════════════════

# 与 MinimaxAI.get_best_move 中 len(game_history) < 30 对齐；
# 用负整数占位，避免误入重复局面检测或 Zobrist 碰撞
OPENING_BOOK_BYPASS_HISTORY: List[int] = [-(i + 1) for i in range(30)]


def _print_top(
    pr: cProfile.Profile,
    title: str,
    sort_key: pstats.SortKey,
    top_n: int = 30,
) -> None:
    """格式化输出 cProfile 统计结果。

    Args:
        pr: 已完成采集的 cProfile.Profile 实例。
        title: 本段统计的标题描述。
        sort_key: pstats 排序键（如 TIME / CUMULATIVE）。
        top_n: 打印前 N 条热点函数。
    """
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats(sort_key).print_stats(top_n)
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(buf.getvalue())


def _play_legal_plies(board, n: int) -> int:
    """自当前局面起执行至多 ``n`` 步完全合法走法（确定性：按坐标排序取首步）。

    用于从初始局面快速推进到中局，为性能剖析提供更有代表性的棋盘状态。

    Args:
        board: 棋盘实例（会被原地修改）。
        n: 最多执行的步数。

    Returns:
        实际执行的合法步数。
    """
    from chinese_chess.model.rules import Rules

    played = 0
    for _ in range(n):
        moves = list(Rules.get_pseudo_legal_moves(board, board.current_player))
        moves.sort()
        has_progressed = False
        for move in moves:
            sr, sc, er, ec = move
            mover = board.current_player
            captured = board.apply_move(sr, sc, er, ec)
            if Rules.is_king_in_check(board, mover) or Rules._jiang_face_to_face(board):
                board.undo_move(sr, sc, er, ec, captured)
                continue
            has_progressed = True
            played += 1
            break
        if not has_progressed:
            break
    return played


def build_midgame_board(plies: int = 10):
    """标准开局叠加 ``plies`` 步合法走子，得到较复杂的中局局面。

    Args:
        plies: 从开局起推进的步数。

    Returns:
        处于中局阶段的 Board 实例。
    """
    from chinese_chess.model.board import Board

    board = Board()
    _play_legal_plies(board, plies)
    return board


def build_sparse_endgame_board():
    """手动构造残局：双方仅保留将 + 车 / 马 / 炮，子力稀少但仍有战术空间。

    该局面的 ``phase`` 值明显低于满编开局，适合剖析残局阶段的评估开销。

    Returns:
        处于残局阶段的 Board 实例。
    """
    from chinese_chess.model import zobrist
    from chinese_chess.model.board import Board
    from chinese_chess.model.piece import Piece

    b = Board()
    for r in range(b.rows):
        for c in range(b.cols):
            b.board[r][c] = None
    b.active_pieces["red"].clear()
    b.active_pieces["black"].clear()

    # 黑方：将 + 车 + 炮 + 马
    b.board[0][4] = Piece("black", "jiang")
    b.board[0][0] = Piece("black", "che")
    b.board[2][4] = Piece("black", "pao")
    b.board[3][1] = Piece("black", "ma")
    # 红方：帅 + 车 + 炮 + 马
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


# ═══════════════════════════════════════════════════════════════
#  子命令一：minimax（Minimax 性能分析）
# ═══════════════════════════════════════════════════════════════


def _run_minimax_scenario(
    name: str,
    board,
    depth: int,
    top_n: int,
) -> None:
    """执行单个 Minimax 场景剖析（绕过开局库）。

    Args:
        name: 场景名称（用于打印标题）。
        board: 待剖析的棋盘状态。
        depth: Minimax 搜索深度。
        top_n: 打印前 N 条热点。
    """
    from chinese_chess.algorithm.minimax import MinimaxAI

    ai = MinimaxAI(depth=depth, verbose=False)
    pr = cProfile.Profile()
    pr.enable()
    ai.get_best_move(board, game_history=OPENING_BOOK_BYPASS_HISTORY, time_limit=None)
    pr.disable()

    print("\n" + "#" * 72)
    print(f"# 场景: {name}  |  depth={depth}  |  开局库已绕过")
    print("#" * 72)
    for title, key in (
        ("tottime（函数自身时间，降序）", pstats.SortKey.TIME),
        ("cumtime（含子调用累积时间，降序）", pstats.SortKey.CUMULATIVE),
    ):
        _print_top(pr, title, key, top_n)


def cmd_profile_minimax(args: argparse.Namespace) -> None:
    """Minimax 性能分析：中局 + 残局两个场景，绕过开局库。

    在中局和残局两种典型局面下分别运行 Minimax 搜索，
    输出 cProfile 热点统计（tottime + cumtime 双视角）。

    Args:
        args: 命令行参数（含 ``depth``、``plies``、``top``）。
    """
    from chinese_chess.model.board import Board
    from chinese_chess.model.rules import Rules

    depth = args.depth
    plies = args.plies
    top_n = args.top

    # 预热 import，避免首次导入开销被误计入剖析
    Rules.is_game_over(Board())

    print(f"[Minimax 性能分析] depth={depth}, 中局推进步数={plies}")
    print()

    mid = build_midgame_board(plies=plies)
    _run_minimax_scenario(f"中局（开局 + {plies} 步合法走法）", mid, depth, top_n)

    end = build_sparse_endgame_board()
    _run_minimax_scenario("残局样例（将车马炮 vs 将车马炮）", end, depth, top_n)


# ═══════════════════════════════════════════════════════════════
#  子命令二：mcts（MCTS 性能分析）
# ═══════════════════════════════════════════════════════════════


def cmd_profile_mcts(args: argparse.Namespace) -> None:
    """MCTS 单核极限热点分析：直接调用底层 ``_run_single_mcts_tree``。

    绕过多进程调度，以 cProfile 收集指定次模拟的完整函数调用热点。

    Args:
        args: 命令行参数（含 ``simulations`` 和 ``top``）。
    """
    from chinese_chess.model.board import Board
    from chinese_chess.algorithm.mcts import _run_single_mcts_tree

    sims = args.simulations
    top_n = args.top

    board = Board()
    print(f"[MCTS 性能分析] 初始局面子力数: {board.piece_count()}")
    print(f"[MCTS 性能分析] cProfile 分析: max_simulations={sims}, time_limit=999（无时间限制）")
    print()

    profiler = cProfile.Profile()
    profiler.enable()

    result = _run_single_mcts_tree(
        board=board,
        max_simulations=sims,
        time_limit=999.0,
        seed_offset=0,
    )

    profiler.disable()

    total_visits = sum(int(s["visits"]) for s in result.values())
    print(f"搜索完成: {len(result)} 个根子节点, 合计 visits={total_visits}")
    print()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumtime")
    stats.print_stats(top_n)
    print(stream.getvalue())

    print("=" * 72)
    print("性能分析完成。请重点检查 get_pseudo_legal_moves、is_king_in_check 等")
    print("规则函数的 tottime 占比。")


# ═══════════════════════════════════════════════════════════════
#  子命令三：mcts_minimax（MCTS-Minimax 搜索性能分析）
# ═══════════════════════════════════════════════════════════════


def cmd_profile_mcts_minimax(args: argparse.Namespace) -> None:
    """MCTS-Minimax 单树性能分析：调用 ``_run_single_mcts_minimax_tree``。"""
    from chinese_chess.model.board import Board
    from chinese_chess.algorithm.mcts_minimax import _run_single_mcts_minimax_tree

    sims = args.simulations
    top_n = args.top

    board = Board()
    print(f"[MCTS-Minimax 性能分析] 初始局面子力数: {board.piece_count()}")
    print(f"[MCTS-Minimax 性能分析] cProfile: max_simulations={sims}, time_limit=999")
    print()

    profiler = cProfile.Profile()
    profiler.enable()

    child_stats, probe_stats = _run_single_mcts_minimax_tree(
        board=board,
        max_simulations=sims,
        time_limit=999.0,
        seed_offset=0,
    )

    profiler.disable()

    total_visits = sum(int(s["visits"]) for s in child_stats.values())
    print(f"搜索完成: {len(child_stats)} 个根子节点, 合计 visits={total_visits}")
    print(
        f"Probe: {probe_stats.get('probes', 0)} 次, "
        f"budget calls {probe_stats.get('budget_calls_used', 0)}/"
        f"{probe_stats.get('budget_calls_max', 0)}"
    )
    print()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumtime")
    stats.print_stats(top_n)
    print(stream.getvalue())

    print("=" * 72)
    print("MCTS-Minimax 性能分析完成。")


# ═══════════════════════════════════════════════════════════════
#  CLI 入口：argparse 子命令
# ═══════════════════════════════════════════════════════════════


def main() -> None:
    """解析命令行参数并分派到对应的子命令处理函数。"""
    parser = argparse.ArgumentParser(
        description="中国象棋 AI 性能分析工具（cProfile 热点剖析）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── minimax ──
    p_minimax = subparsers.add_parser(
        "minimax",
        help="Minimax 性能分析（中局 + 残局，默认 depth=5）",
    )
    p_minimax.add_argument("--depth", type=int, default=5, help="搜索深度（默认 5）")
    p_minimax.add_argument("--plies", type=int, default=10, help="中局推进步数（默认 10）")
    p_minimax.add_argument("--top", type=int, default=30, help="打印热点条数（默认 30）")
    p_minimax.set_defaults(func=cmd_profile_minimax)

    # ── mcts ──
    p_mcts = subparsers.add_parser(
        "mcts",
        help="MCTS 单核性能分析（默认 2000 次模拟）",
    )
    p_mcts.add_argument("--simulations", type=int, default=2000, help="模拟次数（默认 2000）")
    p_mcts.add_argument("--top", type=int, default=30, help="打印热点条数（默认 30）")
    p_mcts.set_defaults(func=cmd_profile_mcts)

    p_mcts_minimax = subparsers.add_parser(
        "mcts_minimax",
        help="MCTS-Minimax 单核性能分析（默认 1500 次模拟）",
    )
    p_mcts_minimax.add_argument("--simulations", type=int, default=1500, help="模拟次数（默认 1500）")
    p_mcts_minimax.add_argument("--top", type=int, default=30, help="打印热点条数（默认 30）")
    p_mcts_minimax.set_defaults(func=cmd_profile_mcts_minimax)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
