"""MCTS 单核极限性能分析脚本。

直接调用底层 _run_single_mcts_tree 函数（绕过多进程调度），
用 cProfile 收集 2000 次模拟的完整函数调用热点。
"""

import cProfile
import io
import pstats
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from chinese_chess.model.board import Board
from chinese_chess.algorithm.mcts import _run_single_mcts_tree


def main():
    board = Board()
    print(f"初始局面子力数: {board.piece_count()}")
    print(f"开始 cProfile 分析: max_simulations=2000, time_limit=999 (无时间限制)")
    print()

    profiler = cProfile.Profile()
    profiler.enable()

    result = _run_single_mcts_tree(
        board=board,
        max_simulations=2000,
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
    stats.print_stats(30)
    print(stream.getvalue())

    print("=" * 72)
    print("性能分析完成。请重点检查 `get_legal_moves`、`is_valid_move` 等")
    print("规则函数的 `tottime` 占比。")


if __name__ == "__main__":
    main()
