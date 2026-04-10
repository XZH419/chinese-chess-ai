"""
分析 MinimaxAI.get_best_move / choose_move 的耗时分布。

用法（仓库根目录）:
  conda run -n chessai python scripts/profile_minimax.py

输出 cProfile 按累计时间排序的前若干项，便于定位占比 >80% 的热点。
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from chinese_chess.algorithm.minimax import MinimaxAI
    from chinese_chess.model.board import Board

    board = Board()
    ai = MinimaxAI(depth=1)

    def run_once() -> None:
        ai.get_best_move(board, time_limit=None)

    pr = cProfile.Profile()
    pr.enable()
    run_once()
    pr.disable()

    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf).sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(50)
    print(buf.getvalue())


if __name__ == "__main__":
    main()
