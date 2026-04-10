"""AI 搜索性能分析：cProfile + pstats，定位 Minimax 热点。

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

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from chinese_chess.algorithm.minimax import MinimaxAI
    from chinese_chess.model.board import Board
    from chinese_chess.model.rules import Rules

    board = Board()
    Rules.is_game_over(board)  # 显式使用 Rules；不计入下方 cProfile

    ai = MinimaxAI(depth=3)
    pr = cProfile.Profile()
    pr.enable()
    ai.get_best_move(board, time_limit=None)
    pr.disable()

    for title, key in (
        ("tottime（函数自身时间，降序 Top 30）", pstats.SortKey.TIME),
        ("cumtime（含子调用累积时间，降序 Top 30）", pstats.SortKey.CUMULATIVE),
    ):
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats(key).print_stats(30)
        print("=" * 72)
        print(title)
        print("=" * 72)
        print(buf.getvalue())


if __name__ == "__main__":
    main()
