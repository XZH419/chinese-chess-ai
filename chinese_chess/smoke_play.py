"""无头冒烟测试：验证 Controller + AI 代理的基本走棋流程。

使用示例（从仓库根目录执行）::

    python -m chinese_chess.main cli --red random --black random

本文件也可由 ``main.py`` 的 CLI 模式调用，传入用户构造的 Controller。
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from chinese_chess.control.controller import GameController

# 直接执行本文件时，Python 会将 `.../chinese_chess/` 目录加入 sys.path，
# 导致 `import chinese_chess...` 无法正确解析（需要的是仓库根目录）。
# 这里显式插入仓库根目录以修正导入路径。
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main(controller: Optional["GameController"] = None) -> None:
    """执行冒烟测试：双方轮流走棋，最多走 40 步（半回合）后终止。

    Args:
        controller: 外部传入的游戏控制器。若为 ``None``，
            则自动创建一个 RandomAI vs RandomAI 的控制器。
    """
    from chinese_chess.algorithm.random_ai import RandomAI
    from chinese_chess.control.controller import GameController

    if controller is None:
        controller = GameController(red_agent=RandomAI(), black_agent=RandomAI())

    max_plies = 40
    plies = 0

    while not controller.is_game_over() and plies < max_plies:
        controller.maybe_play_ai_turn(time_limit=0.1)
        plies += 1

    print("冒烟测试通过")
    print("已执行半回合数:", plies)
    print("终局胜者:", controller.winner())


if __name__ == "__main__":
    main()
