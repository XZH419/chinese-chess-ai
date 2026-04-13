"""中国象棋程序主入口模块。

本模块作为整个中国象棋应用的统一启动入口，通过 argparse 命令行参数
支持灵活切换不同的启动模式（CLI / GUI）和 AI 配置，便于快速进行
不同算法之间的实验对比。

启动示例::

    # GUI 模式：人类 vs Minimax（深度 3）
    python -m chinese_chess.main gui --red human --black minimax --black-depth 3

    # CLI 模式：MCTS vs Minimax
    python -m chinese_chess.main cli --red mcts --black minimax

支持的玩家类型：
    - ``human``: 人类玩家（CLI 手动输入 / GUI 鼠标点击）
    - ``minimax``: 极大极小搜索 AI（可配置搜索深度）
    - ``random``: 随机走子 AI
    - ``mcts``: 蒙特卡洛树搜索 AI
"""

import sys
import os
import argparse

# 确保仓库根目录在导入路径中，使 `import chinese_chess...` 在
# 用户从仓库根目录执行 `python chinese_chess/main.py` 时也能正常工作。
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中国象棋 AI 启动入口")
    parser.add_argument("mode", nargs="?", default="cli", choices=["cli", "gui"], help="启动模式：cli 或 gui")

    parser.add_argument(
        "--red",
        type=str,
        choices=["human", "minimax", "random", "mcts"],
        default="human",
        help="红方玩家类型",
    )
    parser.add_argument(
        "--black",
        type=str,
        choices=["human", "minimax", "random", "mcts"],
        default="minimax",
        help="黑方玩家类型",
    )
    parser.add_argument("--red-depth", type=int, default=3, help="红方为 Minimax 时的搜索深度（默认 3）")
    parser.add_argument("--black-depth", type=int, default=3, help="黑方为 Minimax 时的搜索深度（默认 3）")

    args = parser.parse_args()

    from chinese_chess.control.controller import GameController, format_matchup_line
    from chinese_chess.algorithm.random_ai import RandomAI
    from chinese_chess.algorithm.minimax import MinimaxAI
    from chinese_chess.algorithm.mcts import MCTSAI

    def build_agent(kind: str, *, depth: int):
        """根据玩家类型字符串构建对应的 AI 代理实例。

        Args:
            kind: 玩家类型标识，可选 ``'human'``, ``'random'``,
                ``'minimax'``, ``'mcts'``。
            depth: Minimax 搜索深度（仅对 ``'minimax'`` 类型生效）。

        Returns:
            AI 代理实例；若为 ``'human'`` 则返回 ``None``。

        Raises:
            ValueError: 当 ``kind`` 不在支持的类型列表中时抛出。
        """
        if kind == "human":
            return None
        if kind == "random":
            return RandomAI()
        if kind == "minimax":
            return MinimaxAI(depth=depth)
        if kind == "mcts":
            return MCTSAI(time_limit=3.0, max_simulations=5000)
        raise ValueError(f"unknown player kind: {kind!r}")

    red_agent = build_agent(args.red, depth=args.red_depth)
    black_agent = build_agent(args.black, depth=args.black_depth)

    print("[System] " + format_matchup_line(red_agent, black_agent))

    controller = GameController(red_agent=red_agent, black_agent=black_agent)

    if args.mode == "gui":
        from PyQt5.QtWidgets import QApplication
        from chinese_chess.view.qt.main_window import MainWindow

        app = QApplication(sys.argv)
        window = MainWindow(controller=controller)
        window.show()
        sys.exit(app.exec_())
    else:
        from chinese_chess.smoke_play import main as smoke_main

        smoke_main(controller)
