"""中国象棋程序主入口模块。

本模块作为整个中国象棋应用的统一启动入口，通过 argparse 命令行参数
支持灵活切换不同的启动模式（CLI / GUI）和 AI 配置，便于快速进行
不同算法之间的实验对比。

启动示例::

    # GUI 模式：人类 vs Minimax（深度 3）
    python -m chinese_chess.main gui --red human --black minimax --black-depth 3

    # CLI 模式：MCTS vs Minimax
    python -m chinese_chess.main cli --red mcts --black minimax

    # MCTS+Minimax 混合引擎（别名：hybrid、mcts_minmax、mcts-minmax）
    python -m chinese_chess.main cli --red mcts_minimax --black minimax --red-sims 3000

支持的玩家类型：
    - ``human``: 人类玩家（CLI 手动输入 / GUI 鼠标点击）
    - ``minimax``: 极大极小搜索 AI（可配置搜索深度）
    - ``random``: 随机走子 AI
    - ``mcts``: 蒙特卡洛树搜索 AI
    - ``mcts_minimax`` / ``hybrid`` / ``mcts_minmax`` 等：MCTS 主干 + Minimax 局部精算
"""

import sys
import os
import argparse

# 确保仓库根目录在导入路径中，使 `import chinese_chess...` 在
# 用户从仓库根目录执行 `python chinese_chess/main.py` 时也能正常工作。
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# 命令行可用的玩家类型（含常见别名，在 build 前会规范化为内部键）
_AI_CHOICE_STRINGS = [
    "human",
    "minimax",
    "random",
    "mcts",
    "mcts_minimax",
    "mcts_minmax",
    "hybrid",
    "mcts-minimax",
    "mcts-minmax",
]


def _normalize_ai_kind(kind: str) -> str:
    """将 CLI 别名统一为内部使用的引擎键。"""
    k = kind.strip().lower().replace("-", "_")
    aliases = {
        "hybrid": "mcts_minimax",
        "mcts_minmax": "mcts_minimax",
    }
    return aliases.get(k, k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中国象棋 AI 启动入口")
    parser.add_argument("mode", nargs="?", default="cli", choices=["cli", "gui"], help="启动模式：cli 或 gui")

    parser.add_argument(
        "--red",
        type=str,
        choices=_AI_CHOICE_STRINGS,
        default="human",
        help="红方玩家类型（混合引擎可写 mcts_minimax / hybrid / mcts_minmax）",
    )
    parser.add_argument(
        "--black",
        type=str,
        choices=_AI_CHOICE_STRINGS,
        default="minimax",
        help="黑方玩家类型",
    )
    parser.add_argument("--red-depth", type=int, default=3, help="红方为 Minimax 时的搜索深度（默认 3）")
    parser.add_argument("--black-depth", type=int, default=3, help="黑方为 Minimax 时的搜索深度（默认 3）")
    parser.add_argument(
        "--red-sims",
        type=int,
        default=5000,
        help="红方为 MCTS / 混合引擎时的最大模拟次数（默认 5000）",
    )
    parser.add_argument(
        "--black-sims",
        type=int,
        default=5000,
        help="黑方为 MCTS / 混合引擎时的最大模拟次数（默认 5000）",
    )

    args = parser.parse_args()

    from chinese_chess.control.controller import GameController, format_matchup_line
    from chinese_chess.algorithm.random_ai import RandomAI
    from chinese_chess.algorithm.minimax import MinimaxAI
    from chinese_chess.algorithm.mcts import MCTSAI
    from chinese_chess.algorithm.mcts_minimax import MCTSMinimaxAI

    def build_agent(kind: str, *, depth: int, sims: int):
        """根据玩家类型字符串构建对应的 AI 代理实例。

        Args:
            kind: 规范化后的玩家类型（``mcts_minimax``、``minimax`` 等）。
            depth: Minimax 搜索深度。
            sims: MCTS / 混合引擎的 ``max_simulations``。

        Returns:
            AI 代理实例；若为 ``'human'`` 则返回 ``None``。
        """
        if kind == "human":
            return None
        if kind == "random":
            return RandomAI()
        if kind == "minimax":
            return MinimaxAI(depth=depth)
        if kind == "mcts":
            return MCTSAI(time_limit=10.0, max_simulations=sims, verbose=False)
        if kind == "mcts_minimax":
            return MCTSMinimaxAI(
                max_simulations=sims,
                time_limit=10.0,
                verbose=False,
            )
        raise ValueError(f"unknown player kind: {kind!r}")

    red_k = _normalize_ai_kind(args.red)
    black_k = _normalize_ai_kind(args.black)
    red_agent = build_agent(red_k, depth=args.red_depth, sims=args.red_sims)
    black_agent = build_agent(black_k, depth=args.black_depth, sims=args.black_sims)

    print("[System] " + format_matchup_line(red_agent, black_agent))

    controller = GameController(red_agent=red_agent, black_agent=black_agent)

    if args.mode == "gui":
        try:
            from PyQt5.QtWidgets import QApplication
        except ModuleNotFoundError:
            print(
                "错误：未安装 PyQt5，无法启动图形界面。\n"
                "请执行:  python -m pip install PyQt5\n"
                "或从仓库根目录:  python -m pip install -r requirements.txt",
                file=sys.stderr,
            )
            sys.exit(1)
        from chinese_chess.view.qt.main_window import MainWindow

        app = QApplication(sys.argv)
        window = MainWindow(controller=controller)
        window.show()
        sys.exit(app.exec_())
    else:
        from chinese_chess.smoke_play import main as smoke_main

        smoke_main(controller)
