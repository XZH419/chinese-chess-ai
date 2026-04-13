import sys
import os
import argparse

# Ensure the repo root is importable so `import chinese_chess...` works even when the
# user runs `python chinese_chess/main.py ...` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# 中国象棋程序入口。
# 支持 argparse 参数，便于切换不同 AI 进行实验对比。
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
