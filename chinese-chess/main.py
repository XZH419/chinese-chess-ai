import sys
import os
import argparse

# Ensure the repo root is importable so `import chess...` works even when the
# user runs `python main.py` from inside the `chinese-chess/` directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# 中国象棋程序入口。
# 支持 argparse 参数，便于切换不同 AI 进行实验对比。
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中国象棋 AI 启动入口")
    parser.add_argument("mode", nargs="?", default="cli", choices=["cli", "gui"], help="启动模式：cli 或 gui")

    # 黑方 AI：random / minimax
    parser.add_argument(
        "--black-ai",
        choices=["random", "minimax"],
        default="minimax",
        help="黑方 AI 类型（默认 minimax）",
    )
    # 红方：human / random / minimax
    parser.add_argument(
        "--red-ai",
        choices=["human", "random", "minimax"],
        default="human",
        help="红方类型（默认 human）",
    )
    # Minimax 搜索深度（红黑独立配置，便于不同强度对战）
    parser.add_argument("--red-depth", type=int, default=3, help="红方 Minimax 搜索深度（默认 3）")
    parser.add_argument("--black-depth", type=int, default=3, help="黑方 Minimax 搜索深度（默认 3）")

    args = parser.parse_args()

    from chess.control.controller import GameController
    from chess.algorithm.random_ai import RandomAI
    from chess.algorithm.minimax import MinimaxAI

    def build_agent(name: str, *, depth: int):
        if name == "human":
            return None
        if name == "random":
            return RandomAI()
        if name == "minimax":
            return MinimaxAI(depth=depth)
        raise ValueError(f"Unknown agent: {name}")

    red_agent = build_agent(args.red_ai, depth=args.red_depth)
    black_agent = build_agent(args.black_ai, depth=args.black_depth)

    # 启动前打印明确配置，便于实验记录与对比
    def side_desc(side: str, ai_name: str, depth: int) -> str:
        if ai_name == "human":
            return f"{side}: Human"
        if ai_name == "random":
            return f"{side}: RandomAI"
        if ai_name == "minimax":
            return f"{side}: Minimax (深度:{depth})"
        return f"{side}: {ai_name}"

    print(
        "[System] "
        + side_desc("红方", args.red_ai, args.red_depth)
        + " vs "
        + side_desc("黑方", args.black_ai, args.black_depth)
    )

    controller = GameController(red_agent=red_agent, black_agent=black_agent)

    if args.mode == "gui":
        from gui.main_window import MainWindow
        from PyQt5.QtWidgets import QApplication

        app = QApplication(sys.argv)
        window = MainWindow(controller=controller)
        window.show()
        sys.exit(app.exec_())
    else:
        # CLI 模式暂时沿用旧入口（后续也可 Controller 化）
        from ai.game import Game

        game = Game()
        game.play()
