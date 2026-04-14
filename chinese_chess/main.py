"""中国象棋程序主入口模块。

本模块作为整个中国象棋应用的统一启动入口，通过 argparse 命令行参数
支持灵活切换不同的启动模式（CLI / GUI）和 AI 配置，便于快速进行
不同算法之间的实验对比。

启动示例::

    # GUI 模式：人类 vs Minimax（深度 5）
    python -m chinese_chess.main gui --red human --black minimax --black-depth 5

    # CLI 模式：MCTS vs Minimax
    python -m chinese_chess.main cli --red mcts --black minimax

    # MCTS-Minimax 引擎
    python -m chinese_chess.main cli --red mcts_minimax --black minimax --red-sims 3000

支持的玩家类型：
    - ``human``: 人类玩家（CLI 手动输入 / GUI 鼠标点击）
    - ``minimax``: 极大极小搜索 AI（可配置搜索深度）
    - ``random``: 随机走子 AI
    - ``mcts``: 蒙特卡洛树搜索 AI
    - ``mcts_minimax``: MCTS 主干 + Minimax 叶节点战术精算（统一名称）

旧拼写（如 ``hybrid``、``mcts_minmax``）在解析时仍会规范为 ``mcts_minimax``。
"""

import sys
import os
import argparse
from typing import Optional

# 确保仓库根目录在导入路径中，使 `import chinese_chess...` 在
# 用户从仓库根目录执行 `python chinese_chess/main.py` 时也能正常工作。
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_AI_KIND_ALLOWED = frozenset({"human", "minimax", "random", "mcts", "mcts_minimax"})

# 已废弃的 CLI 写法 → 统一为 mcts_minimax（不改变算法，仅规范化）
_LEGACY_AI_TO_MCTS_MINIMAX = frozenset({"hybrid", "mcts_minmax"})


def _normalize_ai_kind(kind: str) -> str:
    """将 CLI 字符串规范为内部引擎键 ``mcts_minimax`` 等。"""
    k = kind.strip().lower().replace("-", "_")
    if k in _LEGACY_AI_TO_MCTS_MINIMAX:
        return "mcts_minimax"
    return k


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中国象棋 AI 启动入口")
    parser.add_argument("mode", nargs="?", default="cli", choices=["cli", "gui"], help="启动模式：cli 或 gui")

    parser.add_argument(
        "--red",
        type=str,
        default="human",
        metavar="KIND",
        help="红方：human（玩家）|minimax|random|mcts|mcts_minimax",
    )
    parser.add_argument(
        "--black",
        type=str,
        default="minimax",
        metavar="KIND",
        help="黑方：human（玩家）|minimax|random|mcts|mcts_minimax",
    )
    parser.add_argument("--red-depth", type=int, default=5, help="红方为 Minimax 时的搜索深度（默认 5）")
    parser.add_argument("--black-depth", type=int, default=5, help="黑方为 Minimax 时的搜索深度（默认 5）")
    parser.add_argument(
        "--red-sims",
        type=int,
        default=None,
        metavar="N",
        help="红方 MCTS / MCTS-Minimax 的最大模拟次数；省略时 MCTS 默认 5000，MCTS-Minimax 默认 4000",
    )
    parser.add_argument(
        "--black-sims",
        type=int,
        default=None,
        metavar="N",
        help="黑方 MCTS / MCTS-Minimax 的最大模拟次数；省略时 MCTS 默认 5000，MCTS-Minimax 默认 4000",
    )

    args = parser.parse_args()

    red_k = _normalize_ai_kind(args.red)
    black_k = _normalize_ai_kind(args.black)
    if red_k not in _AI_KIND_ALLOWED:
        parser.error(f"未知的 --red: {args.red!r}（允许: {sorted(_AI_KIND_ALLOWED)}）")
    if black_k not in _AI_KIND_ALLOWED:
        parser.error(f"未知的 --black: {args.black!r}（允许: {sorted(_AI_KIND_ALLOWED)}）")

    from chinese_chess.control.controller import GameController, format_matchup_line
    from chinese_chess.algorithm.random_ai import RandomAI
    from chinese_chess.algorithm.minimax import MinimaxAI
    from chinese_chess.algorithm.mcts import MCTSAI
    from chinese_chess.algorithm.mcts_minimax import MCTSMinimaxAI

    def _default_sims(kind: str, sims: Optional[int]) -> int:
        if sims is not None:
            return sims
        return 4000 if kind == "mcts_minimax" else 5000

    def build_agent(kind: str, *, depth: int, sims: Optional[int]):
        """根据玩家类型字符串构建对应的 AI 代理实例。

        Args:
            kind: 规范化后的玩家类型（``mcts_minimax``、``minimax`` 等）。
            depth: Minimax 搜索深度。
            sims: MCTS / MCTS-Minimax 的 ``max_simulations``；``None`` 时按引擎类型取默认。

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
            return MCTSAI(
                time_limit=10.0,
                max_simulations=_default_sims(kind, sims),
                verbose=False,
            )
        if kind == "mcts_minimax":
            return MCTSMinimaxAI(
                max_simulations=_default_sims(kind, sims),
                time_limit=10.0,
                verbose=False,
            )
        raise ValueError(f"未知的玩家类型: {kind!r}")

    red_agent = build_agent(red_k, depth=args.red_depth, sims=args.red_sims)
    black_agent = build_agent(black_k, depth=args.black_depth, sims=args.black_sims)

    print("[系统] " + format_matchup_line(red_agent, black_agent))

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
