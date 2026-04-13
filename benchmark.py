#!/usr/bin/env python3
"""Headless AI vs AI benchmark: no GUI, aggregate win rates and search stats."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional, Tuple

# 保证从仓库根目录执行时 `import chinese_chess` 可用
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from chinese_chess.algorithm.minimax import MinimaxAI
from chinese_chess.algorithm.mcts import MCTSAI
from chinese_chess.algorithm.mcts_minimax import MCTSMinimaxAI
from chinese_chess.algorithm.random_ai import RandomAI
from chinese_chess.model.board import Board
from chinese_chess.model.rules import Rules

MAX_MOVES = 150  # 单方步数上限（着法数），超过判和

_AI_KIND_ALLOWED = frozenset({"minimax", "mcts", "mcts_minimax", "random"})
_LEGACY_TO_MCTS_MINIMAX = frozenset({"hybrid", "mcts_minmax"})


def _normalize_engine(name: str) -> str:
    k = name.strip().lower().replace("-", "_")
    if k in _LEGACY_TO_MCTS_MINIMAX:
        return "mcts_minimax"
    return k


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="无头 AI 对弈基准（支持 Minimax / MCTS / MCTS-Minimax / Random）",
    )
    parser.add_argument("--games", type=int, default=10, help="对局数量")
    parser.add_argument(
        "--red-ai",
        type=str,
        default="minimax",
        metavar="ENGINE",
        help="红方引擎：minimax|mcts|mcts_minimax|random",
    )
    parser.add_argument(
        "--black-ai",
        type=str,
        default="minimax",
        metavar="ENGINE",
        help="黑方引擎：minimax|mcts|mcts_minimax|random",
    )
    parser.add_argument("--red-depth", type=int, default=3, dest="red_depth", help="红方 Minimax 搜索深度")
    parser.add_argument("--black-depth", type=int, default=3, dest="black_depth", help="黑方 Minimax 搜索深度")
    parser.add_argument(
        "--red-sims",
        type=int,
        default=3000,
        dest="red_sims",
        help="红方 MCTS / MCTS-Minimax 的 max_simulations（默认 3000）",
    )
    parser.add_argument(
        "--black-sims",
        type=int,
        default=3000,
        dest="black_sims",
        help="黑方 MCTS / MCTS-Minimax 的 max_simulations",
    )
    parser.set_defaults(stochastic=True)
    parser.add_argument(
        "--stochastic",
        dest="stochastic",
        action="store_true",
        help="Minimax 根节点随机采样（默认开启，仅对 Minimax 生效）",
    )
    parser.add_argument(
        "--no-stochastic",
        dest="stochastic",
        action="store_false",
        help="关闭 Minimax 根节点随机采样",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="每步搜索时间上限（秒），默认不限制",
    )
    ns = parser.parse_args()
    r = _normalize_engine(ns.red_ai)
    b = _normalize_engine(ns.black_ai)
    if r not in _AI_KIND_ALLOWED:
        parser.error(f"未知的 --red-ai: {ns.red_ai!r}")
    if b not in _AI_KIND_ALLOWED:
        parser.error(f"未知的 --black-ai: {ns.black_ai!r}")
    ns.red_ai, ns.black_ai = r, b
    return ns


def _winner_label(w: Optional[str]) -> str:
    if w == "red":
        return "红方"
    if w == "black":
        return "黑方"
    return "和棋"


def build_agent(engine: str, *, depth: int, sims: int, stochastic: bool) -> Any:
    e = _normalize_engine(engine)
    if e == "minimax":
        return MinimaxAI(depth=depth, stochastic=stochastic, verbose=False)
    if e == "mcts":
        return MCTSAI(time_limit=999.0, max_simulations=sims, verbose=False)
    if e == "mcts_minimax":
        return MCTSMinimaxAI(max_simulations=sims, time_limit=999.0, verbose=False)
    if e == "random":
        return RandomAI()
    raise ValueError(f"unsupported engine: {engine!r}")


def _tally_last_stats(agent: Any, bucket: dict) -> None:
    """从 agent.last_stats 累加一步的耗时与节点/模拟量（各引擎字段略有差异）。"""
    ls = getattr(agent, "last_stats", None) or {}
    t = ls.get("time_taken")
    if isinstance(t, (int, float)):
        bucket["time"] += float(t)
    nodes = ls.get("nodes_evaluated")
    if nodes is None:
        nodes = ls.get("simulations", 0)
    if isinstance(nodes, (int, float)):
        bucket["nodes"] += int(nodes)
    bucket["moves"] += 1


def play_one_game(
    red_ai: Any,
    black_ai: Any,
    time_limit: Optional[float],
    agg: Optional[dict] = None,
) -> Tuple[Optional[str], int]:
    """返回 (结果颜色 'red'/'black'/None 表示和棋, 着法数)。

    若传入 ``agg``（含 ``red`` / ``black`` 子字典），每步搜索后根据 ``agent.last_stats`` 累加耗时与节点量。
    """
    board = Board()
    history_hashes = [board.zobrist_hash]
    if hasattr(red_ai, "reset_benchmark_stats"):
        red_ai.reset_benchmark_stats()
    if hasattr(black_ai, "reset_benchmark_stats"):
        black_ai.reset_benchmark_stats()
    plies = 0

    while plies < MAX_MOVES:
        if Rules.is_game_over(board, position_history=history_hashes):
            w = Rules.winner(board)
            return (w, plies)

        if board.current_player == "red":
            agent = red_ai
            agg_key = "red"
        else:
            agent = black_ai
            agg_key = "black"

        if agent is None:
            w = Rules.winner(board)
            return (w, plies)

        move = agent.get_best_move(board, game_history=history_hashes, time_limit=time_limit)
        if move is None:
            w = Rules.winner(board)
            return (w, plies)

        if agg is not None:
            _tally_last_stats(agent, agg[agg_key])

        sr, sc, er, ec = move
        board.apply_move(sr, sc, er, ec)
        history_hashes.append(board.zobrist_hash)
        plies += 1

    return (None, plies)


def main() -> None:
    args = _parse_args()
    red_wins = black_wins = draws = 0
    agg = {
        "red": {"time": 0.0, "nodes": 0, "moves": 0},
        "black": {"time": 0.0, "nodes": 0, "moves": 0},
    }

    r_eng = _normalize_engine(args.red_ai)
    b_eng = _normalize_engine(args.black_ai)

    for g in range(1, args.games + 1):
        red_ai = build_agent(
            r_eng,
            depth=args.red_depth,
            sims=args.red_sims,
            stochastic=args.stochastic,
        )
        black_ai = build_agent(
            b_eng,
            depth=args.black_depth,
            sims=args.black_sims,
            stochastic=args.stochastic,
        )
        winner, plies = play_one_game(red_ai, black_ai, args.time_limit, agg=agg)

        if winner == "red":
            red_wins += 1
        elif winner == "black":
            black_wins += 1
        else:
            draws += 1

        print(f"第 {g} 局结束，胜者：{_winner_label(winner)}，回合数：{plies}")

    n = max(args.games, 1)
    red_rate = 100.0 * red_wins / n
    black_rate = 100.0 * black_wins / n
    draw_rate = 100.0 * draws / n

    def _avg(side: str) -> Tuple[float, float]:
        m = max(agg[side]["moves"], 1)
        return agg[side]["time"] / m, agg[side]["nodes"] / m

    red_avg_t, red_avg_n = _avg("red")
    black_avg_t, black_avg_n = _avg("black")

    report = f"""## 对弈基准汇总

| 指标 | 数值 |
|------|------|
| 对局数 | {args.games} |
| 红方引擎 | {args.red_ai} (depth={args.red_depth}, sims={args.red_sims}) |
| 黑方引擎 | {args.black_ai} (depth={args.black_depth}, sims={args.black_sims}) |
| Minimax 根节点随机 | {'开启' if args.stochastic else '关闭'} |
| **红方胜率** | {red_rate:.1f}% |
| **黑方胜率** | {black_rate:.1f}% |
| **平局率** | {draw_rate:.1f}% |
| **红方单步平均耗时 (s)** | {red_avg_t:.4f} |
| **黑方单步平均耗时 (s)** | {black_avg_t:.4f} |
| **红方单步平均节点/模拟量** | {red_avg_n:.1f} |
| **黑方单步平均节点/模拟量** | {black_avg_n:.1f} |

注：耗时与节点/模拟量来自各引擎每步搜索后的 ``last_stats``（Minimax 为 ``nodes_evaluated``，
MCTS / MCTS-Minimax 为 ``simulations`` 或与 ``nodes_evaluated`` 的合成字段）。
"""
    print(report)


if __name__ == "__main__":
    main()
