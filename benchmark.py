#!/usr/bin/env python3
"""Headless AI vs AI benchmark: no GUI, aggregate win rates and search stats."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

# 保证从仓库根目录执行时 `import chinese_chess` 可用
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from chinese_chess.algorithm.minimax import MinimaxAI
from chinese_chess.model.board import Board
from chinese_chess.model.rules import Rules

MAX_MOVES = 150  # 单方步数上限（着法数），超过判和


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="无头 Minimax 对弈基准")
    parser.add_argument("--games", type=int, default=10, help="对局数量")
    parser.add_argument("--red-depth", type=int, default=3, dest="red_depth", help="红方搜索深度")
    parser.add_argument("--black-depth", type=int, default=3, dest="black_depth", help="黑方搜索深度")
    parser.set_defaults(stochastic=True)
    parser.add_argument(
        "--stochastic",
        dest="stochastic",
        action="store_true",
        help="启用根节点随机采样（默认已启用）",
    )
    parser.add_argument(
        "--no-stochastic",
        dest="stochastic",
        action="store_false",
        help="关闭根节点随机采样",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="每步搜索时间上限（秒），默认不限制",
    )
    return parser.parse_args()


def _winner_label(w: Optional[str]) -> str:
    if w == "red":
        return "红方"
    if w == "black":
        return "黑方"
    return "和棋"


def play_one_game(
    red_ai: MinimaxAI,
    black_ai: MinimaxAI,
    time_limit: Optional[float],
) -> Tuple[Optional[str], int]:
    """返回 (结果颜色 'red'/'black'/None 表示和棋, 着法数)。"""
    board = Board()
    history_hashes = [board.zobrist_hash]
    red_ai.reset_benchmark_stats()
    black_ai.reset_benchmark_stats()
    plies = 0

    while plies < MAX_MOVES:
        if Rules.is_game_over(board, position_history=history_hashes):
            w = Rules.winner(board)
            return (w, plies)

        if board.current_player == "red":
            move = red_ai.get_best_move(board, game_history=history_hashes, time_limit=time_limit)
        else:
            move = black_ai.get_best_move(board, game_history=history_hashes, time_limit=time_limit)
        if move is None:
            w = Rules.winner(board)
            return (w, plies)

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

    for g in range(1, args.games + 1):
        red_ai = MinimaxAI(
            depth=args.red_depth,
            stochastic=args.stochastic,
            verbose=False,
        )
        black_ai = MinimaxAI(
            depth=args.black_depth,
            stochastic=args.stochastic,
            verbose=False,
        )
        winner, plies = play_one_game(red_ai, black_ai, args.time_limit)

        if winner == "red":
            red_wins += 1
        elif winner == "black":
            black_wins += 1
        else:
            draws += 1

        agg["red"]["time"] += red_ai._bench_total_time
        agg["red"]["nodes"] += red_ai._bench_total_nodes
        agg["red"]["moves"] += red_ai._bench_search_count
        agg["black"]["time"] += black_ai._bench_total_time
        agg["black"]["nodes"] += black_ai._bench_total_nodes
        agg["black"]["moves"] += black_ai._bench_search_count

        print(f"第 {g} 局结束，胜者：{_winner_label(winner)}，回合数：{plies}")

    n = max(args.games, 1)
    red_rate = 100.0 * red_wins / n
    black_rate = 100.0 * black_wins / n
    draw_rate = 100.0 * draws / n

    rm = max(agg["red"]["moves"], 1)
    bm = max(agg["black"]["moves"], 1)
    red_avg_t = agg["red"]["time"] / rm
    black_avg_t = agg["black"]["time"] / bm
    red_avg_n = agg["red"]["nodes"] / rm
    black_avg_n = agg["black"]["nodes"] / bm

    report = f"""## 对弈基准汇总

| 指标 | 数值 |
|------|------|
| 对局数 | {args.games} |
| 红方深度 | {args.red_depth} |
| 黑方深度 | {args.black_depth} |
| 根节点随机 | {'开启' if args.stochastic else '关闭'} |
| **红方胜率** | {red_rate:.1f}% |
| **黑方胜率** | {black_rate:.1f}% |
| **平局率** | {draw_rate:.1f}% |
| **红方单步平均耗时 (s)** | {red_avg_t:.4f} |
| **黑方单步平均耗时 (s)** | {black_avg_t:.4f} |
| **红方单步平均评估节点数** | {red_avg_n:.1f} |
| **黑方单步平均评估节点数** | {black_avg_n:.1f} |
"""
    print(report)


if __name__ == "__main__":
    main()
