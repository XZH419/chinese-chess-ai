#!/usr/bin/env python3
"""无图形界面的 AI 对弈基准脚本：汇总胜率与搜索统计。"""

from __future__ import annotations

import argparse
from typing import Any, Optional, Tuple

from ai.minimax_ai import MinimaxAI
from ai.mcts_ai import MCTSAI
from ai.random_ai import RandomAI
from engine.board import Board
from engine.rules import MoveEntry, Rules

MAX_MOVES = 150  # 着法数上限，超过则停止对局（结果 None）

_AI_KIND_ALLOWED = frozenset({"minimax", "mcts", "random"})


def _normalize_engine(name: str) -> str:
    k = name.strip().lower().replace("-", "_")
    return k


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="无头 AI 对弈基准（支持 Minimax / MCTS / Random）",
    )
    parser.add_argument("--games", type=int, default=10, help="对局数量")
    parser.add_argument(
        "--red-ai",
        type=str,
        default="minimax",
        metavar="ENGINE",
        help="红方引擎：minimax|mcts|random",
    )
    parser.add_argument(
        "--black-ai",
        type=str,
        default="minimax",
        metavar="ENGINE",
        help="黑方引擎：minimax|mcts|random",
    )
    parser.add_argument("--red-depth", type=int, default=5, dest="red_depth", help="红方 Minimax 搜索深度（默认 5）")
    parser.add_argument("--black-depth", type=int, default=5, dest="black_depth", help="黑方 Minimax 搜索深度（默认 5）")
    parser.add_argument(
        "--red-sims",
        type=int,
        default=3000,
        dest="red_sims",
        help="红方 MCTS 的 max_simulations（默认 3000）",
    )
    parser.add_argument(
        "--black-sims",
        type=int,
        default=3000,
        dest="black_sims",
        help="黑方 MCTS 的 max_simulations",
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


def _engine_summary_cn(engine: str, depth: int, sims: int) -> str:
    """将 CLI 引擎键转为中文说明（用于基准报告展示）。"""
    e = _normalize_engine(engine)
    if e == "minimax":
        return f"Minimax AI（深度 {depth}）"
    if e == "mcts":
        return f"MCTS AI（模拟上限 {sims}）"
    if e == "random":
        return "随机 AI"
    return engine


def build_agent(engine: str, *, depth: int, sims: int, stochastic: bool) -> Any:
    e = _normalize_engine(engine)
    if e == "minimax":
        return MinimaxAI(depth=depth, stochastic=stochastic, verbose=False)
    if e == "mcts":
        return MCTSAI(time_limit=999.0, max_simulations=sims, verbose=False)
    if e == "random":
        return RandomAI()
    raise ValueError(f"不支持的引擎: {engine!r}")


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
    move_hist: list = [MoveEntry(pos_hash=board.zobrist_hash)]
    if hasattr(red_ai, "reset_benchmark_stats"):
        red_ai.reset_benchmark_stats()
    if hasattr(black_ai, "reset_benchmark_stats"):
        black_ai.reset_benchmark_stats()
    plies = 0

    while plies < MAX_MOVES:
        if Rules.is_game_over(board, move_history=move_hist):
            w = Rules.winner(board, move_history=move_hist)
            return (w, plies)

        if board.current_player == "red":
            agent = red_ai
            agg_key = "red"
        else:
            agent = black_ai
            agg_key = "black"

        if agent is None:
            w = Rules.winner(board, move_history=move_hist)
            return (w, plies)

        move = agent.get_best_move(
            board,
            game_history=history_hashes,
            time_limit=time_limit,
            move_history=move_hist,
        )
        if move is None:
            w = Rules.winner(board, move_history=move_hist)
            return (w, plies)

        if agg is not None:
            _tally_last_stats(agent, agg[agg_key])

        sr, sc, er, ec = move
        mover = board.current_player
        board.apply_move(sr, sc, er, ec)
        opp = board.current_player
        move_hist.append(
            MoveEntry(
                pos_hash=board.zobrist_hash,
                mover=mover,
                gave_check=Rules.is_king_in_check(board, opp),
                last_move=(sr, sc, er, ec),
            )
        )
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

    red_desc = _engine_summary_cn(args.red_ai, args.red_depth, args.red_sims)
    black_desc = _engine_summary_cn(args.black_ai, args.black_depth, args.black_sims)
    report = f"""## 对弈基准汇总

| 指标 | 数值 |
|------|------|
| 对局数 | {args.games} |
| 红方 | {red_desc}（CLI 键: {args.red_ai}） |
| 黑方 | {black_desc}（CLI 键: {args.black_ai}） |
| Minimax 根节点随机着法 | {'开启' if args.stochastic else '关闭'} |
| **红方胜率** | {red_rate:.1f}% |
| **黑方胜率** | {black_rate:.1f}% |
| **和棋率** | {draw_rate:.1f}% |
| **红方单步平均耗时（秒）** | {red_avg_t:.4f} |
| **黑方单步平均耗时（秒）** | {black_avg_t:.4f} |
| **红方单步平均节点/模拟量** | {red_avg_n:.1f} |
| **黑方单步平均节点/模拟量** | {black_avg_n:.1f} |

说明：上表中的耗时与节点或模拟次数，来自每步搜索后各引擎写入的 `last_stats`；
Minimax 以「评估节点数」为主，MCTS / MCTS-Minimax 以「模拟次数」或合成节点统计为主。
"""
    print(report)


if __name__ == "__main__":
    main()
