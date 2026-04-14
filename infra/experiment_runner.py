"""对弈实验数据采集脚本（多轮次、自动换先）。

运行：

```bash
python -m infra.experiment_runner
```

输出：
- `raw_data.csv`：逐局原始数据（便于绘图）
- `summary_report.txt`：汇总统计（胜率/回合数/耗时）
"""

from __future__ import annotations

import csv
import gc
import os
import time
import traceback
import builtins
import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai.mcts_ai import MCTSAI
from ai.mcts_minimax_ai import MCTSMinimaxAI
from ai.minimax_ai import MinimaxAI
from ai.random_ai import RandomAI
from engine.board import Board
from engine.rules import MoveEntry, Rules

Move4 = Tuple[int, int, int, int]


# ──────────────────────────────────────────────────────────────
#  配置
# ──────────────────────────────────────────────────────────────

MAX_PLIES_PER_GAME = 200  # 超过则判 Draw，防止死循环
# 绕过开局库：三种 AI 都在 len(game_history) < 30 时才探测开局库
OPENING_BOOK_BYPASS_HISTORY: List[int] = [-(i + 1) for i in range(30)]


@contextlib.contextmanager
def _silence_output() -> List[str]:
    """屏蔽 stdout/stderr/print，降低噪声与 observer effect。"""
    buf = io.StringIO()
    old_print = builtins.print

    def _noop_print(*_a: Any, **_kw: Any) -> None:
        return None

    try:
        builtins.print = _noop_print  # type: ignore[assignment]
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            yield []
    finally:
        builtins.print = old_print  # type: ignore[assignment]


def _safe_workers(requested: int) -> int:
    """在当前机器上取一个合理的 workers 上限。"""
    try:
        import multiprocessing

        cpu = multiprocessing.cpu_count()
        return max(1, min(int(requested), cpu))
    except Exception:
        return max(1, int(requested))


def _agent_name(agent: Any) -> str:
    if agent is None:
        return "Human"
    return type(agent).__name__


def _material_diff(board: Board) -> int:
    """剩余子力差（红方 - 黑方），按棋子数量。"""
    return len(board.active_pieces["red"]) - len(board.active_pieces["black"])


def _append_history_entry(history: List[MoveEntry], board: Board, move: Move4, mover: str) -> None:
    """对齐 controller 的 MoveEntry 语义：记录走后 hash + mover + gave_check + last_move。"""
    opp = board.current_player
    history.append(
        MoveEntry(
            pos_hash=board.zobrist_hash,
            mover=mover,
            gave_check=Rules.is_king_in_check(board, opp),
            last_move=move,
        )
    )


@dataclass(slots=True)
class GameResult:
    experiment: str
    game_index: int
    red_ai: str
    black_ai: str
    winner: str  # 'red'|'black'|'draw'|'error'
    plies: int
    red_total_time_s: float
    black_total_time_s: float
    red_avg_time_s: float
    black_avg_time_s: float
    material_diff: int
    error: str = ""


def _pick_move(agent: Any, board: Board, *, game_history: List[int], move_history: List[MoveEntry]) -> Optional[Move4]:
    """统一调用 AI 的 move 接口（choose_move / get_best_move）。"""
    if hasattr(agent, "choose_move"):
        return agent.choose_move(
            board,
            time_limit=None,
            game_history=game_history,
            move_history=move_history,
        )
    return agent.get_best_move(
        board,
        time_limit=None,
        game_history=game_history,
        move_history=move_history,
    )


def run_single_game(
    *,
    experiment: str,
    game_index: int,
    red_agent: Any,
    black_agent: Any,
) -> GameResult:
    """运行一局对弈，返回逐局统计。"""

    board = Board()
    history: List[MoveEntry] = [MoveEntry(pos_hash=board.zobrist_hash)]
    # 确保开局库永远不触发：先填充 30 个占位，再追加真实哈希
    game_hash_history: List[int] = list(OPENING_BOOK_BYPASS_HISTORY) + [board.zobrist_hash]

    red_time = 0.0
    black_time = 0.0
    red_moves = 0
    black_moves = 0

    try:
        for ply in range(MAX_PLIES_PER_GAME):
            if Rules.is_game_over(board, move_history=history):
                w = Rules.winner(board, move_history=history)
                winner = w if w is not None else "draw"
                return GameResult(
                    experiment=experiment,
                    game_index=game_index,
                    red_ai=_agent_name(red_agent),
                    black_ai=_agent_name(black_agent),
                    winner=winner,
                    plies=ply,
                    red_total_time_s=red_time,
                    black_total_time_s=black_time,
                    red_avg_time_s=(red_time / red_moves) if red_moves else 0.0,
                    black_avg_time_s=(black_time / black_moves) if black_moves else 0.0,
                    material_diff=_material_diff(board),
                )

            side = board.current_player
            agent = red_agent if side == "red" else black_agent
            if agent is None:
                raise RuntimeError("agent is None (this runner expects AI vs AI)")

            with _silence_output():
                t0 = time.perf_counter()
                move = _pick_move(agent, board, game_history=game_hash_history, move_history=history)
                dt = time.perf_counter() - t0

            if side == "red":
                red_time += dt
                red_moves += 1
            else:
                black_time += dt
                black_moves += 1

            if move is None:
                # 无合法着法：胜负由规则判定；若仍未判定则归为 draw
                w = Rules.winner(board, move_history=history)
                winner = w if w is not None else "draw"
                return GameResult(
                    experiment=experiment,
                    game_index=game_index,
                    red_ai=_agent_name(red_agent),
                    black_ai=_agent_name(black_agent),
                    winner=winner,
                    plies=ply + 1,
                    red_total_time_s=red_time,
                    black_total_time_s=black_time,
                    red_avg_time_s=(red_time / red_moves) if red_moves else 0.0,
                    black_avg_time_s=(black_time / black_moves) if black_moves else 0.0,
                    material_diff=_material_diff(board),
                )

            sr, sc, er, ec = move
            ok, reason = Rules.is_valid_move(
                board,
                sr,
                sc,
                er,
                ec,
                player=side,
                history=history,
            )
            if not ok:
                # 鲁棒性：AI 输出非法走法，改用当前局面第一个合法走法继续，并记录 error
                legal = Rules.get_legal_moves(board, side, history=history)
                if not legal:
                    w = Rules.winner(board, move_history=history)
                    winner = w if w is not None else "draw"
                    return GameResult(
                        experiment=experiment,
                        game_index=game_index,
                        red_ai=_agent_name(red_agent),
                        black_ai=_agent_name(black_agent),
                        winner=winner,
                        plies=ply + 1,
                        red_total_time_s=red_time,
                        black_total_time_s=black_time,
                        red_avg_time_s=(red_time / red_moves) if red_moves else 0.0,
                        black_avg_time_s=(black_time / black_moves) if black_moves else 0.0,
                        material_diff=_material_diff(board),
                        error=f"illegal_move_from_{_agent_name(agent)}: {move} ({reason})",
                    )
                move = legal[0]

            mover = board.current_player
            _captured = board.apply_move(*move)
            _append_history_entry(history, board, move, mover)
            game_hash_history.append(board.zobrist_hash)

        # 超过最大步数，强制和棋
        return GameResult(
            experiment=experiment,
            game_index=game_index,
            red_ai=_agent_name(red_agent),
            black_ai=_agent_name(black_agent),
            winner="draw",
            plies=MAX_PLIES_PER_GAME,
            red_total_time_s=red_time,
            black_total_time_s=black_time,
            red_avg_time_s=(red_time / red_moves) if red_moves else 0.0,
            black_avg_time_s=(black_time / black_moves) if black_moves else 0.0,
            material_diff=_material_diff(board),
        )
    except Exception as e:
        return GameResult(
            experiment=experiment,
            game_index=game_index,
            red_ai=_agent_name(red_agent),
            black_ai=_agent_name(black_agent),
            winner="error",
            plies=len(history) - 1,
            red_total_time_s=red_time,
            black_total_time_s=black_time,
            red_avg_time_s=(red_time / red_moves) if red_moves else 0.0,
            black_avg_time_s=(black_time / black_moves) if black_moves else 0.0,
            material_diff=_material_diff(board),
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


def run_match(ai_red: Any, ai_black: Any, *, rounds: int = 10, experiment: str = "") -> List[GameResult]:
    """批量对局并自动换先（每局交换红黑）。"""

    results: List[GameResult] = []
    for i in range(rounds):
        # 第 1 局：A 红 B 黑；第 2 局：B 红 A 黑；以此类推
        if i % 2 == 0:
            red, black = ai_red, ai_black
        else:
            red, black = ai_black, ai_red

        res = run_single_game(
            experiment=experiment,
            game_index=i + 1,
            red_agent=red,
            black_agent=black,
        )
        results.append(res)

        # 释放跨局对象（并行/board.copy() 会产生不少短命对象）
        gc.collect()

    return results


def _write_raw_csv(path: Path, results: List[GameResult]) -> None:
    fieldnames = [
        "experiment",
        "game_index",
        "red_ai",
        "black_ai",
        "winner",
        "plies",
        "red_total_time_s",
        "black_total_time_s",
        "red_avg_time_s",
        "black_avg_time_s",
        "material_diff",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in fieldnames})


def _summarize(results: List[GameResult]) -> str:
    # 对每个 experiment 分组统计
    by_exp: Dict[str, List[GameResult]] = {}
    for r in results:
        by_exp.setdefault(r.experiment or "Unnamed", []).append(r)

    lines: List[str] = []
    for exp, rs in by_exp.items():
        total = len(rs)
        draws = sum(1 for r in rs if r.winner == "draw")
        errors = sum(1 for r in rs if r.winner == "error")
        red_wins = sum(1 for r in rs if r.winner == "red")
        black_wins = sum(1 for r in rs if r.winner == "black")
        avg_plies = sum(r.plies for r in rs) / max(1, total)

        # 统计每种 AI 的胜局数（无视颜色）
        ai_wins: Dict[str, int] = {}
        for r in rs:
            if r.winner == "red":
                ai_wins[r.red_ai] = ai_wins.get(r.red_ai, 0) + 1
            elif r.winner == "black":
                ai_wins[r.black_ai] = ai_wins.get(r.black_ai, 0) + 1

        # 耗时统计（按每步平均耗时的均值，足够用于粗粒度对比）
        red_avg = [r.red_avg_time_s for r in rs if r.red_avg_time_s > 0]
        black_avg = [r.black_avg_time_s for r in rs if r.black_avg_time_s > 0]
        mean_red_avg = sum(red_avg) / len(red_avg) if red_avg else 0.0
        mean_black_avg = sum(black_avg) / len(black_avg) if black_avg else 0.0

        lines.append(f"== {exp} ==")
        lines.append(f"games: {total} | red_wins: {red_wins} | black_wins: {black_wins} | draws: {draws} | errors: {errors}")
        lines.append(f"avg plies: {avg_plies:.1f}")
        lines.append(f"mean red avg time (s): {mean_red_avg:.4f} | mean black avg time (s): {mean_black_avg:.4f}")
        if ai_wins:
            parts = [
                f"{k}: {v} wins ({v / total * 100:.1f}%)"
                for k, v in sorted(ai_wins.items())
            ]
            lines.append("ai win share: " + ", ".join(parts))
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    # 关闭并行 worker 的观测日志（如需观察可手动 export CHESSAI_PARALLEL_LOG=1）
    os.environ.pop("CHESSAI_PARALLEL_LOG", None)

    results: List[GameResult] = []

    # 实验 A：MinimaxAI(D=5) vs RandomAI，共 4 局
    a1 = MinimaxAI(depth=5)
    a2 = RandomAI()
    results += run_match(a1, a2, rounds=4, experiment="A: MinimaxAI(depth=5) vs RandomAI")

    # 实验 B：MinimaxAI(D=5) vs MCTSMinimaxAI(sims=8000, workers=8)，共 10 局（可自行改为 20）
    b1 = MinimaxAI(depth=5)
    b2 = MCTSMinimaxAI(max_simulations=8000, time_limit=999.0, workers=_safe_workers(8), verbose=False)
    results += run_match(
        b1,
        b2,
        rounds=10,
        experiment="B: MinimaxAI(depth=5) vs MCTSMinimaxAI(sims=8000, workers=8)",
    )

    # 实验 C：MCTSAI(sims=8000, workers=8) vs MCTSMinimaxAI(sims=8000, workers=8)，共 10 局（可自行改为 20）
    c1 = MCTSAI(max_simulations=8000, time_limit=999.0, workers=_safe_workers(8), verbose=False)
    c2 = MCTSMinimaxAI(max_simulations=8000, time_limit=999.0, workers=_safe_workers(8), verbose=False)
    results += run_match(
        c1,
        c2,
        rounds=10,
        experiment="C: MCTSAI(sims=8000, workers=8) vs MCTSMinimaxAI(sims=8000, workers=8)",
    )

    out_csv = Path("raw_data.csv")
    out_txt = Path("summary_report.txt")
    _write_raw_csv(out_csv, results)
    out_txt.write_text(_summarize(results), encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_txt}")


if __name__ == "__main__":
    main()

