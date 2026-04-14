"""对弈实验数据采集脚本（多轮次、自动换先、固定预算）。

运行：

```bash
python -m infra.experiment_runner
```

输出（默认输出到 `runs/<timestamp>/`）：
- `raw_games.csv`：逐局原始数据（便于绘图）
- `summary_report.txt`：汇总统计（胜率/回合数/耗时/错误）

说明：
- 本脚本仅保留 **calibrated** 公平性策略：先测 Minimax(depth=K) 的真实每步耗时中位数，
  再把 MCTS 的 `time_limit` 设为该中位数，使两者总时间预算对齐（无需修改 Minimax 实现）。
- 通过固定 seed + 固定起始局面（可选 midgame）提升可复现性。
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import csv
import gc
import io
import os
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai.mcts_ai import MCTSAI
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

def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


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
    start_kind: str = "initial"  # initial|midgame


def _pick_move(
    agent: Any,
    board: Board,
    *,
    time_limit_s: Optional[float],
    game_history: List[int],
    move_history: List[MoveEntry],
) -> Optional[Move4]:
    """统一调用 AI 的 move 接口（choose_move / get_best_move）。"""
    if hasattr(agent, "choose_move"):
        return agent.choose_move(
            board,
            time_limit=time_limit_s,
            game_history=game_history,
            move_history=move_history,
        )
    return agent.get_best_move(
        board,
        time_limit=time_limit_s,
        game_history=game_history,
        move_history=move_history,
    )


def build_midgame_board(plies: int = 18) -> Board:
    """确定性地从初始局面推进到一个中局局面（避免开局库/开局走法偏差）。"""
    b = Board()
    for _ in range(int(plies)):
        legal = Rules.get_legal_moves(b, b.current_player)
        if not legal:
            break
        captures = [m for m in legal if b.board[m[2]][m[3]] is not None]
        m = sorted(captures or legal)[0]
        b.apply_move(*m)
    return b


def _median(xs: List[float]) -> float:
    xs2 = sorted(float(x) for x in xs)
    n = len(xs2)
    if n == 0:
        return 0.0
    mid = n // 2
    return xs2[mid] if (n % 2 == 1) else (xs2[mid - 1] + xs2[mid]) / 2.0


def calibrate_minimax_seconds_per_move(
    *,
    depth: int,
    start_kind: str,
    start_midgame_plies: int,
    samples: int = 5,
) -> float:
    """不修改 Minimax 的前提下，测一个“该深度的真实每步耗时”中位数。

    用于 `--fairness calibrated`：让 MCTS 的 time_limit 贴近 Minimax(depth=K) 的真实耗时，
    避免因 Minimax 的 time_limit 行为差异导致对比失真。
    """
    board = Board() if start_kind == "initial" else build_midgame_board(plies=start_midgame_plies)
    history: List[MoveEntry] = [MoveEntry(pos_hash=board.zobrist_hash)]
    gh: List[int] = list(OPENING_BOOK_BYPASS_HISTORY) + [board.zobrist_hash]
    ai = MinimaxAI(depth=int(depth))

    times: List[float] = []
    # 取多个不同的局面点：每次选一步，然后推进一步（用合法走法的稳定选择）
    for _ in range(max(1, int(samples))):
        with _silence_output():
            t0 = time.perf_counter()
            _ = ai.get_best_move(board.copy(), time_limit=None, game_history=gh, move_history=history)
            dt = time.perf_counter() - t0
        times.append(dt)

        legal = Rules.get_legal_moves(board, board.current_player, history=history)
        if not legal:
            break
        mv = sorted(legal)[0]
        mover = board.current_player
        _ = board.apply_move(*mv)
        _append_history_entry(history, board, mv, mover)
        gh.append(board.zobrist_hash)

    return max(0.01, float(_median(times)))


def run_single_game(
    *,
    experiment: str,
    game_index: int,
    red_agent: Any,
    black_agent: Any,
    time_limit_s: Optional[float],
    start_kind: str = "initial",
    start_midgame_plies: int = 18,
) -> GameResult:
    """运行一局对弈，返回逐局统计。"""

    _log(
        f"[对局开始] {experiment} | game={game_index} | start={start_kind} | "
        f"red={_agent_name(red_agent)} vs black={_agent_name(black_agent)}"
    )
    board = Board() if start_kind == "initial" else build_midgame_board(plies=start_midgame_plies)
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
                res = GameResult(
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
                    start_kind=start_kind,
                )
                _log(
                    f"[对局结束] {experiment} | game={game_index} | winner={res.winner} | plies={res.plies} | "
                    f"red_avg={res.red_avg_time_s:.3f}s black_avg={res.black_avg_time_s:.3f}s | "
                    f"material_diff={res.material_diff}"
                )
                return res

            side = board.current_player
            agent = red_agent if side == "red" else black_agent
            if agent is None:
                raise RuntimeError("agent is None (this runner expects AI vs AI)")

            with _silence_output():
                t0 = time.perf_counter()
                move = _pick_move(
                    agent,
                    board,
                    time_limit_s=time_limit_s,
                    game_history=game_hash_history,
                    move_history=history,
                )
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
                res = GameResult(
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
                    start_kind=start_kind,
                )
                _log(
                    f"[对局结束] {experiment} | game={game_index} | winner={res.winner} | plies={res.plies} | "
                    f"red_avg={res.red_avg_time_s:.3f}s black_avg={res.black_avg_time_s:.3f}s | "
                    f"material_diff={res.material_diff}"
                )
                return res

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
                    res = GameResult(
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
                        start_kind=start_kind,
                    )
                    _log(
                        f"[对局结束] {experiment} | game={game_index} | winner={res.winner} | plies={res.plies} | "
                        f"ERROR={res.error}"
                    )
                    return res
                move = legal[0]

            mover = board.current_player
            _captured = board.apply_move(*move)
            _append_history_entry(history, board, move, mover)
            game_hash_history.append(board.zobrist_hash)

        # 超过最大步数，强制和棋
        res = GameResult(
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
            start_kind=start_kind,
        )
        _log(
            f"[对局结束] {experiment} | game={game_index} | winner=draw(max_plies) | plies={res.plies} | "
            f"red_avg={res.red_avg_time_s:.3f}s black_avg={res.black_avg_time_s:.3f}s | "
            f"material_diff={res.material_diff}"
        )
        return res
    except Exception as e:
        res = GameResult(
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
            start_kind=start_kind,
        )
        _log(
            f"[对局结束] {experiment} | game={game_index} | winner=error | plies={res.plies} | ERROR={type(e).__name__}: {e}"
        )
        return res


def run_match(
    ai_red: Any,
    ai_black: Any,
    *,
    rounds: int,
    experiment: str,
    time_limit_s: Optional[float],
    start_kind: str,
    start_midgame_plies: int,
) -> List[GameResult]:
    """批量对局并自动换先（每局交换红黑）。"""

    _log(f"[实验开始] {experiment} | rounds={rounds} | start={start_kind}")
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
            time_limit_s=time_limit_s,
            start_kind=start_kind,
            start_midgame_plies=start_midgame_plies,
        )
        results.append(res)

        # 释放跨局对象（并行/board.copy() 会产生不少短命对象）
        gc.collect()

    wins = {"red": 0, "black": 0, "draw": 0, "error": 0}
    for r in results:
        wins[r.winner] = wins.get(r.winner, 0) + 1
    _log(f"[实验结束] {experiment} | results={wins}")
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
        "start_kind",
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
    parser = argparse.ArgumentParser(description="中国象棋 AI 对弈实验采集脚本")
    parser.add_argument("--rounds", type=int, default=10, help="每个实验的对局数（自动换先，默认 10）")
    parser.add_argument(
        "--minimax-depth",
        type=int,
        default=4,
        help="Minimax 深度（默认 4；将用于校准真实每步耗时）",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=5,
        help="校准阶段：测 Minimax 每步耗时的样本数（默认 5）",
    )
    parser.add_argument("--max-plies", type=int, default=200, help="单局最大半回合数（默认 200）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument(
        "--start",
        choices=["initial", "midgame"],
        default="midgame",
        help="起始局面：initial=开局，midgame=固定中局（默认 midgame）",
    )
    parser.add_argument("--midgame-plies", type=int, default=18, help="midgame 推进半回合数（默认 18）")
    parser.add_argument("--out", type=str, default="", help="输出目录（默认 runs/<timestamp>/）")
    args = parser.parse_args()

    t_global0 = time.perf_counter()
    # 关闭并行 worker 的观测日志（如需观察可手动 export CHESSAI_PARALLEL_LOG=1）
    os.environ.pop("CHESSAI_PARALLEL_LOG", None)

    global MAX_PLIES_PER_GAME
    MAX_PLIES_PER_GAME = int(args.max_plies)
    random.seed(int(args.seed))

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path("runs") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    rounds = max(2, int(args.rounds))
    start_kind = str(args.start)
    midgame_plies = int(args.midgame_plies)
    mm_depth = int(args.minimax_depth)

    mm_median = calibrate_minimax_seconds_per_move(
        depth=mm_depth,
        start_kind=start_kind,
        start_midgame_plies=midgame_plies,
        samples=int(args.calib_samples),
    )
    # 让 MCTS 用 Minimax 的真实中位数做 time_limit；Minimax 固定深度，不传 time_limit
    mcts_time_limit_s = mm_median

    _log(
        "[实验总览] "
        f"rounds={rounds} | minimax_depth={mm_depth} | calib_samples={int(args.calib_samples)} | "
        f"calib_median={mm_median:.3f}s | start={start_kind} | midgame_plies={midgame_plies} | "
        f"max_plies={MAX_PLIES_PER_GAME} | seed={int(args.seed)} | out={out_dir}"
    )

    # ── 实验设计（当前项目适配版） ──
    # E1: 随机基线（验证规则/runner 稳定）
    # E2: Minimax vs Random（检验确定性搜索基本有效性）
    # E3: MCTS vs Random（检验采样搜索基本有效性）
    # E4: Minimax vs MCTS（核心对比：同一 time_limit 下的强度）
    #
    # 所有实验均使用同一 per-move time_limit 以保证公平；MCTS 设置一个“足够高”的 max_simulations 作为兜底。
    mcts_max_sims = 200000  # 主要由 time_limit 控制；上限只用于防止极端情况下过早耗尽

    results: List[GameResult] = []

    e1a, e1b = RandomAI(), RandomAI()
    results += run_match(
        e1a,
        e1b,
        rounds=rounds,
        experiment=f"E1: RandomAI vs RandomAI | start={start_kind}",
        time_limit_s=None,
        start_kind=start_kind,
        start_midgame_plies=midgame_plies,
    )

    e2a, e2b = MinimaxAI(depth=mm_depth), RandomAI()
    results += run_match(
        e2a,
        e2b,
        rounds=rounds,
        experiment=f"E2: MinimaxAI(depth={mm_depth}) vs RandomAI | calib_median={mm_median:.3f}s | start={start_kind}",
        time_limit_s=None,
        start_kind=start_kind,
        start_midgame_plies=midgame_plies,
    )

    e3a = MCTSAI(
        max_simulations=mcts_max_sims,
        time_limit=mcts_time_limit_s,
        verbose=False,
    )
    e3b = RandomAI()
    results += run_match(
        e3a,
        e3b,
        rounds=rounds,
        experiment=f"E3: MCTSAI vs RandomAI | tl={mcts_time_limit_s:.3f}s | start={start_kind}",
        time_limit_s=mcts_time_limit_s,
        start_kind=start_kind,
        start_midgame_plies=midgame_plies,
    )

    e4a = MinimaxAI(depth=mm_depth)
    e4b = MCTSAI(
        max_simulations=mcts_max_sims,
        time_limit=mcts_time_limit_s,
        verbose=False,
    )
    results += run_match(
        e4a,
        e4b,
        rounds=rounds,
        experiment=f"E4: MinimaxAI(depth={mm_depth}) vs MCTSAI | tl={mcts_time_limit_s:.3f}s | start={start_kind}",
        time_limit_s=None,
        start_kind=start_kind,
        start_midgame_plies=midgame_plies,
    )

    out_csv = out_dir / "raw_games.csv"
    out_txt = out_dir / "summary_report.txt"
    _write_raw_csv(out_csv, results)
    out_txt.write_text(_summarize(results), encoding="utf-8")

    _log(f"[写入完成] {out_csv}")
    _log(f"[写入完成] {out_txt}")
    _log(f"[实验结束] elapsed={time.perf_counter() - t_global0:.1f}s")


if __name__ == "__main__":
    main()

