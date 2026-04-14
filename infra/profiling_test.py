"""性能预测试（Profiling）——为后续对弈实验“定标”参数与预算。

目标：
- 尽量减少观察者效应（Observer Effect）
- 使用固定“中局局面”作为输入，避免开局库干扰
- 对 Minimax / MCTS 采集吞吐量与耗时指标
- 3~5 次重复取中位数，输出轻量 Markdown 表格
- 给 `infra.experiment_runner` 提供“每步 time_limit + 保险的 max_simulations”建议

运行方式（Windows/conda/venv 均可）：

```bash
python -m infra.profiling_test
```
"""

from __future__ import annotations

import builtins
import contextlib
import gc
import io
import os
import statistics
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ai.mcts_ai import MCTSAI
from ai.minimax_ai import MinimaxAI
from engine.board import Board
from engine.rules import MoveEntry, Rules


# ──────────────────────────────────────────────────────────────
#  1) 测试环境标准化
# ──────────────────────────────────────────────────────────────

WARMUP_SECONDS = 2.0
REPEATS = 5  # 3~5 次取中位数

# 统一关闭开局库：三种 AI 都在 len(game_history) < 30 时才探测开局库
OPENING_BOOK_BYPASS_HISTORY: List[int] = [-(i + 1) for i in range(30)]


@contextlib.contextmanager
def _silence_output() -> Iterable[None]:
    """在测试循环内屏蔽所有 stdout/stderr/print，减少观察者效应与噪声。"""

    buf = io.StringIO()
    old_print = builtins.print

    def _noop_print(*_a: Any, **_kw: Any) -> None:
        return None

    try:
        builtins.print = _noop_print  # type: ignore[assignment]
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            yield
    finally:
        builtins.print = old_print  # type: ignore[assignment]


def _cpu_workers_max() -> int:
    try:
        import multiprocessing

        return max(1, min(8, multiprocessing.cpu_count()))
    except Exception:
        return 1


# ──────────────────────────────────────────────────────────────
#  2) 固定中局局面构造（避免开局库干扰）
# ──────────────────────────────────────────────────────────────


def build_midgame_board(plies: int = 18) -> Board:
    """从初始局面 deterministically 推进到一个相对复杂的中局局面。"""

    b = Board()
    for _ in range(plies):
        legal = Rules.get_legal_moves(b, b.current_player)
        if not legal:
            break
        # 为了产生更“中局化”的局面：优先吃子，其次按字典序稳定选择
        captures = [m for m in legal if b.board[m[2]][m[3]] is not None]
        m = sorted(captures or legal)[0]
        b.apply_move(*m)
    return b


def _root_move_history_for(board: Board) -> List[MoveEntry]:
    """生成与当前局面一致的最小 MoveEntry 链。"""

    return [MoveEntry(pos_hash=board.zobrist_hash)]


# ──────────────────────────────────────────────────────────────
#  3) 计量模型
# ──────────────────────────────────────────────────────────────


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _run_warmup(run_once: Callable[[], None], seconds: float = WARMUP_SECONDS) -> None:
    """预热：运行 seconds 秒，不计入正式统计。"""

    t0 = time.perf_counter()
    while (time.perf_counter() - t0) < seconds:
        run_once()


def _measure_repeated(
    run_once: Callable[[], Tuple[float, Optional[float]]],
    repeats: int = REPEATS,
) -> Tuple[float, Optional[float]]:
    """重复测量并取中位数。"""

    times: List[float] = []
    metrics: List[float] = []
    for _ in range(repeats):
        gc.collect()
        dt, mv = run_once()
        times.append(dt)
        if mv is not None:
            metrics.append(float(mv))
    t_med = _median(times)
    m_med = _median(metrics) if metrics else None
    return t_med, m_med


# ──────────────────────────────────────────────────────────────
#  4) 各 AI 的测量逻辑
# ──────────────────────────────────────────────────────────────


def _measure_minimax(depth: int, board: Board) -> Tuple[float, float]:
    """返回 (median_seconds, median_nodes_per_sec)。"""

    mh = _root_move_history_for(board)
    gh = list(OPENING_BOOK_BYPASS_HISTORY) + [board.zobrist_hash]

    def _mk() -> MinimaxAI:
        return MinimaxAI(depth=depth)

    def _run_once() -> Tuple[float, Optional[float]]:
        ai = _mk()
        with _silence_output():
            t0 = time.perf_counter()
            # 使用“无时间限制”测一次，得到该深度的真实耗时与节点吞吐
            ai.get_best_move(
                board.copy(),
                time_limit=None,
                game_history=gh,
                move_history=mh,
            )
            dt = time.perf_counter() - t0
        stats = getattr(ai, "last_stats", None) or {}
        nodes = stats.get("nodes_evaluated")
        return dt, float(nodes) if isinstance(nodes, (int, float)) else None

    # warm-up（不计时）
    with _silence_output():
        _run_warmup(lambda: _mk().get_best_move(board.copy(), time_limit=None, game_history=gh, move_history=mh))

    t_med, nodes_med = _measure_repeated(_run_once)
    nodes_per_s = float(nodes_med) / max(t_med, 1e-9) if nodes_med is not None else 0.0
    return t_med, nodes_per_s


def _measure_mcts_like(
    *,
    cls_name: str,
    make_ai: Callable[[int], Any],
    board: Board,
    simulations: int,
    workers: int,
) -> Tuple[float, float]:
    """返回 (median_seconds, median_sims_per_sec)。"""

    mh = _root_move_history_for(board)
    gh = list(OPENING_BOOK_BYPASS_HISTORY) + [board.zobrist_hash]

    def _run_once() -> Tuple[float, Optional[float]]:
        ai = make_ai(workers)
        with _silence_output():
            t0 = time.perf_counter()
            ai.get_best_move(
                board.copy(),
                time_limit=999.0,  # 用 sims 定标吞吐；避免提前 time-limit
                game_history=gh,
                move_history=mh,
            )
            dt = time.perf_counter() - t0
        stats = getattr(ai, "last_stats", None) or {}
        sims = stats.get("simulations")
        return dt, float(sims) if isinstance(sims, (int, float)) else float(simulations)

    # warm-up（不计时）
    with _silence_output():
        _run_warmup(
            lambda: make_ai(workers).get_best_move(
                board.copy(),
                time_limit=0.2,
                game_history=gh,
                move_history=mh,
            )
        )

    t_med, sims_med = _measure_repeated(_run_once)
    sims_per_s = float(sims_med) / max(t_med, 1e-9) if sims_med is not None else 0.0
    return t_med, sims_per_s


# ──────────────────────────────────────────────────────────────
#  5) 输出与建议参数
# ──────────────────────────────────────────────────────────────


def _fmt_float(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def _suggest_simulations(sps: float, time_limit_s: float, safety: float = 1.5) -> int:
    """给 time-limit 实验准备一个“不会先被 sims 卡住”的 max_simulations。"""
    return max(200, int(sps * time_limit_s * safety))


def _suggest_depth(minimax_rows: List[Tuple[int, float]]) -> int:
    """选择最接近 3 秒的深度（minimax_rows: [(depth, median_seconds), ...]）。"""

    target = 3.0
    best_d = minimax_rows[0][0]
    best_err = abs(minimax_rows[0][1] - target)
    for d, sec in minimax_rows:
        err = abs(sec - target)
        if err < best_err:
            best_err = err
            best_d = d
    return best_d


def main() -> None:
    # 禁用并行 worker 的观测打印（避免污染测量）
    os.environ.pop("CHESSAI_PARALLEL_LOG", None)

    board = build_midgame_board(plies=18)
    wmax = _cpu_workers_max()

    rows: List[Dict[str, Any]] = []

    # Minimax：D in {3,4,5}（你也可以按需扩展）
    minimax_rows: List[Tuple[int, float]] = []
    for d in (3, 4, 5):
        gc.collect()
        t_med, nps = _measure_minimax(d, board)
        minimax_rows.append((d, t_med))
        rows.append(
            {
                "AI": "MinimaxAI",
                "Config": f"depth={d}",
                "Median time (s)": _fmt_float(t_med),
                "Throughput": f"{int(nps):,} nodes/s",
            }
        )

    # MCTS：同 simulations=1600，workers=1 vs workers=max
    sims = 1600

    def _mk_mcts(workers: int) -> MCTSAI:
        return MCTSAI(max_simulations=sims, time_limit=999.0, workers=workers, verbose=False)

    for workers in (1, wmax):
        t_med, sps = _measure_mcts_like(
            cls_name="MCTSAI",
            make_ai=_mk_mcts,
            board=board,
            simulations=sims,
            workers=workers,
        )
        rows.append(
            {
                "AI": "MCTSAI",
                "Config": f"sims={sims}, workers={workers}",
                "Median time (s)": _fmt_float(t_med),
                "Throughput": f"{int(sps):,} sims/s",
            }
        )

    mcts_sps_workers1: Optional[float] = None
    mcts_time_workers1: Optional[float] = None

    # 取回 MCTS workers=1 数据用于开销分析（从 rows 中解析最简单）
    for r in rows:
        if r["AI"] == "MCTSAI" and r["Config"] == f"sims={sims}, workers=1":
            mcts_time_workers1 = float(r["Median time (s)"])
            # Throughput: "{int(sps):,} sims/s"
            mcts_sps_workers1 = float(r["Throughput"].split()[0].replace(",", ""))
            break

    # 输出 Markdown 表格（仅脚本末尾输出）
    headers = ["AI", "Config", "Median time (s)", "Throughput"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        print("| " + " | ".join(str(r[h]) for h in headers) + " |")

    print()

    # ── 实验参数建议（为 experiment_runner 服务） ──
    # 推荐用“固定每步 time_limit”做对弈公平对比；max_simulations 只作为兜底上限
    budget_s = 0.50
    suggested_depth = _suggest_depth(minimax_rows)
    print(f"- **建议对弈 time_limit**：每步 `time_limit={budget_s:.2f}s`（公平对比用）")
    print(f"- **建议 MinimaxAI 深度**：`depth={suggested_depth}`（接近中局单步耗时基准）")

    mcts_sps_max = None
    for r in rows:
        if r["AI"] == "MCTSAI" and r["Config"] == f"sims={sims}, workers={wmax}":
            mcts_sps_max = float(r["Throughput"].split()[0].replace(",", ""))
    if mcts_sps_max is not None:
        rec = _suggest_simulations(mcts_sps_max, budget_s, safety=1.8)
        print(
            f"- **建议 MCTSAI 上限模拟次数**：`max_simulations≈{rec:d}`（workers={wmax}，主要由 time_limit 控制，sims 仅防极端情况）"
        )


if __name__ == "__main__":
    main()

