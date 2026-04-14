"""独立 AI 子进程：在 **非 Qt** 环境中执行 ``get_best_move`` / ``choose_move``。

GUI 主进程仅通过 ``multiprocessing.Queue`` 发送纯数据请求并取回结果，
避免在 ``QThread`` 中直接调用 ``ProcessPoolExecutor``（Windows 上易死锁）。

`infra.ai_state_codec` 负责棋盘与历史的序列化。
"""

from __future__ import annotations

import multiprocessing
import traceback
from queue import Empty
from typing import Any, Dict, Optional

from infra.ai_state_codec import (
    deserialize_board,
    deserialize_move_history,
)
from ai.ai_registry import create_ai_from_config


def process_search_request(msg: Dict[str, Any]) -> Dict[str, Any]:
    """执行单次搜索，返回可放入 ``response_queue`` 的纯数据 ``dict``。"""
    request_id = msg["request_id"]
    try:
        board = deserialize_board(msg["board"])
        move_history = deserialize_move_history(msg.get("move_history") or [])
        game_history = list(msg.get("game_history") or [])
        ai_cfg = msg["ai_config"]
        ai_color = str(msg["ai_color"])
        time_limit_s = msg.get("time_limit_s")
        ai = create_ai_from_config(ai_cfg)
        board.current_player = ai_color
        if hasattr(ai, "choose_move"):
            move = ai.choose_move(
                board,
                time_limit=time_limit_s,
                game_history=game_history,
                move_history=move_history,
            )
        else:
            move = ai.get_best_move(
                board,
                time_limit=time_limit_s,
                game_history=game_history,
                move_history=move_history,
            )
        stats = getattr(ai, "last_stats", None)
        return {
            "request_id": request_id,
            "ok": True,
            "move": move,
            "stats": stats,
            "error": None,
        }
    except Exception as e:
        return {
            "request_id": request_id,
            "ok": False,
            "move": None,
            "stats": {"error": str(e)},
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def ai_worker_main(request_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue) -> None:
    """子进程入口：循环处理请求，直到收到 ``None`` 哨兵。"""
    print("[AI子进程] 已启动，等待搜索请求…")
    while True:
        try:
            msg = request_queue.get()
        except (EOFError, OSError):
            break
        if msg is None:
            print("[AI子进程] 收到关闭指令，退出。")
            break
        rid = msg.get("request_id", "?")
        print(f"[AI子进程] 开始搜索 request_id={rid}")
        resp = process_search_request(msg)
        try:
            response_queue.put(resp)
        except Exception:
            pass
        print(f"[AI子进程] 已完成 request_id={rid} ok={resp.get('ok')}")


def drain_queue(q: multiprocessing.Queue) -> int:
    """丢弃队列中待处理消息（用于丢弃过期搜索结果）。返回丢弃条数。"""
    n = 0
    while True:
        try:
            q.get_nowait()
            n += 1
        except Empty:
            break
    return n


def shutdown_worker(request_queue: Optional[multiprocessing.Queue]) -> None:
    """向子进程发送 ``None`` 哨兵。"""
    if request_queue is None:
        return
    try:
        request_queue.put(None)
    except Exception:
        pass
