"""AI 注册表：统一 AI 类型映射与可序列化配置。

本模块的目标是消除 GUI 与子进程 worker 中重复的 if/else 构造逻辑，
把 “ai_type 字符串 ↔ AI 类 ↔ 参数” 维护成**单一数据源**。

约定：
- ``ai_type`` 使用稳定的短字符串（例如 ``"minimax"``、``"mcts"``）。
- 配置字典必须只包含可通过 ``multiprocessing.Queue`` 传递的纯数据类型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

from ai.minimax_ai import MinimaxAI
from ai.mcts_ai import MCTSAI
from ai.random_ai import RandomAI


@dataclass(frozen=True, slots=True)
class AIRegistryEntry:
    """注册表条目：描述一种可构造的 AI。"""

    ai_type: str
    display_name: str
    cls: Type[Any]
    from_config: Callable[[Dict[str, Any]], Any]
    to_config: Callable[[Any], Dict[str, Any]]


def _cfg_random(_: Dict[str, Any]) -> RandomAI:
    return RandomAI()


def _cfg_minimax(cfg: Dict[str, Any]) -> MinimaxAI:
    return MinimaxAI(depth=int(cfg.get("depth", 5)))


def _cfg_mcts(cfg: Dict[str, Any]) -> MCTSAI:
    return MCTSAI(
        max_simulations=int(cfg.get("max_simulations", 5000)),
        time_limit=float(cfg.get("time_limit", 5.0)),
        workers=cfg.get("workers"),
        verbose=bool(cfg.get("verbose", False)),
    )


def _cfg_mcts_minimax(cfg: Dict[str, Any]) -> MCTSAI:
    """兼容旧键：mcts_minimax 已合并为纯 MCTS，统一走 MCTSAI。"""
    return MCTSAI(
        max_simulations=int(cfg.get("max_simulations", 5000)),
        time_limit=float(cfg.get("time_limit", 10.0)),
        workers=cfg.get("workers"),
        verbose=bool(cfg.get("verbose", False)),
    )


def _to_cfg_random(_: RandomAI) -> Dict[str, Any]:
    return {"ai_type": "random"}


def _to_cfg_minimax(agent: MinimaxAI) -> Dict[str, Any]:
    return {"ai_type": "minimax", "depth": int(getattr(agent, "depth", 5))}


def _to_cfg_mcts(agent: MCTSAI) -> Dict[str, Any]:
    w = getattr(agent, "workers", None)
    return {
        "ai_type": "mcts",
        "max_simulations": int(getattr(agent, "max_simulations", 5000)),
        "time_limit": float(getattr(agent, "time_limit", 5.0)),
        "workers": (int(w) if w is not None else None),
        "verbose": bool(getattr(agent, "verbose", False)),
    }


def _to_cfg_mcts_minimax(agent: MCTSAI) -> Dict[str, Any]:
    """兼容旧键：保持 ai_type=mcts_minimax 以兼容 GUI/脚本配置。"""
    w = getattr(agent, "workers", None)
    return {
        "ai_type": "mcts_minimax",
        "max_simulations": int(getattr(agent, "max_simulations", 5000)),
        "time_limit": float(getattr(agent, "time_limit", 10.0)),
        "workers": (int(w) if w is not None else None),
        "verbose": bool(getattr(agent, "verbose", False)),
    }


AI_REGISTRY: Dict[str, AIRegistryEntry] = {
    "random": AIRegistryEntry(
        ai_type="random",
        display_name="随机 AI",
        cls=RandomAI,
        from_config=_cfg_random,
        to_config=_to_cfg_random,
    ),
    "minimax": AIRegistryEntry(
        ai_type="minimax",
        display_name="Minimax AI",
        cls=MinimaxAI,
        from_config=_cfg_minimax,
        to_config=_to_cfg_minimax,
    ),
    "mcts": AIRegistryEntry(
        ai_type="mcts",
        display_name="MCTS AI",
        cls=MCTSAI,
        from_config=_cfg_mcts,
        to_config=_to_cfg_mcts,
    ),
    "mcts_minimax": AIRegistryEntry(
        ai_type="mcts_minimax",
        display_name="MCTS AI",
        cls=MCTSAI,
        from_config=_cfg_mcts_minimax,
        to_config=_to_cfg_mcts_minimax,
    ),
}


def create_ai_from_config(cfg: Dict[str, Any]) -> Any:
    """由纯数据配置构造 AI 实例。

    Args:
        cfg: 至少包含 ``ai_type`` 键的配置字典。

    Returns:
        对应的 AI 实例。
    """
    t = str(cfg["ai_type"])
    try:
        entry = AI_REGISTRY[t]
    except KeyError as e:
        raise ValueError(f"unknown ai_type: {t!r}") from e
    return entry.from_config(cfg)


def build_ai_config_dict(agent: Any) -> Dict[str, Any]:
    """从 AI 实例提取可重建配置（仅数据）。

    Args:
        agent: AI 实例（不可为 ``None``）。

    Returns:
        纯数据配置字典，可用于子进程重建同类 AI。
    """
    if agent is None:
        raise ValueError("agent is None")
    name = type(agent).__name__
    for entry in AI_REGISTRY.values():
        if entry.cls.__name__ == name:
            return entry.to_config(agent)
    raise TypeError(f"unsupported AI type: {name}")


def engine_key_for_agent(agent: Optional[Any]) -> str:
    """将 agent 映射为 engine key（与 GUI 下拉框/配置使用的 key 一致）。"""
    if agent is None:
        return "human"
    name = type(agent).__name__
    for k, entry in AI_REGISTRY.items():
        if entry.cls.__name__ == name:
            return k
    return "human"

