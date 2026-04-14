"""MCTS 系列引擎的共享内部实现入口（内部模块）。

当前代码库中 `mcts_minimax.py` 需要复用 `mcts.py` 的若干“内部”工具函数与常量。
若直接从 `mcts.py` 导入下划线私有符号，会导致两个模块强耦合、重构脆弱。

本模块提供一个**稳定的内部转接层**：
- 允许 `mcts.py` 与 `mcts_minimax.py` 共享实现细节
- 便于后续把共享逻辑从 `mcts.py` 逐步迁移到独立文件，而不影响调用方

注意：这不是对外 API；但它是项目内部的“共享入口”，应避免随意破坏其导出符号。
"""

from __future__ import annotations

# 目前先做“转发/再导出”，以最小改动消除跨文件直接 import 私有符号。
# 后续重构可把实现从 `mcts.py` 挪到本模块，再反向让 `mcts.py` 导入。

from .mcts import Move4
from .mcts import _append_path_move_entry
from .mcts import _is_aggressive_push
from .mcts import _move_gives_check
from .mcts import _order_untried_moves_policy
from .mcts import _parallel_workers_when_safe
from .mcts import _pick_rollout_move_fast
from .mcts import _policy_attack_bias
from .mcts import _POLICY_HVCAP_VALUE
from .mcts import _ROOT_BIAS_SCALE
from .mcts import _ROOT_VISITS_TIE_FRAC
from .mcts import _SELECTION_MAX_PLIES

__all__ = [
    "Move4",
    "_append_path_move_entry",
    "_is_aggressive_push",
    "_move_gives_check",
    "_order_untried_moves_policy",
    "_parallel_workers_when_safe",
    "_pick_rollout_move_fast",
    "_policy_attack_bias",
    "_POLICY_HVCAP_VALUE",
    "_ROOT_BIAS_SCALE",
    "_ROOT_VISITS_TIE_FRAC",
    "_SELECTION_MAX_PLIES",
]

