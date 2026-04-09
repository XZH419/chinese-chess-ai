"""Compatibility wrapper.

The actual implementation has been migrated to `chess.algorithm.mcts`.
Keep this module so older imports (`from ai.ai_mcts import MCTSAI`) continue
to work during the refactor.
"""

from chess.algorithm.mcts import MCTSAI, MCTSNode

__all__ = ["MCTSAI", "MCTSNode"]