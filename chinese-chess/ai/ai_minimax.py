"""Compatibility wrapper.

The actual implementation has been migrated to `chess.algorithm.minimax`.
Keep this module so older imports (`from ai.ai_minimax import MinimaxAI`)
continue to work during the refactor.
"""

from chess.algorithm.minimax import MinimaxAI

__all__ = ["MinimaxAI"]