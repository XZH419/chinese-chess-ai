"""Compatibility wrapper.

The actual implementation has been migrated to `chess.algorithm.evaluation`.
Keep this module so older imports (`from ai.evaluation import Evaluation`)
continue to work during the refactor.
"""

from chess.algorithm.evaluation import Evaluation

__all__ = ["Evaluation"]