"""Compatibility wrapper.

The actual implementation has been migrated to `chess.algorithm.endgame_book`.
"""

from chess.algorithm.endgame_book import EndgameBook

__all__ = ["EndgameBook"]