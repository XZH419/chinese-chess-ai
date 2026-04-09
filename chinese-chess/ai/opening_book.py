"""Compatibility wrapper.

The actual implementation has been migrated to `chess.algorithm.opening_book`.
"""

from chess.algorithm.opening_book import OpeningBook

__all__ = ["OpeningBook"]