"""Compatibility wrapper.

The actual GUI is implemented in `chess.view.qt.main_window` to keep the GUI
aligned with the new MVC-ish architecture. This module remains so
`python chinese-chess/main.py gui` keeps working.
"""

from chess.view.qt.main_window import MainWindow

__all__ = ["MainWindow"]