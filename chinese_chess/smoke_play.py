"""Headless smoke test for controller + agents.

Run (repo root):
    python -m chinese_chess.main cli --red random --black random

This file is also invoked from `main.py` cli mode with the user-built controller.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from chinese_chess.control.controller import GameController

# When running this file directly, Python puts this directory (`.../chinese_chess/`)
# on sys.path, which prevents `import chinese_chess...` from resolving (it needs the
# repo root on sys.path). Insert the repo root explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main(controller: Optional["GameController"] = None) -> None:
    from chinese_chess.algorithm.random_ai import RandomAI
    from chinese_chess.control.controller import GameController

    if controller is None:
        controller = GameController(red_agent=RandomAI(), black_agent=RandomAI())

    max_plies = 40
    plies = 0

    while not controller.is_game_over() and plies < max_plies:
        controller.maybe_play_ai_turn(time_limit=0.1)
        plies += 1

    print("smoke ok")
    print("plies", plies)
    print("winner", controller.winner())


if __name__ == "__main__":
    main()
