"""Headless smoke test for controller + RandomAI.

Run (inside conda env):
    conda run -n chessai python chess/smoke_play.py

This script is non-interactive and is meant to catch obvious import/runtime
errors after refactors.
"""

from __future__ import annotations

import os
import sys

# When running this file directly, Python puts this directory (`.../chess/`)
# on sys.path, which prevents `import chess...` from resolving (it needs the
# repo root on sys.path). Insert the repo root explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chess.control.controller import GameController


def main() -> None:
    controller = GameController()

    # Let the controller's default RandomAI play as black.
    # For smoke testing, we'll also let red play random by temporarily swapping
    # the ai_color when it's red's turn.
    max_plies = 40
    plies = 0

    while not controller.is_game_over() and plies < max_plies:
        if controller.board.current_player == "red":
            controller.ai_color = "red"
            controller.maybe_play_ai_turn(time_limit=0.1)
            controller.ai_color = "black"
        else:
            controller.maybe_play_ai_turn(time_limit=0.1)
        plies += 1

    print("smoke ok")
    print("plies", plies)
    print("winner", controller.winner())


if __name__ == "__main__":
    main()

