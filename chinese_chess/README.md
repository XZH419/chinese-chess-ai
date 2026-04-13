# Chinese Chess AI

A complete Chinese Chess (Xiangqi) AI implementation with rule framework, AI algorithms, and GUI.

## Features

- Complete Chinese Chess rule implementation
- Random AI for testing
- Minimax AI with alpha-beta pruning
- MCTS (Monte Carlo Tree Search) AI
- **MCTS+Minimax hybrid** (`MCTSMinimaxAI`): MCTS 全局探索 + 叶节点浅层 Minimax 战术精算
- Heuristic evaluation function with piece values and position bonuses
- Opening and endgame book support
- PyQt5 graphical user interface
- Console and GUI modes

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Console Mode
```bash
python main.py
```

### GUI Mode
```bash
python main.py gui
```

## Project Structure

```
chinese-chess/
├── ai/
│   ├── __init__.py
│   ├── board.py          # Game board and rules
│   ├── pieces.py         # Piece classes
│   ├── game.py           # Game logic
│   ├── ai_minimax.py     # Minimax AI
│   ├── ai_mcts.py        # MCTS AI
│   ├── evaluation.py     # Evaluation functions
│   ├── opening_book.py   # Opening book
│   └── endgame_book.py   # Endgame book
├── gui/
│   ├── __init__.py
│   └── main_window.py    # PyQt5 GUI
├── main.py               # Entry point
└── requirements.txt      # Dependencies
```

## AI Algorithms

1. **Random AI**: Selects random valid moves for testing
2. **Minimax**: Depth-limited search with alpha-beta pruning
3. **MCTS**: Monte Carlo Tree Search with UCB selection
4. **MCTS+Minimax (Hybrid)**: `MCTSMinimaxAI` in `algorithm/mcts_minimax.py` — same `choose_move` / `get_best_move` API as other engines

CLI examples::

    python -m chinese_chess.main cli --red mcts_minimax --black minimax --red-sims 3000
    python -m chinese_chess.main gui --red human --black mcts_minimax --black-sims 5000

Benchmark (root `benchmark.py`)::

    python benchmark.py --games 2 --red-ai mcts_minimax --black-ai minimax --red-sims 500 --black-depth 3

Profiler::

    python -m chinese_chess.scripts.profile_tool mcts_minimax --simulations 800

## Evaluation Function

- Piece values: Jiang(10000), Shi/Xiang(120/110), Ma/Pao(300), Che(600), Bing(70)
- Position bonuses for key pieces
- Mobility bonus based on number of legal moves

## Rules Implemented

- All piece movement rules (Che, Ma, Xiang, Shi, Jiang, Pao, Bing)
- Special rules: Ma leg blocking, Xiang eye blocking, Pao cannon rule
- Jiang/Shuai palace restrictions
- Jiang/Shuai face-to-face check
- Check and checkmate detection
- Game over conditions

## GUI Features

- Visual board display
- AI move button
- Game reset
- Status display
