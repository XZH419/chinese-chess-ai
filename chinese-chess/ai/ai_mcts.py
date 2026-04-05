import math
import random
import time
from .evaluation import Evaluation

class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_moves = board.get_all_moves(board.current_player)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.4):
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                choices_weights.append(float('inf'))
                continue
            exploit = child.value / child.visits
            explore = c_param * math.sqrt((2 * math.log(self.visits)) / child.visits)
            choices_weights.append(exploit + explore)
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        move = self.untried_moves.pop()
        new_board = self.board.copy()  # Need to implement copy
        new_board.make_move(*move)
        child_node = MCTSNode(new_board, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.value += result

class MCTSAI:
    def __init__(self, time_limit=10):
        self.time_limit = time_limit

    def get_best_move(self, board):
        root = MCTSNode(board)
        start_time = time.time()

        while time.time() - start_time < self.time_limit:
            node = root
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            # Expansion
            if not node.is_fully_expanded():
                node = node.expand()

            # Simulation
            result = self.simulate(node.board)

            # Backpropagation
            while node:
                node.update(result)
                node = node.parent

        if not root.children:
            return None
        return root.best_child(c_param=0).move

    def simulate(self, board):
        # Random playout
        current_board = board.copy()
        while not current_board.is_game_over():
            moves = current_board.get_all_moves(current_board.current_player)
            if not moves:
                break
            move = random.choice(moves)
            current_board.make_move(*move)
        return Evaluation.evaluate(current_board)