import time
from .evaluation import Evaluation

class MinimaxAI:
    def __init__(self, depth=3):
        self.depth = depth

    def get_best_move(self, board, time_limit=10):
        start_time = time.time()
        best_move = None
        original_player = board.current_player
        best_value = float('-inf') if original_player == 'red' else float('inf')

        moves = board.get_legal_moves(original_player)
        for move in moves:
            captured = board.make_move(*move)
            value = self.minimax(board, self.depth - 1, float('-inf'), float('inf'), original_player != 'red')
            board.undo_move(*move, captured)

            if original_player == 'red':  # Maximizing
                if value > best_value:
                    best_value = value
                    best_move = move
            else:  # Minimizing
                if value < best_value:
                    best_value = value
                    best_move = move

            if time.time() - start_time > time_limit:
                break

        return best_move

    def minimax(self, board, depth, alpha, beta, maximizing):
        if depth == 0 or board.is_game_over():
            return Evaluation.evaluate(board)

        if maximizing:
            max_eval = float('-inf')
            moves = board.get_all_moves(board.current_player)
            for move in moves:
                captured = board.make_move(*move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.undo_move(*move, captured)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            moves = board.get_all_moves(board.current_player)
            for move in moves:
                captured = board.make_move(*move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.undo_move(*move, captured)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval