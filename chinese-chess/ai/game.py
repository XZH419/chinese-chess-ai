from .board import Board
from .ai_minimax import MinimaxAI

class Game:
    def __init__(self):
        self.board = Board()
        self.ai = MinimaxAI(depth=3)

    def play(self):
        while not self.board.is_game_over():
            print(str(self.board))
            print(f"{self.board.current_player}'s turn")
            if self.board.current_player == 'red':
                move = self.get_human_move()
            else:
                move = self.ai.get_best_move(self.board, time_limit=8)
                if move:
                    print(f"AI chooses: {move}")
            if move:
                captured = self.board.make_move(*move)
                if self.board.is_check(self.board.current_player):
                    self.board.undo_move(*move, captured)
                    print("Illegal move: leaves king in check")
                    continue
            else:
                print("No moves available")
                break
        print(str(self.board))
        print("Game over")
        if not any(piece and piece.color == 'red' and piece.piece_type == 'jiang' for row in self.board.board for piece in row):
            print("Black wins")
        elif not any(piece and piece.color == 'black' and piece.piece_type == 'jiang' for row in self.board.board for piece in row):
            print("Red wins")
        else:
            print("Stalemate or no legal moves")

    def get_human_move(self):
        while True:
            try:
                start = input("Enter start position (row col): ").split()
                end = input("Enter end position (row col): ").split()
                sr, sc = int(start[0]), int(start[1])
                er, ec = int(end[0]), int(end[1])
                if self.board.is_valid_move(sr, sc, er, ec, player='red'):
                    captured = self.board.make_move(sr, sc, er, ec)
                    if self.board.is_check('red'):
                        self.board.undo_move(sr, sc, er, ec, captured)
                        print("Illegal move: red king would be in check")
                        continue
                    self.board.undo_move(sr, sc, er, ec, captured)
                    return (sr, sc, er, ec)
                print("Invalid move")
            except ValueError:
                print("Invalid input: please enter two integers")
            except Exception as exc:
                print(f"Invalid input: {exc}")