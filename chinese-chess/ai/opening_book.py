# Opening book for Chinese Chess
# Simple dictionary of board states to recommended moves

class OpeningBook:
    def __init__(self):
        self.book = {
            # Initial position -> common opening moves
            "initial": [(9, 6, 7, 8), (9, 2, 7, 0), (9, 7, 7, 6)]  # Example moves
        }

    def get_move(self, board):
        # For simplicity, return a random move from book if available
        if "initial" in self.book:
            import random
            return random.choice(self.book["initial"])
        return None