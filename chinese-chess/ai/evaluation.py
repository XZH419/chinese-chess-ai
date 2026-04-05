from .pieces import Piece

class Evaluation:
    PIECE_VALUES = {
        'jiang': 10000,
        'shi': 120,
        'xiang': 110,
        'ma': 300,
        'che': 600,
        'pao': 300,
        'bing': 70
    }

    POSITION_VALUES = {
        'jiang': [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 2, 2, 2, 0, 0, 0],
            [0, 0, 0, 11, 15, 11, 0, 0, 0]
        ],
        'bing': [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        # Add more position values for other pieces
    }

    @staticmethod
    def evaluate(board):
        score = 0
        for r in range(board.rows):
            for c in range(board.cols):
                piece = board.board[r][c]
                if piece:
                    value = Evaluation.PIECE_VALUES[piece.piece_type]
                    if piece.color == 'red':
                        score += value
                        # Add position value
                        if piece.piece_type in Evaluation.POSITION_VALUES:
                            score += Evaluation.POSITION_VALUES[piece.piece_type][r][c]
                    else:
                        score -= value
                        # Add position value (flip for black)
                        if piece.piece_type in Evaluation.POSITION_VALUES:
                            score -= Evaluation.POSITION_VALUES[piece.piece_type][9-r][c]

        # Add mobility bonus
        red_moves = len(board.get_all_moves('red'))
        black_moves = len(board.get_all_moves('black'))
        score += (red_moves - black_moves) * 0.1

        return score