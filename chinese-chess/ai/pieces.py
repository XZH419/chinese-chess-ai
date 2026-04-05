class Piece:
    def __init__(self, color, piece_type):
        self.color = color  # 'red' or 'black'
        self.piece_type = piece_type  # 'jiang', 'shi', 'xiang', 'ma', 'che', 'pao', 'bing'

    def __str__(self):
        symbols = {
            'red': {'jiang': '帅', 'shi': '仕', 'xiang': '相', 'ma': '马', 'che': '车', 'pao': '炮', 'bing': '兵'},
            'black': {'jiang': '将', 'shi': '士', 'xiang': '象', 'ma': '马', 'che': '车', 'pao': '炮', 'bing': '卒'}
        }
        return symbols[self.color][self.piece_type]

    def __repr__(self):
        return f"{self.color}_{self.piece_type}"