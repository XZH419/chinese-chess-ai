class Piece:
    """表示一个中国象棋棋子。"""

    def __init__(self, color, piece_type):
        # 棋子颜色，值为 'red' 或 'black'。
        self.color = color
        # 棋子类型，用于区分不同棋子功能。
        self.piece_type = piece_type

    def __str__(self):
        # 将棋子转换为可显示的中文符号。
        symbols = {
            'red': {'jiang': '帅', 'shi': '仕', 'xiang': '相', 'ma': '马', 'che': '车', 'pao': '炮', 'bing': '兵'},
            'black': {'jiang': '将', 'shi': '士', 'xiang': '象', 'ma': '马', 'che': '车', 'pao': '炮', 'bing': '卒'}
        }
        return symbols[self.color][self.piece_type]

    def __repr__(self):
        return f"{self.color}_{self.piece_type}"