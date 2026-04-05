"""控制台模式下的人机对战循环模块。"""

from .board import Board
from .ai_minimax import MinimaxAI

class Game:
    def __init__(self):
        self.board = Board()
        self.ai = MinimaxAI(depth=3)

    def play(self):
        """运行主控制台循环，直到游戏结束。"""
        while not self.board.is_game_over():
            print(str(self.board))
            print(f"{self.board.current_player}'s turn")
            if self.board.current_player == 'red':
                move = self.get_human_move()
            else:
                move = self.ai.get_best_move(self.board, time_limit=8)
                if move:
                    print(f"AI 选择: {move}")
            if move:
                captured = self.board.make_move(*move)
                if self.board.is_check(self.board.current_player):
                    self.board.undo_move(*move, captured)
                    print("非法走子：将帅被将军")
                    continue
            else:
                print("无合法走法")
                break
        print(str(self.board))
        print("游戏结束")
        if not any(piece and piece.color == 'red' and piece.piece_type == 'jiang' for row in self.board.board for piece in row):
            print("黑方获胜")
        elif not any(piece and piece.color == 'black' and piece.piece_type == 'jiang' for row in self.board.board for piece in row):
            print("红方获胜")
        else:
            print("和棋或无合法走法")

    def get_human_move(self):
        """提示玩家输入合法走法，坐标格式为行 列。"""
        while True:
            try:
                start = input("输入起始位置（行 列）：").split()
                end = input("输入目标位置（行 列）：").split()
                sr, sc = int(start[0]), int(start[1])
                er, ec = int(end[0]), int(end[1])
                if self.board.is_valid_move(sr, sc, er, ec, player='red'):
                    captured = self.board.make_move(sr, sc, er, ec)
                    if self.board.is_check('red'):
                        self.board.undo_move(sr, sc, er, ec, captured)
                        print("非法走子：红方将帅会被将军")
                        continue
                    self.board.undo_move(sr, sc, er, ec, captured)
                    return (sr, sc, er, ec)
                print("无效走子")
            except ValueError:
                print("输入无效：请输入两个整数")
            except Exception as exc:
                print(f"输入无效：{exc}")
