"""控制台模式下的人机对战循环模块。"""

from chess.control.controller import GameController

class Game:
    def __init__(self):
        # 迁移到 Controller 驱动：棋盘/规则/AI 回合由 controller 统一管理
        self.controller = GameController()

    def play(self):
        """运行主控制台循环，直到游戏结束。"""
        while not self.controller.is_game_over():
            print(str(self.controller.board))
            print(f"{self.controller.board.current_player}'s turn")
            if self.controller.board.current_player == "red":
                move = self.get_human_move()
                outcome = self.controller.apply_move(move, player="red")
                if not outcome.ok:
                    print("无效走子")
                    continue
            else:
                # 默认对手：RandomAI（controller 内部）
                outcome = self.controller.maybe_play_ai_turn(time_limit=8)
                if outcome.ok:
                    print(f"AI 选择: {outcome}")
                else:
                    # AI 无法行动（多为无合法走法/终局）
                    pass

        print(str(self.controller.board))
        print("游戏结束")
        winner = self.controller.winner()
        if winner == "red":
            print("红方获胜")
        elif winner == "black":
            print("黑方获胜")
        else:
            print("未判定胜者")

    def get_human_move(self):
        """提示玩家输入合法走法，坐标格式为行 列。"""
        while True:
            try:
                start = input("输入起始位置（行 列）：").split()
                end = input("输入目标位置（行 列）：").split()
                sr, sc = int(start[0]), int(start[1])
                er, ec = int(end[0]), int(end[1])
                move = (sr, sc, er, ec)
                if self.controller.can_move(move, player="red"):
                    return move
                print("无效走子")
            except ValueError:
                print("输入无效：请输入两个整数")
            except Exception as exc:
                print(f"输入无效：{exc}")
