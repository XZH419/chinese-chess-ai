"""中国象棋的蒙特卡洛树搜索实现模块。"""

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
        # 当前状态下尚未扩展的走法列表。
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
        # 通过执行一个未尝试走法创建新子节点。
        move = self.untried_moves.pop()
        new_board = self.board.copy()
        new_board.make_move(*move)
        child_node = MCTSNode(new_board, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.value += result

class MCTSAI:
    """中国象棋的蒙特卡洛树搜索AI。

    该AI使用UCT策略进行选择、扩展、模拟和回传。
    """

    def __init__(self, time_limit=10):
        self.time_limit = time_limit

    def get_best_move(self, board):
        # 通过大量随机模拟搜索最优走法，直到达到时间上限。
        root = MCTSNode(board)
        start_time = time.time()

        while time.time() - start_time < self.time_limit:
            node = root
            # 选择阶段：从根节点沿最佳UCT子节点向下选择直到叶节点。
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            # 扩展阶段：如果当前节点还有未尝试的走法，则扩展一个新子节点。
            if not node.is_fully_expanded():
                node = node.expand()

            # 模拟阶段：从扩展后的节点随机模拟对弈结果。
            result = self.simulate(node.board)

            # 回传阶段：将模拟结果逐层累积到根节点。
            while node:
                node.update(result)
                node = node.parent

        if not root.children:
            return None
        return root.best_child(c_param=0).move

    def simulate(self, board):
        # 从当前局面随机模拟对弈直到游戏结束。
        current_board = board.copy()
        while not current_board.is_game_over():
            moves = current_board.get_all_moves(current_board.current_player)
            if not moves:
                break
            move = random.choice(moves)
            current_board.make_move(*move)
        return Evaluation.evaluate(current_board)