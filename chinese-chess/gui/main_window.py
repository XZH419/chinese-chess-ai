import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QLabel, QPushButton, QSpinBox, QHBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ai.board import Board
from ai.ai_minimax import MinimaxAI

# 人机对战图形界面模块。
# 棋盘使用按钮网格显示，AI 思考在后台线程运行，避免界面阻塞。

class AIMoveThread(QThread):
    """后台线程，用于执行AI搜索，不阻塞主窗口界面。"""

    move_found = pyqtSignal(tuple)

    def __init__(self, ai, board, time_limit):
        super().__init__()
        self.ai = ai
        self.board = board.copy()
        self.time_limit = time_limit

    def run(self):
        move = self.ai.get_best_move(self.board, time_limit=self.time_limit)
        self.move_found.emit(move)

class ChessBoardWidget(QWidget):
    """棋盘显示控件，将棋盘每个格子渲染为可点击的按钮。"""

    def __init__(self, board, click_handler):
        super().__init__()
        self.board = board
        self.click_handler = click_handler
        self.init_ui()

    def init_ui(self):
        # 构建一个 10x9 的按钮网格，用于表示棋盘上的每个格子。
        layout = QGridLayout()
        self.buttons = []
        for r in range(self.board.rows):
            row_buttons = []
            for c in range(self.board.cols):
                button = QPushButton()
                button.setFixedSize(50, 50)
                button.setStyleSheet("font-size: 18px; border: 1px solid black;")
                button.clicked.connect(lambda _, rr=r, cc=c: self.click_handler(rr, cc))
                layout.addWidget(button, r, c)
                row_buttons.append(button)
            self.buttons.append(row_buttons)
        self.setLayout(layout)
        self.update_board()

    def update_board(self):
        # 根据当前棋盘状态刷新所有格子的文本和颜色显示。
        for r in range(self.board.rows):
            for c in range(self.board.cols):
                piece = self.board.board[r][c]
                text = str(piece) if piece else ''
                if piece:
                    if piece.color == 'red':
                        style = "font-size: 18px; border: 1px solid black; color: red; background-color: #ffe6e6;"
                    else:
                        style = "font-size: 18px; border: 1px solid black; color: black; background-color: #e6e6e6;"
                else:
                    style = "font-size: 18px; border: 1px solid black; background-color: white;"
                self.buttons[r][c].setText(text)
                self.buttons[r][c].setStyleSheet(style)

    def highlight_square(self, row, col):
        # 将选中的棋盘格高亮显示，便于玩家确认所选棋子位置。
        if 0 <= row < self.board.rows and 0 <= col < self.board.cols:
            piece = self.board.board[row][col]
            if piece:
                if piece.color == 'red':
                    style = "font-size: 18px; border: 2px solid red; background: #ffcccc; color: red;"
                else:
                    style = "font-size: 18px; border: 2px solid red; background: #ffcccc; color: black;"
            else:
                style = "font-size: 18px; border: 2px solid red; background: #ffcccc;"
            self.buttons[row][col].setStyleSheet(style)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.board = Board()
        self.ai = MinimaxAI(depth=2)
        self.human_color = 'red'
        self.selected = None
        self.ai_thread = None
        self.time_limit = 5
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("中国象棋 AI")
        self.setGeometry(100, 100, 600, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        self.board_widget = ChessBoardWidget(self.board, self.on_square_clicked)
        layout.addWidget(self.board_widget)

        player_name = '红方' if self.board.current_player == 'red' else '黑方'
        self.status_label = QLabel(f"{player_name}回合")
        layout.addWidget(self.status_label)

        time_layout = QHBoxLayout()
        time_label = QLabel("AI 思考时间（秒）：")
        self.time_spinbox = QSpinBox()
        self.time_spinbox.setRange(1, 60)
        self.time_spinbox.setValue(self.time_limit)
        self.time_spinbox.valueChanged.connect(self.on_time_changed)
        time_layout.addWidget(time_label)
        time_layout.addWidget(self.time_spinbox)
        layout.addLayout(time_layout)

        button_layout = QVBoxLayout()
        self.ai_button = QPushButton("AI 走子")
        self.ai_button.clicked.connect(self.ai_move)
        button_layout.addWidget(self.ai_button)

        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.reset_game)
        button_layout.addWidget(self.reset_button)

        layout.addLayout(button_layout)
        central_widget.setLayout(layout)

    def get_winner(self):
        """判断游戏是否结束，返回获胜方，否则返回 None。"""
        red_jiang = any(piece and piece.color == 'red' and piece.piece_type == 'jiang' for row in self.board.board for piece in row)
        black_jiang = any(piece and piece.color == 'black' and piece.piece_type == 'jiang' for row in self.board.board for piece in row)
        if not red_jiang:
            return 'black'
        if not black_jiang:
            return 'red'
        if not self.board.has_legal_moves('red'):
            return 'black'
        if not self.board.has_legal_moves('black'):
            return 'red'
        return None

    def on_time_changed(self, value):
        self.time_limit = value

    def on_square_clicked(self, row, col):
        # 处理用户点击事件，支持选中棋子和落子操作。
        winner = self.get_winner()
        if winner:
            winner_name = '红方' if winner == 'red' else '黑方'
            self.status_label.setText(f"{winner_name}获胜！")
            return
        if self.board.current_player != self.human_color:
            return
        piece = self.board.get_piece(row, col)
        if self.selected is None:
            if piece and piece.color == self.human_color:
                self.selected = (row, col)
                self.board_widget.highlight_square(row, col)
                self.status_label.setText(f"选中 {row},{col}")
        else:
            sr, sc = self.selected
            er, ec = row, col
            if self.board.is_valid_move(sr, sc, er, ec, player=self.human_color):
                captured = self.board.make_move(sr, sc, er, ec)
                if self.board.is_check(self.human_color):
                    self.board.undo_move(sr, sc, er, ec, captured)
                    self.status_label.setText("非法走子：将帅被将军")
                else:
                    self.board_widget.board = self.board
                    self.board_widget.update_board()
                    self.selected = None
                    winner = self.get_winner()
                    if winner:
                        winner_name = '红方' if winner == 'red' else '黑方'
                        self.status_label.setText(f"{winner_name}获胜！")
                    else:
                        player_name = '红方' if self.board.current_player == 'red' else '黑方'
                        self.status_label.setText(f"{player_name}回合")
                        self.ai_move()
            else:
                self.status_label.setText("无效走子")
                self.selected = None
            self.board_widget.board = self.board
            self.board_widget.update_board()

    def ai_move(self):
        # 在后台线程中启动AI搜索，避免界面卡顿或无响应。
        if self.board.is_game_over():
            self.status_label.setText("游戏结束")
            return
        if self.ai_thread and self.ai_thread.isRunning():
            return
        self.status_label.setText("AI 思考中...")
        self.ai_thread = AIMoveThread(self.ai, self.board, time_limit=self.time_limit)
        self.ai_thread.move_found.connect(self.on_ai_move_found)
        self.ai_thread.start()

    def on_ai_move_found(self, move):
        # AI搜索完成后应用走法并刷新界面显示。
        if move:
            self.board.make_move(*move)
            self.board_widget.board = self.board
            self.board_widget.update_board()
            winner = self.get_winner()
            if winner:
                winner_name = '红方' if winner == 'red' else '黑方'
                self.status_label.setText(f"{winner_name}获胜！")
            else:
                player_name = '红方' if self.board.current_player == 'red' else '黑方'
                self.status_label.setText(f"{player_name}回合")
        else:
            self.status_label.setText("AI 无合法走法")

    def reset_game(self):
        # 停止正在运行的AI计算并复位棋盘到初始状态。
        if self.ai_thread and self.ai_thread.isRunning():
            self.ai_thread.terminate()
            self.ai_thread.wait()
        self.board = Board()
        self.board_widget.board = self.board
        self.selected = None
        self.board_widget.update_board()
        player_name = '红方' if self.board.current_player == 'red' else '黑方'
        self.status_label.setText(f"{player_name}回合")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())