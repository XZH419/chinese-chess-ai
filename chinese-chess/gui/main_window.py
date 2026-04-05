import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt
from ai.board import Board
from ai.ai_minimax import MinimaxAI

class ChessBoardWidget(QWidget):
    def __init__(self, board, click_handler):
        super().__init__()
        self.board = board
        self.click_handler = click_handler
        self.init_ui()

    def init_ui(self):
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
        for r in range(self.board.rows):
            for c in range(self.board.cols):
                piece = self.board.board[r][c]
                text = str(piece) if piece else ''
                self.buttons[r][c].setText(text)
                self.buttons[r][c].setStyleSheet("font-size: 18px; border: 1px solid black;")

    def highlight_square(self, row, col):
        if 0 <= row < self.board.rows and 0 <= col < self.board.cols:
            self.buttons[row][col].setStyleSheet("font-size: 18px; border: 2px solid red; background: #ffe6e6;")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.board = Board()
        self.ai = MinimaxAI(depth=3)
        self.human_color = 'red'
        self.selected = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Chinese Chess AI")
        self.setGeometry(100, 100, 600, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        self.board_widget = ChessBoardWidget(self.board, self.on_square_clicked)
        layout.addWidget(self.board_widget)

        self.status_label = QLabel(f"{self.board.current_player}'s turn")
        layout.addWidget(self.status_label)

        button_layout = QVBoxLayout()
        self.ai_button = QPushButton("AI Move")
        self.ai_button.clicked.connect(self.ai_move)
        button_layout.addWidget(self.ai_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_game)
        button_layout.addWidget(self.reset_button)

        layout.addLayout(button_layout)
        central_widget.setLayout(layout)

    def on_square_clicked(self, row, col):
        if self.board.is_game_over():
            return
        if self.board.current_player != self.human_color:
            return
        piece = self.board.get_piece(row, col)
        if self.selected is None:
            if piece and piece.color == self.human_color:
                self.selected = (row, col)
                self.board_widget.highlight_square(row, col)
                self.status_label.setText(f"Selected {row},{col}")
        else:
            sr, sc = self.selected
            er, ec = row, col
            if self.board.is_valid_move(sr, sc, er, ec, player=self.human_color):
                captured = self.board.make_move(sr, sc, er, ec)
                if self.board.is_check(self.human_color):
                    self.board.undo_move(sr, sc, er, ec, captured)
                    self.status_label.setText("Illegal move: king in check")
                else:
                    self.board_widget.update_board()
                    self.selected = None
                    self.status_label.setText(f"{self.board.current_player}'s turn")
                    if not self.board.is_game_over():
                        self.ai_move()
            else:
                self.status_label.setText("Invalid move")
                self.selected = None
            self.board_widget.update_board()

    def ai_move(self):
        if self.board.is_game_over():
            self.status_label.setText("Game Over")
            return
        move = self.ai.get_best_move(self.board, time_limit=5)
        if move:
            self.board.make_move(*move)
            self.board_widget.update_board()
            if self.board.is_game_over():
                self.status_label.setText("Game Over")
            else:
                self.status_label.setText(f"{self.board.current_player}'s turn")
        else:
            self.status_label.setText("AI has no legal move")

    def reset_game(self):
        self.board = Board()
        self.board_widget.board = self.board
        self.selected = None
        self.board_widget.update_board()
        self.status_label.setText(f"{self.board.current_player}'s turn")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())