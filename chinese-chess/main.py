import sys
import os

# Ensure the repo root is importable so `import chess...` works even when the
# user runs `python main.py` from inside the `chinese-chess/` directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# 中国象棋程序入口。
# 如果传入参数 "gui"，则启动图形界面；否则启动控制台对战。
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        from gui.main_window import MainWindow
        from PyQt5.QtWidgets import QApplication

        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    else:
        from ai.game import Game

        game = Game()
        game.play()
