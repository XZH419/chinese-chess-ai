import sys

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
