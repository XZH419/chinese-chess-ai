from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import traceback

from PyQt5.QtCore import QEasingCurve, QPointF, QPropertyAnimation, QRectF, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsObject,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from chess.control.controller import GameController


Move = Tuple[int, int, int, int]
Pos = Tuple[int, int]


def _assets_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "assets", "img")


def _img(name: str) -> str:
    return os.path.join(_assets_dir(), name)


def _piece_code(color: str, piece_type: str) -> str:
    """Map our model Piece to the reference image naming scheme.

    Reference images (from IntelligentChineseChessSystem/res/img):
    - Red:  rb rj rm rp rs rx rz
    - Black: bb bj bm bp bs bx bz

    The second letter matches the reference project's internal piece char:
    - b: boss(king), s: advisor, x: elephant, m: horse, j: rook, p: cannon, z: pawn
    """

    type_to_char = {
        "jiang": "b",
        "shi": "s",
        "xiang": "x",
        "ma": "m",
        "che": "j",
        "pao": "p",
        "bing": "z",
    }
    prefix = "r" if color == "red" else "b"
    return prefix + type_to_char[piece_type]


class AIMoveThread(QThread):
    """后台线程：计算 AI 的下一步走法（不阻塞 UI）。"""

    move_ready = pyqtSignal(object, object, int)  # (Move|None, stats:dict|None, run_id)

    def __init__(self, ai, board_snapshot, ai_color: str, time_limit_s: int, run_id: int):
        super().__init__()
        # 线程中绝对不持有任何 UI / controller 对象，避免隐式触碰主线程资源
        self._ai = ai
        self._board = board_snapshot
        self._ai_color = ai_color
        self._time_limit_s = time_limit_s
        self._run_id = run_id

    def run(self) -> None:
        # 仅在纯数据上计算走法：board_snapshot / ai
        # 注意：AI 计算不应该触碰任何 Qt 对象
        try:
            if hasattr(self._ai, "choose_move"):
                self._board.current_player = self._ai_color
                move = self._ai.choose_move(self._board, time_limit=self._time_limit_s)
            else:
                self._board.current_player = self._ai_color
                move = self._ai.get_best_move(self._board, time_limit=self._time_limit_s)
            print("[AI Thread] finished, emitting signal")
            stats = getattr(self._ai, "last_stats", None)
            self.move_ready.emit(move, stats, self._run_id)
        except Exception as e:
            print("[AI Thread] exception:", e)
            traceback.print_exc()
            self.move_ready.emit(None, {"error": str(e)}, self._run_id)


class PixmapPieceItem(QGraphicsObject):
    """可动画的棋子图元。

    关键：QPropertyAnimation 需要 QObject，而 QGraphicsPixmapItem 不是 QObject，
    所以必须用 QGraphicsObject 来承载 pixmap 并实现 paint/boundingRect。
    """

    def __init__(self, pixmap: QPixmap):
        super().__init__()
        self._pixmap = pixmap
        # 以“图元中心”为锚点：后续 setPos(x, y) 直接给中心点坐标即可
        self._half_w = self._pixmap.width() / 2.0
        self._half_h = self._pixmap.height() / 2.0

    def boundingRect(self):
        # PyQt5 要求 boundingRect() **必须**返回 QRectF（不能是 QRect）
        # 同时把包围盒改成“以中心为原点”，与 setPos 的中心点语义保持一致
        return QRectF(-self._half_w, -self._half_h, self._pixmap.width(), self._pixmap.height())

    def paint(self, painter: QPainter, option, widget=None):
        # 以中心为锚点绘制
        painter.drawPixmap(int(-self._half_w), int(-self._half_h), self._pixmap)


class XiangqiBoardView(QGraphicsView):
    """对齐参考项目的“贴图棋盘 + 贴图棋子”界面。

    重要约束：
    - 本类只做渲染与鼠标交互坐标转换
    - 走子是否合法/胜负判定/AI 决策：全部交给 controller
    """

    # 这些参数用于“模型坐标(row,col) <-> 场景坐标(x,y)”换算。
    # 由于 Swing(JFrame) 与 Qt(QGraphicsView) 的坐标/边距行为不同，必须可调。
    VIEW_WIDTH = 700
    VIEW_HEIGHT = 712
    PIECE_W = 67
    PIECE_H = 67

    # 强制应用你实测的精准映射数据（渲染/点击必须统一使用）
    SX_OFFSET = 85
    SX_COE = 67.1
    SY_OFFSET = 42.1
    SY_COE = 68.9

    # 保留可选点击微调（默认 0），你后续若还要细调可以用
    CLICK_X_OFFSET = 0.0
    CLICK_Y_OFFSET = 0.0

    square_clicked = pyqtSignal(int, int)  # row, col

    def __init__(self, controller: GameController):
        super().__init__()
        self._controller = controller

        self.setFixedSize(self.VIEW_WIDTH, self.VIEW_HEIGHT)
        self.setHorizontalScrollBarPolicy(1)  # ScrollBarAlwaysOff
        self.setVerticalScrollBarPolicy(1)

        self._scene = QGraphicsScene(0, 0, self.VIEW_WIDTH, self.VIEW_HEIGHT)
        self.setScene(self._scene)

        self._piece_items: Dict[Pos, PixmapPieceItem] = {}

        self._load_background()
        self.rebuild_from_model()

    def _load_background(self) -> None:
        board_pix = QPixmap(_img("board.png"))
        bg = QGraphicsPixmapItem(board_pix)
        bg.setZValue(-10)
        self._scene.addItem(bg)

    def model_to_view(self, row: int, col: int) -> QPointF:
        x = col * float(self.SX_COE) + float(self.SX_OFFSET)
        y = row * float(self.SY_COE) + float(self.SY_OFFSET)
        return QPointF(x, y)

    def view_to_model(self, x: float, y: float) -> Optional[Pos]:
        # 点击坐标 -> 格子坐标（row,col）
        # 为便于调试：先做可调偏移，再做“就近吸附”。
        x = x + self.CLICK_X_OFFSET
        y = y + self.CLICK_Y_OFFSET
        col = int((x - float(self.SX_OFFSET) + float(self.SX_COE) / 2) / float(self.SX_COE))
        row = int((y - float(self.SY_OFFSET) + float(self.SY_COE) / 2) / float(self.SY_COE))
        if 0 <= row < self._controller.board.rows and 0 <= col < self._controller.board.cols:
            return (row, col)
        return None

    def mousePressEvent(self, event) -> None:
        pos = self.mapToScene(event.pos())
        x = pos.x()
        y = pos.y()
        rc = self.view_to_model(x, y)
        if rc is not None:
            row, col = rc
            # 便于你调试点击错位：打印原始坐标与换算后的格子
            print(f"Mouse Click: x={x}, y={y} -> Board: row={row}, col={col}")
            self.square_clicked.emit(row, col)
        super().mousePressEvent(event)

    def piece_item_at(self, row: int, col: int) -> Optional[PixmapPieceItem]:
        """返回该格子的棋子图元（为空格则返回 None）。"""
        return self._piece_items.get((row, col))

    def rebuild_from_model(self) -> None:
        # Remove existing pieces
        for item in list(self._piece_items.values()):
            self._scene.removeItem(item)
        self._piece_items.clear()

        board = self._controller.board
        for r in range(board.rows):
            for c in range(board.cols):
                piece = board.board[r][c]
                if not piece:
                    continue
                code = _piece_code(piece.color, piece.piece_type)
                pix = QPixmap(_img(f"{code}.png"))
                it = PixmapPieceItem(pix)
                it.setZValue(1)
                p = self.model_to_view(r, c)
                it.setPos(p.x(), p.y())
                self._scene.addItem(it)
                self._piece_items[(r, c)] = it

    # 兼容你希望的命名（语义更清晰）
    def update_all(self) -> None:
        self.rebuild_from_model()

    def animate_move(self, move: Move) -> None:
        sr, sc, er, ec = move
        src = (sr, sc)
        dst = (er, ec)

        moving_item = self._piece_items.get(src)
        # If we couldn't resolve the item (shouldn't happen), rebuild.
        if moving_item is None:
            self.rebuild_from_model()
            return

        # Capture: remove any item currently at destination before moving.
        captured_item = self._piece_items.get(dst)
        if captured_item is not None and captured_item is not moving_item:
            self._scene.removeItem(captured_item)
            self._piece_items.pop(dst, None)

        # 移动过程中保持放大（像“拿起来”悬浮移动）
        moving_item.setScale(1.2)

        anim = QPropertyAnimation(moving_item, b"pos")
        anim.setDuration(160)
        # 物理缓动：加速启动 -> 减速轻放
        anim.setEasingCurve(QEasingCurve.InOutCubic)
        anim.setStartValue(moving_item.pos())
        p = self.model_to_view(er, ec)
        anim.setEndValue(QPointF(p.x(), p.y()))
        # 非阻塞：start() 只会把动画注册进事件循环，不会卡住主线程
        anim.start()
        # 动画结束后缩回，模拟“落地”
        anim.finished.connect(lambda: moving_item.setScale(1.0))
        moving_item._anim = anim  # type: ignore[attr-defined]

        self._piece_items.pop(src, None)
        self._piece_items[dst] = moving_item


class MainWindow(QMainWindow):
    """新 GUI：风格对齐参考 GameView（贴图棋盘/棋子 + 侧边信息区）。"""

    def __init__(self, controller: Optional[GameController] = None):
        super().__init__()

        self.controller = controller or GameController()
        self.human_color = "red"

        self._selected: Optional[Pos] = None
        self._selected_item: Optional[PixmapPieceItem] = None
        self._ai_thread: Optional[AIMoveThread] = None
        # 用于防止“幽灵走子”：每次 reset 或启动新线程都会递增
        self._run_id = 0

        self._init_window()
        self._init_ui()
        self._refresh_status()
        # 启动时立即检查：如果红方是 AI，则自动走子
        self.check_and_run_ai()

    def _init_window(self) -> None:
        self.setWindowTitle("中国象棋 AI (Controller 驱动)")
        icon_path = _img("icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    def _init_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        main = QHBoxLayout()
        root.setLayout(main)

        # Left: board graphics
        self.board_view = XiangqiBoardView(self.controller)
        self.board_view.square_clicked.connect(self._on_square_clicked)
        main.addWidget(self.board_view, 1)

        # Right: info + buttons (keep some valuable existing controls)
        self.info_panel = QWidget()
        self.info_panel.setMinimumWidth(250)
        right = QVBoxLayout()
        self.info_panel.setLayout(right)
        main.addWidget(self.info_panel, 0)

        self.status_label = QLabel("")
        # 长状态文本自动换行，避免被截断/折叠
        self.status_label.setWordWrap(True)
        right.addWidget(self.status_label)

        # GUI 实时控制台（Dashboard）
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.log_console.setStyleSheet(
            "QPlainTextEdit { background: #1e1e1e; color: #d4d4d4; border: 1px solid #444; }"
        )
        self.log_console.setFont(QFont("Consolas", 10))
        right.addWidget(self.log_console, 1)

        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self._reset_game)
        right.addWidget(self.reset_btn)

        right.addStretch(1)

    def append_log(self, text: str) -> None:
        """向右侧 log_console 追加一条带时间戳的日志，并自动滚动到底部。"""
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_console.appendPlainText(f"[{ts}] {text}")
        self.log_console.ensureCursorVisible()

    def _side_name(self, color: str) -> str:
        return "红方" if color == "red" else "黑方"

    def _agent_depth_hint(self, agent) -> Optional[int]:
        # 仅用于 UI 展示；不影响底层逻辑
        d = getattr(agent, "depth", None)
        return int(d) if isinstance(d, int) else None

    def _refresh_status(self) -> None:
        result = self.controller.current_result()
        if result["game_over"]:
            winner = result["winner"]
            if winner == "red":
                self.status_label.setText("红方获胜！")
            elif winner == "black":
                self.status_label.setText("黑方获胜！")
            else:
                self.status_label.setText("游戏结束")
            return

        player = result["current_player"]
        agent = self.controller.agent_for(player)
        if agent is None:
            self.status_label.setText(f"[{self._side_name(player)}] 请走棋")
        else:
            depth = self._agent_depth_hint(agent)
            if depth is not None:
                self.status_label.setText(f"[{self._side_name(player)}] AI 正在思考 (深度: {depth})...")
            else:
                self.status_label.setText(f"[{self._side_name(player)}] AI 正在思考...")

    def _reset_game(self) -> None:
        # 1) 安全中断当前计算：防止旧线程回调导致“幽灵走子”
        self._run_id += 1
        if self._ai_thread and self._ai_thread.isRunning():
            self._ai_thread.requestInterruption()
            try:
                self._ai_thread.move_ready.disconnect(self._on_ai_move_ready)
            except Exception:
                pass
        self._ai_thread = None

        # 2) 不重建 Controller：只重置棋盘（保持 red/black agent 配置）
        self.controller.reset_game()
        self.board_view._controller = self.controller
        self.board_view.update_all()
        if self._selected_item is not None:
            self._selected_item.setScale(1.0)
        self._selected = None
        self._selected_item = None
        self._refresh_status()
        self.append_log("[UI] 重置对局（保持当前 AI 配置）")
        # 3) 重置后立刻接力：AI vs AI 会自动开新局第一步
        self.check_and_run_ai()

    def _on_square_clicked(self, row: int, col: int) -> None:
        if self.controller.is_game_over():
            self._refresh_status()
            return
        # 权限控制：如果当前回合是 AI，则忽略一切鼠标点击
        if self.controller.agent_for(self.controller.board.current_player) is not None:
            return
        if self.controller.board.current_player != self.human_color:
            return

        piece = self.controller.board.get_piece(row, col)
        clicked_item = self.board_view.piece_item_at(row, col)

        # 1) selecting
        if self._selected is None:
            if piece and piece.color == self.human_color:
                self._selected = (row, col)
                self._selected_item = clicked_item
                if self._selected_item is not None:
                    # 点击放大（被“拿起来”）
                    self._selected_item.setScale(1.2)
            return

        # 再次点击同一枚棋子：取消选中并缩小
        if self._selected == (row, col):
            if self._selected_item is not None:
                self._selected_item.setScale(1.0)
            self._selected = None
            self._selected_item = None
            return

        # 2) moving
        sr, sc = self._selected
        move: Move = (sr, sc, row, col)
        # 取消“选中态”（移动动画内部会再放大并在结束时缩回）
        if self._selected_item is not None:
            self._selected_item.setScale(1.0)
        self._selected = None
        self._selected_item = None

        outcome = self.controller.try_apply_player_move(move, player=self.human_color)
        if not outcome.ok:
            self.status_label.setText("无效走子")
            return

        print(f"[UI] player move applied: {move}")
        self.append_log(f"[UI] 玩家落子: {move}")
        # Animate based on model coordinates; do NOT compute rules here.
        self.board_view.animate_move(move)
        self._refresh_status()
        # 玩家走完后，触发下一手（可能是黑方 AI，也可能是红方 AI vs AI 场景）
        self.check_and_run_ai()

    def check_and_run_ai(self) -> None:
        """接力棒机制：如果当前回合是 AI，则自动启动计算并落子。"""
        if self.controller.is_game_over():
            self._refresh_status()
            return

        current_agent = self.controller.agent_for(self.controller.board.current_player)
        if current_agent is None:
            # 人类回合：等待点击
            self._refresh_status()
            return

        # 避免重复启动线程
        if self._ai_thread and self._ai_thread.isRunning():
            return

        side = self._side_name(self.controller.board.current_player)
        depth = self._agent_depth_hint(current_agent)
        if depth is not None:
            print(f"[UI] 检测到 AI 回合（{side}，深度={depth}），启动计算...")
            self.status_label.setText(f"[{side}] AI 正在思考 (深度: {depth})...")
            self.append_log(f"[UI] 检测到 AI 回合 ({side})，开始计算...")
        else:
            print(f"[UI] 检测到 AI 回合（{side}），启动计算...")
            self.status_label.setText(f"[{side}] AI 正在思考...")
            self.append_log(f"[UI] 检测到 AI 回合 ({side})，开始计算...")

        board_snapshot = self.controller.board.copy()
        ai_color = self.controller.board.current_player
        run_id = self._run_id
        self._ai_thread = AIMoveThread(
            ai=current_agent,
            board_snapshot=board_snapshot,
            ai_color=ai_color,
            # 不再暴露 UI 控件；这里保留一个温和的固定上限，避免极端卡死
            time_limit_s=10,
            run_id=run_id,
        )
        self._ai_thread.move_ready.connect(self._on_ai_move_ready)
        self._ai_thread.start()

    def _on_ai_move_ready(self, move: Optional[Move], stats: Optional[dict], run_id: int) -> None:
        # 如果这是旧局/旧线程的回调，直接忽略
        if run_id != self._run_id:
            return
        print(f"[UI] AI move signal received: {move}")
        if stats:
            depth = stats.get("depth", "?")
            time_taken = stats.get("time_taken", "?")
            nodes = stats.get("nodes_evaluated", "?")
            self.append_log("本次搜索深度: " + str(depth))
            self.append_log("搜索耗时 (秒): " + (f"{time_taken:.3f}" if isinstance(time_taken, (int, float)) else str(time_taken)))
            self.append_log("评估的节点总数: " + str(nodes))
        self.append_log("[UI] AI 信号接收完成，执行落子。")
        self.append_log("--------------------------")
        if move:
            self.controller.apply_move(move, player=self.controller.board.current_player)
            self.board_view.animate_move(move)
        else:
            # AI 无合法走法（困毙/将死由 Rules.winner 判定）
            pass
        self._refresh_status()
        # 极速接力：本方落子后立刻轮到下一方计算
        self.check_and_run_ai()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

