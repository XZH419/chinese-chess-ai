from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import traceback

from PyQt5.QtCore import QEasingCurve, QPointF, QPropertyAnimation, QRectF, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPainter, QPixmap, QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGraphicsObject,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from chinese_chess.algorithm.minimax import MinimaxAI
from chinese_chess.algorithm.mcts import MCTSAI
from chinese_chess.algorithm.random_ai import RandomAI
from chinese_chess.control.controller import GameController, MoveOutcome


Move = Tuple[int, int, int, int]
Pos = Tuple[int, int]


def _assets_dir() -> str:
    # 统一资源路径：`chinese_chess/resources/img`
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "resources", "img")
    )


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

    def __init__(
        self,
        ai,
        board_snapshot,
        ai_color: str,
        time_limit_s: int,
        run_id: int,
        game_history_hashes: Optional[list] = None,
    ):
        super().__init__()
        # 线程中绝对不持有任何 UI / controller 对象，避免隐式触碰主线程资源
        self._ai = ai
        self._board = board_snapshot
        self._ai_color = ai_color
        self._time_limit_s = time_limit_s
        self._run_id = run_id
        self._game_history_hashes = list(game_history_hashes) if game_history_hashes else []

    def run(self) -> None:
        # 仅在纯数据上计算走法：board_snapshot / ai
        # 注意：AI 计算不应该触碰任何 Qt 对象
        try:
            if hasattr(self._ai, "choose_move"):
                self._board.current_player = self._ai_color
                move = self._ai.choose_move(
                    self._board,
                    time_limit=self._time_limit_s,
                    game_history=self._game_history_hashes,
                )
            else:
                self._board.current_player = self._ai_color
                move = self._ai.get_best_move(
                    self._board,
                    time_limit=self._time_limit_s,
                    game_history=self._game_history_hashes,
                )
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
        # 以"图元中心"为锚点：后续 setPos(x, y) 直接给中心点坐标即可
        self._half_w = self._pixmap.width() / 2.0
        self._half_h = self._pixmap.height() / 2.0

    def boundingRect(self):
        # PyQt5 要求 boundingRect() **必须**返回 QRectF（不能是 QRect）
        # 同时把包围盒改成"以中心为原点"，与 setPos 的中心点语义保持一致
        return QRectF(-self._half_w, -self._half_h, self._pixmap.width(), self._pixmap.height())

    def paint(self, painter: QPainter, option, widget=None):
        # 以中心为锚点绘制
        painter.drawPixmap(int(-self._half_w), int(-self._half_h), self._pixmap)


class XiangqiBoardView(QGraphicsView):
    """对齐参考项目的"贴图棋盘 + 贴图棋子"界面。

    重要约束：
    - 本类只做渲染与鼠标交互坐标转换
    - 走子是否合法/胜负判定/AI 决策：全部交给 controller
    """

    # 这些参数用于"模型坐标(row,col) <-> 场景坐标(x,y)"换算。
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
        # 为便于调试：先做可调偏移，再做"就近吸附"。
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

        # 移动过程中保持放大（像"拿起来"悬浮移动）
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
        # 动画结束后缩回，模拟"落地"
        anim.finished.connect(lambda: moving_item.setScale(1.0))
        moving_item._anim = anim  # type: ignore[attr-defined]

        self._piece_items.pop(src, None)
        self._piece_items[dst] = moving_item


# ═══════════════════════════════════════════════════════════════
#  MainWindow：配置 → 开始 → 对局 → 结束 的完整生命周期
# ═══════════════════════════════════════════════════════════════


class MainWindow(QMainWindow):
    """中国象棋 GUI 主窗口。

    启动后处于「配置阶段」——用户可在右侧面板选择红/黑双方 AI 类型及参数，
    点击「开始对局」后进入「对局阶段」，棋盘点击和 AI 自动行棋才被激活。
    """

    _AI_TYPES = ["Human (人类)", "Random (随机)", "Minimax (极大极小)", "MCTS (蒙特卡洛)"]
    _IDX_HUMAN, _IDX_RANDOM, _IDX_MINIMAX, _IDX_MCTS = 0, 1, 2, 3

    def __init__(self, controller: Optional[GameController] = None):
        super().__init__()

        self.controller = controller or GameController(red_agent=None, black_agent=None)
        self.human_color = "red"
        self.is_game_running = False

        self._selected: Optional[Pos] = None
        self._selected_item: Optional[PixmapPieceItem] = None
        self._ai_thread: Optional[AIMoveThread] = None
        self._run_id = 0

        self._init_window()
        self._init_ui()
        self.status_label.setText("请配置红黑双方，然后点击「开始对局」")

        # 外部注入已带 agent 的 controller 时：同步 UI 并自动开局
        if controller is not None and (
            controller.red_agent is not None or controller.black_agent is not None
        ):
            self._red_combo.setCurrentIndex(self._agent_to_combo_index(controller.red_agent))
            self._black_combo.setCurrentIndex(self._agent_to_combo_index(controller.black_agent))
            self._sync_param_from_agent(
                controller.red_agent, self._red_param_label, self._red_param_spin
            )
            self._sync_param_from_agent(
                controller.black_agent, self._black_param_label, self._black_param_spin
            )
            self._on_start_stop()

    # ────────────────────── 窗口 / 布局 ──────────────────────

    def _init_window(self) -> None:
        self.setWindowTitle("中国象棋")
        icon_path = _img("icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    def _init_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout()
        root.setLayout(main_layout)

        # ── 左侧：棋盘 ──
        self.board_view = XiangqiBoardView(self.controller)
        self.board_view.square_clicked.connect(self._on_square_clicked)
        main_layout.addWidget(self.board_view, 1)

        # ── 右侧：配置 + 状态 + 日志 ──
        self.info_panel = QWidget()
        self.info_panel.setMinimumWidth(270)
        right = QVBoxLayout()
        self.info_panel.setLayout(right)
        main_layout.addWidget(self.info_panel, 0)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-weight: bold;")
        right.addWidget(self.status_label)

        # 红方配置 GroupBox
        (
            self._red_group,
            self._red_combo,
            self._red_param_label,
            self._red_param_spin,
        ) = self._build_side_group("红方设置")
        self._red_combo.setCurrentIndex(self._IDX_HUMAN)
        self._red_combo.currentIndexChanged.connect(
            lambda idx: self._on_type_combo_changed(
                idx, self._red_param_label, self._red_param_spin
            )
        )
        self._on_type_combo_changed(
            self._red_combo.currentIndex(), self._red_param_label, self._red_param_spin
        )
        right.addWidget(self._red_group)

        # 黑方配置 GroupBox
        (
            self._black_group,
            self._black_combo,
            self._black_param_label,
            self._black_param_spin,
        ) = self._build_side_group("黑方设置")
        self._black_combo.setCurrentIndex(self._IDX_MINIMAX)
        self._black_combo.currentIndexChanged.connect(
            lambda idx: self._on_type_combo_changed(
                idx, self._black_param_label, self._black_param_spin
            )
        )
        self._on_type_combo_changed(
            self._black_combo.currentIndex(), self._black_param_label, self._black_param_spin
        )
        right.addWidget(self._black_group)

        # 开始 / 结束 按钮
        self.start_btn = QPushButton("开始对局")
        self.start_btn.setStyleSheet(
            "QPushButton { font-size: 14px; padding: 6px; font-weight: bold; }"
        )
        self.start_btn.clicked.connect(self._on_start_stop)
        right.addWidget(self.start_btn)

        # 日志控制台
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.log_console.setStyleSheet(
            "QPlainTextEdit { background: #1e1e1e; color: #d4d4d4; border: 1px solid #444; }"
        )
        self.log_console.setFont(QFont("Consolas", 10))
        right.addWidget(self.log_console, 1)

    # ────────────────────── UI 构建辅助 ──────────────────────

    def _build_side_group(self, title: str):
        """构建单侧配置 GroupBox，返回 (group, combo, param_label, param_spin)。"""
        group = QGroupBox(title)
        layout = QVBoxLayout()
        group.setLayout(layout)

        combo = QComboBox()
        combo.addItems(self._AI_TYPES)
        layout.addWidget(combo)

        param_label = QLabel("")
        layout.addWidget(param_label)

        param_spin = QSpinBox()
        param_spin.setMinimum(1)
        param_spin.setMaximum(100000)
        layout.addWidget(param_spin)

        return group, combo, param_label, param_spin

    def _on_type_combo_changed(self, index: int, label: QLabel, spin: QSpinBox) -> None:
        """AI 类型切换时动态调整参数控件的可见性和范围。"""
        if index == self._IDX_MINIMAX:
            label.setText("搜索深度：")
            label.show()
            spin.setRange(1, 8)
            spin.setValue(3)
            spin.setSingleStep(1)
            spin.show()
        elif index == self._IDX_MCTS:
            label.setText("模拟次数：")
            label.show()
            spin.setRange(100, 100000)
            spin.setValue(5000)
            spin.setSingleStep(500)
            spin.show()
        else:
            label.hide()
            spin.hide()

    @staticmethod
    def _sync_param_from_agent(agent, label: QLabel, spin: QSpinBox) -> None:
        """根据已有 agent 实例回写参数控件的值（用于外部注入 controller 时）。"""
        if agent is None:
            label.hide()
            spin.hide()
            return
        cls = type(agent).__name__
        if cls == "MinimaxAI":
            d = getattr(agent, "depth", 3)
            label.setText("搜索深度：")
            label.show()
            spin.setRange(1, 8)
            spin.setValue(int(d))
            spin.show()
        elif cls == "MCTSAI":
            s = getattr(agent, "max_simulations", 5000)
            label.setText("模拟次数：")
            label.show()
            spin.setRange(100, 100000)
            spin.setValue(int(s))
            spin.show()
        else:
            label.hide()
            spin.hide()

    # ────────────────────── 开始 / 结束 对局 ──────────────────────

    def _build_agent_from_ui(self, combo: QComboBox, spin: QSpinBox):
        """根据当前下拉框和 SpinBox 实例化 agent（Human 返回 None）。"""
        idx = combo.currentIndex()
        if idx == self._IDX_HUMAN:
            return None
        if idx == self._IDX_RANDOM:
            return RandomAI()
        if idx == self._IDX_MINIMAX:
            return MinimaxAI(depth=spin.value())
        if idx == self._IDX_MCTS:
            return MCTSAI(time_limit=5.0, max_simulations=spin.value())
        return None

    def _on_start_stop(self) -> None:
        """「开始对局」/「结束/重置对局」切换按钮的统一入口。"""
        if not self.is_game_running:
            self._start_game()
        else:
            self._stop_game()

    def _start_game(self) -> None:
        # 1) 从 UI 读取并实例化 agents
        self.controller.red_agent = self._build_agent_from_ui(
            self._red_combo, self._red_param_spin
        )
        self.controller.black_agent = self._build_agent_from_ui(
            self._black_combo, self._black_param_spin
        )
        self._sync_human_color_from_controller()

        # 2) 重置棋盘
        self.controller.reset_game()
        self.board_view._controller = self.controller
        self.board_view.update_all()
        self._selected = None
        self._selected_item = None

        # 3) 切换为"对局中"状态
        self.is_game_running = True
        self._set_config_enabled(False)
        self.start_btn.setText("结束 / 重置对局")

        # 4) UI 反馈
        matchup = self.controller.matchup_line()
        self.setWindowTitle(f"中国象棋 — {matchup}")
        self.log_console.clear()
        self.append_log(f"[对局] {matchup}")
        self._refresh_status()
        self.check_and_run_ai()

    def _stop_game(self) -> None:
        # 1) 中断 AI 线程
        self._run_id += 1
        if self._ai_thread and self._ai_thread.isRunning():
            self._ai_thread.requestInterruption()
            try:
                self._ai_thread.move_ready.disconnect(self._on_ai_move_ready)
            except Exception:
                pass
        self._ai_thread = None

        # 2) 切换为"配置中"状态
        self.is_game_running = False
        self._set_config_enabled(True)
        self.start_btn.setText("开始对局")

        # 3) 清除棋盘交互状态
        if self._selected_item is not None:
            self._selected_item.setScale(1.0)
        self._selected = None
        self._selected_item = None

        self.status_label.setText("对局已结束，请重新配置后点击「开始对局」")
        self.append_log("[UI] 对局已结束")

    def _set_config_enabled(self, enabled: bool) -> None:
        """启用 / 禁用所有配置面板控件。"""
        for w in (
            self._red_combo,
            self._red_param_spin,
            self._black_combo,
            self._black_param_spin,
        ):
            w.setEnabled(enabled)

    # ────────────────────── agent 描述 / 映射 ──────────────────────

    def _sync_human_color_from_controller(self) -> None:
        r, b = self.controller.red_agent, self.controller.black_agent
        if r is None and b is not None:
            self.human_color = "red"
        elif b is None and r is not None:
            self.human_color = "black"
        else:
            self.human_color = "red"

    @staticmethod
    def _agent_label(agent) -> str:
        """生成用于状态栏的简短 AI 描述。"""
        if agent is None:
            return "Human"
        cls = type(agent).__name__
        if cls == "MinimaxAI":
            d = getattr(agent, "depth", None)
            return f"Minimax 深度={d}" if isinstance(d, int) else "Minimax"
        if cls == "MCTSAI":
            s = getattr(agent, "max_simulations", None)
            w = getattr(agent, "workers", 1)
            return f"MCTS {s}sims/{w}w" if s else "MCTS"
        if cls == "RandomAI":
            return "Random"
        return cls

    @staticmethod
    def _agent_to_combo_index(agent) -> int:
        if agent is None:
            return 0
        cls = type(agent).__name__
        if cls == "RandomAI":
            return 1
        if cls == "MinimaxAI":
            return 2
        if cls == "MCTSAI":
            return 3
        return 0

    def _side_name(self, color: str) -> str:
        return "红方" if color == "red" else "黑方"

    # ────────────────────── 游戏内逻辑 ──────────────────────

    def _finalize_after_legal_move(self, outcome: MoveOutcome) -> None:
        if not outcome.ok:
            return
        self._refresh_status()
        if outcome.game_over:
            self._run_id += 1
            if outcome.winner == "red":
                msg = "游戏结束：红方获胜！"
            elif outcome.winner == "black":
                msg = "游戏结束：黑方获胜！"
            else:
                msg = "游戏结束：和棋（死局或三次重复）！"
            self.append_log(msg)
            self.append_log("==========================")
            QMessageBox.information(self, "对局结束", msg)
            self._game_over_ui()
            return
        self.check_and_run_ai()

    def _game_over_ui(self) -> None:
        """对局自然结束后的 UI 收尾：启用配置面板、按钮复位。"""
        self.is_game_running = False
        self._set_config_enabled(True)
        self.start_btn.setText("开始对局")

    def append_log(self, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_console.appendPlainText(f"[{ts}] {text}")
        cursor = self.log_console.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_console.setTextCursor(cursor)
        sb = self.log_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _refresh_status(self) -> None:
        result = self.controller.current_result()
        if result["game_over"]:
            winner = result["winner"]
            if winner == "red":
                self.status_label.setText("红方获胜！")
            elif winner == "black":
                self.status_label.setText("黑方获胜！")
            else:
                self.status_label.setText("游戏结束：和棋！")
            return

        player = result["current_player"]
        agent = self.controller.agent_for(player)
        if agent is None:
            self.status_label.setText(f"[{self._side_name(player)}] 请走棋")
        else:
            self.status_label.setText(
                f"[{self._side_name(player)}] AI 正在思考 ({self._agent_label(agent)})..."
            )

    # ────────────────────── 棋盘点击 ──────────────────────

    def _on_square_clicked(self, row: int, col: int) -> None:
        if not self.is_game_running:
            return
        if self.controller.is_game_over():
            self._refresh_status()
            return
        if self.controller.agent_for(self.controller.board.current_player) is not None:
            return
        if self.controller.board.current_player != self.human_color:
            return

        piece = self.controller.board.get_piece(row, col)
        clicked_item = self.board_view.piece_item_at(row, col)

        if self._selected is None:
            if piece and piece.color == self.human_color:
                self._selected = (row, col)
                self._selected_item = clicked_item
                if self._selected_item is not None:
                    self._selected_item.setScale(1.2)
            return

        if self._selected == (row, col):
            if self._selected_item is not None:
                self._selected_item.setScale(1.0)
            self._selected = None
            self._selected_item = None
            return

        sr, sc = self._selected
        move: Move = (sr, sc, row, col)
        if self._selected_item is not None:
            self._selected_item.setScale(1.0)
        self._selected = None
        self._selected_item = None

        outcome = self.controller.try_apply_player_move(move, player=self.human_color)
        if not outcome.ok:
            self.status_label.setText(outcome.message or "无效走子")
            if outcome.message:
                self.append_log(f"[UI] {outcome.message}")
            return

        print(f"[UI] player move applied: {move}")
        self.append_log(f"[UI] 玩家落子: {move}")
        self.append_log("--------------------------")
        self.board_view.animate_move(move)
        self._finalize_after_legal_move(outcome)

    # ────────────────────── AI 后台线程 ──────────────────────

    def check_and_run_ai(self) -> None:
        """仅当 is_game_running 且轮到 AI 时，才启动后台计算线程。"""
        if not self.is_game_running:
            return
        if self.controller.is_game_over():
            self._refresh_status()
            return

        cp = self.controller.board.current_player
        current_agent = (
            self.controller.red_agent if cp == "red" else self.controller.black_agent
        )
        if current_agent is None:
            self._refresh_status()
            return
        if self._ai_thread and self._ai_thread.isRunning():
            return

        side = self._side_name(cp)
        label = self._agent_label(current_agent)
        print(f"[UI] 检测到 AI 回合（{side}，{label}），启动计算...")
        self.status_label.setText(f"[{side}] AI 正在思考 ({label})...")
        self.append_log(f"[UI] 检测到 AI 回合 ({side}, {label})，开始计算...")

        board_snapshot = self.controller.board.copy()
        run_id = self._run_id
        game_hist = list(self.controller.game_history_hashes)
        self._ai_thread = AIMoveThread(
            ai=current_agent,
            board_snapshot=board_snapshot,
            ai_color=cp,
            time_limit_s=10,
            run_id=run_id,
            game_history_hashes=game_hist,
        )
        self._ai_thread.move_ready.connect(self._on_ai_move_ready)
        self._ai_thread.start()

    def _on_ai_move_ready(
        self, move: Optional[Move], stats: Optional[dict], run_id: int
    ) -> None:
        if run_id != self._run_id:
            return
        print(f"[UI] AI move signal received: {move}")
        if stats:
            if stats.get("opening_book"):
                self.append_log("命中开局库 | 耗时: 0.0s")
            elif stats.get("random"):
                self._log_random_stats(stats)
            elif stats.get("simulations") is not None:
                self._log_mcts_stats(stats)
            else:
                self._log_minimax_stats(stats)
        self.append_log("[UI] AI 信号接收完成，执行落子。")
        self.append_log("--------------------------")
        if move:

            outcome = self.controller.apply_move(
                move, player=self.controller.board.current_player
            )
            self.board_view.animate_move(move)
            self._finalize_after_legal_move(outcome)
        else:
            self._refresh_status()
            self.check_and_run_ai()

    def _log_random_stats(self, stats: dict) -> None:
        """Random AI 极简日志。"""
        time_taken = stats.get("time_taken", 0)
        time_str = (
            f"{time_taken:.3f}" if isinstance(time_taken, (int, float)) else str(time_taken)
        )
        self.append_log(f"Random AI 随机落子")
        self.append_log(f"耗时 (秒): {time_str}s")
    def _log_mcts_stats(self, stats: dict) -> None:
        """格式化 MCTS 搜索统计并写入日志（多行，与 Minimax 风格对齐）。"""
        time_taken = stats.get("time_taken", 0)
        time_str = (
            f"{time_taken:.3f}" if isinstance(time_taken, (int, float)) else str(time_taken)
        )
        self.append_log(f"搜索耗时 (秒): {time_str}")
        self.append_log(f"MCTS 模拟次数: {stats.get('simulations', 0)}")
        self.append_log(f"并行 Workers: {stats.get('workers', 1)}")
        win_rate = stats.get("win_rate", "")
        if win_rate:
            self.append_log(f"当前胜率: {win_rate}")

    def _log_minimax_stats(self, stats: dict) -> None:
        """格式化 Minimax 搜索统计并写入日志。"""
        time_taken = stats.get("time_taken", "?")
        time_str = (
            f"{time_taken:.3f}" if isinstance(time_taken, (int, float)) else str(time_taken)
        )
        self.append_log(f"搜索耗时 (秒): {time_str}")
        self.append_log(f"本次搜索深度: {stats.get('depth', '?')}")
        self.append_log(f"评估的节点总数: {stats.get('nodes_evaluated', '?')}")
        tt_hits = stats.get("tt_hits")
        if tt_hits is not None:
            self.append_log(f"置换表命中次数: {tt_hits}")


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
