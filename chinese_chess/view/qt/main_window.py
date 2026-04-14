"""中国象棋 PyQt5 图形界面主窗口模块。

本模块实现基于 PyQt5 的完整中国象棋 GUI，包含棋盘渲染、棋子动画、
玩家交互、AI 后台计算以及对局配置面板等功能。

核心组件：
    - ``AIMoveThread``: AI 走法计算的后台线程，避免阻塞 UI。
    - ``PixmapPieceItem``: 可动画的棋子图元，支持平滑移动效果。
    - ``XiangqiBoardView``: 棋盘视图，负责渲染和鼠标交互坐标转换。
    - ``MainWindow``: 主窗口，管理"配置 → 开始 → 对局 → 结束"的完整生命周期。
"""

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
from chinese_chess.algorithm.mcts_minimax import MCTSMinimaxAI
from chinese_chess.algorithm.random_ai import RandomAI
from chinese_chess.control.controller import GameController, MoveOutcome
from chinese_chess.model.rules import Rules


Move = Tuple[int, int, int, int]
Pos = Tuple[int, int]

# ── GUI 展示用引擎名 ↔ 内部 key（勿用于非界面逻辑）──
ENGINE_TO_DISPLAY_NAME: Dict[str, str] = {
    "human": "玩家",
    "mcts": "MCTS AI",
    "minimax": "Minimax AI",
    "mcts_minimax": "MCTS-Minimax AI",
    "random": "随机 AI",
}
# 下拉框等「展示名 → 内部 key」；与上表互逆，单一数据源避免不一致
DISPLAY_NAME_TO_ENGINE: Dict[str, str] = {
    v: k for k, v in ENGINE_TO_DISPLAY_NAME.items()
}


def _assets_dir() -> str:
    """获取统一资源图片目录的绝对路径。

    Returns:
        str: ``chinese_chess/resources/img`` 目录的规范化绝对路径。
    """
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "resources", "img")
    )


def _img(name: str) -> str:
    """拼接资源图片的完整路径。

    Args:
        name: 图片文件名（如 ``"board.png"``）。

    Returns:
        str: 资源图片的完整路径。
    """
    return os.path.join(_assets_dir(), name)


def _piece_code(color: str, piece_type: str) -> str:
    """将棋盘模型的棋子信息映射为参考项目的图片命名编码。

    参考项目（IntelligentChineseChessSystem/res/img）的命名规则：
        - 红方：rb rj rm rp rs rx rz
        - 黑方：bb bj bm bp bs bx bz

    第二个字母对应参考项目内部的棋子字符：
        b=帅/将, s=仕/士, x=相/象, m=马, j=车, p=炮, z=兵/卒

    Args:
        color: 棋子颜色，``'red'`` 或 ``'black'``。
        piece_type: 棋子类型标识符（如 ``'jiang'``, ``'ma'``, ``'che'`` 等）。

    Returns:
        str: 两字符的图片编码（如 ``'rb'`` 表示红方帅）。
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
    """AI 走法计算后台线程。

    在独立线程中执行 AI 搜索计算，避免阻塞 GUI 主线程。
    计算完成后通过 ``move_ready`` 信号将结果回传给主线程。

    Signals:
        move_ready: 计算完成时发射，携带参数 ``(走法|None, 统计信息字典|None, 运行ID)``。
    """

    move_ready = pyqtSignal(object, object, int)  # (Move|None, stats:dict|None, run_id)

    def __init__(
        self,
        ai,
        board_snapshot,
        ai_color: str,
        time_limit_s: int,
        run_id: int,
        game_history_hashes: Optional[list] = None,
        move_history: Optional[list] = None,
    ):
        """初始化 AI 计算线程。

        线程内部仅持有棋盘快照和 AI 实例等纯数据对象，
        绝不持有任何 UI / Controller 引用，避免跨线程资源竞争。

        Args:
            ai: AI 代理实例（需实现 ``choose_move`` 或 ``get_best_move``）。
            board_snapshot: 当前棋盘的深拷贝快照。
            ai_color: AI 所执颜色（``'red'`` 或 ``'black'``）。
            time_limit_s: 思考时间上限（秒）。
            run_id: 本次计算的唯一标识，用于主线程过滤过期信号。
            game_history_hashes: 从开局至今的 Zobrist 哈希列表（开局库等）。
            move_history: 控制器 ``MoveEntry`` 列表副本，与终局长将判负规则对齐。
        """
        super().__init__()
        self._ai = ai
        self._board = board_snapshot
        self._ai_color = ai_color
        self._time_limit_s = time_limit_s
        self._run_id = run_id
        self._game_history_hashes = list(game_history_hashes) if game_history_hashes else []
        self._move_history = list(move_history) if move_history else []

    def run(self) -> None:
        """线程执行体：在纯数据上计算走法，不触碰任何 Qt 对象。

        计算完成后通过 ``move_ready`` 信号发射结果；
        若发生异常，将错误信息封装在 stats 字典中一并发射。
        """
        try:
            if hasattr(self._ai, "choose_move"):
                self._board.current_player = self._ai_color
                move = self._ai.choose_move(
                    self._board,
                    time_limit=self._time_limit_s,
                    game_history=self._game_history_hashes,
                    move_history=self._move_history,
                )
            else:
                self._board.current_player = self._ai_color
                move = self._ai.get_best_move(
                    self._board,
                    time_limit=self._time_limit_s,
                    game_history=self._game_history_hashes,
                    move_history=self._move_history,
                )
            print("[后台计算] 已完成，正在发送结果信号")
            stats = getattr(self._ai, "last_stats", None)
            self.move_ready.emit(move, stats, self._run_id)
        except Exception as e:
            print("[后台计算] 发生异常:", e)
            traceback.print_exc()
            self.move_ready.emit(None, {"error": str(e)}, self._run_id)


class PixmapPieceItem(QGraphicsObject):
    """可动画的棋子图元。

    由于 ``QPropertyAnimation`` 需要 ``QObject`` 支撑，而 ``QGraphicsPixmapItem``
    并非 ``QObject`` 子类，因此使用 ``QGraphicsObject`` 来承载 pixmap 并实现
    ``paint`` / ``boundingRect``，从而支持平滑的位置动画。

    以图元中心为锚点——``setPos(x, y)`` 直接指定中心点坐标即可。
    """

    def __init__(self, pixmap: QPixmap):
        """初始化棋子图元。

        Args:
            pixmap: 棋子的位图资源。
        """
        super().__init__()
        self._pixmap = pixmap
        self._half_w = self._pixmap.width() / 2.0
        self._half_h = self._pixmap.height() / 2.0

    def boundingRect(self):
        """返回图元的包围矩形（以中心为原点）。

        PyQt5 要求 ``boundingRect()`` 必须返回 ``QRectF``，
        且包围盒采用"以中心为原点"的坐标系，与 ``setPos`` 的中心点语义保持一致。

        Returns:
            QRectF: 以中心为原点的包围矩形。
        """
        return QRectF(-self._half_w, -self._half_h, self._pixmap.width(), self._pixmap.height())

    def paint(self, painter: QPainter, option, widget=None):
        """以中心为锚点绘制棋子位图。

        Args:
            painter: Qt 绘图器。
            option: 样式选项（未使用）。
            widget: 目标窗口部件（未使用）。
        """
        painter.drawPixmap(int(-self._half_w), int(-self._half_h), self._pixmap)


class XiangqiBoardView(QGraphicsView):
    """中国象棋棋盘视图——贴图棋盘与贴图棋子的渲染层。

    本类严格只负责棋盘图像渲染和鼠标交互坐标转换；
    走子合法性校验、胜负判定、AI 决策等逻辑全部委托给 Controller。

    坐标映射说明：
        由于 Swing(JFrame) 与 Qt(QGraphicsView) 的坐标/边距行为不同，
        "模型坐标 (row, col) ↔ 场景坐标 (x, y)" 的换算参数必须可调。

    Signals:
        square_clicked: 用户点击棋盘格子时发射，携带 ``(行号, 列号)``。
    """

    VIEW_WIDTH = 700
    VIEW_HEIGHT = 712
    PIECE_W = 67
    PIECE_H = 67

    # 实测校准后的精准坐标映射参数（渲染与点击必须统一使用）
    SX_OFFSET = 85
    SX_COE = 67.1
    SY_OFFSET = 42.1
    SY_COE = 68.9

    # 可选的点击微调偏移（默认 0），便于后续精细调整
    CLICK_X_OFFSET = 0.0
    CLICK_Y_OFFSET = 0.0

    square_clicked = pyqtSignal(int, int)  # row, col

    def __init__(self, controller: GameController):
        """初始化棋盘视图。

        Args:
            controller: 游戏控制器实例，用于获取棋盘模型数据。
        """
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
        """加载并显示棋盘背景图片。"""
        board_pix = QPixmap(_img("board.png"))
        bg = QGraphicsPixmapItem(board_pix)
        bg.setZValue(-10)
        self._scene.addItem(bg)

    def model_to_view(self, row: int, col: int) -> QPointF:
        """将棋盘模型坐标转换为场景渲染坐标。

        Args:
            row: 棋盘行号（0~9）。
            col: 棋盘列号（0~8）。

        Returns:
            QPointF: 对应的场景坐标点。
        """
        x = col * float(self.SX_COE) + float(self.SX_OFFSET)
        y = row * float(self.SY_COE) + float(self.SY_OFFSET)
        return QPointF(x, y)

    def view_to_model(self, x: float, y: float) -> Optional[Pos]:
        """将场景坐标转换为棋盘模型坐标（就近吸附到最近格点）。

        Args:
            x: 场景 X 坐标。
            y: 场景 Y 坐标。

        Returns:
            Optional[Pos]: 棋盘格坐标 ``(行号, 列号)``，超出范围则返回 ``None``。
        """
        x = x + self.CLICK_X_OFFSET
        y = y + self.CLICK_Y_OFFSET
        col = int((x - float(self.SX_OFFSET) + float(self.SX_COE) / 2) / float(self.SX_COE))
        row = int((y - float(self.SY_OFFSET) + float(self.SY_COE) / 2) / float(self.SY_COE))
        if 0 <= row < self._controller.board.rows and 0 <= col < self._controller.board.cols:
            return (row, col)
        return None

    def mousePressEvent(self, event) -> None:
        """处理鼠标点击事件：将场景坐标转换为棋盘格并发射信号。

        Args:
            event: Qt 鼠标事件对象。
        """
        pos = self.mapToScene(event.pos())
        x = pos.x()
        y = pos.y()
        rc = self.view_to_model(x, y)
        if rc is not None:
            row, col = rc
            print(f"[视图] 鼠标点击 场景坐标=({x:.0f},{y:.0f}) → 棋盘格 row={row}, col={col}")
            self.square_clicked.emit(row, col)
        super().mousePressEvent(event)

    def piece_item_at(self, row: int, col: int) -> Optional[PixmapPieceItem]:
        """获取指定格子上的棋子图元。

        Args:
            row: 棋盘行号。
            col: 棋盘列号。

        Returns:
            Optional[PixmapPieceItem]: 该位置的棋子图元，空格则返回 ``None``。
        """
        return self._piece_items.get((row, col))

    def rebuild_from_model(self) -> None:
        """根据棋盘模型数据完全重建所有棋子图元。

        清除当前场景中的全部棋子，然后遍历棋盘模型逐一创建新图元。
        适用于开局、重置或需要完全刷新的场景。
        """
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

    def update_all(self) -> None:
        """刷新全部棋子显示（``rebuild_from_model`` 的语义别名）。"""
        self.rebuild_from_model()

    def animate_move(self, move: Move) -> None:
        """以平滑动画方式执行一步棋子移动。

        动画流程：放大棋子（模拟"拿起"）→ 缓动移动到目标位置 → 恢复原始大小（模拟"落地"）。
        若目标位置已有棋子（吃子），先将其从场景中移除。

        Args:
            move: 走法四元组 ``(起始行, 起始列, 目标行, 目标列)``。
        """
        sr, sc, er, ec = move
        src = (sr, sc)
        dst = (er, ec)

        moving_item = self._piece_items.get(src)
        if moving_item is None:
            self.rebuild_from_model()
            return

        # 吃子：先移除目标位置上的被吃棋子图元
        captured_item = self._piece_items.get(dst)
        if captured_item is not None and captured_item is not moving_item:
            self._scene.removeItem(captured_item)
            self._piece_items.pop(dst, None)

        # 移动过程中放大棋子，模拟"拿起悬浮"效果
        moving_item.setScale(MainWindow._PIECE_HIGHLIGHT_SCALE)

        anim = QPropertyAnimation(moving_item, b"pos")
        anim.setDuration(MainWindow._ANIM_DURATION_MS)
        # 三次缓动曲线：加速启动 → 减速轻放
        anim.setEasingCurve(QEasingCurve.InOutCubic)
        anim.setStartValue(moving_item.pos())
        p = self.model_to_view(er, ec)
        anim.setEndValue(QPointF(p.x(), p.y()))
        # 非阻塞调用：start() 将动画注册进 Qt 事件循环
        anim.start()
        # 动画结束后恢复原始大小，模拟"落地"
        anim.finished.connect(lambda: moving_item.setScale(1.0))
        moving_item._anim = anim  # type: ignore[attr-defined]

        self._piece_items.pop(src, None)
        self._piece_items[dst] = moving_item


# ═══════════════════════════════════════════════════════════════
#  MainWindow：配置 → 开始 → 对局 → 结束 的完整生命周期
# ═══════════════════════════════════════════════════════════════


class MainWindow(QMainWindow):
    """中国象棋 GUI 主窗口。

    管理从「配置阶段」到「对局阶段」再到「结束/重置」的完整生命周期。

    配置阶段：
        用户可在右侧面板选择红/黑双方 AI 类型及参数。
    对局阶段：
        点击「开始对局」后进入——棋盘点击和 AI 自动行棋被激活。
    结束/重置：
        对局自然结束或用户手动结束后回到配置阶段。

    Attributes:
        controller: 游戏控制器实例。
        human_color: 人机模式下人类所执色（红/黑）；双方均为玩家时可为 ``None``，
            走子以 ``board.current_player`` 为准。
        is_game_running: 是否处于对局进行中状态。
    """

    # 下拉框顺序 = 内部引擎 key 顺序；展示名来自 ENGINE_TO_DISPLAY_NAME
    _ENGINE_KEYS_ORDERED = ("human", "random", "minimax", "mcts", "mcts_minimax")
    _AI_TYPES = [ENGINE_TO_DISPLAY_NAME[k] for k in _ENGINE_KEYS_ORDERED]
    _IDX_HUMAN, _IDX_RANDOM, _IDX_MINIMAX, _IDX_MCTS, _IDX_MCTS_MINIMAX = 0, 1, 2, 3, 4

    # 棋子选中 / 拿起时的放大倍率（模拟"悬浮"视觉效果）
    _PIECE_HIGHLIGHT_SCALE = 1.2
    # 棋子移动动画时长（毫秒）
    _ANIM_DURATION_MS = 160
    # AI 后台线程的默认思考时间上限（秒）
    _AI_TIME_LIMIT_S = 10

    def __init__(self, controller: Optional[GameController] = None):
        """初始化主窗口。

        Args:
            controller: 外部注入的游戏控制器实例。若为 ``None``，
                则创建默认的玩家对玩家控制器。若注入的控制器
                已绑定 AI 代理，窗口将自动同步 UI 配置并开始对局。
        """
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
        """初始化窗口基本属性（标题、图标）。"""
        self.setWindowTitle("中国象棋")
        icon_path = _img("icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    def _init_ui(self) -> None:
        """构建完整的 UI 布局：左侧棋盘 + 右侧配置/状态/日志面板。"""
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout()
        root.setLayout(main_layout)

        # ── 左侧：棋盘视图 ──
        self.board_view = XiangqiBoardView(self.controller)
        self.board_view.square_clicked.connect(self._on_square_clicked)
        main_layout.addWidget(self.board_view, 1)

        # ── 右侧：配置 + 状态 + 日志面板 ──
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
        """构建单侧（红方/黑方）的 AI 配置面板 GroupBox。

        Args:
            title: 分组标题（如 ``"红方设置"``）。

        Returns:
            tuple: ``(QGroupBox, QComboBox, QLabel, QSpinBox)`` 四元组，
            分别为分组容器、AI 类型下拉框、参数标签和参数数值框。
        """
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
        """AI 类型下拉框切换时的回调——动态调整参数控件的可见性和数值范围。

        Args:
            index: 当前选中的 AI 类型索引。
            label: 对应的参数标签控件。
            spin: 对应的参数数值输入框控件。
        """
        if index == self._IDX_MINIMAX:
            label.setText("深度：")
            label.show()
            spin.setRange(1, 8)
            spin.setValue(5)
            spin.setSingleStep(1)
            spin.show()
        elif index == self._IDX_MCTS:
            label.setText("模拟次数：")
            label.show()
            spin.setRange(100, 100000)
            spin.setValue(5000)
            spin.setSingleStep(500)
            spin.show()
        elif index == self._IDX_MCTS_MINIMAX:
            label.setText("模拟次数：")
            label.show()
            spin.setRange(100, 100000)
            spin.setValue(4000)
            spin.setSingleStep(500)
            spin.show()
        else:
            label.hide()
            spin.hide()

    @staticmethod
    def _sync_param_from_agent(agent, label: QLabel, spin: QSpinBox) -> None:
        """根据已有 AI 代理实例回写参数控件的值。

        用于外部注入已配置好的 Controller 时，将 AI 参数同步到 UI 控件。

        Args:
            agent: AI 代理实例，``None`` 表示玩家（human）。
            label: 对应的参数标签控件。
            spin: 对应的参数数值输入框控件。
        """
        if agent is None:
            label.hide()
            spin.hide()
            return
        cls = type(agent).__name__
        if cls == "MinimaxAI":
            d = getattr(agent, "depth", 5)
            label.setText("深度：")
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
        elif cls == "MCTSMinimaxAI":
            s = getattr(agent, "max_simulations", 4000)
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
        """根据当前 UI 配置实例化 AI 代理。

        Args:
            combo: AI 类型下拉框控件。
            spin: 参数数值输入框控件。

        Returns:
            AI 代理实例；若选择玩家（human）则返回 ``None``。
        """
        idx = combo.currentIndex()
        key = self._ENGINE_KEYS_ORDERED[idx]
        if key == "human":
            return None
        if key == "random":
            return RandomAI()
        if key == "minimax":
            return MinimaxAI(depth=spin.value())
        if key == "mcts":
            return MCTSAI(time_limit=5.0, max_simulations=spin.value())
        if key == "mcts_minimax":
            return MCTSMinimaxAI(
                time_limit=10.0,
                max_simulations=spin.value(),
                verbose=False,
            )
        return None

    def _on_start_stop(self) -> None:
        """「开始对局」/「结束/重置对局」切换按钮的统一入口回调。"""
        if not self.is_game_running:
            self._start_game()
        else:
            self._stop_game()

    def _start_game(self) -> None:
        """启动新对局：读取 UI 配置 → 实例化 AI → 重置棋盘 → 进入对局模式。"""
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

        # 4) UI 反馈（对局标题/日志使用 GUI 专用文案，不沿用控制台 matchup_line）
        matchup = self._gui_matchup_line()
        self.setWindowTitle(f"中国象棋 — {matchup}")
        self.log_console.clear()
        self.append_log(f"[对局] {matchup}")
        self._refresh_status()
        self.check_and_run_ai()

    def _stop_game(self) -> None:
        """结束当前对局：中断 AI 线程 → 恢复配置模式 → 清除交互状态。"""
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
        self.append_log("[界面] 对局已结束")

    def _set_config_enabled(self, enabled: bool) -> None:
        """批量启用或禁用所有配置面板控件。

        Args:
            enabled: ``True`` 启用，``False`` 禁用。
        """
        for w in (
            self._red_combo,
            self._red_param_spin,
            self._black_combo,
            self._black_param_spin,
        ):
            w.setEnabled(enabled)

    # ────────────────────── 代理描述 / 映射 ──────────────────────

    @staticmethod
    def _engine_key_for_agent(agent) -> str:
        """``None`` / 各类 AI 实例 → ``human`` / ``mcts`` / …（内部 key）。"""
        if agent is None:
            return "human"
        cls = type(agent).__name__
        if cls == "RandomAI":
            return "random"
        if cls == "MinimaxAI":
            return "minimax"
        if cls == "MCTSAI":
            return "mcts"
        if cls == "MCTSMinimaxAI":
            return "mcts_minimax"
        return "human"

    @staticmethod
    def _gui_agent_brief(agent) -> str:
        """对局标题、状态栏用的简短展示名（含可选参数）。"""
        key = MainWindow._engine_key_for_agent(agent)
        base = ENGINE_TO_DISPLAY_NAME[key]
        if key == "human":
            return base
        if key == "random":
            return base
        if key == "minimax":
            d = getattr(agent, "depth", None)
            return f"{base} · 深度 {d}" if isinstance(d, int) else base
        if key == "mcts":
            s = getattr(agent, "max_simulations", None)
            w = getattr(agent, "workers", 1)
            return f"{base} · 模拟上限 {s} · 并行进程 {w}" if s is not None else base
        if key == "mcts_minimax":
            s = getattr(agent, "max_simulations", None)
            w = getattr(agent, "workers", 1)
            return f"{base} · 模拟上限 {s} · 并行进程 {w}" if s is not None else base
        return base

    def _gui_matchup_line(self) -> str:
        """窗口标题与对局日志中的对阵行。"""
        r = self._gui_agent_brief(self.controller.red_agent)
        b = self._gui_agent_brief(self.controller.black_agent)
        return f"红方 · {r} 对阵 黑方 · {b}"

    def _sync_human_color_from_controller(self) -> None:
        """根据控制器推断人机模式下人类所执色；双方均为玩家时为 ``None``。"""
        r, b = self.controller.red_agent, self.controller.black_agent
        if r is None and b is not None:
            self.human_color = "red"
        elif b is None and r is not None:
            self.human_color = "black"
        elif r is None and b is None:
            self.human_color = None
        else:
            self.human_color = "red"

    @staticmethod
    def _agent_label(agent) -> str:
        """状态栏 / 日志中当前思考方的简短描述（与 _gui_agent_brief 一致）。"""
        return MainWindow._gui_agent_brief(agent)

    @staticmethod
    def _agent_to_combo_index(agent) -> int:
        """将 AI 代理实例映射为 UI 下拉框对应的索引值。

        Args:
            agent: AI 代理实例，``None`` 表示玩家（human）。

        Returns:
            int: 与 ``_ENGINE_KEYS_ORDERED`` 对齐的下拉框索引。
        """
        k = MainWindow._engine_key_for_agent(agent)
        return MainWindow._ENGINE_KEYS_ORDERED.index(k)

    def _side_name(self, color: str) -> str:
        """将颜色代码转换为中文阵营名称。

        Args:
            color: 阵营颜色（``'red'`` 或 ``'black'``）。

        Returns:
            str: ``"红方"`` 或 ``"黑方"``。
        """
        return "红方" if color == "red" else "黑方"

    # ────────────────────── 游戏内逻辑 ──────────────────────

    def _finalize_after_legal_move(self, outcome: MoveOutcome) -> None:
        """合法走子后的统一收尾处理：刷新状态、检测终局、触发下一轮 AI。

        Args:
            outcome: 本步走子的结果对象。
        """
        if not outcome.ok:
            return
        self._refresh_status()
        if outcome.perpetual_warning and not outcome.game_over:
            off = outcome.perpetual_offender
            who = "红方" if off == "red" else "黑方" if off else "某方"
            wtxt = f"长将警告：{who}持续将军；第三次出现相同局面将判负。"
            self.append_log(f"⚠ {wtxt}")
            self.status_label.setText(f"⚠ {wtxt}")
            QMessageBox.warning(self, "长将警告", wtxt)

        if outcome.game_over:
            self._run_id += 1
            if outcome.move_limit_draw:
                msg = (
                    f"对局结束：已达 {Rules.MAX_PLIES_AUTODRAW} 手限着（半回合计），和棋。"
                )
                self.append_log(msg)
                self.append_log("==========================")
                self.status_label.setText(msg)
                QMessageBox.information(self, "对局结束", msg)
                self._game_over_ui()
                return
            if outcome.perpetual_forfeit:
                side = "红" if outcome.winner == "red" else "黑"
                msg = f"对局结束：长将判负，{side}方获胜！"
                self.append_log(msg)
                self.append_log("==========================")
                self.status_label.setText(msg)
                self._game_over_ui()
                return
            if outcome.winner == "red":
                msg = "对局结束：红方获胜！"
            elif outcome.winner == "black":
                msg = "对局结束：黑方获胜！"
            else:
                msg = "对局结束：和棋（死局）！"
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
        """向日志控制台追加一条带时间戳的消息，并自动滚动到底部。

        Args:
            text: 日志消息文本。
        """
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_console.appendPlainText(f"[{ts}] {text}")
        cursor = self.log_console.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_console.setTextCursor(cursor)
        sb = self.log_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _refresh_status(self) -> None:
        """根据当前对局状态刷新状态栏显示文本。"""
        result = self.controller.current_result()
        if result["game_over"]:
            winner = result["winner"]
            if winner == "red":
                self.status_label.setText("红方获胜！")
            elif winner == "black":
                self.status_label.setText("黑方获胜！")
            else:
                self.status_label.setText("对局结束：和棋！")
            return

        player = result["current_player"]
        agent = self.controller.agent_for(player)
        if agent is None:
            self.status_label.setText(
                f"{self._side_name(player)} · {ENGINE_TO_DISPLAY_NAME['human']} · 请走棋"
            )
        else:
            self.status_label.setText(
                f"{self._side_name(player)} · {self._agent_label(agent)} 思考中…"
            )

    # ────────────────────── 棋盘点击 ──────────────────────

    def _on_square_clicked(self, row: int, col: int) -> None:
        """棋盘格点击事件处理器：实现选子 → 落子的两步交互流程。

        第一次点击选择己方棋子（放大高亮），第二次点击指定目标位置完成走子。
        点击已选中棋子可取消选择。

        Args:
            row: 被点击格的行号。
            col: 被点击格的列号。
        """
        if not self.is_game_running:
            return
        if self.controller.is_game_over():
            self._refresh_status()
            return
        cp = self.controller.board.current_player
        if self.controller.agent_for(cp) is not None:
            return

        piece = self.controller.board.get_piece(row, col)
        clicked_item = self.board_view.piece_item_at(row, col)

        if self._selected is None:
            if piece and piece.color == cp:
                self._selected = (row, col)
                self._selected_item = clicked_item
                if self._selected_item is not None:
                    self._selected_item.setScale(self._PIECE_HIGHLIGHT_SCALE)
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

        outcome = self.controller.try_apply_player_move(move, player=cp)
        if not outcome.ok:
            self.status_label.setText(outcome.message or "无效走子")
            if outcome.message:
                self.append_log(f"[界面] {outcome.message}")
            return

        print(f"[界面] 玩家已落子: {move}")
        self.append_log(f"[界面] 玩家走法: {move}")
        self.append_log("--------------------------")
        self.board_view.animate_move(move)
        self._finalize_after_legal_move(outcome)

    # ────────────────────── AI 后台线程 ──────────────────────

    def check_and_run_ai(self) -> None:
        """检查是否轮到 AI 行棋，若是则启动后台计算线程。

        仅当对局正在进行、未终局且当前轮次为 AI 方时才启动线程。
        若已有 AI 线程正在运行则跳过，避免重复启动。
        """
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
        print(f"[界面] {side} · {label} 思考中，启动计算…")
        self.status_label.setText(f"{side} · {label} 思考中…")
        self.append_log(f"[界面] {side} · {label}，开始计算…")

        board_snapshot = self.controller.board.copy()
        run_id = self._run_id
        game_hist = list(self.controller.game_history_hashes)
        move_hist = list(self.controller.history)
        self._ai_thread = AIMoveThread(
            ai=current_agent,
            board_snapshot=board_snapshot,
            ai_color=cp,
            time_limit_s=self._AI_TIME_LIMIT_S,
            run_id=run_id,
            game_history_hashes=game_hist,
            move_history=move_hist,
        )
        self._ai_thread.move_ready.connect(self._on_ai_move_ready)
        self._ai_thread.start()

    def _on_ai_move_ready(
        self, move: Optional[Move], stats: Optional[dict], run_id: int
    ) -> None:
        """AI 计算完成的信号槽：接收走法并应用到棋盘。

        通过 ``run_id`` 过滤过期信号（防止旧线程的结果污染当前对局）。

        Args:
            move: AI 选择的走法，``None`` 表示无合法走法。
            stats: AI 搜索统计信息字典（耗时、节点数等）。
            run_id: 本次计算的唯一标识，用于判断信号是否仍然有效。
        """
        if run_id != self._run_id:
            return
        print(f"[界面] 已收到 AI 走法信号: {move}")
        if stats:
            if stats.get("error"):
                self.append_log(f"[界面] AI 计算异常：{stats['error']}")
                QMessageBox.critical(self, "计算异常", str(stats["error"]))
                self._refresh_status()
                return
            if stats.get("opening_book"):
                self.append_log("命中开局库 | 耗时: 0.0s")
            elif stats.get("random"):
                self._log_random_stats(stats)
            elif stats.get("probe_count") is not None and not stats.get("opening_book"):
                self._log_mcts_minimax_stats(stats)
            elif stats.get("simulations") is not None:
                self._log_mcts_stats(stats)
            else:
                self._log_minimax_stats(stats)
        self.append_log("[界面] 已收到走法，执行落子。")
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
        """将 Random AI 的极简统计信息写入日志。

        Args:
            stats: 包含 ``time_taken`` 等键的统计字典。
        """
        time_taken = stats.get("time_taken", 0)
        time_str = (
            f"{time_taken:.3f}" if isinstance(time_taken, (int, float)) else str(time_taken)
        )
        self.append_log(f"{ENGINE_TO_DISPLAY_NAME['random']} — 已选出走法")
        self.append_log(f"{ENGINE_TO_DISPLAY_NAME['random']} — 耗时（秒）: {time_str}")

    def _log_mcts_stats(self, stats: dict) -> None:
        """将 MCTS 搜索统计信息格式化后写入日志。

        Args:
            stats: 包含 ``time_taken``, ``simulations``, ``workers``,
                ``win_rate`` 等键的统计字典。
        """
        time_taken = stats.get("time_taken", 0)
        time_str = (
            f"{time_taken:.3f}" if isinstance(time_taken, (int, float)) else str(time_taken)
        )
        self.append_log(f"{ENGINE_TO_DISPLAY_NAME['mcts']} — 搜索耗时（秒）: {time_str}")
        self.append_log(f"{ENGINE_TO_DISPLAY_NAME['mcts']} — 模拟次数: {stats.get('simulations', 0)}")
        self.append_log(f"{ENGINE_TO_DISPLAY_NAME['mcts']} — 并行进程数: {stats.get('workers', 1)}")
        win_rate = stats.get("win_rate", "")
        if win_rate:
            self.append_log(f"{ENGINE_TO_DISPLAY_NAME['mcts']} — 估计胜率: {win_rate}")

    def _log_mcts_minimax_stats(self, stats: dict) -> None:
        """将 MCTS_Minimax_AI 搜索的统计信息写入日志。"""
        mx = ENGINE_TO_DISPLAY_NAME["mcts_minimax"]
        time_taken = stats.get("time_taken", 0)
        time_str = (
            f"{time_taken:.3f}" if isinstance(time_taken, (int, float)) else str(time_taken)
        )
        self.append_log(f"{mx} — 搜索耗时（秒）: {time_str}")
        self.append_log(f"{mx} — 模拟次数: {stats.get('simulations', 0)}")
        self.append_log(f"{mx} — 并行进程数: {stats.get('workers', 1)}")
        win_rate = stats.get("win_rate", "")
        if win_rate:
            self.append_log(f"{mx} — 估计胜率: {win_rate}")
        pc = stats.get("probe_count")
        pn = stats.get("probe_nodes")
        if pc is not None:
            self.append_log(
                f"{mx} — Minimax 子搜索次数: {pc}"
                + (f"，子搜索节点数: {pn}" if pn is not None else "")
            )
        bc = stats.get("budget_calls_used")
        bm = stats.get("budget_calls_max")
        if bc is not None and bm is not None:
            self.append_log(f"{mx} — 子搜索预算（调用次数）: {bc}/{bm}")

    def _log_minimax_stats(self, stats: dict) -> None:
        """将 Minimax 搜索统计信息格式化后写入日志。

        Args:
            stats: 包含 ``time_taken``, ``depth``, ``nodes_evaluated``,
                ``tt_hits`` 等键的统计字典。
        """
        time_taken = stats.get("time_taken", "?")
        time_str = (
            f"{time_taken:.3f}" if isinstance(time_taken, (int, float)) else str(time_taken)
        )
        mn = ENGINE_TO_DISPLAY_NAME["minimax"]
        self.append_log(f"{mn} — 搜索耗时（秒）: {time_str}")
        self.append_log(f"{mn} — 搜索深度: {stats.get('depth', '?')}")
        self.append_log(f"{mn} — 评估节点数: {stats.get('nodes_evaluated', '?')}")
        tt_hits = stats.get("tt_hits")
        if tt_hits is not None:
            self.append_log(f"{mn} — 置换表命中次数: {tt_hits}")


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
