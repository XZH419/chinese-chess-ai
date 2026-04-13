"""中国象棋启发式评估函数模块。

本模块实现了一套面向中国象棋的静态局面评估体系，核心思路如下：

1. **Tapered Evaluation（锥形评估）**：
   维护中局（MG）与残局（EG）两套独立评分轨道，再根据当前盘面
   剩余重子数量计算阶段因子 ``phase``，对两轨道做线性插值得到最终
   分值。这样做的原因是同一棋子在中局和残局的价值差异显著——例如
   兵在中局价值较低，但残局因可助攻九宫而大幅增值；马在残局因子力
   稀疏、空间开阔而优于中局。

2. **子力价值表（Material Values）**：
   ``MG_VALUES`` / ``EG_VALUES`` 分别存储中局与残局子力基础分，
   两套表在锥形插值中与 PST 合并。

3. **位置分值表（Piece-Square Tables, PST）**：
   ``PST_MG_MAP`` / ``PST_EG_MAP`` 为每种棋子在 10×9 棋盘上的
   每个位置赋予额外奖惩分。设计依据：
   - 马 PST：行 0–2、列 3/6 给予高分，反映挂角马和卧槽马的战术
     价值（这两个位置可直接威胁九宫）。
   - 车 PST：前三行（深入敌阵）分值递增，体现"车入敌营价值千金"
     的棋理。
   - 兵 PST：过河兵按接近九宫程度递增，中路兵（列 3–5）额外加分，
     体现"小卒过河顶大车"的残局优势。
   - 炮 PST：己方阵地偏正，深入敌方中路则扣分（炮怕空心，需要
     炮架），翼侧略加分。
   - 士/象/将 PST：鼓励居中守护九宫的稳固阵型。

4. **战术协同分（Tactical Synergy）**：
   ``_tactical_synergy`` 方法在不进行着法生成的前提下，纯粹通过
   坐标几何检测以下战术模式：
   - 车与敌将同列且无遮挡 → 纵线车威胁。
   - 炮与敌将同列且恰有一枚炮架 → 炮架已成。
   - 马处于挂角/卧槽高价值格且马腿畅通 → 即时战术威胁。
   - 马深入敌阵 + 己方车亦在敌方底线/侧翼 → 车马协同加分。
   这样可以在不调用完整着法生成器的情况下捕获关键战术态势，
   在保持评估速度的同时提升棋力。

5. **兑子惩罚（Anti-Trade Penalty）**：
   当一方子力领先（差值 > 50）时，若总子力减少则给领先方施加惩罚。
   原因是子力领先方应当避免无意义的兑子——保持子力优势比简化局面
   更有利。

6. **将军奖励（Check Bonus）**：
   当走子方正在将军对方时，给予固定评分偏置，使搜索倾向于保持
   对敌方的压迫。

整个评估函数采用 Negamax 视角——返回值始终以当前走子方为正。
"""

from __future__ import annotations

from typing import Optional

from chinese_chess.model.board import Board
from chinese_chess.model.rules import Rules


def _pst_ma_mg() -> list[list[int]]:
    """生成马的中局位置分值表（PST）。

    功能说明：
        构建马在中局阶段的 10×9 位置奖励矩阵。表以红方视角编写
        （行 0 = 对方底线，行 9 = 己方底线），黑方使用时需翻转行号。

        设计要点——为什么行 0–2、列 3/6 分值最高：
        这些格子对应"挂角马"与"卧槽马"的经典站位。挂角马从对方
        九宫斜角直接威胁将帅，卧槽马则卡在九宫底角形成难以解除的
        攻势。中局阶段子力密集，马到达这些高价值格的威胁远大于残局，
        因此 MG 表的峰值（18–22）高于 EG 表。

    Returns:
        10×9 的二维整数列表，表示马在中局每个位置的额外奖励分值。
    """
    t = [
        [0, 1, 4, 18, 6, 18, 4, 1, 0],
        [1, 4, 8, 22, 10, 22, 8, 4, 1],
        [2, 8, 12, 16, 13, 16, 12, 8, 2],
        [3, 9, 12, 14, 15, 14, 12, 9, 3],
        [3, 9, 13, 15, 16, 15, 13, 9, 3],
        [3, 8, 12, 14, 15, 14, 12, 8, 3],
        [2, 6, 10, 12, 13, 12, 10, 6, 2],
        [1, 4, 6, 8, 9, 8, 6, 4, 1],
        [0, 2, 3, 4, 5, 4, 3, 2, 0],
        [0, 0, 1, 2, 3, 2, 1, 0, 0],
    ]
    return t


def _pst_ma_eg() -> list[list[int]]:
    """生成马的残局位置分值表（PST）。

    功能说明：
        构建马在残局阶段的 10×9 位置奖励矩阵。结构与中局表相同，
        但整体数值略低。

        设计原因——为什么残局马的 PST 弱于中局：
        残局阶段子力稀疏，马的机动性因蹩腿概率降低而天然增强，
        因此子力基础分 ``EG_VALUES["ma"]`` 已从 400 提升到 430。
        相应地，PST 奖励适当下调，避免与基础分叠加后过度膨胀。
        尽管如此，前沿高位马（靠近对方底线）仍保留一定奖励，
        体现残局中马深入的进攻价值。

    Returns:
        10×9 的二维整数列表，表示马在残局每个位置的额外奖励分值。
    """
    t = [
        [0, 1, 3, 14, 5, 14, 3, 1, 0],
        [1, 3, 6, 18, 8, 18, 6, 3, 1],
        [2, 6, 10, 13, 12, 13, 10, 6, 2],
        [3, 8, 11, 13, 14, 13, 11, 8, 3],
        [3, 8, 12, 14, 15, 14, 12, 8, 3],
        [3, 7, 11, 13, 14, 13, 11, 7, 3],
        [2, 5, 9, 11, 12, 11, 9, 5, 2],
        [1, 3, 5, 7, 8, 7, 5, 3, 1],
        [0, 2, 3, 4, 5, 4, 3, 2, 0],
        [0, 0, 1, 2, 3, 2, 1, 0, 0],
    ]
    return t


class Evaluation:
    """中国象棋静态局面评估器。

    功能说明：
        基于 Tapered Evaluation 框架，综合子力价值、位置分值表（PST）、
        战术协同、兑子惩罚和将军奖励，对给定局面输出一个以当前走子方
        为正的浮点评估分。

        **Tapered Evaluation 的工作原理：**
        游戏阶段因子 ``phase`` 由盘面剩余重子（车权重 2、马/炮权重 1）
        决定，满子值为 16。评估时分别计算中局轨道分 ``mg`` 和残局轨道
        分 ``eg``，最终得分 = ``(mg × phase + eg × (16 − phase)) / 16``。
        当重子齐全时 phase ≈ 16，偏向中局分；随着兑子推进 phase 下降，
        逐渐过渡到残局分。这比硬切阈值更平滑，能避免评估函数在阶段
        边界处发生突变。

    Attributes:
        _eval_cache: 静态评估置换表，键为 Zobrist 哈希值，值为评估分。
            由 ``MinimaxAI.get_best_move`` 在每次根搜索入口处清空，
            避免跨局面的缓存污染。
        ATTACKING_PIECE_TYPES: 具备进攻能力的棋子类型集合（车、马、炮、兵）。
            若双方均无此类棋子，则判定为无法将杀的和棋局面。
        PHASE_WEIGHTS: 各棋子对游戏阶段因子的贡献权重。车作为最强重子
            权重为 2，马和炮各为 1。
        TOTAL_PHASE: 满子局面的阶段总权重（双方各 2 车 + 2 马 + 2 炮 = 16）。
        MG_VALUES: 中局子力基础价值表。
        EG_VALUES: 残局子力基础价值表。
        PIECE_VALUES: 对外暴露的子力价值表（等同于中局表），供走法排序等
            模块引用。
        CHECK_EVAL_BONUS: 将军评估奖励分。
    """

    # 静态评估置换表：``zobrist_hash`` → 评估分
    # MinimaxAI.get_best_move 在每次根搜索入口处会清空此缓存
    _eval_cache: dict[int, float] = {}

    # 具备将杀/实质性进攻能力的棋子类型
    # 若双方均只剩将、士、象，则视为无法将死对方，评估直接返回 0（和棋）
    ATTACKING_PIECE_TYPES = frozenset({"che", "ma", "pao", "bing"})

    # 游戏阶段权重：车权重 2、马权重 1、炮权重 1
    # 满子状态下双方共计 phase = (2+2+2)×2 = 16
    PHASE_WEIGHTS: dict[str, int] = {"che": 2, "ma": 1, "pao": 1}
    TOTAL_PHASE: float = 16.0

    # 中局子力基础价值
    # 车最强（900），炮略高于马（470 vs 400），反映中局炮远程打击的优势
    MG_VALUES: dict[str, int] = {
        "che": 900,
        "pao": 470,
        "ma": 400,
        "xiang": 200,
        "shi": 200,
        "bing": 90,
        "jiang": 0,
    }
    # 残局子力基础价值
    # 马升至 430（残局空间开阔、蹩腿少），炮降至 410（缺炮架），兵升至 140（可助攻九宫）
    EG_VALUES: dict[str, int] = {
        "che": 920,
        "pao": 410,
        "ma": 430,
        "xiang": 200,
        "shi": 200,
        "bing": 140,
        "jiang": 0,
    }
    # 对外暴露的子力价值表（等同于中局表），供走法排序等外部模块引用
    PIECE_VALUES: dict[str, int] = MG_VALUES

    # 开局子力总和的近似量级，用于兑子惩罚公式中的归一化
    _REF_TOTAL_MATERIAL = 4400
    # 兑子惩罚系数：当子力差 > 50 时，领先方因总子力减少而受到的惩罚倍率
    # 目的是让领先方倾向于保持子力优势，不做无意义兑子
    _ANTI_TRADE_COEFF = 0.018

    # ── 兵的位置分值表 ──
    # 中局兵 PST：过河兵按接近九宫的程度递增奖励，中路列（3–5）额外加分
    # 设计原因：过河兵越靠近对方九宫威胁越大，中路兵可配合车炮直攻将帅
    PST_BING_MG = [
        [18, 18, 22, 38, 48, 38, 22, 18, 18],
        [20, 20, 26, 40, 50, 40, 26, 20, 20],
        [16, 16, 24, 36, 46, 36, 24, 16, 16],
        [12, 12, 20, 32, 40, 32, 20, 12, 12],
        [8, 8, 14, 22, 28, 22, 14, 8, 8],
        [4, 4, 8, 10, 12, 10, 8, 4, 4],
        [2, 2, 4, 6, 6, 6, 4, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    # 残局兵 PST：整体数值大幅高于中局表
    # 设计原因：残局阶段兵的推进价值急剧上升，"小卒过河顶大车"在残局尤为突出
    PST_BING_EG = [
        [38, 38, 48, 68, 82, 68, 48, 38, 38],
        [32, 32, 42, 60, 72, 60, 42, 32, 32],
        [26, 26, 36, 52, 62, 52, 36, 26, 26],
        [20, 20, 28, 42, 50, 42, 28, 20, 20],
        [14, 14, 20, 32, 38, 32, 20, 14, 14],
        [8, 8, 10, 16, 18, 16, 10, 8, 8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    # ── 马的位置分值表（由工厂函数生成） ──
    PST_MA_MG = _pst_ma_mg()
    PST_MA_EG = _pst_ma_eg()

    # ── 车的位置分值表（中局/残局共用） ──
    # 行 0–2（深入敌阵）分值最高，体现"车入敌营价值千金"的棋理
    # 原先评估函数中 r<=1 / r>=8 的 +30 深入奖励已并入此表前三行
    PST_CHE = [
        [18, 19, 20, 21, 22, 21, 20, 19, 18],
        [16, 17, 18, 19, 20, 19, 18, 17, 16],
        [12, 13, 14, 15, 16, 15, 14, 13, 12],
        [7, 8, 9, 10, 11, 10, 9, 8, 7],
        [7, 8, 10, 11, 12, 11, 10, 8, 7],
        [7, 8, 10, 11, 12, 11, 10, 8, 7],
        [7, 8, 9, 10, 11, 10, 9, 8, 7],
        [6, 7, 8, 9, 10, 9, 8, 7, 6],
        [5, 6, 7, 8, 9, 8, 7, 6, 5],
        [4, 5, 6, 7, 8, 7, 6, 5, 4],
    ]

    # ── 炮的位置分值表（红方视角，行 0=黑侧上方，行 9=红侧底线） ──
    # 设计原因：炮需要炮架才能发挥威力，深入对方空旷中路反而不利（负分），
    # 留在己方阵地或侧翼有炮架可借时更佳（正分）
    PST_PAO = [
        [6, 4, 0, -10, -12, -10, 0, 4, 6],
        [4, 2, 0, -8, -14, -8, 0, 2, 4],
        [2, 0, 0, -6, -8, -6, 0, 0, 2],
        [0, 0, 0, -2, -4, -2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-2, 0, 4, 2, 6, 2, 4, 0, -2],
        [0, 0, 0, 2, 4, 2, 0, 0, 0],
        [2, 2, 0, 4, 6, 4, 0, 2, 2],
        [2, 2, 0, 4, 6, 4, 0, 2, 2],
        [0, 0, 0, 2, 4, 2, 0, 0, 0],
    ]

    # ── 士的位置分值表 ──
    # 行 8 列 4（九宫中心）给予 +15 奖励，鼓励士守中
    PST_SHI = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 15, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    # ── 象的位置分值表 ──
    # 行 7 列 4（九宫前方中心点）给予 +15 奖励，行 5/9 列 2/6（象眼位）各 +2
    # 鼓励象守护九宫前沿，形成稳固防线
    PST_XIANG = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 15, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 2, 0, 0],
    ]

    # ── 将/帅的位置分值表 ──
    # 行 9 列 4（九宫底线正中）给予最高 +15 奖励，鼓励将帅居中
    # 行 7（前沿）给予负分 -10，惩罚将帅过度前移导致的暴露风险
    PST_JIANG = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -10, -10, -10, 0, 0, 0],
        [0, 0, 0, -5, 0, -5, 0, 0, 0],
        [0, 0, 0, 5, 15, 5, 0, 0, 0],
    ]

    # 默认 PST：全零矩阵，用于未配置专用表的棋子类型
    PST_DEFAULT = [[0] * 9 for _ in range(10)]

    # 中局 PST 映射表：棋子类型 → 对应的中局位置分值表
    PST_MG_MAP: dict[str, list[list[int]]] = {
        "bing": PST_BING_MG,
        "ma": PST_MA_MG,
        "che": PST_CHE,
        "shi": PST_SHI,
        "xiang": PST_XIANG,
        "pao": PST_PAO,
        "jiang": PST_JIANG,
    }
    # 残局 PST 映射表：棋子类型 → 对应的残局位置分值表
    PST_EG_MAP: dict[str, list[list[int]]] = {
        "bing": PST_BING_EG,
        "ma": PST_MA_EG,
        "che": PST_CHE,
        "shi": PST_SHI,
        "xiang": PST_XIANG,
        "pao": PST_PAO,
        "jiang": PST_JIANG,
    }

    # 马的八个跳跃方向增量（行偏移, 列偏移）
    _MA_DELTAS = (
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
    )
    # 四个正交方向（用于车/炮的纵横线扫描）
    _ORTH = ((0, 1), (0, -1), (1, 0), (-1, 0))

    # 将军评估奖励：走子方正在将军对方时给予的固定偏置
    # 使搜索偏好保持对敌方将帅的压迫，依赖 Rules.is_king_in_check 的正确性
    CHECK_EVAL_BONUS: float = 15.0

    # ── 战术协同分常量（同时加在 MG/EG 轨道） ──
    # 车与敌将同列且无遮挡时的纵线威胁奖励
    _T_ROOK_FILE = 32.0
    # 炮与敌将同列且恰有一枚炮架时的奖励
    _T_PAO_SCREEN = 26.0
    # 马处于挂角/卧槽高价值格且马腿畅通时的奖励
    _T_MA_LEG_PREMIUM = 18.0
    # 马深入敌阵且己方车也在敌方底线/侧翼时的车马协同奖励
    _T_MA_CHE_SYNERGY = 24.0

    @staticmethod
    def repetition_leaf_score(board: Board) -> float:
        """计算搜索中遇到重复局面时的叶子节点评估分。

        功能说明：
            当 Minimax 搜索检测到当前路径将重演已出现过的局面时，调用
            此函数获取一个惩罚/接受分值，而非继续深搜。其策略为：
            - 己方大劣（评估 < -200）时接受和棋，返回 0。
            - 己方领先（评估 > 50）时厌战，返回负分以惩罚重复。
            - 其余情况返回 0（不鼓励也不惩罚）。

            这与 ``Rules.is_threefold_repetition_draw``（三次重复判和的
            布尔终局判定）不同，本函数仅用于搜索树内部的启发式估值。

        Args:
            board: 当前棋盘局面。

        Returns:
            叶子节点的评估分（浮点数）。大优时为负值（惩罚重复），
            大劣时为 0（接受和棋）。
        """
        e = Evaluation.evaluate(board)
        if e < -200:
            return 0.0
        if e > 50:
            return min(0.0, 50.0 - 0.65 * e)
        return 0.0

    @staticmethod
    def _ma_mobility(board: Board, sr: int, sc: int, color: str) -> float:
        """计算指定马的机动性奖励分。

        功能说明：
            遍历马的八个跳跃方向，逐一判定马腿是否被蹩（中国象棋
            马走"日"字需经过一个中间格，若该格有子则不可跳）。对于
            马腿畅通的每个落点：
            - 若落点为空或对方棋子，加 5 分（可用机动步）。
            - 若落点为己方的马，加 15 分（连环马协同奖励——两马
              相互保护，战术价值高于单马）。
            - 若落点为己方其他棋子，不计分（该方向被己方阻塞）。

        Args:
            board: 当前棋盘局面。
            sr: 马所在行坐标。
            sc: 马所在列坐标。
            color: 马的颜色（"red" 或 "black"）。

        Returns:
            该马的机动性总奖励分（非负浮点数）。
        """
        b = board.board
        bonus = 0.0
        for dr, dc in Evaluation._MA_DELTAS:
            er, ec = sr + dr, sc + dc
            if not (0 <= er < 10 and 0 <= ec < 9):
                continue
            # 判定马腿位置：日字长边方向上紧邻起点的那个格
            if abs(dr) == 2:
                leg_r, leg_c = sr + (1 if dr > 0 else -1), sc
            else:
                leg_r, leg_c = sr, sc + (1 if dc > 0 else -1)
            # 马腿被蹩，跳过此方向
            if b[leg_r][leg_c] is not None:
                continue
            tgt = b[er][ec]
            if tgt is not None and tgt.color == color:
                # 落点为己方马 → 连环马协同奖励
                if tgt.piece_type == "ma":
                    bonus += 15.0
                continue
            # 落点为空或对方棋子 → 可用机动步
            bonus += 5.0
        return bonus

    @staticmethod
    def _pao_screen_bonus(board: Board, r: int, c: int, color: str) -> float:
        """计算指定炮的炮架（屏风）奖励分。

        功能说明：
            沿四个正交方向扫描，若在该方向上遇到的第一枚棋子是己方棋子，
            则视为潜在炮架，给予 +18 分奖励。

            设计原因：炮的攻击依赖隔子打，拥有己方棋子作为炮架意味着
            该方向上的攻击通道已具备条件。此奖励鼓励炮保持在有炮架
            可借的位置，而非孤立无援。

        Args:
            board: 当前棋盘局面。
            r: 炮所在行坐标。
            c: 炮所在列坐标。
            color: 炮的颜色（"red" 或 "black"）。

        Returns:
            该炮的炮架总奖励分（非负浮点数）。
        """
        b = board.board
        bonus = 0.0
        for dr, dc in Evaluation._ORTH:
            tr, tc = r + dr, c + dc
            while 0 <= tr < 10 and 0 <= tc < 9:
                p = b[tr][tc]
                if p is None:
                    tr += dr
                    tc += dc
                    continue
                # 遇到的第一枚棋子为己方 → 可作为炮架
                if p.color == color:
                    bonus += 18.0
                break
        return bonus

    @staticmethod
    def _col_pieces_between(b, r0: int, r1: int, c: int) -> int:
        """统计同一列上两行之间的棋子数量（不含端点）。

        功能说明：
            用于车/炮的纵线威胁判定——检查己方车/炮与敌方将之间是否
            有遮挡棋子。车需要 0 个遮挡（直线攻击），炮需要恰好 1 个
            遮挡（炮架）。

        Args:
            b: 棋盘二维数组（``board.board`` 的引用）。
            r0: 第一个端点行坐标。
            r1: 第二个端点行坐标。
            c: 列坐标。

        Returns:
            严格位于 r0 与 r1 之间（不含端点）的棋子数量。
        """
        if r0 > r1:
            r0, r1 = r1, r0
        n = 0
        for r in range(r0 + 1, r1):
            if b[r][c] is not None:
                n += 1
        return n

    @staticmethod
    def _ma_premium_square(player: str, r: int, c: int) -> bool:
        """判断马是否处于挂角/卧槽高价值格。

        功能说明：
            检查马是否位于对方底线前三行（PST 行 0–2）的列 3 或列 6。
            这些格子对应中国象棋中经典的"挂角马"和"卧槽马"站位，
            可直接威胁对方九宫内的将帅。

        Args:
            player: 马的所属方（"red" 或 "black"）。
            r: 马的行坐标（绝对棋盘坐标）。
            c: 马的列坐标。

        Returns:
            若马位于高价值格则返回 True，否则返回 False。
        """
        pst_r = r if player == "red" else 9 - r
        return pst_r <= 2 and c in (3, 6)

    @staticmethod
    def _ma_any_leg_clear(b, sr: int, sc: int) -> bool:
        """判断马是否至少有一条马腿畅通。

        功能说明：
            遍历马的八个跳跃方向，只要任意一个方向的马腿格为空即返回
            True。用于配合 ``_ma_premium_square`` 判断高价值格上的马
            是否具备实际威胁——若马腿全部被蹩则即使位于好格也无法
            发动进攻。

        Args:
            b: 棋盘二维数组（``board.board`` 的引用）。
            sr: 马所在行坐标。
            sc: 马所在列坐标。

        Returns:
            若至少有一条马腿畅通则返回 True，否则返回 False。
        """
        for dr, dc in Evaluation._MA_DELTAS:
            er, ec = sr + dr, sc + dc
            if not (0 <= er < 10 and 0 <= ec < 9):
                continue
            if abs(dr) == 2:
                leg_r, leg_c = sr + (1 if dr > 0 else -1), sc
            else:
                leg_r, leg_c = sr, sc + (1 if dc > 0 else -1)
            if b[leg_r][leg_c] is None:
                return True
        return False

    @staticmethod
    def _tactical_synergy(
        board: Board,
        player: str,
        b,
        enemy_kr: Optional[int],
        enemy_kc: Optional[int],
    ) -> float:
        """计算轻量级战术协同评估分（无需着法生成，纯坐标运算）。

        功能说明：
            遍历指定方的所有活跃棋子，检测以下四种战术模式并累加奖励：

            1. **纵线车威胁**（+32 分）：
               己方车与敌方将处于同一列，且两者之间无任何棋子遮挡。
               这意味着车对将形成直接的纵线牵制或攻杀威胁。

            2. **炮架已成**（+26 分）：
               己方炮与敌方将处于同一列，且两者之间恰有一枚棋子。
               炮隔子打的条件已满足，对方将帅面临炮击威胁。

            3. **马占高价值格**（+18 分）：
               己方马位于挂角/卧槽格（对方底线前三行的列 3 或列 6），
               且至少有一条马腿畅通。这表示马已到位且具备实际进攻能力。

            4. **车马协同**（+24 分）：
               己方马深入对方阵地，且己方车也处于对方底线或侧翼。
               车马联合进攻是中国象棋中最强大的战术组合之一。

            设计原因——为什么不使用完整的着法生成：
            完整着法生成的计算开销较大，而评估函数在搜索树的每个
            节点都会被调用。通过纯坐标几何判定可以在 O(棋子数) 的
            时间内捕获关键战术态势，在速度和准确性之间取得良好平衡。

        Args:
            board: 当前棋盘局面。
            player: 计算己方的颜色（"red" 或 "black"）。
            b: ``board.board`` 的局部引用（避免重复属性查找以提升性能）。
            enemy_kr: 敌方将帅的行坐标（若已被吃则为 ``None``）。
            enemy_kc: 敌方将帅的列坐标（若已被吃则为 ``None``）。

        Returns:
            己方战术协同总分（非负浮点数）。
        """
        if enemy_kr is None or enemy_kc is None:
            return 0.0
        ekr, ekc = enemy_kr, enemy_kc
        score = 0.0
        ma_deep = False
        che_flank = False

        # 根据己方颜色确定"深入敌阵"的行范围
        if player == "red":
            deep_lo, deep_hi = 0, 2
            che_deep_lo, che_deep_hi = 0, 2
        else:
            deep_lo, deep_hi = 7, 9
            che_deep_lo, che_deep_hi = 7, 9

        for r, c in board.active_pieces.get(player, ()):
            p = b[r][c]
            if p is None or p.color != player:
                continue
            pt = p.piece_type

            if pt == "che":
                # 检测纵线车威胁：车与敌将同列且无遮挡
                if c == ekc:
                    between = Evaluation._col_pieces_between(b, r, ekr, c)
                    if between == 0:
                        score += Evaluation._T_ROOK_FILE
                # 检测车是否深入敌方底线区域
                if che_deep_lo <= r <= che_deep_hi:
                    che_flank = True
                elif c in (0, 8) and (
                    (player == "red" and r <= 4) or (player == "black" and r >= 5)
                ):
                    # 车在边路（列 0 或列 8）且已过河 → 侧翼威胁
                    che_flank = True

            elif pt == "pao":
                # 检测炮架威胁：炮与敌将同列且恰有一枚中间棋子
                if c == ekc:
                    between = Evaluation._col_pieces_between(b, r, ekr, c)
                    if between == 1:
                        score += Evaluation._T_PAO_SCREEN

            elif pt == "ma":
                # 检测马是否占据挂角/卧槽高价值格且马腿畅通
                if Evaluation._ma_premium_square(player, r, c) and Evaluation._ma_any_leg_clear(
                    b, r, c
                ):
                    score += Evaluation._T_MA_LEG_PREMIUM
                # 记录马是否深入敌阵（用于后续车马协同判定）
                if deep_lo <= r <= deep_hi:
                    ma_deep = True

        # 车马协同：马已深入 + 车在敌方底线/侧翼 → 额外加分
        if ma_deep and che_flank:
            score += Evaluation._T_MA_CHE_SYNERGY
        return score

    @staticmethod
    def evaluate(board: Board) -> float:
        """对给定局面进行 Tapered Evaluation 静态评估。

        功能说明：
            这是评估器的核心入口。按以下步骤计算评估分：

            1. **缓存查询**：通过 Zobrist 哈希检查置换表，命中则直接返回。

            2. **子力与位置遍历**：遍历双方所有活跃棋子，分别累加：
               - 中局/残局子力基础分（``MG_VALUES`` / ``EG_VALUES``）。
               - 中局/残局位置奖惩分（``PST_MG_MAP`` / ``PST_EG_MAP``）。
               - 马的机动性奖励和炮的炮架奖励。
               - 阶段因子 ``phase`` 和总子力 ``total_mat``。
               同时统计双方的攻击棋子数和各兵种计数。

            3. **和棋判定**：若双方均无攻击棋子（车/马/炮/兵全灭），
               直接返回 0（无法将杀）。

            4. **战术协同加分**：调用 ``_tactical_synergy`` 为双方各自
               计算纵线车、炮架、高位马、车马协同等战术分。

            5. **阵型惩罚/奖励**：
               - 缺象 + 对方有炮 → 对方加 30 分（炮打空头效率高）。
               - 缺士 + 对方双车 → 对方加 50 分（双车杀缺士效率极高）。

            6. **Tapered 插值**：
               ``score = (mg × phase + eg × (16 − phase)) / 16``。
               阶段因子 phase 越大，越偏向中局分；phase 越小，越偏向残局分。

            7. **兑子惩罚**：若子力差 > 50 且总子力低于参考值，对领先方
               施加惩罚。原因是领先方应保持子力优势，避免无谓兑子。

            8. **将军奖励**：走子方正在将军对方时加固定偏置。

            9. **Negamax 转换**：始终以当前走子方为正返回评估分。

            10. **缓存写入**：结果存入置换表，超过 20 万条时清空。

        Args:
            board: 当前棋盘局面。

        Returns:
            以当前走子方为正的评估分（浮点数）。正值表示当前走子方
            局面占优，负值表示劣势，0 表示均势或和棋。
        """
        # ── 步骤 1：置换表缓存查询 ──
        h = board.zobrist_hash
        if h in Evaluation._eval_cache:
            return Evaluation._eval_cache[h]

        b = board.board
        rk = board.red_king_pos
        bk = board.black_king_pos
        red_kr = rk[0] if rk else None
        red_kc = rk[1] if rk else None
        black_kr = bk[0] if bk else None
        black_kc = bk[1] if bk else None

        # 缓存类属性的局部引用，减少重复的属性查找开销
        attacking = Evaluation.ATTACKING_PIECE_TYPES
        mg_map = Evaluation.PST_MG_MAP
        eg_map = Evaluation.PST_EG_MAP
        mgv = Evaluation.MG_VALUES
        egv = Evaluation.EG_VALUES
        pw = Evaluation.PHASE_WEIGHTS

        # 双方攻击棋子计数
        red_attack = black_attack = 0
        # 双方中局/残局评估分累加器
        red_mg = red_eg = black_mg = black_eg = 0.0
        # 游戏阶段因子和总子力
        phase = 0.0
        total_mat = 0.0
        # 双方各兵种计数（用于阵型惩罚判定）
        red_shi = red_xiang = red_che = red_pao = 0
        black_shi = black_xiang = black_che = black_pao = 0

        # ── 步骤 2：遍历双方所有活跃棋子 ──
        for color_key in ("red", "black"):
            for r, c in board.active_pieces.get(color_key, ()):
                p = b[r][c]
                if p is None or p.color != color_key:
                    continue
                pt = p.piece_type
                # 统计攻击棋子数
                if pt in attacking:
                    if color_key == "red":
                        red_attack += 1
                    else:
                        black_attack += 1
                # 统计各兵种数量
                if pt == "shi":
                    if color_key == "red":
                        red_shi += 1
                    else:
                        black_shi += 1
                elif pt == "xiang":
                    if color_key == "red":
                        red_xiang += 1
                    else:
                        black_xiang += 1
                elif pt == "che":
                    if color_key == "red":
                        red_che += 1
                    else:
                        black_che += 1
                elif pt == "pao":
                    if color_key == "red":
                        red_pao += 1
                    else:
                        black_pao += 1

                # 累加阶段因子和总子力
                phase += float(pw.get(pt, 0))
                total_mat += float(mgv.get(pt, 0))

                # 计算该棋子的中局/残局综合得分（基础分 + PST + 机动奖励）
                mg_base = float(mgv.get(pt, 0))
                eg_base = float(egv.get(pt, 0))
                # PST 行翻转：红方直接用行号，黑方翻转（9 - r）以共用同一张表
                pst_r = r if color_key == "red" else 9 - r
                mg_tbl = mg_map.get(pt, Evaluation.PST_DEFAULT)
                eg_tbl = eg_map.get(pt, Evaluation.PST_DEFAULT)
                mg_pst = float(mg_tbl[pst_r][c])
                eg_pst = float(eg_tbl[pst_r][c])

                # 特定棋子的额外机动性奖励
                bonus = 0.0
                if pt == "ma":
                    bonus = Evaluation._ma_mobility(board, r, c, color_key)
                elif pt == "pao":
                    bonus = Evaluation._pao_screen_bonus(board, r, c, color_key)

                # 累加到对应方的中局/残局评估分
                acc_mg = mg_base + mg_pst + bonus
                acc_eg = eg_base + eg_pst + bonus
                if color_key == "red":
                    red_mg += acc_mg
                    red_eg += acc_eg
                else:
                    black_mg += acc_mg
                    black_eg += acc_eg

        # ── 步骤 3：和棋判定——双方均无攻击棋子 ──
        if red_attack == 0 and black_attack == 0:
            Evaluation._eval_cache[h] = 0.0
            return 0.0

        # ── 步骤 4：战术协同加分 ──
        tact_r = Evaluation._tactical_synergy(board, "red", b, black_kr, black_kc)
        tact_b = Evaluation._tactical_synergy(board, "black", b, red_kr, red_kc)
        # 战术分同时加在 MG 和 EG 轨道，确保不因阶段因子而被削弱
        red_mg += tact_r
        red_eg += tact_r
        black_mg += tact_b
        black_eg += tact_b

        # ── 步骤 5：阵型惩罚/奖励 ──
        # 缺象时对方炮威力倍增（炮打空头效率高，象不全则九宫缺乏屏障）
        if red_xiang < 2 and black_pao > 0:
            black_mg += 30.0
            black_eg += 30.0
        if black_xiang < 2 and red_pao > 0:
            red_mg += 30.0
            red_eg += 30.0
        # 缺士面对双车时极度危险（双车错杀、铁门栓等杀法对缺士方几乎无解）
        if red_shi < 2 and black_che == 2:
            black_mg += 50.0
            black_eg += 50.0
        if black_shi < 2 and red_che == 2:
            red_mg += 50.0
            red_eg += 50.0

        # ── 步骤 6：Tapered 线性插值 ──
        # phase 越大（重子越多）越偏向中局分，phase 越小越偏向残局分
        phase = min(phase, Evaluation.TOTAL_PHASE)
        tp = Evaluation.TOTAL_PHASE
        red_score = (red_mg * phase + red_eg * (tp - phase)) / tp
        black_score = (black_mg * phase + black_eg * (tp - phase)) / tp

        # ── 步骤 7：兑子惩罚 ──
        # 领先方在子力减少时受惩罚，迫使其避免无意义兑子以保持优势
        diff = red_score - black_score
        deficit = max(0.0, Evaluation._REF_TOTAL_MATERIAL - total_mat)
        pen = Evaluation._ANTI_TRADE_COEFF * deficit
        if diff > 50:
            red_score -= pen
        elif diff < -50:
            black_score -= pen

        # ── 步骤 8：将军奖励 ──
        opp = "black" if board.current_player == "red" else "red"
        check_bonus = (
            Evaluation.CHECK_EVAL_BONUS if Rules.is_king_in_check(board, opp) else 0.0
        )
        # ── 步骤 9：Negamax 转换——始终以当前走子方为正 ──
        if board.current_player == "red":
            res = red_score - black_score + check_bonus
        else:
            res = black_score - red_score + check_bonus
        # ── 步骤 10：缓存写入（超过 20 万条时清空，防止内存膨胀） ──
        if len(Evaluation._eval_cache) > 200_000:
            Evaluation._eval_cache.clear()
        Evaluation._eval_cache[h] = res
        return res
