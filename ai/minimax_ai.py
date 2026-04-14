"""Minimax AI 搜索引擎（Negamax + Alpha-Beta 剪枝）。

本模块实现了一个面向中国象棋的对抗搜索引擎，以 Negamax 框架为核心，
通过 Alpha-Beta 剪枝大幅削减搜索树规模。在此基础上集成了多项工业级
优化技术以在有限时间内获得更深、更精准的搜索：

    - **迭代加深（Iterative Deepening）**：从浅层到目标深度逐层搜索，
      保证任何时刻都有可用着法，同时为期望窗口和走法排序提供先验信息。
    - **PVS（Principal Variation Search）**：对排序后的首着使用全窗口
      搜索，后续着法用零窗口快速验证；若零窗口 fail-high 则回退全窗口
      重搜。在排序准确时可节省大量节点。
    - **期望窗口（Aspiration Windows）**：利用上一迭代层的分数设置
      窄窗口，加速剪枝；若发生 fail-low/fail-high 则逐侧放宽重搜。
    - **置换表（Transposition Table）**：以 Zobrist 哈希为键缓存已搜索
      局面的深度、分数与最佳着法，避免对同一局面的重复搜索。
    - **Null Move Pruning（空步剪枝）**：假设"跳过一手仍 fail-high"
      则当前节点几乎必然 fail-high，直接剪枝；受限于深水区且检查大子
      充足度以规避残局 Zugzwang 风险。
    - **杀手走法（Killer Moves）**：记录同一深度层上引发 Beta 截断的
      非吃子着法，在后续兄弟节点中优先尝试以提高剪枝概率。
    - **历史启发（History Heuristic）**：按 depth² 权重累加引发截断的
      着法分数，跨节点积累经验，辅助排序。
    - **将军延伸（Check Extension）**：被将军时不消耗深度余量，确保不
      遗漏战术杀招；每条根分支设延伸次数上限以防指数膨胀。
    - **静止搜索（Quiescence Search）**：主搜索到达叶节点后，继续搜索
      吃子序列直至局面"安静"，缓解水平线效应；内含 Delta Pruning 以
      提前裁剪无望的吃子分支。
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

from engine.rules import MoveEntry, Rules

from ai.evaluation import Evaluation
from ai.search_move_helpers import (
    MoveGivesCheckCache,
    PostApplyFlagsCache,
    apply_pseudo_legal_with_rule_cache,
)
from ai.opening_book import OPENING_BOOK, mirror_move

# ==================== 置换表边界类型常量 ====================
# 置换表中需要记录该分数与搜索窗口 [alpha, beta] 的关系，
# 以便后续查询时判断缓存值能否直接使用。
_TT_EXACT = 0  # 精确值：搜索时 alpha < score < beta，可直接采信
_TT_LOWER = 1  # 下界（fail-high）：score >= beta，实际值可能更高
_TT_UPPER = 2  # 上界（fail-low）：score <= alpha，实际值可能更低

# ==================== 走法排序权重常量 ====================
# 吃子走法与普通走法之间需要拉开明显差距，使得即使最低价值的吃子
# 在排序中仍优先于最高分的非吃子走法，从而保证 Alpha-Beta 剪枝
# 优先搜索战术性强的吃子序列。
_CAPTURE_SORT_BIAS = 10000

# 杀手走法排序权重：
# 设计在吃子基线 (10000) 以下、普通走子 (0) 以上的区间内，
# 保证杀手走法在所有非吃子着法中享有最高优先级。
_KILLER_PRIMARY_BONUS = 5000    # 主杀手（slot 0）
_KILLER_SECONDARY_BONUS = 4000  # 次杀手（slot 1）

# 置换表着法排序分：必须远高于历史+吃子+杀手三者之和，
# 确保 TT 中记录的最佳着法始终被第一个搜索。
_TT_MOVE_SORT_SCORE = 1_000_000

# 历史分量排序上限：防止历史表在深度较大时累计过高压过 TT 着法排序分。
_HISTORY_SORT_CAP = 500_000

# ==================== 搜索控制常量 ====================
# 杀手表按「当前 _alphabeta 的剩余深度 depth」索引，
# 数组大小与预期最大搜索深度对齐即可。
MAX_KILLER_DEPTH = 10

# 将军延伸上限：每条根分支上最多额外延伸的层数。
# 若不设上限，反复将军会导致搜索树指数爆炸。
_MAX_CHECK_EXTENSIONS = 2

# 历史表老化阈值：当条目总数超过此值时，在新一轮迭代加深开始前
# 对所有条目做半值衰减，淘汰过时信息、控制内存增长。
_HISTORY_AGING_THRESHOLD = 10_000

# ==================== 静止搜索常量 ====================
# Delta Pruning 裁剪阈值：约等于一车的价值（900 + 安全余量）。
# 若 stand_pat + DELTA_MARGIN 仍低于 alpha，说明即使吃到最值钱的子也
# 无法改善局面，可提前返回以节省搜索开销。
_QS_DELTA_MARGIN = 1100

# ==================== 将杀评估常量 ====================
# 被将死时的基础负分。加上深度偏移量后可区分远近杀棋——
# 越早将死得分越高，引导引擎选择最短杀棋路径。
_MATE_BASE_SCORE = 10000

# ==================== 根节点随机化常量 ====================
# 当最优分数的绝对值超过此阈值时，认为局面已有明显优劣，
# 关闭根节点随机化以确保搜索确定性。
_RANDOM_SCORE_THRESHOLD = 300

# ==================== 期望窗口常量 ====================
# 初始窗口宽度约一兵 / 半马量级；过窄则频繁重搜，过宽则失去加速效果
_ASPIRATION_WINDOW = 100.0


class SearchTimeoutException(Exception):
    """搜索超时异常。

    在迭代加深搜索过程中，当已用时间超过预设的 time_limit 时抛出，
    由上层 get_best_move 捕获并回退到上一完整迭代层的最佳着法。
    这一机制保证了即使在严格时限下也总能返回合法着法。
    """


class MinimaxAI:
    """基于 Negamax + Alpha-Beta 剪枝的中国象棋搜索引擎。

    本类封装了完整的对抗搜索流程：开局库查询 → 迭代加深 →
    Alpha-Beta 主搜索 → 静止搜索，以及置换表、杀手走法、
    历史启发等辅助数据结构的生命周期管理。

    Attributes:
        depth: 迭代加深的目标深度上限。
        stochastic: 是否在根节点对近最优着法做随机化选择。
        top_k: 随机化时从得分最高的前 k 个着法中选取。
        tolerance: 随机化容差阈值（分），得分在最优分 ± tolerance
            范围内的着法均有资格被选中。
        verbose: 是否输出搜索统计信息到标准输出。
        last_stats: 最近一次搜索的统计摘要（深度、耗时、节点数等），
            供 GUI 仪表盘或日志系统读取。
        killer_moves: 杀手走法表，按剩余深度索引，每层保存 2 个。
        transposition_table: 置换表（dict 实现），键为 Zobrist 哈希。
        history_table: 历史启发表，键为着法四元组，值为累加权重。
        history_hashes: 当前搜索路径上的局面 Zobrist 哈希序列，
            用于路径内重复局面检测。
    """

    def __init__(
        self,
        depth=5,
        stochastic: bool = False,
        top_k: int = 2,
        tolerance: int = 5,
        verbose: bool = True,
    ):
        """初始化搜索引擎实例。

        Args:
            depth: 迭代加深的目标深度上限，值越大搜索越深但耗时指数增长。
            stochastic: 若为 True，则在根节点对得分接近的着法做随机选择，
                用于打破纯确定性开局、增加棋力多样性。
            top_k: 随机化模式下，从排序后的前 top_k 个近最优着法中随机选取。
            tolerance: 随机化容差（单位：评估分）。得分在 best_score - tolerance
                范围内的着法均纳入随机池。
            verbose: 若为 True，在每次搜索完成后输出深度、耗时、节点数等统计。
        """
        self.depth = depth
        # 根节点随机化配置：在「近最优」走法中随机选择，打破纯确定性开局
        self.stochastic = stochastic
        self.top_k = top_k
        self.tolerance = tolerance
        self.verbose = verbose
        self._nodes = 0
        self._tt_hits = 0
        # 供 GUI Dashboard / 日志读取的最近一次搜索统计
        self.last_stats: Dict[str, Any] = {}
        # 无头基准模式下：跨多局累加本实例每次 get_best_move 的耗时与节点
        # （由 reset_benchmark_stats 清零）
        self._bench_total_time: float = 0.0
        self._bench_total_nodes: int = 0
        self._bench_search_count: int = 0
        # 杀手走法表：每层深度最多保存 2 个杀手走法。
        # 新杀手写入 slot[0]，原 slot[0] 下沉到 slot[1]（位移覆盖策略）。
        self.killer_moves: List[List[Optional[Tuple[int, int, int, int]]]] = [
            [None, None] for _ in range(MAX_KILLER_DEPTH)
        ]
        # 普通 dict 实现的置换表；值格式为 (stored_depth, score, flag, best_move)
        self.transposition_table: Dict[
            int, Tuple[int, float, int, Optional[Tuple[int, int, int, int]]]
        ] = {}
        # 搜索期间使用的上下文变量（避免每层函数调用层层传递）
        self.start_time: float = 0.0
        # 重复局面检测：保存"从根到当前节点"的 Zobrist 哈希路径，
        # 可叠加外部传入的 game_history 以覆盖整局历史。
        self.history_hashes: List[int] = []
        # 与 history_hashes 同步的 MoveEntry 栈，供「无获益纯长将」叶评估使用
        self._move_history_stack: List[MoveEntry] = []
        # 历史启发表：Beta 截断着法按 depth² 加权累加，跨着法积累经验。
        # 与置换表不同，此表不在每步根搜索清空，而是通过老化机制衰减。
        self.history_table: Dict[Tuple[int, int, int, int], int] = {}

    def _current_tolerance(self, is_midgame: bool, best_score: float) -> int:
        """根据局面阶段和分差计算当前根节点随机化容差。

        中局或分差明显时关闭随机化（返回 0），开局平稳期允许
        小范围随机以增加棋路多样性。

        Args:
            is_midgame: 是否已进入中局阶段（手数 > 20）。
            best_score: 当前搜索的最优分数。

        Returns:
            容差值：0 表示确定性搜索，正值表示随机化范围。
        """
        if is_midgame:
            return 0
        if best_score > float("-inf") and abs(best_score) >= _RANDOM_SCORE_THRESHOLD:
            return 0
        return self.tolerance

    def reset_benchmark_stats(self) -> None:
        """清零基准统计计数器。

        在无头基准测试模式下，每局对弈开始前调用此方法，将累计耗时、
        累计节点数和搜索次数归零，以便精确统计本局的搜索性能。
        """
        self._bench_total_time = 0.0
        self._bench_total_nodes = 0
        self._bench_search_count = 0

    def _tt_probe(self, key: int, depth: int, alpha: float, beta: float) -> Optional[float]:
        """查询置换表，尝试直接获取已缓存的搜索结果。

        置换表命中需同时满足两个条件：
        1. 存储深度 >= 当前搜索剩余深度（浅层搜索结果对深层不可信）；
        2. 存储的边界类型与当前搜索窗口 [alpha, beta] 匹配：
           - EXACT：精确值，直接返回。
           - LOWER（下界）：score >= beta 时可触发截断。
           - UPPER（上界）：score <= alpha 时可确认 fail-low。

        Args:
            key: 局面 Zobrist 哈希值，作为置换表的索引键。
            depth: 当前剩余搜索深度，仅 stored_depth >= depth 的条目才可信。
            alpha: 当前搜索窗口下界。
            beta: 当前搜索窗口上界。

        Returns:
            命中且满足窗口条件时返回缓存分数；否则返回 None，需继续搜索。
        """
        entry = self.transposition_table.get(key)
        if entry is None:
            return None
        stored_depth, score, flag, _best_move = entry
        if stored_depth < depth:
            return None
        if flag == _TT_EXACT:
            return score
        if flag == _TT_LOWER and score >= beta:
            return score
        if flag == _TT_UPPER and score <= alpha:
            return score
        return None

    def _tt_store(
        self,
        key: int,
        depth: int,
        score: float,
        alpha_orig: float,
        beta_orig: float,
        best_move: Optional[Tuple[int, int, int, int]],
    ) -> None:
        """将搜索结果写入置换表，自动推断边界类型标志位。

        根据最终分数与搜索开始时的原始窗口 [alpha_orig, beta_orig] 的关系，
        推断出该分数属于精确值、下界还是上界：
        - score <= alpha_orig → UPPER（搜索未能提升下界，实际值可能更低）
        - score >= beta_orig  → LOWER（触发截断，实际值可能更高）
        - 其他情况           → EXACT（落在窗口内的精确值）

        注意使用 alpha_orig / beta_orig（搜索入口时的快照）而非搜索过程中
        不断收窄的 alpha/beta，以正确反映该分数对于原始窗口的意义。

        Args:
            key: 局面 Zobrist 哈希值。
            depth: 搜索剩余深度。
            score: 搜索返回的评估分数。
            alpha_orig: 搜索入口时的原始 alpha 值。
            beta_orig: 搜索入口时的原始 beta 值。
            best_move: 搜索过程中找到的最佳着法；截断或叶节点时可能为 None。
        """
        if score <= alpha_orig and alpha_orig != float("-inf"):
            flag = _TT_UPPER
        elif score >= beta_orig and beta_orig != float("inf"):
            flag = _TT_LOWER
        else:
            flag = _TT_EXACT
        self._tt_write_entry(key, depth, score, flag, best_move)

    def _tt_store_exact(
        self,
        key: int,
        depth: int,
        score: float,
        best_move: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """将精确评估值写入置换表（固定为 EXACT 标志位）。

        适用于叶节点评估、根节点显式记录等场景——这些分数不依赖于搜索窗口，
        始终为精确值，无需通过窗口关系推断边界类型。

        Args:
            key: 局面 Zobrist 哈希值。
            depth: 搜索剩余深度。
            score: 精确评估分数。
            best_move: 对应的最佳着法（可选）。
        """
        self._tt_write_entry(key, depth, score, _TT_EXACT, best_move)

    _TT_MAX_SIZE = 200_000

    def _tt_write_entry(
        self,
        key: int,
        depth: int,
        score: float,
        flag: int,
        best_move: Optional[Tuple[int, int, int, int]],
    ) -> None:
        """置换表底层写入方法，附带容量保护。

        当置换表条目数超过 _TT_MAX_SIZE 时，采用全清策略释放内存。
        相比 LRU/深度替换等精细策略，全清实现最简且在 Python dict
        场景下性能开销最低，对搜索精度影响有限。

        Args:
            key: 局面 Zobrist 哈希值。
            depth: 搜索剩余深度。
            score: 评估分数。
            flag: 边界类型标志位（_TT_EXACT / _TT_LOWER / _TT_UPPER）。
            best_move: 最佳着法（可选）。
        """
        if len(self.transposition_table) > self._TT_MAX_SIZE:
            self.transposition_table.clear()
        self.transposition_table[key] = (depth, score, flag, best_move)

    @staticmethod
    def _killer_index(depth: int) -> int:
        """将搜索深度映射为杀手表的有效索引。

        将负深度钳位到 0，超过 MAX_KILLER_DEPTH 的深度钳位到最大索引，
        保证数组访问始终安全。

        Args:
            depth: 当前搜索剩余深度（可能为负值，如 QS 阶段）。

        Returns:
            杀手走法表的有效索引，范围 [0, MAX_KILLER_DEPTH - 1]。
        """
        if depth < 0:
            return 0
        return min(depth, MAX_KILLER_DEPTH - 1)

    def _reset_killers(self) -> None:
        """清空所有深度层的杀手走法表。

        在每次根搜索开始时调用，避免上一步残留的杀手走法干扰
        新局面的搜索排序。
        """
        for slot in self.killer_moves:
            slot[0] = None
            slot[1] = None

    def _push_killer(self, depth: int, move: Tuple[int, int, int, int]) -> None:
        """将一个引发 Beta 截断的非吃子着法记录为杀手走法。

        采用二槽位移覆盖策略：新杀手写入 slot[0]，原 slot[0] 下沉至 slot[1]。
        若该着法已存在于任一槽位则跳过，避免重复记录浪费空间。

        之所以只保留非吃子着法，是因为吃子着法已通过 MVV-LVA 获得高排序分，
        无需杀手表额外提权。

        Args:
            depth: 当前搜索剩余深度，用于定位杀手表索引。
            move: 引发截断的着法四元组 (起始行, 起始列, 目标行, 目标列)。
        """
        i = self._killer_index(depth)
        k0 = self.killer_moves[i][0]
        if move == k0 or move == self.killer_moves[i][1]:
            return
        self.killer_moves[i][1] = k0
        self.killer_moves[i][0] = move

    @staticmethod
    def _is_capture(board, move: Tuple[int, int, int, int]) -> bool:
        """判断给定着法是否为吃子走法。

        Args:
            board: 当前棋盘对象。
            move: 着法四元组 (起始行, 起始列, 目标行, 目标列)。

        Returns:
            若目标格上存在棋子则为 True（吃子），否则为 False。
        """
        _, _, er, ec = move
        return board.get_piece(er, ec) is not None

    _MAJOR_PIECE_TYPES = frozenset({"che", "ma", "pao"})

    def has_enough_material(self, board, player: str) -> bool:
        """检查指定方是否拥有足够的大子（车/马/炮）。

        此方法用于 Null Move Pruning 的前置条件判断：在残局中，若己方
        仅剩将+兵/士/象等小子，跳过一手的假设可能导致 Zugzwang
        （被迫走坏棋）误判。仅在大子数 >= 2 时才认为空步假设安全。

        Args:
            board: 当前棋盘对象。
            player: 待检查的玩家颜色标识（"red" 或 "black"）。

        Returns:
            若该方车/马/炮数量 >= 2 则返回 True，否则返回 False。
        """
        n = 0
        b = board.board
        for r, c in board.active_pieces.get(player, ()):
            p = b[r][c]
            if p is None or p.color != player:
                continue
            if p.piece_type in self._MAJOR_PIECE_TYPES:
                n += 1
                if n >= 2:
                    return True
        return False

    def order_moves(
        self, board, moves: List[Tuple[int, int, int, int]], depth: int
    ) -> None:
        """就地排序走法列表，以最大化 Alpha-Beta 剪枝效率。

        良好的走法排序是 Alpha-Beta 搜索性能的核心。排序越接近"最优着法
        在前"，剪枝越多、搜索越快。本方法综合以下信息源按降序排列：

        1. **TT 最佳着法**（排序分 1,000,000）：置换表中记录的该局面最佳着法，
           来自更浅或相同深度的搜索结果，几乎总是当前最优候选。
        2. **MVV-LVA 吃子分 + 历史启发**（排序分 ~10,000+）：
           吃子着法按"被吃子价值 - 吃子方价值"（Most Valuable Victim,
           Least Valuable Attacker）排序，优先搜索高性价比的吃子。
           历史启发分作为附加权重叠加。
        3. **杀手走法**（排序分 4,000~5,000）：同深度层曾引发 Beta 截断的
           非吃子着法，经验性地优先尝试。
        4. **其余着法**（排序分 ~0）：按历史启发分微调顺序。

        历史分量设有上限 _HISTORY_SORT_CAP = 500,000，确保永远低于 TT 着法。

        Args:
            board: 当前棋盘，用于读取目标格棋子以计算 MVV-LVA 分。
            moves: 待排序的走法列表（原地修改，不返回新列表）。
            depth: 当前剩余搜索深度，用于索引杀手走法表。
        """
        pv = Evaluation.PIECE_VALUES
        ki = self._killer_index(depth)
        killers = self.killer_moves[ki]
        k0, k1 = killers[0], killers[1]
        entry = self.transposition_table.get(board.zobrist_hash)
        tt_move = entry[3] if entry is not None else None

        def move_score(m: Tuple[int, int, int, int]) -> int:
            # TT 最佳着法享有绝对最高优先级
            if tt_move is not None and m == tt_move:
                return _TT_MOVE_SORT_SCORE
            hist = min(self.history_table.get(m, 0), _HISTORY_SORT_CAP)
            sr, sc, er, ec = m
            victim = board.get_piece(er, ec)
            if victim is None:
                cap = 0
            else:
                # MVV-LVA：被吃子价值越高、吃子方价值越低，排序分越高
                attacker = board.get_piece(sr, sc)
                victim_value = int(pv.get(victim.piece_type, 0))
                attacker_value = int(pv.get(attacker.piece_type, 0)) if attacker else 0
                cap = _CAPTURE_SORT_BIAS + victim_value - attacker_value
            score = hist + cap
            # 杀手走法加成：仅对非吃子着法有实际意义（吃子分已足够高）
            if m == k0:
                score += _KILLER_PRIMARY_BONUS
            elif m == k1:
                score += _KILLER_SECONDARY_BONUS
            return score

        moves.sort(key=move_score, reverse=True)

    def choose_move(
        self,
        board,
        time_limit: Optional[float] = 10.0,
        game_history: Optional[List[int]] = None,
        move_history: Optional[List[MoveEntry]] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """外部搜索接口：为当前行棋方选择一步最佳着法。

        此方法是适配 Searcher 统一接口的入口点，内部直接委托
        给 get_best_move 执行实际搜索。

        Args:
            board: 当前棋盘对象。
            time_limit: 搜索时限（秒），传递给 get_best_move。
            game_history: 从开局至今的 Zobrist 哈希序列，用于路径重复检测。
            move_history: 可选；控制器 ``MoveEntry`` 链，与长将判负叶评估对齐。

        Returns:
            最佳着法四元组 (起始行, 起始列, 目标行, 目标列)；
            无合法着法时返回 None。
        """
        return self.get_best_move(
            board,
            game_history=game_history,
            move_history=move_history,
            time_limit=time_limit,
        )

    def get_best_move(
        self,
        board,
        game_history: Optional[List[int]] = None,
        time_limit: Optional[float] = 10.0,
        move_history: Optional[List[MoveEntry]] = None,
    ):
        """迭代加深搜索主入口：从深度 1 到 self.depth 逐层搜索，选择最佳着法。

        搜索流程概述：
        1. **开局库查询**：若当前局面（或其镜像）命中开局库则直接出棋，
           跳过搜索以节省时间并保证开局质量。
        2. **迭代加深**：从深度 1 开始逐层增加搜索深度。每层搜索均使用
           Negamax + Alpha-Beta + PVS 框架，并在深度 >= 3 时启用
           期望窗口以利用上一层分数加速剪枝。
        3. **期望窗口重搜**：若某层搜索结果 fail-low 或 fail-high（超出
           窄窗口范围），则将对应侧窗口放宽至无穷并重新搜索该层。
        4. **根节点随机化**（可选）：在 stochastic 模式下，从得分接近
           最优的 top_k 个着法中随机选取，增加棋路多样性。

        time_limit 参数仅为接口兼容性保留，本实现中根迭代加深路径上
        不启用超时控制（由固定深度决定搜索量）。

        Args:
            board: 当前棋盘对象。
            game_history: 从开局至今的 Zobrist 哈希序列（可选），
                用于扩展路径内重复局面检测的覆盖范围。
            move_history: 可选；完整 ``MoveEntry`` 列表（须与当前 ``board`` 一致），
                用于与终局规则一致的「无获益纯长将」叶节点评分。
            time_limit: 搜索时限（秒），None 表示不限时。

        Returns:
            最佳着法四元组 (起始行, 起始列, 目标行, 目标列)；
            无合法着法时返回 None（被将死或困毙）。
        """
        Evaluation._eval_cache.clear()
        hash_history: List[int] = [] if game_history is None else list(game_history)
        if len(hash_history) < 30:
            zkey = board.zobrist_hash
            book_moves = OPENING_BOOK.get(zkey)
            if book_moves is None:
                # 尝试镜像局面查找开局库：对称开局只需存储一侧
                zm = board.column_mirror_copy().zobrist_hash
                alt = OPENING_BOOK.get(zm)
                if alt is not None:
                    book_moves = [mirror_move(m) for m in alt]
            if book_moves is None and len(hash_history) == 0:
                keys = list(OPENING_BOOK.keys())
                disp = keys if len(keys) <= 48 else keys[:24] + ["..."] + keys[-16:]
                print(
                    f"[Minimax 开局库] 根局面未命中（局面键 zkey={zkey:#x}）；"
                    f"开局库共 {len(keys)} 个局面键，示例: {disp}"
                )
            if book_moves:
                valid = [
                    m
                    for m in book_moves
                    if Rules.is_valid_move(board, m[0], m[1], m[2], m[3])[0]
                ]
                if valid:
                    picked = random.choice(valid)
                    if self.verbose:
                        print(f"命中开局库！瞬间出棋: {picked}")
                    self.last_stats = {
                        "depth": int(self.depth),
                        "time_taken": 0.0,
                        "nodes_evaluated": 0,
                        "tt_hits": 0,
                        "opening_book": True,
                    }
                    return picked

        bench_t0 = time.time()
        self._nodes = 0
        self._tt_hits = 0
        # 每步根搜索清空置换表：跨回合复用同键曾导致子树全命中、
        # nodes=0 与着法异常等难以调试的 bug
        self.transposition_table.clear()
        self._reset_killers()
        # 本步根搜索：走子前 (hash,move) + 走后 hash 双 LRU，压低 Rules 调用与重复 apply/undo
        self._post_apply_flags_cache = PostApplyFlagsCache(65536)
        self._pre_move_flags_cache = MoveGivesCheckCache(
            131072, post_apply_cache=self._post_apply_flags_cache
        )

        # 外部可传入从开局到当前局面的完整哈希链，
        # 若末尾已是当前局面则不再重复追加
        self.history_hashes = list(hash_history)
        if not self.history_hashes or self.history_hashes[-1] != board.zobrist_hash:
            self.history_hashes.append(board.zobrist_hash)

        if move_history is not None and (
            len(move_history) > 0 and move_history[-1].pos_hash == board.zobrist_hash
        ):
            self._move_history_stack = list(move_history)
        else:
            self._move_history_stack = [MoveEntry(pos_hash=board.zobrist_hash)]

        global_best_move: Optional[Tuple[int, int, int, int]] = None
        gh_len = len(hash_history)
        # 中局阶段（手数 > 20）关闭根节点随机化，切换为纯确定性搜索
        is_midgame = gh_len > 20

        # ---- 期望窗口（Aspiration Windows）配置 ----
        window_size = _ASPIRATION_WINDOW
        alpha_full = float("-inf")
        beta_full = float("inf")
        previous_score: Optional[float] = None

        for current_depth in range(1, self.depth + 1):
            # 历史表老化：当条目过多时做半值衰减，淘汰远古经验以腾出排序权重空间
            if len(self.history_table) > _HISTORY_AGING_THRESHOLD:
                for k in list(self.history_table.keys()):
                    self.history_table[k] = self.history_table[k] // 2
                self.history_table = {k: v for k, v in self.history_table.items() if v > 0}

            moves = list(Rules.get_pseudo_legal_moves(board, board.current_player))
            self.order_moves(board, moves, current_depth)

            # 深度 >= 3 且有上一层分数时启用窄窗口；浅层搜索开销极低，无需加速
            if current_depth >= 3 and previous_score is not None:
                asp_alpha = previous_score - window_size
                asp_beta = previous_score + window_size
            else:
                asp_alpha = alpha_full
                asp_beta = beta_full

            # 期望窗口主循环：若当前层最优分数超出窗口，则放宽对应侧后整层重搜
            while True:
                alpha = asp_alpha
                beta = asp_beta
                best_score_this_depth = float("-inf")
                best_move_this_depth: Optional[Tuple[int, int, int, int]] = None
                scored_moves = []
                best_score_so_far = float("-inf")
                has_legal_move = False

                for move in moves:
                    mover = board.current_player
                    applied = apply_pseudo_legal_with_rule_cache(
                        board,
                        move,
                        mover,
                        pre_move_cache=self._pre_move_flags_cache,
                        post_apply_cache=self._post_apply_flags_cache,
                    )
                    if applied is None:
                        continue
                    captured, _ = applied
                    self.history_hashes.append(board.zobrist_hash)
                    opp = board.current_player
                    self._move_history_stack.append(
                        MoveEntry(
                            pos_hash=board.zobrist_hash,
                            mover=mover,
                            gave_check=Rules.is_king_in_check(board, opp),
                            last_move=move,
                        )
                    )

                    has_legal_move = True
                    # 搜索下界快照：与容差/记录用的「业务 alpha」解耦，
                    # 避免与 PVS 根窗口互相污染
                    search_alpha = alpha
                    try:
                        score = -self._alphabeta(
                            board,
                            current_depth - 1,
                            -beta,
                            -search_alpha,
                            0.0,
                            None,
                            check_ext_left=_MAX_CHECK_EXTENSIONS,
                        )
                    finally:
                        self.history_hashes.pop()
                        self._move_history_stack.pop()
                        board.undo_move(*move, captured)

                    if score > best_score_this_depth:
                        best_score_this_depth = score
                        best_move_this_depth = move

                    # Fail-Low / 窗口剪枝返回的界值不可信为真实分数，
                    # 标记为 -inf 以禁止其进入随机选择池
                    actual_record_score = score
                    if score <= search_alpha:
                        actual_record_score = float("-inf")

                    scored_moves.append((actual_record_score, move))

                    if actual_record_score > best_score_so_far:
                        best_score_so_far = actual_record_score

                    # 动态容差策略：中局或分差明显时关闭随机化（current_tol=0），
                    # 开局平稳期允许小范围随机以增加棋路多样性
                    current_tol = self._current_tolerance(is_midgame, best_score_so_far)
                    new_alpha = best_score_so_far - current_tol
                    if new_alpha > alpha:
                        alpha = new_alpha

                    if score > alpha:
                        alpha = score

                if not has_legal_move:
                    break

                # 期望窗口 fail-low：最优分数落在窗口下方，说明窄窗口下界太高，
                # 放宽 alpha 侧至负无穷后重新搜索本层
                if best_score_this_depth <= asp_alpha:
                    asp_alpha = float("-inf")
                    continue
                # 期望窗口 fail-high：最优分数落在窗口上方，说明窄窗口上界太低，
                # 放宽 beta 侧至正无穷后重新搜索本层
                if best_score_this_depth >= asp_beta:
                    asp_beta = float("inf")
                    continue
                break

            if not has_legal_move:
                break  # 根节点无合法着法：被将死或困毙

            # 确定本层的最终容差设置
            current_tol = self._current_tolerance(is_midgame, best_score_so_far)

            # 过滤掉 fail-low 标记为 -inf 的无效分数
            finite_scored = [(s, m) for s, m in scored_moves if s > float("-inf")]
            if not finite_scored:
                break

            if not self.stochastic:
                # 确定性模式：直接取最高分着法
                finite_scored.sort(key=lambda x: x[0], reverse=True)
                max_score = finite_scored[0][0]
                current_best_move = finite_scored[0][1]
                root_tt_score = max_score
            else:
                # 随机化模式：从得分在 best - tolerance 范围内的着法中随机选取
                near_best = [
                    (s, m)
                    for s, m in scored_moves
                    if s > float("-inf") and s >= best_score_so_far - current_tol
                ]
                near_best.sort(key=lambda x: x[0], reverse=True)
                pool = near_best[: self.top_k]
                picked = random.choice(pool)
                current_best_move = picked[1]
                root_tt_score = picked[0]

            global_best_move = current_best_move
            # 根节点显式写入置换表：确保即使 depth=1 直接进入 QS 后超时回退，
            # 仍有上一完整迭代层的最佳着法可用
            self._tt_store_exact(
                board.zobrist_hash, current_depth, float(root_tt_score), current_best_move
            )
            previous_score = float(best_score_this_depth)

        elapsed = time.time() - bench_t0
        self.last_stats = {
            "depth": int(self.depth),
            "time_taken": float(elapsed),
            "nodes_evaluated": int(self._nodes),
            "tt_hits": int(self._tt_hits),
        }
        self._bench_total_time += float(elapsed)
        self._bench_total_nodes += int(self._nodes)
        self._bench_search_count += 1
        if self.verbose:
            print(f"本次搜索深度: {self.depth}")
            print(f"搜索耗时 (秒): {elapsed:.3f}")
            print(f"评估的节点总数: {self._nodes} (置换表命中: {self._tt_hits})")
        return global_best_move

    def _is_repeated(self, board, *, skip_rep_count: bool = False) -> Optional[float]:
        """路径重复与长将第三次判负叶分（``Rules.perpetual_check_status``）。

        Args:
            skip_rep_count: 空步剪枝子树为 True 时跳过长将叶判定（虚拟停一手，
                ``_move_history_stack`` 与棋盘不同步）。
        """
        if not skip_rep_count:
            pf = Evaluation.perpetual_forfeit_leaf_score(
                board, self._move_history_stack
            )
            if pf is not None:
                return pf
        if self.history_hashes and board.zobrist_hash in self.history_hashes[:-1]:
            return Evaluation.repetition_leaf_score(board)
        return None

    def _quiescence_search(
        self,
        board,
        alpha: float,
        beta: float,
        depth_limit: int = 4,
        *,
        skip_rep_count: bool = False,
    ) -> float:
        """静止搜索（Quiescence Search, QS）：仅扩展吃子走法，缓解水平线效应。

        水平线效应（Horizon Effect）是固定深度搜索的固有缺陷：主搜索在叶节点
        处可能恰好截断在一个激烈吃子序列的中间，导致评估严重失真。静止搜索
        通过在叶节点之后继续搜索所有吃子着法，直到局面"安静"为止。

        核心机制：
        - **Stand-Pat 评估**：在任何节点处，行棋方都有权选择"不吃"（即直接
          以当前静态评估分作为下界）。若 stand_pat >= beta 则直接截断。
        - **Delta Pruning**：若 stand_pat + 最大可能捕获增益（DELTA_MARGIN）
          仍低于 alpha，说明即使吃到最值钱的子也无法改善局面，提前返回。
        - **深度限制**：防止罕见的无限吃子链导致栈溢出，默认最多递归 4 层。

        Args:
            board: 当前棋盘对象。
            alpha: 搜索窗口下界。
            beta: 搜索窗口上界。
            depth_limit: QS 递归深度上限（防止无限吃子链），默认为 4。
            skip_rep_count: 是否跳过全局重复计数检测（空步剪枝子树使用）。

        Returns:
            当前行棋方视角的评估分数（分越高越好）。
        """
        if len(self._move_history_stack) - 1 >= Rules.MAX_PLIES_AUTODRAW:
            return 0.0
        rep = self._is_repeated(board, skip_rep_count=skip_rep_count)
        if rep is not None:
            return rep

        if depth_limit <= 0:
            self._nodes += 1
            return Evaluation.evaluate(board)

        self._nodes += 1
        # Stand-Pat：行棋方总可以选择不吃子，以当前评估分作为下界
        stand_pat = Evaluation.evaluate(board)
        if stand_pat >= beta:
            return beta

        # Delta Pruning：若即使获得最大捕获增益（约一车 ≈ _QS_DELTA_MARGIN 分）
        # 仍无法超越 alpha，则此节点无望改善局面，直接返回
        if stand_pat + _QS_DELTA_MARGIN <= alpha:
            return alpha

        if stand_pat > alpha:
            alpha = stand_pat

        moves = list(Rules.get_pseudo_legal_moves(board, board.current_player))
        # 仅保留吃子走法——静止搜索的核心假设是非吃子走法不会剧烈改变评估
        captures = [m for m in moves if board.board[m[2]][m[3]] is not None]
        if not captures:
            return alpha

        self.order_moves(board, captures, depth_limit)

        for move in captures:
            mover = board.current_player
            applied = apply_pseudo_legal_with_rule_cache(
                board,
                move,
                mover,
                pre_move_cache=self._pre_move_flags_cache,
                post_apply_cache=self._post_apply_flags_cache,
            )
            if applied is None:
                continue
            captured, _ = applied
            self.history_hashes.append(board.zobrist_hash)
            opp = board.current_player
            self._move_history_stack.append(
                MoveEntry(
                    pos_hash=board.zobrist_hash,
                    mover=mover,
                    gave_check=Rules.is_king_in_check(board, opp),
                    last_move=move,
                )
            )

            try:
                score = -self._quiescence_search(
                    board, -beta, -alpha, depth_limit - 1, skip_rep_count=skip_rep_count
                )
            finally:
                self.history_hashes.pop()
                self._move_history_stack.pop()
                board.undo_move(*move, captured)

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def _alphabeta(
        self,
        board,
        depth: int,
        alpha: float,
        beta: float,
        start_time: float,
        time_limit: Optional[float],
        *,
        allow_null: bool = True,
        use_tt: bool = True,
        check_ext_left: int = _MAX_CHECK_EXTENSIONS,
        skip_rep_count: bool = False,
    ) -> float:
        """Negamax Alpha-Beta 搜索核心函数。

        以 Negamax 框架统一处理双方视角（当前行棋方总是最大化 score），
        集成以下优化技术：

        **置换表（Transposition Table）**
            在搜索开始时查询 TT，若命中且深度和窗口条件满足则直接返回缓存分数，
            避免对同一局面重复展开整棵子树。搜索结束后将结果写回 TT。

        **Null Move Pruning（空步剪枝）**
            核心思想："如果我跳过一手（让对手连走两步）仍然 fail-high，
            那么正常走棋几乎必然也会 fail-high"。以极低的搜索代价（减少 R+1 层）
            探测这一假设，成功则直接剪枝。
            安全性限制：仅在深水区 (depth >= 5)、非将军状态、且己方大子充足时启用，
            并禁止连续两次空步（allow_null=False 递归关闭）。

        **PVS（Principal Variation Search）**
            排序后的首着最可能是主变化（PV），用全窗口 [alpha, beta] 搜索。
            后续着法只用零窗口 [-alpha-1, -alpha] 快速验证"是否真的不优于首着"；
            若零窗口 fail-high（发现确实更优），再用全窗口重搜获取精确分数。
            在走法排序准确时，绝大多数着法在零窗口阶段即被裁剪。

        **将军延伸（Check Extension）**
            当走完一步后对手处于被将军状态时，不消耗搜索深度余量（next_depth = depth），
            确保不会因固定深度限制而遗漏强制性战术杀招。每条根分支设有延伸次数
            上限 check_ext_left 以防止反复将军导致的搜索树指数膨胀。

        **杀手走法 / 历史启发**
            Beta 截断时记录引发截断的着法：非吃子着法存入杀手表（同深度层优先尝试），
            所有着法按 depth² 累加历史启发分（跨节点辅助排序）。

        Args:
            board: 当前棋盘对象。
            depth: 剩余搜索深度。当 depth == 0 时转入静止搜索（QS）。
            alpha: 搜索窗口下界（当前行棋方已知的最低保证分数）。
            beta: 搜索窗口上界（对手能容忍的最高分数）。
            start_time: 搜索启动时间戳（time.time()），用于超时检测。
            time_limit: 超时秒数。为 None 时不限时。
            allow_null: 是否允许空步剪枝。递归调用时设为 False 以防止连续空步。
            use_tt: 是否使用置换表读写。空步子树中禁用以避免缓存污染。
            check_ext_left: 本路径上剩余的将军延伸次数配额。
            skip_rep_count: 是否跳过全局重复计数检测（空步子树使用）。

        Returns:
            当前行棋方视角的评估分数（Negamax 约定：分越高对行棋方越有利）。

        Raises:
            SearchTimeoutException: 搜索超时时抛出，由迭代加深框架捕获后
                回退到上一完整迭代层的结果。
        """
        if len(self._move_history_stack) - 1 >= Rules.MAX_PLIES_AUTODRAW:
            return 0.0
        rep = self._is_repeated(board, skip_rep_count=skip_rep_count)
        if rep is not None:
            return rep
        if time_limit is not None and (time.time() - start_time) > time_limit:
            raise SearchTimeoutException()

        # 保存搜索入口时的原始窗口，用于后续 TT 写入时推断边界类型
        alpha_orig, beta_orig = alpha, beta
        pos_key = board.zobrist_hash
        if use_tt:
            tt_hit = self._tt_probe(pos_key, depth, alpha, beta)
            if tt_hit is not None:
                self._tt_hits += 1
                return tt_hit

        # ---- Null Move Pruning（空步剪枝）----
        # 条件：深水区 (depth >= 5)、非将军状态、己方大子充足。
        # 减少 R+1 层深度（R=2，即减少 3 层），用零窗口 [-beta, -beta+1] 探测。
        # 若空步后仍 fail-high，说明当前局面对行棋方极度有利，正常走棋必然截断。
        player = board.current_player
        if (
            allow_null
            and depth >= 5
            and not Rules.is_king_in_check(board, player)
            and self.has_enough_material(board, player)
        ):
            R = 2
            reduced_depth = max(depth - 1 - R, 0)
            board.toggle_player()
            try:
                null_score = -self._alphabeta(
                    board,
                    reduced_depth,
                    -beta,
                    -beta + 1,
                    start_time,
                    time_limit,
                    allow_null=False,
                    use_tt=False,
                    check_ext_left=check_ext_left,
                    skip_rep_count=True,
                )
            finally:
                board.toggle_player()
            if null_score >= beta:
                return beta

        # 深度耗尽，转入静止搜索以消除水平线效应
        if depth == 0:
            return self._quiescence_search(board, alpha, beta, skip_rep_count=skip_rep_count)

        moves = list(Rules.get_pseudo_legal_moves(board, board.current_player))
        self.order_moves(board, moves, depth)

        best = float("-inf")
        best_move = None
        has_legal_move = False
        is_first_move = True

        for move in moves:
            if time_limit is not None and (time.time() - start_time) > time_limit:
                raise SearchTimeoutException()
            is_cap = self._is_capture(board, move)
            mover = board.current_player
            applied = apply_pseudo_legal_with_rule_cache(
                board,
                move,
                mover,
                pre_move_cache=self._pre_move_flags_cache,
                post_apply_cache=self._post_apply_flags_cache,
            )
            if applied is None:
                continue
            captured, opp_in_check = applied
            self.history_hashes.append(board.zobrist_hash)
            opp = board.current_player
            self._move_history_stack.append(
                MoveEntry(
                    pos_hash=board.zobrist_hash,
                    mover=mover,
                    gave_check=Rules.is_king_in_check(board, opp),
                    last_move=move,
                )
            )
            has_legal_move = True

            # ---- 将军延伸 ----
            # 若走完这步后对手处于被将军状态，则不消耗深度余量，
            # 保证战术杀招序列不会因深度截断而被遗漏。
            next_depth = depth - 1
            next_check_ext = check_ext_left
            if opp_in_check and check_ext_left > 0:
                next_depth = depth
                next_check_ext = check_ext_left - 1

            try:
                # ---- PVS（Principal Variation Search）----
                # 排序后的首步最可能位于主变化线上，使用全窗口 [-beta, -alpha] 搜索。
                # 后续着法先用零窗口 [-alpha-1, -alpha] 快速探测：
                #   - 若 score <= alpha（未 fail-high），确认不优于首着，剪枝成功。
                #   - 若 alpha < score < beta（fail-high），需用全窗口重搜获取精确分。
                # 特殊情况：alpha == -inf 时零窗口 (-inf-1, -inf) 无意义，强制全窗口。
                use_full_window = is_first_move or alpha == float("-inf")
                if use_full_window:
                    score = -self._alphabeta(
                        board,
                        next_depth,
                        -beta,
                        -alpha,
                        start_time,
                        time_limit,
                        allow_null=allow_null,
                        use_tt=use_tt,
                        check_ext_left=next_check_ext,
                        skip_rep_count=skip_rep_count,
                    )
                else:
                    # 零窗口探测阶段
                    score = -self._alphabeta(
                        board,
                        next_depth,
                        -alpha - 1,
                        -alpha,
                        start_time,
                        time_limit,
                        allow_null=allow_null,
                        use_tt=use_tt,
                        check_ext_left=next_check_ext,
                        skip_rep_count=skip_rep_count,
                    )
                    if alpha < score < beta:
                        # 零窗口 fail-high：该着法可能比首着更优，全窗口重搜确认
                        score = -self._alphabeta(
                            board,
                            next_depth,
                            -beta,
                            -alpha,
                            start_time,
                            time_limit,
                            allow_null=allow_null,
                            use_tt=use_tt,
                            check_ext_left=next_check_ext,
                            skip_rep_count=skip_rep_count,
                        )
            finally:
                self.history_hashes.pop()
                self._move_history_stack.pop()
                board.undo_move(*move, captured)

            is_first_move = False

            if score > best:
                best = score
                best_move = move
            if best > alpha:
                alpha = best
            if alpha >= beta:
                # Beta 截断：当前着法已足够好，对手不会允许到达此节点。
                # 记录历史启发（按 depth² 加权）和杀手走法以加速兄弟节点搜索。
                self.history_table[move] = self.history_table.get(move, 0) + depth * depth
                if not is_cap:
                    self._push_killer(depth, move)
                break

        if not has_legal_move:
            # 无合法着法：被将死或困毙。返回负向极大值，加上深度偏移量
            # 使引擎倾向于选择最短的杀棋路径（越早将死得分越高）。
            self._nodes += 1
            mate_score = float(-_MATE_BASE_SCORE + (self.depth - depth))
            if use_tt:
                self._tt_store_exact(pos_key, depth, mate_score, None)
            return mate_score

        if use_tt:
            self._tt_store(pos_key, depth, best, alpha_orig, beta_orig, best_move)
        return best
