# 中国象棋 AI（搜索算法）课程项目

本项目为《人工智能原理》课程作业 **题目二：象棋 AI** 的实现：在标准 **9×10** 中国象棋棋盘上实现完整规则框架、图形化界面，并提供基于搜索算法的 AI（含随机基线、Alpha-Beta Minimax、MCTS）。

---

## 功能完成情况（对照题目要求）

### 1.（必做）中国象棋规则框架

- **棋盘表示（9×10，含楚河汉界）**  
  - 棋盘网格与开局摆放：`engine/board.py` 的 `Board`（10 行 × 9 列）。  
  - GUI 背景棋盘贴图（含楚河汉界）：`ui/resources/img/board.png`，加载逻辑在 `ui/qt/main_window.py`。

- **所有棋子的基本走法与吃子规则（含关键特例）**（核心实现：`engine/rules.py`）  
  - **将/帅**：九宫内上下左右一步（`Rules._geometry_error` 中 `pt == "jiang"`）。  
  - **士/仕**：九宫内斜走一步（`pt == "shi"`）。  
  - **象/相**：田字两步、**不可过河**、**塞象眼**（`pt == "xiang"`）。  
  - **马**：日字走法、**蹩马腿**（`pt == "ma"` + `Rules._ma_leg_square`）。  
  - **车**：直线行走、路径不可有子（`pt == "che"`）。  
  - **炮**：直线行走，吃子需 **恰好隔一子（炮架）**（`pt == "pao"`）。  
  - **兵/卒**：不得后退；未过河不得横移，过河后可横移（`Rules._bing_geometry_error`）。

- **将帅对面（无遮挡时不可直接相对）**  
  - 走后局面检测：`Rules._jiang_face_to_face`（`engine/rules.py`）。  
  - 在 `Rules.is_valid_move(..., check_legality=True)` 中通过“模拟走子→检测→撤销”过滤非法走法。

- **胜负判定**（核心：`engine/rules.py` 的 `Rules.winner` / `Rules.is_game_over`）  
  - **一方将（帅）被吃掉**：`Board.apply_move` 会把 `red_king_pos` / `black_king_pos` 置为 `None`，`Rules.winner` 以此 O(1) 判定。  
  - **一方无合法下法（困毙/将死）即判负**：`Rules.has_legal_moves` 为假时判负（中国象棋困毙判负）。  
  - 规则框架的基线测试：随机 AI（`ai/random_ai.py`）+ CLI 冒烟测试（`app/smoke_play.py`）。

> 说明：题目不要求“长将判负”。本项目在规则引擎中 **额外实现了长将警告/判负**（见下方“扩展项”），GUI 也会弹窗提示。

### 2.（必做）可对弈 AI：MCTS 或 Minimax（含剪枝）+ 启发式评估

- **Minimax（Negamax + Alpha-Beta 剪枝）**：`ai/minimax_ai.py` 的 `MinimaxAI`  
  - 迭代加深、PVS、期望窗口、置换表（Zobrist）、Null Move、杀手走法、历史启发、将军延伸、静止搜索（QS）等优化均在该文件内实现（文件头部 docstring 有完整说明）。  
  - 走法生成：大量节点使用 `Rules.get_pseudo_legal_moves`（伪合法，仅做几何/吃子颜色层面的快速过滤），关键位置再通过规则/将军检测保证正确性。

- **MCTS（Monte Carlo Tree Search）**：`ai/mcts_ai.py` 的 `MCTSAI`  
  - UCB 选择 + 扩展 + rollout + 回传。rollout 中使用轻量随机策略，并用评估函数对截断局面做 sigmoid 映射。  
  - 也集成了开局库命中（见 `ai/opening_book.py`）。

- **启发式评估函数**：`ai/evaluation.py` 的 `Evaluation.evaluate`  
  - Tapered Evaluation（中局/残局锥形插值：`MG_VALUES` / `EG_VALUES` + `phase`）。  
  - PST（Piece-Square Tables）：兵/马/车/炮/士/象/将分别有位置分表。  
  - 机动性（马）与炮架奖励（炮），战术协同（车直线牵制、炮架成势、高位马、车马协同），兑子惩罚与将军奖励等。

- **开局库（可选加成，但不影响规则与搜索正确性）**：`ai/opening_book.py`  
  - 手写 `BASE_BOOK` + 左右镜像自动扩展；对 `Zobrist Hash → 推荐着法` 做 O(1) 查表。  
  - Minimax / MCTS 在开局前若干步会优先探测开局库以提升开局质量。

### 3.（选做）扩展项（本项目已实现其中至少一项）

- **长将判负**（高级规则）  
  - 核心检测：`engine/rules.py` 的 `Rules.perpetual_check_status`  
    - 同形第 2 次：warning（GUI 弹窗提醒“必须变招”）  
    - 同形第 3 次：forfeit（长将方判负）  
  - 终局落点：`Rules.winner(..., move_history=history)` 会将长将判负纳入胜负判定。

- **更复杂的评估函数**（棋子价值 + 位置 + 机动性 + 战术协同）  
  - 实现在 `ai/evaluation.py`：Tapered Evaluation（中局/残局插值）、PST、马机动性、炮架奖励、战术协同（纵线车/炮架/高位马/车马联动）、兑子惩罚、将军奖励等。

- **开局库**（含左右对称自动扩展）  
  - 实现在 `ai/opening_book.py`：手写 `BASE_BOOK`，用 `mirror_move` 自动生成左右镜像变例，并投影为 `OPENING_BOOK: zobrist_hash -> [moves]` 供 `MinimaxAI`/`MCTSAI` O(1) 查表使用。

- **多种搜索算法对比实验与分析**  
  - 实现在 `infra/experiment_runner.py`：自动对弈采集 `raw_games.csv` + `summary_report.txt`（默认输出目录 `runs/<timestamp>/`）。  
  - 内置实验组包含 Random / Minimax / MCTS 的多组对比，并提供 “calibrated” 公平性策略：先测 `Minimax(depth=K)` 的真实每步耗时中位数，再把 MCTS 的 `time_limit` 校准到同一时间预算。


## 项目结构

- `app/`：程序入口与控制器  
  - `app/main.py`：统一入口（CLI/GUI + AI 参数）  
  - `app/controller.py`：`GameController`，负责走子流程、历史记录、终局判定、AI 调度  
  - `app/smoke_play.py`：CLI 冒烟对弈（用于快速验证规则框架）
- `engine/`：棋盘与规则引擎  
  - `engine/board.py`：`Board`（状态容器、走子/悔棋、Zobrist 增量维护）  
  - `engine/rules.py`：`Rules`（走法合法性、将军检测、走法生成、终局判定）  
  - `engine/zobrist.py`：Zobrist 哈希
- `ai/`：AI 与评估  
  - `ai/random_ai.py`：随机基线  
  - `ai/minimax_ai.py`：Alpha-Beta Minimax（大量剪枝与工程优化）  
  - `ai/mcts_ai.py`：MCTS  
  - `ai/evaluation.py`：启发式评估函数  
  - `ai/opening_book.py`：开局库（含镜像扩展）
- `ui/qt/`：PyQt5 图形界面  
  - `ui/qt/main_window.py`：棋盘渲染、鼠标交互、后台 AI 子进程通信、对局配置面板
- `infra/`：基础设施与实验脚本  
  - `infra/ai_worker.py`：AI 搜索子进程入口（GUI 为避免卡顿会把搜索放到独立进程）  
  - `infra/experiment_runner.py`：自动对弈实验，输出到 `runs/<timestamp>/`

---

## 运行环境

- Python 3.10+（建议 3.11）
- Windows 10/11（本仓库的 GUI 线程/多进程逻辑已针对 Windows 的 `spawn` 做了处理）
- 图形界面依赖：`PyQt5`



---

## 安装与运行

### 安装依赖（GUI 需要）

在仓库根目录执行：

```bash
python -m pip install -U pip
python -m pip install PyQt5
```

### 启动 GUI（推荐）

```bash
# 默认：CLI
# GUI：红方玩家 vs 黑方 Minimax（默认深度 5）
python -m app.main gui

# GUI：人类 vs MCTS（模拟次数上限 5000）
python -m app.main gui --black mcts --black-sims 5000

# GUI：AI vs AI（例如：MCTS vs Minimax）
python -m app.main gui --red mcts --black minimax --black-depth 5
```

GUI 操作方式：

- 点击己方棋子以选中（会放大高亮），再点击目标格完成落子
- 右侧面板可配置红/黑双方为：玩家 / 随机 AI / Minimax / MCTS，并设置对应参数（深度或模拟次数）

### 启动 CLI（规则/AI 冒烟测试）

```bash
# 随机 vs 随机（用于验证规则框架、走子流程）
python -m app.main cli --red random --black random

# MCTS vs Minimax
python -m app.main cli --red mcts --black minimax --black-depth 5
```

---

## 关键实现说明

### 规则引擎如何保证正确性

- **两层走法过滤**  
  - 伪合法走法：`Rules.get_pseudo_legal_moves`（快，供搜索用；不检查自将、白脸将等）  
  - 完整合法走法：`Rules.is_valid_move(..., check_legality=True)`（慢但严谨；通过“模拟走子→检测将军/将帅对面→撤销”确保走后局面合法）

- **将军检测**  
  - `Rules.is_king_in_check` 使用从将/帅出发的反向扫描：车/炮射线 + 马腿判定 + 兵/卒攻击格判定。

- **终局判定**  
  - `Rules.winner`：将帅被吃、长将判负、无合法着法（困毙/将死）等统一在这里裁决。  
  - `GameController.apply_move` 每走一步都会记录 `MoveEntry`（Zobrist hash、行棋方、是否将军、走法），用于 GUI 提示与长将判断。

### Minimax 的核心剪枝与工程优化点（摘要）

代码在 `ai/minimax_ai.py`，包含：

- Alpha-Beta + PVS（主变化搜索）
- 置换表（Zobrist hash 为键）
- 走法排序：TT 最佳着法、MVV-LVA 吃子排序、杀手走法、历史启发
- Null Move Pruning
- 将军延伸（Check Extension）
- 静止搜索（Quiescence Search）+ Delta Pruning

### 启发式评估函数（摘要）

代码在 `ai/evaluation.py`，主要由以下部分组成：

- 子力价值（中局/残局两套）+ 阶段因子线性插值（Tapered Evaluation）
- PST（位置分）
- 马机动性、炮架奖励
- 战术协同（纵线车、炮架、高位马、车马协同）
- 领先方兑子惩罚、将军奖励

---

## 对弈实验与数据输出（可选）

运行自动对弈实验（输出到 `runs/<timestamp>/`）：

```bash
python -m infra.experiment_runner
```

默认输出：

- `runs/<timestamp>/raw_games.csv`：逐局数据
- `runs/<timestamp>/summary_report.txt`：汇总统计

---

## 备注

- 本项目坐标系采用 **0-based**：行 `0..9`、列 `0..8`，走法格式为四元组 `(sr, sc, er, ec)`（可在 `engine/board.py` / `engine/rules.py` / `ai/*` 中看到一致用法）。
- GUI 为避免界面卡顿，将 AI 搜索放到独立进程（见 `infra/ai_worker.py` 与 `ui/qt/main_window.py` 的 `AIMoveThread` 说明）。

