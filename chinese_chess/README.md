# 中国象棋与多种 AI 搜索

本仓库主体为 **中国象棋** 实现：完整规则、PyQt5 图形界面，以及 **MCTS**、**Minimax**、**MCTS-Minimax**、**随机 AI** 等多种对弈引擎。另含无头基准脚本与基于 cProfile 的性能分析工具。

> **说明**：仓库中若存在其他课程目录（例如围棋相关作业），与本象棋子项目相互独立；本 README 默认描述 `chinese_chess/` 包与根目录 `benchmark.py`。

---

## 1. 项目简介

- **图形界面对弈**：棋盘渲染、选子落子动画、右侧配置面板与对局日志。
- **玩家 / AI**：任一侧可为「玩家」或上述任一 AI；支持玩家对玩家、人机、AI 自对弈。
- **搜索算法**：极大极小（Minimax）、蒙特卡洛树搜索（MCTS）、MCTS 与 Minimax 结合的混合搜索（MCTS-Minimax），以及随机走子基线。

---

## 2. 主要功能

| 类别 | 说明 |
|------|------|
| 规则 | 中国象棋完整走子与合法性校验；将军、应将、飞将（王不见王）；长将检测与处罚；限着和棋等。 |
| 界面 | PyQt5 主窗口：红黑方引擎下拉框、参数、开始对局/结束重置、状态栏、日志区。 |
| 开局库 | Minimax / MCTS / MCTS-Minimax 在开局阶段可命中开局库着法。 |
| 性能 | `scripts/profile_tool.py`：对 Minimax、MCTS、MCTS-Minimax 做 cProfile 热点统计。 |
| 基准 | 根目录 `benchmark.py`：无 GUI 多局 AI 对弈，汇总胜率与单步平均耗时、节点/模拟量。 |

---

## 3. 项目结构（概要）

```
chinese_chess/
  model/          # 棋盘、棋子、Zobrist、规则与终局判定
  control/        # GameController：走子、历史、AI 调度
  view/qt/        # PyQt5 界面（主窗口、棋盘视图）
  algorithm/      # Minimax、MCTS、MCTS-Minimax、随机 AI、开局库数据
  scripts/        # profile_tool 等辅助脚本
  main.py         # CLI / GUI 统一入口
benchmark.py      # 仓库根目录：无头对弈基准
requirements.txt  # 根目录：PyQt5 等（与 chinese_chess/requirements.txt 可二选一安装）
```

---

## 4. AI 说明

- **玩家**：由用户在 GUI 点击走子；CLI 模式下无玩家输入流时请勿将双方均设为玩家。
- **随机 AI**：从全部合法着法中均匀随机选取，用于基线测试与冒烟流程。
- **Minimax AI**：固定深度迭代加深风格的极大极小搜索，带置换表与局面评估；可通过 UI 或 CLI 配置深度。
- **MCTS AI**：根并行多进程可选；以模拟次数与时间上限控制算力；verbose 模式下在终端输出中文搜索摘要。
- **MCTS-Minimax AI**：在 MCTS 展开/模拟链路中嵌入 Minimax 类局面探查（probe）与预算控制；强度与开销介于纯 MCTS 与深 Minimax 之间，适合实验对比。

算法英文名 **MCTS / Minimax / MCTS-Minimax** 在界面与文档中保留；其余面向用户的标签、日志统一为中文（例如「玩家」「并行进程数」「命中开局库」等）。

---

## 5. 运行方式

**环境**：建议 Python **3.10+**（以本机已测版本为准）。安装依赖：

```bash
python -m pip install -r requirements.txt
```

或在包目录：

```bash
python -m pip install -r chinese_chess/requirements.txt
```

### 5.1 启动 GUI

在仓库根目录执行：

```bash
python -m chinese_chess.main gui --red human --black minimax
```

也可先构造控制器再启动窗口（见 `main.py` 与 `view/qt/main_window.py`）。若未安装 PyQt5，程序会提示中文错误信息并退出。命令行中 Minimax 默认深度为 **5**；`--red-sims` / `--black-sims` 可省略，未指定时 **MCTS** 默认 5000 次模拟，**MCTS-Minimax** 默认 4000 次。

### 5.2 CLI 冒烟（无界面）

```bash
python -m chinese_chess.main cli --red random --black random
```

### 5.3 无头基准 `benchmark.py`

```bash
python benchmark.py --games 20 --red-ai mcts --black-ai minimax --red-sims 5000 --red-depth 4 --black-depth 5
```

引擎参数：`--red-ai` / `--black-ai` 取 `minimax`、`mcts`、`mcts_minimax`、`random`（旧别名 `hybrid` / `mcts_minmax` 会规范为 `mcts_minimax`）。Minimax 深度与 MCTS 类模拟上限分别用 `--*-depth`、`*-sims` 指定。

### 5.4 性能分析 `profile_tool`

```bash
python -m chinese_chess.scripts.profile_tool minimax --depth 5 --plies 10
python -m chinese_chess.scripts.profile_tool mcts --simulations 3000
python -m chinese_chess.scripts.profile_tool mcts_minimax --simulations 1500
```

---

## 6. 规则说明（与实现对齐）

- **基本走子**：按棋子类型生成几何走法，再经规则层过滤（蹩马、塞象眼、兵过河与否、将/帅九宫等）。
- **将军与应将**：走子后若己方老将仍被将军或送将，则着法非法；对方被将时应应将。
- **飞将（王不见王）**：同一纵线上双将无子阻隔时判非法。
- **长将**：同一单方连续将军形成的重复局面，**第二次**同形时给出**长将警告**并要求变招；**第三次**仍同形则**长将判负**（具体以 `Rules.perpetual_check_status` 与 `MoveOutcome` 为准）。
- **限着和棋**：半回合数达到 `Rules.MAX_PLIES_AUTODRAW`（如 150）时判和，优先于其他未完结分支。

更细的报错文案（如「友军误伤」「蹩马腿」「长将违规，必须变招」等）由规则层常量提供，GUI 会原样展示。

---

## 7. 开发说明

- **依赖**：核心为 Python 标准库；图形界面依赖 **PyQt5**。性能剖析依赖标准库 `cProfile` / `pstats`。
- **分层**：`Board` 维护局面与子力位置；`Rules` 负责合法着法、终局、长将、和棋条件；评估与搜索在 `algorithm/`；`GameController` 串联走子历史与 AI 调用；Qt 视图只负责展示与输入。
- **日志与界面语言**：用户可见的菜单、状态栏、弹窗、终端 `print`、基准与 profile 输出以**中文**为主；内部变量名、类名、统计字典键名（如 `workers`、`last_stats`）仍为英文以保持代码一致性。

---

## 8. 后续可优化方向（可选）

- 搜索与多进程调度性能（置换表共享、模拟批量等）。
- 评估函数与 MCTS 策略（UCT 系数、先验、开局库规模）。
- 自动化测试覆盖（规则边界、长将序列、限着和棋）。
- 规则与棋规细则进一步对齐官方竞赛文本（若有需要）。

---

## 9. 本次文案与 README 的取舍说明

- **改为中文**：所有面向终端用户与对局者的字符串（GUI、QMessageBox、`print`、argparse 的 `help` 说明、基准报告、profile 说明性输出等）。
- **保留英文**：Python 标识符、类型注解、第三方库 API、cProfile 输出的函数名栈（无法也不应强行翻译）、以及 CLI 引擎键名（`human`、`mcts_minimax` 等）以便脚本与代码一致。
- **注释与 docstring**：以开发者阅读为主，仅将其中明显面向「产品说明」的示例句改为中文；大量 API 文档仍为中文技术说明，不逐句英译中。

若需将 **围棋作业目录** 一并中文化，可在该子项目内单独提需求，避免与象棋包混淆。
