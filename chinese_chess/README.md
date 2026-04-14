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
    ai_registry.py # AI 注册表：GUI/worker 共享的 AI 构造与配置协议
    mcts_common.py # MCTS 系列内部共享入口（避免跨文件导入私有符号）
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
- **MCTS-Minimax AI**：在 MCTS 主干中引入静态评估与轻量战术偏置（以及可选的 probe 机制，视实现为准）；强度与开销通常介于纯 MCTS 与深 Minimax 之间，适合实验对比。

算法英文名 **MCTS / Minimax / MCTS-Minimax** 在界面与文档中保留；其余面向用户的标签、日志统一为中文（例如「玩家」「并行进程数」「命中开局库」等）。

### GUI 与 AI 搜索子进程

- **PyQt 主窗口不在 `QThread` 内直接调用 `get_best_move`**。棋盘与 `MoveEntry` 历史经 `algorithm/ai_state_codec.py` 序列化为纯数据结构后，由 `multiprocessing.Queue` 发往 **独立 AI 子进程**（`scripts/ai_worker.py`：反序列化 → `create_ai_from_config` → `choose_move` / `get_best_move` → 回传结果）。
- GUI/worker 的 AI 配置协议由 `algorithm/ai_registry.py` 统一维护：GUI 负责把“当前 AI 实例”编码成纯数据 `ai_config`，子进程据此重建同类 AI 并执行搜索，避免两处重复 if/else 导致的参数漂移问题。
- **QThread** 只负责投递请求并阻塞等待响应，避免在 Qt 后台线程中再嵌套 `ProcessPoolExecutor`（Windows 上常见卡死）。
- AI 子进程必须为 **非 daemon**（默认即非守护）：否则无法在进程内再创建 MCTS 的 worker 子进程。关闭窗口（`closeEvent`）或结束对局（`_stop_game`）时会向队列发送哨兵并 `join` / 必要时 `terminate`，避免残留 `python.exe`。
- 真实搜索运行在子进程的 **主线程** 上，MCTS / MCTS-Minimax 可正常使用多进程 worker（`mcts._parallel_workers_when_safe` 不再因「非主线程」被压成 1）。
- 每次搜索携带 **`request_id`**，与界面 `_run_id` 一致；若结果晚于悔棋/重开/换引擎，主窗口丢弃过期响应。

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

## 7. 算法架构分析（优势与瓶颈）

### 7.1 Minimax（Negamax + Alpha-Beta）

- **优势**
  - **可解释性强**：搜索深度、走法排序、剪枝命中等都可用统计量直观看到效果。
  - **战术稳定**：在评估函数质量较好且深度足够时，对短期杀棋、强制交换更敏感。
  - **可复用缓存**：Zobrist + 置换表能显著减少重复局面的搜索开销。

- **潜在瓶颈**
  - **评估函数成本**：`Evaluation.evaluate()` 很丰富但也相对重，深度增加时评估次数会迅速放大。
  - **分支因子限制**：中局走法多，纯 Python 下深搜速度容易成为瓶颈；走法生成与合法性校验的常数项很关键。
  - **视野局限**：固定深度在复杂局面可能出现“深度够不到的威胁”（需要依赖延伸/静止搜索缓解）。

### 7.2 MCTS（RAVE + 根并行 + DAG 合并）

- **优势**
  - **任意时间可返回**：模拟次数越多结果越稳，适合“可中断”的思考模型。
  - **对评估函数依赖较弱**：主要依赖 rollout/统计回传；在评估难以精确的局面上也能逐步收敛。
  - **并行友好**：根并行天然适合多核（Windows 下需注意线程/进程层级，项目已通过“独立 AI 子进程”规避 Qt 子线程 spawn 风险）。

- **潜在瓶颈**
  - **rollout 质量影响上限**：若模拟策略过弱或偏差较大，收敛速度与最终强度会受限。
  - **合法性与将军检测开销**：为追求吞吐量可能使用伪合法走法生成；需要在关键路径上做足兜底，避免统计被大量“无意义自杀”污染。
  - **合并/DAG 维护成本**：哈希合并、缓存与统计聚合会引入额外常数项；在小模拟次数下可能不如精简实现。

### 7.3 MCTS-Minimax（混合）

- **优势**
  - **更快偏向“看起来更像人类”的着法**：静态评估与轻量偏置能在模拟量不大时减少纯随机探索的盲目性。
  - **中短期战术更稳**：在部分局面下，比纯 MCTS 更不容易错过显而易见的将军/吃子机会。

- **潜在瓶颈**
  - **评估函数的额外成本**：一旦在扩展/回传中更频繁地调用 `Evaluation.evaluate()`，吞吐量会下降。
  - **参数敏感**：偏置比例、模拟次数、probe 深度/预算等设置不当时可能“过拟合某类局面”或导致搜索不稳定。

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
