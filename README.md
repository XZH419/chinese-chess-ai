## 项目简介

本仓库包含一个中国象棋（Xiangqi）AI 子项目：**完整规则引擎**、**PyQt5 GUI**、以及多种对弈算法（**Minimax**、**MCTS**、**MCTS-Minimax**、**随机基线**）。其核心实现位于 `chinese_chess/`。

- **核心文档**：请优先阅读 `chinese_chess/README.md`
- **代码入口**：`python -m chinese_chess.main gui` / `python -m chinese_chess.main cli`

## 快速开始（Windows）

在仓库根目录：

```bash
python -m pip install -r requirements.txt
python -m chinese_chess.main gui --red human --black minimax --black-depth 5
```

更完整的依赖、运行方式、算法与规则说明见 `chinese_chess/README.md`。

