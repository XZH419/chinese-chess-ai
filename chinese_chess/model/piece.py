"""棋子实体（模型层 Model）。

从旧实现 ``chinese-chess/ai/pieces.py`` 物理搬运而来。
本模块仅负责棋子数据表示与显示符号映射，不包含任何规则逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Piece:
    """中国象棋棋子数据对象。

    以不可变风格表示单枚棋子的颜色与兵种，并提供中文符号映射。

    Attributes:
        color: 棋子颜色，``"red"``（红方）或 ``"black"``（黑方）。
        piece_type: 兵种标识字符串，取值为
            ``"jiang"``（将/帅）、``"shi"``（士/仕）、``"xiang"``（象/相）、
            ``"ma"``（马）、``"che"``（车）、``"pao"``（炮）、``"bing"``（兵/卒）。
    """

    color: str
    piece_type: str

    def __str__(self) -> str:
        """返回棋子的中文显示符号（如红"帅"、黑"将"）。

        Returns:
            对应颜色和兵种的单个中文汉字。
        """
        symbols = {
            "red": {
                "jiang": "帅",
                "shi": "仕",
                "xiang": "相",
                "ma": "马",
                "che": "车",
                "pao": "炮",
                "bing": "兵",
            },
            "black": {
                "jiang": "将",
                "shi": "士",
                "xiang": "象",
                "ma": "马",
                "che": "车",
                "pao": "炮",
                "bing": "卒",
            },
        }
        return symbols[self.color][self.piece_type]

    def __repr__(self) -> str:
        """返回棋子的调试表示（如 ``red_jiang``）。

        Returns:
            ``"颜色_兵种"`` 格式的字符串。
        """
        return f"{self.color}_{self.piece_type}"
