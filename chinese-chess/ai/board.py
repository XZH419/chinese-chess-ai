"""中国象棋棋盘和走棋规则逻辑模块。"""

from .pieces import Piece

class Board:
    def __init__(self):
        # 标准中国象棋棋盘大小：10 行 9 列。
        self.rows = 10
        self.cols = 9
        self.board = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        # 本实现中红方先走。
        self.current_player = 'red'
        self.init_board()

    def init_board(self):
        # 初始化黑方棋子在上方。
        self.board[0][0] = Piece('black', 'che')
        self.board[0][1] = Piece('black', 'ma')
        self.board[0][2] = Piece('black', 'xiang')
        self.board[0][3] = Piece('black', 'shi')
        self.board[0][4] = Piece('black', 'jiang')
        self.board[0][5] = Piece('black', 'shi')
        self.board[0][6] = Piece('black', 'xiang')
        self.board[0][7] = Piece('black', 'ma')
        self.board[0][8] = Piece('black', 'che')

        self.board[2][1] = Piece('black', 'pao')
        self.board[2][7] = Piece('black', 'pao')

        self.board[3][0] = Piece('black', 'bing')
        self.board[3][2] = Piece('black', 'bing')
        self.board[3][4] = Piece('black', 'bing')
        self.board[3][6] = Piece('black', 'bing')
        self.board[3][8] = Piece('black', 'bing')

        # 初始化红方棋子在下方。
        self.board[9][0] = Piece('red', 'che')
        self.board[9][1] = Piece('red', 'ma')
        self.board[9][2] = Piece('red', 'xiang')
        self.board[9][3] = Piece('red', 'shi')
        self.board[9][4] = Piece('red', 'jiang')
        self.board[9][5] = Piece('red', 'shi')
        self.board[9][6] = Piece('red', 'xiang')
        self.board[9][7] = Piece('red', 'ma')
        self.board[9][8] = Piece('red', 'che')

        self.board[7][1] = Piece('red', 'pao')
        self.board[7][7] = Piece('red', 'pao')

        self.board[6][0] = Piece('red', 'bing')
        self.board[6][2] = Piece('red', 'bing')
        self.board[6][4] = Piece('red', 'bing')
        self.board[6][6] = Piece('red', 'bing')
        self.board[6][8] = Piece('red', 'bing')

    def get_piece(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.board[row][col]
        return None

    def _inside_board(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_valid_move(self, start_row, start_col, end_row, end_col, player=None, check_legality=True):
        """检查指定棋子走子是否合法。

        如果 check_legality 为 True，还会验证走子后是否导致己方将帅被将军，
        或者是否出现对面将帅见面的非法局面。
        """
        player = player or self.current_player
        if not self._inside_board(start_row, start_col) or not self._inside_board(end_row, end_col):
            return False

        piece = self.get_piece(start_row, start_col)
        if not piece or piece.color != player:
            return False

        target = self.get_piece(end_row, end_col)
        if target and target.color == piece.color:
            return False

        if piece.piece_type == 'jiang':
            valid = self._is_valid_jiang_move(start_row, start_col, end_row, end_col, player)
        elif piece.piece_type == 'shi':
            valid = self._is_valid_shi_move(start_row, start_col, end_row, end_col, player)
        elif piece.piece_type == 'xiang':
            valid = self._is_valid_xiang_move(start_row, start_col, end_row, end_col, player)
        elif piece.piece_type == 'ma':
            valid = self._is_valid_ma_move(start_row, start_col, end_row, end_col)
        elif piece.piece_type == 'che':
            valid = self._is_valid_che_move(start_row, start_col, end_row, end_col)
        elif piece.piece_type == 'pao':
            valid = self._is_valid_pao_move(start_row, start_col, end_row, end_col)
        elif piece.piece_type == 'bing':
            valid = self._is_valid_bing_move(start_row, start_col, end_row, end_col, player)
        else:
            valid = False

        if not valid:
            return False

        if not check_legality:
            return True

        # 模拟走子后检查是否出现将帅见面或己方将被将军。
        captured = self.board[end_row][end_col]
        self.board[end_row][end_col] = piece
        self.board[start_row][start_col] = None
        kings_facing = self._jiang_face_to_face()
        in_check = self.is_check(player)
        self.board[start_row][start_col] = piece
        self.board[end_row][end_col] = captured
        if kings_facing or in_check:
            return False

        return True

    def _is_valid_jiang_move(self, sr, sc, er, ec, player):
        # 将帅只能在九宫内沿直线走一步。
        if abs(sr - er) + abs(sc - ec) != 1:
            return False
        if player == 'red':
            return 7 <= er <= 9 and 3 <= ec <= 5
        return 0 <= er <= 2 and 3 <= ec <= 5

    def _is_valid_shi_move(self, sr, sc, er, ec, player):
        # 士在九宫内沿斜线走一步。
        if abs(sr - er) != 1 or abs(sc - ec) != 1:
            return False
        if player == 'red':
            return 7 <= er <= 9 and 3 <= ec <= 5
        return 0 <= er <= 2 and 3 <= ec <= 5

    def _is_valid_xiang_move(self, sr, sc, er, ec, player):
        # 象走田字格，不能过河，且眼位不能被堵住。
        if abs(sr - er) != 2 or abs(sc - ec) != 2:
            return False
        eye_r = (sr + er) // 2
        eye_c = (sc + ec) // 2
        if self.get_piece(eye_r, eye_c):
            return False
        if player == 'red':
            return er >= 5
        return er <= 4

    def _is_valid_ma_move(self, sr, sc, er, ec):
        # 马走日字，腿部被堵时不能走。
        dr = abs(sr - er)
        dc = abs(sc - ec)
        if not ((dr == 2 and dc == 1) or (dr == 1 and dc == 2)):
            return False
        if dr == 2:
            leg_r = (sr + er) // 2
            leg_c = sc
        else:
            leg_r = sr
            leg_c = (sc + ec) // 2
        if self.get_piece(leg_r, leg_c):
            return False
        return True

    def _is_valid_che_move(self, sr, sc, er, ec):
        # 车直线移动，路径上不能有阻挡棋子。
        if sr != er and sc != ec:
            return False
        if sr == er:
            step = 1 if ec > sc else -1
            for c in range(sc + step, ec, step):
                if self.get_piece(sr, c):
                    return False
        else:
            step = 1 if er > sr else -1
            for r in range(sr + step, er, step):
                if self.get_piece(r, sc):
                    return False
        return True

    def _is_valid_pao_move(self, sr, sc, er, ec):
        # 炮不吃子时走直线，吃子时必须隔一个子。
        if sr != er and sc != ec:
            return False
        target = self.get_piece(er, ec)
        count = 0
        if sr == er:
            step = 1 if ec > sc else -1
            for c in range(sc + step, ec, step):
                if self.get_piece(sr, c):
                    count += 1
        else:
            step = 1 if er > sr else -1
            for r in range(sr + step, er, step):
                if self.get_piece(r, sc):
                    count += 1
        if target:
            return count == 1
        return count == 0

    def _is_valid_bing_move(self, sr, sc, er, ec, player):
        # 兵/卒向前一步，过河后可以横着走一步。
        if player == 'red':
            if er > sr:
                return False
            if sr <= 4:  # 红兵过河后允许横走。
                return (er == sr - 1 and sc == ec) or (er == sr and abs(sc - ec) == 1)
            return er == sr - 1 and sc == ec
        else:
            if er < sr:
                return False
            if sr >= 5:  # 黑卒过河后允许横走。
                return (er == sr + 1 and sc == ec) or (er == sr and abs(sc - ec) == 1)
            return er == sr + 1 and sc == ec

    def _jiang_face_to_face(self):
        # 检查“将帅见面”特殊规则：同一列上无棋子阻挡时不能直接对峙。
        jiang_pos = None
        shuai_pos = None
        for r in range(self.rows):
            for c in range(self.cols):
                piece = self.board[r][c]
                if piece and piece.piece_type == 'jiang':
                    if piece.color == 'red':
                        jiang_pos = (r, c)
                    else:
                        shuai_pos = (r, c)
        if not jiang_pos or not shuai_pos:
            return False
        if jiang_pos[1] != shuai_pos[1]:
            return False
        col = jiang_pos[1]
        start = min(jiang_pos[0], shuai_pos[0]) + 1
        end = max(jiang_pos[0], shuai_pos[0])
        for r in range(start, end):
            if self.board[r][col] is not None:
                return False
        return True

    def make_move(self, start_row, start_col, end_row, end_col):
        # 执行走子，同时处理吃子逻辑并切换当前执子方。
        piece = self.board[start_row][start_col]
        if piece is None:
            return None
        captured = self.board[end_row][end_col]
        self.board[end_row][end_col] = piece
        self.board[start_row][start_col] = None
        self.current_player = 'black' if self.current_player == 'red' else 'red'
        return captured

    def undo_move(self, start_row, start_col, end_row, end_col, captured):
        # 撤销刚才的走子，还原被吃棋子和当前执子方。
        piece = self.board[end_row][end_col]
        self.board[start_row][start_col] = piece
        self.board[end_row][end_col] = captured
        self.current_player = 'black' if self.current_player == 'red' else 'red'

    def get_all_moves(self, player, validate_self_check=True):
        # 生成指定方的所有合法走法。
        # 如果 validate_self_check 为 True，则返回的走法不会使己方被将军。
        moves = []
        for r in range(self.rows):
            for c in range(self.cols):
                piece = self.board[r][c]
                if piece and piece.color == player:
                    for er in range(self.rows):
                        for ec in range(self.cols):
                            if self.is_valid_move(r, c, er, ec, player=player, check_legality=validate_self_check):
                                moves.append((r, c, er, ec))
        return moves

    def get_legal_moves(self, player):
        return self.get_all_moves(player, validate_self_check=True)

    def is_check(self, player):
        # 判断指定方的将帅是否处于被将军状态。
        opponent = 'black' if player == 'red' else 'red'
        jiang_pos = None
        for r in range(self.rows):
            for c in range(self.cols):
                piece = self.board[r][c]
                if piece and piece.color == player and piece.piece_type == 'jiang':
                    jiang_pos = (r, c)
                    break
        if not jiang_pos:
            return True
        for move in self.get_all_moves(opponent, validate_self_check=False):
            if move[2] == jiang_pos[0] and move[3] == jiang_pos[1]:
                return True
        return False

    def has_legal_moves(self, player):
        return len(self.get_legal_moves(player)) > 0

    def is_checkmate(self, player):
        return self.is_check(player) and not self.has_legal_moves(player)

    def is_stalemate(self, player):
        return not self.is_check(player) and not self.has_legal_moves(player)

    def is_game_over(self):
        # 当任意一方将被吃掉或任意一方无合法走法时，游戏结束。
        red_jiang = any(piece and piece.color == 'red' and piece.piece_type == 'jiang' for row in self.board for piece in row)
        black_jiang = any(piece and piece.color == 'black' and piece.piece_type == 'jiang' for row in self.board for piece in row)
        if not red_jiang or not black_jiang:
            return True
        if not self.has_legal_moves('red') or not self.has_legal_moves('black'):
            return True
        return False

    def copy(self):
        # 创建棋盘的深拷贝，用于搜索或模拟而不影响原棋盘状态。
        import copy
        new_board = Board()
        new_board.board = copy.deepcopy(self.board)
        new_board.current_player = self.current_player
        return new_board

    def __str__(self):
        # 将棋盘渲染为简单文本表格，便于在控制台打印查看。
        result = ''
        for r in range(self.rows):
            result += f'{r} '
            for c in range(self.cols):
                piece = self.board[r][c]
                if piece:
                    result += f'{piece} '
                else:
                    result += '· '
            result += '\n'
        result += '  a b c d e f g h i\n'
        return result