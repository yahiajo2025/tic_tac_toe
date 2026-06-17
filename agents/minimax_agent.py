class MinimaxAgent:
    def __init__(self, player=1):
        self.player = player

    def choose_action(self, board):
        best_score = -999
        best_move = None

        for i in range(9):
            if board[i] == 0:
                board[i] = self.player
                score = self.minimax(board, False)
                board[i] = 0

                if score > best_score:
                    best_score = score
                    best_move = i

        return best_move

    def minimax(self, board, is_maximizing):
        winner = self.check_winner(board)

        if winner == self.player:
            return 1
        elif winner == -self.player:
            return -1
        elif 0 not in board:
            return 0

        if is_maximizing:
            best = -999
            for i in range(9):
                if board[i] == 0:
                    board[i] = self.player
                    best = max(best, self.minimax(board, False))
                    board[i] = 0
            return best
        else:
            best = 999
            for i in range(9):
                if board[i] == 0:
                    board[i] = -self.player
                    best = min(best, self.minimax(board, True))
                    board[i] = 0
            return best

    def check_winner(self, board):
        wins = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]

        for a,b,c in wins:
            if board[a] == board[b] == board[c] != 0:
                return board[a]
        return None
