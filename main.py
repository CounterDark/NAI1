from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax
import managers.board_manager as board_manager


class TntFrog(TwoPlayerGame):
    """In turn, the players remove one, two or three bones from a
    pile of bones. The player who removes the last bone loses."""

    def __init__(self, players, size=3):
        self.players = players
        self.board = board_manager.create_board(size)
        self.last_move = (
            board_manager.get_center_index(size),
            board_manager.get_center_index(size),
        )
        self.current_player = 1  # player 1 starts

    def possible_moves(self):
        return board_manager.get_moves(self.board, self.last_move)

    def make_move(self, move):
        self.board = board_manager.move_piece(self.board, self.last_move, move)
        self.last_move = move

    def lose(self):
        return len(self.possible_moves()) <= 0

    def is_over(self):
        return self.lose()  # Game stops when someone loses.

    def show(self):
        board_manager.print_board(self.board)
        moves = self.possible_moves()
        if len(moves) > 0:
            board_manager.print_move_options(self.possible_moves())

    def scoring(self):
        return 0 if self.lose() else 1  # For the AI


# Start a match (and store the history of moves when it ends)
ai = Negamax(4)  # The AI will think 13 moves in advance
game = TntFrog([Human_Player(), AI_Player(ai)], 3)
history = game.play()
