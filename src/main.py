from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax  # type: ignore
from board_manager import BoardManager


class TNTFrog(TwoPlayerGame):
    """In turn, the players remove one, two or three bones from a
    pile of bones. The player who removes the last bone loses."""

    def __init__(self, players, size=5):
        self.players = players
        self.game = BoardManager(size)
        self.current_player = 1  # player 1 starts

    def possible_moves(self):
        return self.game.get_possible_moves(self.current_player)

    def make_move(self, move):
        self.game.make_move(move, self.current_player)

    def lose(self):
        return len(self.game.get_possible_moves(self.current_player)) <= 0

    def is_over(self):
        return self.lose()  # Game stops when someone loses.

    def show(self):
        self.game.print()

    def scoring(self):
        return 0 if self.lose() else 1  # For the AI


# Start a match (and store the history of moves when it ends)
ai = Negamax(10)  # The AI will think 10 moves in advance
game = TNTFrog([Human_Player(), AI_Player(ai)], 5)
history = game.play()
