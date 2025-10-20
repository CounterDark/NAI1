"""
Main module for the TNTFrog game.

## Prerequisites

1. Python 3.13 installed (ensure `python --version` shows 3.13+)
2. `uv` installed, instructions to install uv:
https://docs.astral.sh/uv/getting-started/installation/

## Setup

Use `uv` to create a virtual environment and install dependencies
from `pyproject.toml`.

Option A: using Makefile

```sh
make setup
```

Option B: using uv directly

```sh
uv sync
```

## Run

Start the game in the console.

Option A: using Makefile

```sh
make run
```

Option B: using uv directly

```sh
uv run python src/main.py
```

While playing, type one of the allowed moves: `up`, `down`,
`left`, `right`, `upleft`, `upright`, `downleft`, `downright`.

### CLI arguments

You can configure the game via command line arguments:

- `--size <int>`: Board size (odd integer, minimum 5). Default: `5`.
- `--depth <int>`: AI search depth for Negamax (number of moves the
AI thinks ahead). Default: `10`.
- `--help`: Show help and exit.

Examples:

```
uv run python src/main.py --size 7 --depth 12
make run ARGS="--size 9 --depth 8"
```

## Game rules
The game is played on the Square board of odd size (minimum 5x5).
Player 1 begins first turn. Each player starts in the opposite
corner of the board and take alternating turns moving on the
board. Each player can move to any of the 8 neighboring cells
(orthogonal or diagonal), using the commands listed above. A
move is considered legal if it's within board bounds, target
square is unoccupied by other player and and no player was
positioned on this field in the previous turns. Cells previously
occupied by a player become blocked and are marked as `-1`. A
player loses if they have no legal moves on their turn; thus,
the opponent wins.

Authors: Mateusz Anikiej and Aleksander Kunkowski
"""

from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax  # type: ignore
from board_manager import BoardManager
import argparse


class TNTFrog(TwoPlayerGame):
    """In turn, the players remove one, two or three bones from a
    pile of bones. The player who removes the last bone loses."""

    def __init__(self, players, size=5):
        self.players = players
        self.game = BoardManager(size)
        self.current_player = 1  # player 1 starts
        self.first_round = True

    def possible_moves(self):
        """
        Get all the possible moves for the current player.

        :return possible_moves: A list of possible moves as Move objects.
        """
        return self.game.get_possible_moves(self.current_player)

    def make_move(self, move):
        """
        Make a move on the board.

        :param move: A move as a string.
        """
        self.game.make_move(move, self.current_player)

    def lose(self):
        """
        Check if the current player has no legal moves.

        :return lose: True if the current player has no legal moves, False otherwise.
        """
        return len(self.game.get_possible_moves(self.current_player)) <= 0

    def is_over(self):
        """
        Check if the game is over.

        :return is_over: True if the game is over, False otherwise.
        """
        return self.lose()  # Game stops when someone loses.

    def show(self):
        """
        Show the current state of the game.
        """
        self.game.print()
        if self.first_round:
            if moves := self.game.get_possible_moves(self.current_player):
                self.game.print_move_options(moves, self.current_player)
            self.first_round = False
        else:
            if moves := self.game.get_possible_moves(self.opponent_index):
                self.game.print_move_options(moves, self.opponent_index)

    def scoring(self):
        """
        Score the game.

        :return scoring: 0 if the current player has no legal moves, 1 otherwise.
        """
        return 0 if self.lose() else 1  # For the AI


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for configuring the game.

    --size: Board size (odd integer, >= 5)
    --depth: AI search depth (how many moves ahead the AI thinks)
    --help: Show help and exit
    """
    parser = argparse.ArgumentParser(
        prog="uv run python src/main.py",
        description=(
            "Console board game with a basic AI opponent. "
            "Player 1 (human) plays against Player 2 (AI)."
        ),
    )
    parser.add_argument(
        "--size",
        type=int,
        default=5,
        help="Board size (odd integer, minimum 5). Default: 5",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help="AI search depth for Negamax. Default: 10",
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point for running the TNTFrog game with configurable options.
    """
    args = parse_args()
    ai = Negamax(args.depth)
    game = TNTFrog([Human_Player(), AI_Player(ai)], args.size)
    game.play()
    winner = 2 if game.current_player == 1 else 1
    print(f"Player {winner} wins!")


if __name__ == "__main__":
    main()
