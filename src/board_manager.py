from dataclasses import KW_ONLY, dataclass
from typing import Literal


def _validate_board_size(size: int) -> int:
    if not isinstance(size, int):
        raise TypeError("Size must be of type int!")
    if size < BoardManager.MINIMAL_SIZE:
        raise ValueError(
            f"The board size cannot be less than {BoardManager.MINIMAL_SIZE}!"
        )
    if size % 2 == 0:
        raise ValueError("The board size must be an odd integer!")
    return size


def _validate_player(player: Literal[1, 2]) -> Literal[1, 2]:
    if player not in [1, 2]:
        raise ValueError("Invalid player!")
    return player


type Board = list[list[int]]
type Direction = Literal[-1, 0, 1]


@dataclass
class Position:
    _: KW_ONLY
    x: int
    y: int


type Players = dict[int, Position]


@dataclass(frozen=True)
class Move:
    x: Direction
    y: Direction

    def __str__(self):
        return f"Move(x={self.x}, y={self.y})"


_INSTRUCTION_TO_MOVE: dict[str, Move] = {
    "up": Move(-1, 0),
    "down": Move(1, 0),
    "left": Move(0, -1),
    "right": Move(0, 1),
    "upleft": Move(-1, -1),
    "upright": Move(-1, 1),
    "downleft": Move(1, -1),
    "downright": Move(1, 1),
}

_MOVE_TO_INSTRUCTION: dict[Move, str] = {
    move: name for name, move in _INSTRUCTION_TO_MOVE.items()
}


def to_move(instruction: str) -> Move:
    """
    Translate a textual direction instruction into a Move object.

    Accepted values (case-insensitive): "up", "down", "left", "right",
    "upleft", "upright", "downleft", "downright".
    """
    key = instruction.strip().lower()
    if key not in _INSTRUCTION_TO_MOVE:
        raise ValueError(
            "Invalid instruction. Use one of: "
            "up, down, left, right, upleft, upright, downleft, downright"
        )
    return _INSTRUCTION_TO_MOVE[key]


def from_move(move: Move) -> str:
    """
    Translate a Move object into its textual instruction representation.
    """
    if move not in _MOVE_TO_INSTRUCTION:
        raise ValueError("Unsupported move vector for translation")
    return _MOVE_TO_INSTRUCTION[move]


class BoardManager:
    MINIMAL_SIZE = 5

    def __init__(self, size: int):
        self._size = _validate_board_size(size)
        self._players: Players = {
            # Player 1 starts at top left corner
            1: Position(x=0, y=0),
            # Player 2 starts at bottom right corner
            2: Position(x=self._size - 1, y=self._size - 1),
        }
        self._board = self._create_board()

    def print(self) -> None:
        """
        Prints the game board in the console.
        """
        for row in self._board:
            for number in row:
                print(f" {number}" if number != -1 else number, end=", ")
            print("\n")

    def print_move_options(self, moves: list[str], player: Literal[1, 2]) -> None:
        """
        Prints the possible moves for the given player.
        """
        print(f"Possible moves for player {player}: {", ".join(moves)}", end="\n")

    def _create_board(self) -> Board:
        """
        Create a board with the given size.
        Numer 1 reprezents player 1, number 2 represents player 2 and
        0 represents neutral field.

        :return board: The created board. A 2D List of numbers.
        """
        board = []

        for i in range(self._size):
            row = []
            for j in range(self._size):
                # Place player 1 in a starting position
                if i == self._players[1].x and j == self._players[1].y:
                    row.append(1)
                # Place player 2 in a starting position
                elif i == self._players[2].x and j == self._players[2].y:
                    row.append(2)
                # Place neutral field
                else:
                    row.append(0)
            board.append(row)
        return board

    def get_possible_moves(self, player: Literal[1, 2]) -> list[str]:
        """
        Function to get all the possible moves for the given player.

        :param player: A player, 1 or 2.
        :return possible_moves: A list of possible moves as Move objects.
        """
        player = _validate_player(player)

        possible_directions = [
            Move(-1, -1),
            Move(-1, 0),
            Move(-1, 1),
            Move(0, -1),
            Move(0, 1),
            Move(1, -1),
            Move(1, 0),
            Move(1, 1),
        ]

        possible_moves: list[str] = []
        _player = self._players[player]
        for move in possible_directions:
            next_x, next_y = _player.x + move.x, _player.y + move.y
            if self._is_in_board_bounds(next_x, next_y):
                if self._is_empty_field(next_x, next_y):
                    possible_moves.append(from_move(move))
        return possible_moves

    def _is_in_board_bounds(self, x: int, y: int) -> bool:
        """
        Check if the given coordinates are in the board bounds.
        """
        return 0 <= x < self._size and 0 <= y < self._size

    def _is_empty_field(self, x: int, y: int) -> bool:
        """
        Check if the given field is empty.
        """
        return self._board[x][y] == 0

    def make_move(self, move: str, player: Literal[1, 2]) -> Board:
        """
        Make a move on the board.

        :param move: A move as a string.
        :param player: A player, 1 or 2.
        :return board: A modified board.
        """
        player = _validate_player(player)
        _move = to_move(move)

        new_position = Position(
            x=self._players[player].x + _move.x,
            y=self._players[player].y + _move.y,
        )
        if self._is_in_board_bounds(new_position.x, new_position.y):
            if self._is_empty_field(new_position.x, new_position.y):
                # Place player on the new position
                self._board[new_position.x][new_position.y] = player
                # Remove player from the old position
                self._board[self._players[player].x][self._players[player].y] = -1
                # Update player's position
                self._players[player] = new_position
                return self._board
        raise ValueError("The move is not valid!")
