import math
from typing import Tuple


def get_center_index(size: int):
    """
    Function that returns the center index of a given size.
    :param size: The size of the board.
    :return: The center index.
    """
    return math.floor(size / 2)


def create_board(size: int):
    """
    Function to create a board with the given size.
    :param size: The size of the board. Must be a positive odd integer.
    :return: The created board. A 2D List of numbers.
    """
    if size < 3:
        print("Minimalna wielkość to 3!")
        exit(0)
    if size % 2 == 0:
        print("Wielkość musi byś nieparzysta!")
        exit(0)
    board = []

    center = get_center_index(size)

    for i in range(size):
        row = []
        for j in range(size):
            row.append(2 if (j == center and i == center) else 0)
        board.append(row)
    return board


def print_board(board):
    """
    Function to print the given board.
    :param board: The board to print. A 2D List of numbers.
    """
    print("Current board:", end="\n")
    for row in board:
        for number in row:
            print(number, end=", ")
        print("\n")


def get_moves(board: list[list[int]], last_move: Tuple[int, int]):
    """
    Function to get all the possible moves for the given board.
    :param board: A 2D list of numbers.
    :param last_move: A tuple containing the last move. Row, column coordinates.
    :return: A list of possible moves as coordinates in tuples.
    """
    size = len(board)
    # 8-way-simple
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    moves = []
    x, y = last_move
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            if board[nx][ny] == 0:
                moves.append((nx, ny))
    return moves


def move_piece(
    board: list[list[int]],
    last_move: Tuple[int, int],
    move_coordinates: Tuple[int, int],
):
    """
    Function to move the given piece.
    :param board: The board to move. A 2D List of numbers.
    :param move_coordinates: The coordinates of the piece to move. Row, column coordinates.
    :param last_move: The last move. Row, column coordinates.
    :return: A modified board.
    """
    board[move_coordinates[0]][move_coordinates[1]] = 2
    board[last_move[0]][last_move[1]] = 1
    return board


def print_move_options(moves: list[Tuple[int, int]]):
    """
    Function to print the given moves.
    :param moves: A list of tuples containing the coordinates of the piece to move. Row, column coordinates.
    """
    print("Choose an option:")
    for index, move in enumerate(moves):
        print("Option#" + str(index) + ": " + str(move), end="\n")
