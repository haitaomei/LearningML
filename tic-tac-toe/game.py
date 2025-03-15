import numpy as np


def calculate_state(x) -> int:
    state = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            state *= 3
            state += x[i, j]
    return state


def to_board(state):
    x = np.zeros((3, 3))
    for i in range(x.shape[0] - 1, -1, -1):
        for j in range(x.shape[1] - 1, -1, -1):
            x[i, j] = state % 3
            state //= 3
    return x


def print_board(x):
    board = x.astype(str)
    board[board == "0.0"] = " "
    board[board == "1.0"] = "X"
    board[board == "2.0"] = "O"
    print("-" * 7)
    for row in board:
        print("|" + "|".join(row) + "|")
        print("-" * 7)


def check_winner(board):
    # Check rows
    if board[0][0] == board[0][1] == board[0][2] and board[0][0] != 0:
        return board[0][0]
    if board[1][0] == board[1][1] == board[1][2] and board[1][0] != 0:
        return board[1][0]
    if board[2][0] == board[2][1] == board[2][2] and board[2][0] != 0:
        return board[2][0]

    # Check columns
    if board[0][0] == board[1][0] == board[2][0] and board[0][0] != 0:
        return board[0][0]
    if board[0][1] == board[1][1] == board[2][1] and board[0][1] != 0:
        return board[0][1]
    if board[0][2] == board[1][2] == board[2][2] and board[0][2] != 0:
        return board[0][2]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != 0:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != 0:
        return board[0][2]

    # No winner
    return None


def generate_next_moves(board):
    next_moves = []

    if check_winner(board) is not None:
        return next_moves

    for row in range(3):
        for col in range(3):
            if board[row][col] == 0:
                next_moves.append((row, col))

    return next_moves


def get_next_player(board):
    """
    Assumptions: X must always take the first move
    """
    # Count occurrences of 'X' and 'O'
    board = np.array(board)
    x_count = np.count_nonzero(board == 1)
    o_count = np.count_nonzero(board == 2)

    if check_winner(board) is not None or x_count + o_count == 9:
        return None

    if x_count == o_count:
        return 1
    elif x_count == o_count + 1:
        return 2
    else:
        raise ValueError("Invalid board state when determining next player")


def state_after_move(board, action, player):
    """
    Assume valid game state
    """
    if player is None:
        raise ValueError("Player must be either 1 or 2")

    board[action[0], action[1]] = player
    state = int(calculate_state(board))
    board[action[0], action[1]] = 0  # reset board
    return state
