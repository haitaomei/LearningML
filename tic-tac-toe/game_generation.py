import numpy as np
from itertools import product


def is_winner(board, player):
    """Check if the given player has won on this board."""
    # Check rows
    for i in range(3):
        if np.all(board[i, :] == player):
            return True

    # Check columns
    for i in range(3):
        if np.all(board[:, i] == player):
            return True

    # Check diagonals
    if np.all(np.diag(board) == player):
        return True
    if np.all(np.diag(np.fliplr(board)) == player):
        return True

    return False


def is_valid_state(board):
    """Check if a given board state is valid."""
    # Count the number of X's and O's
    num_x = np.sum(board == 1)
    num_o = np.sum(board == 2)

    # In a valid state, either X has the same number of marks as O
    # or X has exactly one more mark than O (as X goes first)
    if not (num_x == num_o or num_x == num_o + 1):
        return False

    # Check if X has won
    x_won = is_winner(board, 1)

    # Check if O has won
    o_won = is_winner(board, 2)

    # If X has won, then X should have one more mark than O
    if x_won and num_x != num_o + 1:
        return False

    # If O has won, then X and O should have the same number of marks
    if o_won and num_x != num_o:
        return False

    # X and O cannot both have won
    if x_won and o_won:
        return False

    return True


def generate_valid_board():
    """Generate all valid Tic Tac Toe board states."""
    # Initialize an empty list to store valid states
    valid_states = []

    # Generate all possible board configurations
    # Each cell can be 'X', 'O', or ' ' (empty)
    for config in product([1, 2, 0], repeat=9):
        # Convert the configuration to a 3x3 numpy array
        board = np.array(config).reshape(3, 3)

        # Check if the board state is valid
        if is_valid_state(board):
            valid_states.append(board)

    return valid_states


def count_valid_states():
    """Count and categorize valid Tic Tac Toe states."""
    valid_states = generate_valid_board()

    # Initialize counters
    total_states = len(valid_states)
    x_wins = 0
    o_wins = 0
    draws = 0
    ongoing = 0

    for board in valid_states:
        x_won = is_winner(board, 1)
        o_won = is_winner(board, 2)

        # If the board is full (no empty spaces) and nobody won, it's a draw
        is_full = np.all(board != 0)

        if x_won:
            x_wins += 1
        elif o_won:
            o_wins += 1
        elif is_full:
            draws += 1
        else:
            ongoing += 1

    return {
        "total_valid_states": total_states,
        "x_wins": x_wins,
        "o_wins": o_wins,
        "draws": draws,
        "ongoing": ongoing,
    }


# Example usage
if __name__ == "__main__":
    stats = count_valid_states()
    print(f"Total valid Tic Tac Toe states: {stats['total_valid_states']}")
    print(f"X wins: {stats['x_wins']}")
    print(f"O wins: {stats['o_wins']}")
    print(f"Draws: {stats['draws']}")
    print(f"Ongoing games: {stats['ongoing']}")

    # Print a few example valid states
    print("\nExample valid states:")
    valid_states = generate_valid_board()
    for i in range(min(5, len(valid_states))):
        print(f"\nState {i + 1}:")
        print(valid_states[i])
