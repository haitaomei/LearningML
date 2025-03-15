import numpy as np

from game import (
    calculate_state,
    to_board,
    print_board,
    generate_next_moves,
    get_next_player,
    check_winner,
    state_after_move,
)

from game_generation import generate_valid_board, is_valid_state


DISCOUNT = 0.9
DELTA = 1e-4


def init_state_values_and_policy():
    boards = generate_valid_board()

    states = []
    state_values = {}
    policy = {}

    x_wins = 0
    o_wins = 0
    draws = 0
    ongoing = 0
    for board in boards:
        state = calculate_state(board)

        states.append(state)
        winner = check_winner(board)
        if winner == 1:
            x_wins += 1
            state_values[state] = 1
        elif winner == 2:
            o_wins += 1
            state_values[state] = -1
        elif np.all(board != 0):
            draws += 1
            state_values[state] = 0
        else:
            ongoing += 1
            next_player = get_next_player(board)
            if next_player == 1:
                state_values[state] = -10000
            else:
                state_values[state] = 10000

        next_moves = generate_next_moves(board)
        policy[state] = next_moves
        if winner == 1 or winner == 2 or np.all(board != 0):
            assert len(policy[state]) == 0

    for state in states:
        assert is_valid_state(to_board(state))

    print(f"Number of valid states: {len(states)}")
    print(f"X wins: {x_wins}")
    print(f"O wins: {o_wins}")
    print(f"Draws: {draws}")
    print(f"Ongoing: {ongoing}")

    total_actions = 0
    for _key, value_list in policy.items():
        total_actions += len(value_list)
    print(f"Total actions: {total_actions}")

    return states, state_values, policy


def value_iteration(states, state_values, policy):
    iteration = 0
    reward = 0  # state transition reward always zero

    while True:
        delta = 0
        for state in states:
            old_value = state_values[state]

            board = to_board(state)
            winner = check_winner(board)
            if winner is not None or np.all(board != 0):
                continue
            else:
                next_player = get_next_player(board)
                assert next_player == 1 or next_player == 2

                for action in policy[state]:
                    next_state = state_after_move(
                        board=board, action=action, player=next_player
                    )

                    if next_player == 1:
                        state_values[state] = max(
                            state_values[state],
                            reward + DISCOUNT * state_values[next_state],
                        )
                    else:
                        state_values[state] = min(
                            state_values[state],
                            reward + DISCOUNT * state_values[next_state],
                        )

            delta = max(delta, abs(old_value - state_values[state]))

        iteration += 1
        print(f"Iteration: {iteration}, Delta: {delta}")
        if delta < DELTA:
            break

    for state in states:
        actions = policy[state]
        values = []

        for action in actions:
            board = to_board(state)
            next_player = get_next_player(board)
            assert next_player == 1 or next_player == 2
            next_state = state_after_move(
                board=board, action=action, player=next_player
            )
            values.append(reward + DISCOUNT * state_values[next_state])

        if len(values) == 0:
            board = to_board(state)
            assert check_winner(board) is not None or np.all(board != 0)
        else:
            best_action_index = np.argmax(values)
            policy[state] = [actions[best_action_index]]


def main():
    # board = np.zeros((3, 3))
    # board[1, 2] = 1
    # board[2, 1] = 2
    # state = calculate_state(board)
    # print(state)
    # print_board(board)

    states, state_values, policy = init_state_values_and_policy()
    value_iteration(states=states, state_values=state_values, policy=policy)

    total_actions = 0
    for _key, value_list in policy.items():
        total_actions += len(value_list)
    print(f"Total actions after value iteration: {total_actions}")
    # demo_game(state_values, [(1, 1), (1, 0), (0, 1), (2, 1), (0, 2), (2, 0)])
    # demo_game(state_values, [(1,1),(2,1),(1,0),(1,2),(0,0),(2,2),(2,0),])
    # demo_game(state_values, [(2, 1), (1, 2), (1, 1), (0, 1), (2, 0), (0, 2)])


def demo_game(state_values, actions):
    board = np.zeros((3, 3))

    turn = 1
    for action in actions:
        board[action] = turn
        turn = 2 if turn == 1 else 1
        print_board(board)
        print(state_values[calculate_state(board)], "\n")


if __name__ == "__main__":
    main()
