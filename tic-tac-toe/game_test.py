import numpy as np
import unittest

from game import calculate_state, to_board, check_winner, get_next_player


class GameTest(unittest.TestCase):
    def test_calculate_state(self):
        for k in range(3000):
            x = np.zeros((3, 3))
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    x[i, j] = np.random.randint(0, 3)
            state = calculate_state(x)
            assert np.array_equal(x, to_board(state))

    def test_row_win_x(self):
        board = [[1, 1, 1], [2, 2, 0], [0, 2, 0]]
        self.assertEqual(check_winner(board), 1)

    def test_row_win_o(self):
        board = [[1, 1, 0], [2, 2, 2], [1, 0, 0]]
        self.assertEqual(check_winner(board), 2)

    def test_column_win_x(self):
        board = [[1, 2, 0], [1, 2, 0], [1, 0, 0]]
        self.assertEqual(check_winner(board), 1)

    def test_column_win_o(self):
        board = [[1, 2, 0], [1, 2, 0], [0, 2, 0]]
        self.assertEqual(check_winner(board), 2)

    def test_diagonal_win_x(self):
        board = [[1, 2, 0], [2, 1, 0], [0, 2, 1]]
        self.assertEqual(check_winner(board), 1)

    def test_diagonal_win_x2(self):
        board = [[1, 2, 1], [2, 1, 0], [1, 0, 2]]
        self.assertEqual(check_winner(board), 1)

    def test_draw(self):
        board = [[1, 2, 2], [2, 1, 1], [1, 2, 2]]
        self.assertEqual(check_winner(board), None)

    def test_incomplete_game(self):
        board = [[1, 2, 1], [2, 0, 0], [1, 2, 0]]
        self.assertEqual(check_winner(board), None)

    def test_empty_board(self):
        # Test case where the board is empty (no winner)
        board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.assertEqual(check_winner(board), None)

    def test_get_next_player(self):
        board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.assertEqual(get_next_player(board), 1)

        board = [[1, 2, 0], [0, 0, 0], [0, 0, 0]]
        self.assertEqual(get_next_player(board), 1)

        board = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.assertEqual(get_next_player(board), 2)

        board = [[1, 2, 1], [2, 0, 0], [0, 0, 0]]
        self.assertEqual(get_next_player(board), 1)

        board = [[1, 2, 1], [2, 1, 2], [2, 1, 1]]
        self.assertEqual(get_next_player(board), None)

        board = [[1, 2, 1], [0, 2, 0], [0, 0, 0]]
        self.assertEqual(get_next_player(board), 1)

        board = [[1, 1, 2], [2, 1, 0], [0, 0, 2]]
        self.assertEqual(get_next_player(board), 1)


# python -m unittest game_test.py

if __name__ == "__main__":
    unittest.main()
