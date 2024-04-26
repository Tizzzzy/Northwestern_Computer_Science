from main import minimax, minimax_with_alpha_beta
from tictactoe import TicTacToe
import math
import unittest
import time


class TestTicTacToeAI(unittest.TestCase):

    def setUp(self):
        self.game = TicTacToe()

    def test_minimax_easy_win(self):
        """ Minimax should find an immediate winning move. """
        self.game.board = ['X', 'X', ' ',
                           'O', 'O', ' ',
                           ' ', ' ', ' ']
        best_move = minimax(self.game, len(self.game.available_moves()), True, 'O')
        self.assertEqual(best_move['position'], 5)

    def test_minimax_block_win(self):
        """ Minimax should block opponent's winning move. """
        self.game.board = ['X', 'X', ' ',
                           ' ', 'O', ' ',
                           ' ', ' ', ' ']
        best_move = minimax(self.game, len(self.game.available_moves()), True, 'O')
        self.assertEqual(best_move['position'], 2)

    def test_minimax_prevent_fork(self):
        """ Minimax should prevent a fork. """
        self.game.board = ['X', ' ', ' ',
                           ' ', 'O', ' ',
                           ' ', ' ', 'X']
        best_move = minimax(self.game, len(self.game.available_moves()), True, 'O')
        self.assertIn(best_move['position'], [1, 3, 5, 7])

    def test_minimax_optimal_move(self):
        """ Minimax should choose the optimal move in a non-trivial scenario. """
        self.game.board = ['X', 'O', 'X',
                           'X', 'O', ' ',
                           ' ', ' ', ' ']
        best_move = minimax(self.game, len(self.game.available_moves()), True, 'O')
        self.assertEqual(best_move['position'], 7)

    def test_alpha_beta_easy_win(self):
        """ Minimax with Alpha-Beta Pruning should find an immediate winning move. """
        self.game.board = ['X', 'X', ' ',
                           'O', 'O', ' ',
                           ' ', ' ', ' ']
        best_move = minimax_with_alpha_beta(self.game, len(self.game.available_moves()), -math.inf, math.inf, True, 'O')
        self.assertEqual(best_move['position'], 5)

    def test_alpha_beta_block_win(self):
        """ Minimax with Alpha-Beta Pruning should block opponent's winning move. """
        self.game.board = ['X', 'X', ' ',
                           ' ', 'O', ' ',
                           ' ', ' ', ' ']
        best_move = minimax_with_alpha_beta(self.game, len(self.game.available_moves()), -math.inf, math.inf, True, 'O')
        self.assertEqual(best_move['position'], 2)

    def test_alpha_beta_prevent_fork(self):
        """ Minimax with Alpha-Beta Pruning should prevent a fork. """
        self.game.board = ['X', ' ', ' ',
                           ' ', 'O', ' ',
                           ' ', ' ', 'X']
        best_move = minimax_with_alpha_beta(self.game, len(self.game.available_moves()), -math.inf, math.inf, True, 'O')
        self.assertIn(best_move['position'], [1, 3, 5, 7])

    def test_alpha_beta_optimal_move(self):
        """ Minimax with Alpha-Beta Pruning should choose the optimal move in a non-trivial scenario. """
        self.game.board = ['X', 'O', 'X',
                           'X', 'O', ' ',
                           ' ', ' ', ' ']
        best_move = minimax_with_alpha_beta(self.game, len(self.game.available_moves()), -math.inf, math.inf, True, 'O')
        self.assertEqual(best_move['position'], 7)

    def test_correctness(self):
        """ Minimax with Alpha-Beta Pruning should find the same moves as Minimax. """
        depth = 3
        regular_move = minimax(self.game, depth, True, 'O')
        alpha_beta_move = minimax_with_alpha_beta(self.game, depth, float('-inf'), float('inf'), True, 'O')
        self.assertEqual(regular_move, alpha_beta_move)

    def test_performance(self):
        """ Minimax with Alpha-Beta Pruning should find a move faster than Minimax. """
        depth = 7
        start = time.time()
        minimax(self.game, depth, True, 'O')
        regular_time = time.time() - start

        start = time.time()
        minimax_with_alpha_beta(self.game, depth, float('-inf'), float('inf'), True, 'O')
        alpha_beta_time = time.time() - start

        self.assertTrue(alpha_beta_time < regular_time)

    def test_minimax_ad_hoc(self):
        """ Minimax with Alpha-Beta Pruning should choose to win when faced with winning vs. blocking a win. """
        self.game.board = ['X', ' ', 'X',
                           'X', 'O', ' ',
                           ' ', 'O', ' ']
        best_move = minimax_with_alpha_beta(self.game, len(self.game.available_moves()), -math.inf, math.inf, True, 'O')
        self.assertEqual(best_move['position'], 1)

    def test_alpha_beta_ad_hoc(self):
        """ Minimax with Alpha-Beta Pruning should pick a move that preemptively guards against a sure loss. """
        self.game.board = ['X', ' ', ' ',
                           ' ', 'O', 'X',
                           ' ', ' ', ' ']
        best_move = minimax_with_alpha_beta(self.game, len(self.game.available_moves()), -math.inf, math.inf, True, 'O')
        self.assertEqual(best_move['position'], 1)


if __name__ == '__main__':
    unittest.main()
