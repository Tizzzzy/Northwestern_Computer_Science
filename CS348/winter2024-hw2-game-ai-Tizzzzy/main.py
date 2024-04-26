from tictactoe import TicTacToe
import math
import random
import time


def minimax(board, depth, maximizing_player, ai_player):

    if ai_player == 'X':
        human_player = 'O'
    else:
        human_player = 'X'

    if board.empty_squares_available() is False:
        return {'position': None, 'score': 0}
    elif board.current_winner == ai_player:
        return {'position': None, 'score': 1 * (board.num_empty_squares() + 1)}
    elif board.current_winner == human_player:
        return {'position': None, 'score': -1 * (board.num_empty_squares() + 1)}
    

    if maximizing_player:
        best_move = {'position': None, 'score': float('-inf')}
        for position in board.available_moves():
            board.make_move(position, ai_player)
            temp_move = minimax(board, depth - 1, not maximizing_player, ai_player)
            board.board[position] = ' '
            board.current_winner = None
            temp_move['position'] = position

            if temp_move['score'] > best_move['score']:
                best_move = temp_move
    
    else:
        best_move = {'position': None, 'score': float('inf')}
        for position in board.available_moves():
            board.make_move(position, human_player)
            temp_move = minimax(board, depth - 1, not maximizing_player, ai_player)
            board.board[position] = ' '
            board.current_winner = None
            temp_move['position'] = position

            if temp_move['score'] < best_move['score']:
                best_move = temp_move
    
    return best_move

def minimax_with_alpha_beta(board, depth, alpha, beta, maximizing_player, ai_player):

    if ai_player == 'X':
        human_player = 'O'
    else:
        human_player = 'X'

    if board.empty_squares_available() is False:
        return {'position': None, 'score': 0}
    elif board.current_winner == ai_player:
        return {'position': None, 'score': 1 * (board.num_empty_squares() + 1)}
    elif board.current_winner == human_player:
        return {'position': None, 'score': -1 * (board.num_empty_squares() + 1)}

    if maximizing_player:
        best_move = {'position': None, 'score': float('-inf')}
        for position in board.available_moves():
            board.make_move(position, ai_player)
            temp_move = minimax_with_alpha_beta(board, depth - 1, alpha, beta, not maximizing_player, ai_player)
            board.board[position] = ' '
            board.current_winner = None
            temp_move['position'] = position

            if temp_move['score'] > best_move['score']:
                best_move = temp_move
            alpha = max(alpha, best_move['score'])
            if beta <= alpha:
                break
            
    
    else:
        best_move = {'position': None, 'score': float('inf')}
        for position in board.available_moves():
            board.make_move(position, human_player)
            temp_move = minimax_with_alpha_beta(board, depth - 1, alpha, beta, not maximizing_player, ai_player)
            board.board[position] = ' '
            board.current_winner = None
            temp_move['position'] = position

            if temp_move['score'] < best_move['score']:
                best_move = temp_move
            beta = min(beta, best_move['score'])
            if beta <= alpha:
                break
    
    return best_move

def play_game_human_moves_first():

    game = TicTacToe()
    print("\nInitial Board:")
    game.print_board()

    letter = 'X'  # Human player starts first.
    while game.empty_squares_available():
        if letter == 'O':  # AI's turn
            square = minimax_with_alpha_beta(game, len(game.available_moves()), -math.inf, math.inf, True, 'O')['position']
            if square is None:
                print("\nGame is a draw!")
                break
            game.make_move(square, letter)
            print(f"\nAI (O) chooses square {square + 1}")
        else:
            valid_square = False
            while not valid_square:
                square = input(f"\n{letter}'s turn. Input move (1-9): ")
                try:
                    square = int(square) - 1
                    if square not in game.available_moves():
                        raise ValueError
                    valid_square = True
                    game.make_move(square, letter)
                except ValueError:
                    print("\nInvalid square. Try again.")

        game.print_board()

        if game.current_winner:
            print(f"\n{letter} wins!")
            break

        letter = 'O' if letter == 'X' else 'X'  # Switch turns.
    else:
        print("\nIt's a draw!")

def play_game_ai_moves_first():

    game = TicTacToe()
    print("\nInitial Board:")
    game.print_board()

    first_move = True

    letter = 'O'  # AI player starts first.
    while game.empty_squares_available():
        if letter == 'O':  # AI's turn
            if first_move:
                square = random.randint(0, 8)
                first_move = False
            else:
                square = minimax_with_alpha_beta(game, len(game.available_moves()), -math.inf, math.inf, True, 'O')['position']
            if square is None:
                print("\nGame is a draw!")
                break
            game.make_move(square, letter)
            print(f"\nAI (O) chooses square {square + 1}")
        else:
            valid_square = False
            while not valid_square:
                square = input(f"\n{letter}'s turn. Input move (1-9): ")
                try:
                    square = int(square) - 1
                    if square not in game.available_moves():
                        raise ValueError
                    valid_square = True
                    game.make_move(square, letter)
                except ValueError:
                    print("\nInvalid square. Try again.")

        game.print_board()

        if game.current_winner:
            print(f"\n{letter} wins!")
            break

        letter = 'O' if letter == 'X' else 'X'  # Switch turns.
    else:
        print("\nIt's a draw!")

def play_game_human_vs_human():

    game = TicTacToe()
    print("\nInitial Board:")
    game.print_board()

    letter = 'O'  # Human (O) player starts first.
    while game.empty_squares_available():
        if letter == 'O':  # Human (O)'s turn
            valid_square = False
            while not valid_square:
                square = input(f"\n{letter}'s turn. Input move (1-9): ")
                try:
                    square = int(square) - 1
                    if square not in game.available_moves():
                        raise ValueError
                    valid_square = True
                    game.make_move(square, letter)
                except ValueError:
                    print("\nInvalid square. Try again.")

                if square is None:
                    print("\nGame is a draw!")
                    break
                game.make_move(square, letter)
                print(f"\nAI (O) chooses square {square + 1}")
        else:
            valid_square = False
            while not valid_square:
                square = input(f"\n{letter}'s turn. Input move (1-9): ")
                try:
                    square = int(square) - 1
                    if square not in game.available_moves():
                        raise ValueError
                    valid_square = True
                    game.make_move(square, letter)
                except ValueError:
                    print("\nInvalid square. Try again.")

        game.print_board()

        if game.current_winner:
            print(f"\n{letter} wins!")
            break

        letter = 'O' if letter == 'X' else 'X'  # Switch turns.
    else:
        print("\nIt's a draw!")

def play_game_ai_vs_ai():

    game = TicTacToe()
    print("\nInitial Board:")
    game.print_board()

    first_move = True

    letter = 'O'  # AI (O) player starts first.
    while game.empty_squares_available():
        if letter == 'O':  # AI (O)'s turn
            if first_move:
                square = random.randint(0, 8)
                first_move = False
            else:
                square = minimax_with_alpha_beta(game, len(game.available_moves()), -math.inf, math.inf, True, 'O')['position']
            if square is None:
                print("\nGame is a draw!")
                break
            game.make_move(square, letter)
            print(f"\nAI (O) chooses square {square + 1}")
            time.sleep(0.75)
        else:
            square = minimax_with_alpha_beta(game, len(game.available_moves()), -math.inf, math.inf, True, 'O')['position']
            if square is None:
                print("\nGame is a draw!")
                break
            game.make_move(square, letter)
            print(f"\nAI (X) chooses square {square + 1}")
            time.sleep(0.75)

        game.print_board()

        if game.current_winner:
            print(f"\n{letter} wins!")
            break

        letter = 'O' if letter == 'X' else 'X'  # Switch turns.
    else:
        print("\nIt's a draw!")


if __name__ == '__main__':

    print("""
Modes of play available:

    hh: Hooman vs. hooman
    ha: Hooman vs. AI
    ah: AI vs. Hooman - AI makes first move
    aa: AI vs. AI""")

    valid_move = False
    while not valid_move:
        mode = input("\nEnter preferred mode of play (e.g., aa): ")
        try:
            if mode not in ["hh", "ha", "ah", "aa"]:
                raise ValueError
            valid_move = True
            if mode == "hh":
                play_game_human_vs_human()
            elif mode == "ha":
                play_game_human_moves_first()
            elif mode == "ah":
                play_game_ai_moves_first()
            else:
                play_game_ai_vs_ai()
        except ValueError:
            print("\nInvalid option entered. Try again.")

