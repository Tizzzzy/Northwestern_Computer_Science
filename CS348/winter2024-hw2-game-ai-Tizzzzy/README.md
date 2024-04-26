# HW 2 - Game AI

## Context

You're asked to create an AI-powered Tic-tac-toe, one that's hard (impossible?) for humans to beat. You up for the challenge? If you're thinking you're going to power its "brain" using Minimax, then you're thinking right!


## Your Task

We want you to focus on the smarts for the game, so we've implemented the game shell for you, but it lacks the engine that can power its thinking. As such, you're asked to implement two functions: `minimax()`, which is vanilla Minimax, and `minimax_with_alpha_beta()`, which is Minimax with Alpha-Beta Pruning. They should both pick the same moves when playing Tic-tac-toe, but the latter should prove to be faster than the former.

Some things to notice:

1. `main.py` implements four modes of game play:
* **Human-human**, whereby you can launch the game and play it against another human (while sharing the same keyboard),
* **Human-AI**, whereby you get to face off against an AI, with you getting to make the first move,
* **AI-Human**, whereby you get to face off against an AI as well, but the AI gets to make the first move, and
* **AI-AI**, whereby you can see an AI battle it out with another AI (Think T-800 vs. T-1000. What? You don't get this reference? Drop everything and watch Terminator 2. NOW!!!).

Your Minimax and Minimax with Alpha-Beta Pruning implementations will apply to all but the first (human-human) mode, and the different modes should come in handy when debugging your implementation, which will eventually have to pass the tests in `test.py`.

2. Notice that both the `minimax()` and `minimax_with_alpha_beta()` functions share the same arguments except that the latter also accepts *Alpha* and *Beta* variables. You can learn about how the arguments are to be used by inspecting the functions that implement the four different game plays. Specifically, `board` is simply the game that you will create using the `TicTacToe` class. Also, `depth` is actually not used, but it's there in case you want to try out Minimax or Minimax with Alpha-Beta Pruning that stops traversing the game tree after a pre-set depth as opposed to looking all the way down at the leaves. The tests in `test.py` assume going down all the way to the leaves, though. Notice that when any of the game play functions calls the Minimax or Minimax with Alpha-Beta Pruning function, it passes `len(game.available_moves())` into the `depth` argument, which would typically be decremented by your Minimax or Minimax with Alpha-Beta Pruning algorithm for every level of the game tree traversed. You can ignore the handling of the variable if you want to.

3. In `test.py`, there are 12 tests, with the last two tests ending in the substring "ad_hoc". Those two tests carry 0.5 points each and bring the total number of points for this homework to 11, which contains the regular 10 points and 1 extra-credit point.
  
4. To calculate utility at the leaves, you may use +1, 0, and -1 for win, draw, and loss, or you may choose any other similar scheme.
