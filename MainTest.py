import time

import numpy as np
from game import connect4

# play 4 random moves and print the board
board = connect4.BitBoard()
move = board.random_move()
print("play move ", move)
board.play_move(move)

move = board.random_move()
print("play move ", move)
board.play_move(move)

move = board.random_move()
print("play move ", move)
board.play_move(move)

move = board.random_move()
print("play move ", move)
board.play_move(move)
print("board matrix after random moves: ")
board.print()


# load the board form a matrix
mat = np.array(
    [[0, 1, 0, 0, 0, 0],
     [0, 0, 2, 0, 0, 0],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 2, 0, 0, 0]]
)
board = connect4.BitBoard()
board.from_board_matrix(mat)
print("loaded board matrix: ")
board.print()

# play a random game until the finish
board = connect4.BitBoard()
while not board.terminal:
    move = board.random_move()
    print("play move ", move)
    board.play_move(move)
print("board after finshed: ")
board.print()


# play a few random games to measure the time needed
n_games = 10000
start_time = time.time()
for i in range(n_games):
    board = connect4.BitBoard()
    while not board.terminal:
        move = board.random_move()
        board.play_move(move)

end_time = time.time()
elapsed_time = end_time - start_time
time_per_game = elapsed_time / n_games
print("time per random game: ", time_per_game)


mat = np.array(
    [[1, 0, 0, 0, 2, 0],
     [0, 0, 0, 0, 1, 0],
     [2, 2, 0, 2, 0, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0]]
)
board = connect4.BitBoard()
board.from_board_matrix(mat)
board.print()

board.play_move(31)
board.print()

