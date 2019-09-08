import time
import pickle
import torch

import numpy as np
from game import connect4
import mcts
from alpha_zero_learning import Network
import data_storage
from game.globals import Globals





# # play 4 moves and print the board
# board = connect4.BitBoard()
# move = 3
# print("play move ", move)
# board.play_move(move)
#
# move = 1
# print("play move ", move)
# board.play_move(move)
#
# move = 2
# print("play move ", move)
# board.play_move(move)
#
# move = 2
# print("play move ", move)
# board.play_move(move)
# print("board matrix after random moves: ")
# board.print()
#
#
# # play all in one row in order to see if a legal move is removed
# board = connect4.BitBoard()
# board.play_move(5)
# board.print()
# board.play_move(5)
# board.print()
# board.play_move(5)
# board.print()
# board.play_move(5)
# board.print()
# board.play_move(5)
# board.print()
# board.play_move(5)
# board.print()
# print(board.legal_moves)
#
#
#
#
#
# # load the board form a matrix
# mat = np.array(
#     [[0, 0, 0, 2, 0, 0, 0],
#      [0, 0, 0, 1, 0, 0, 0],
#      [0, 0, 0, 2, 0, 0, 0],
#      [0, 0, 0, 1, 0, 0, 0],
#      [2, 0, 0, 2, 0, 0, 0],
#      [1, 0, 0, 1, 0, 0, 0]]
# )
# board = connect4.BitBoard()
# board.from_board_matrix(mat)
# print("loaded board matrix: ")
# board.print()
#
# # play a random game until the finish
# board = connect4.BitBoard()
# while not board.terminal:
#     move = board.random_move()
#     print("play move ", move)
#     board.play_move(move)
# print("board after finshed: ")
# board.print()
#
#
# #play a few random games to measure the time needed
# n_games = 10000
# start_time = time.time()
# for i in range(n_games):
#     board = connect4.BitBoard()
#     while not board.terminal:
#         move = board.random_move()
#         board.play_move(move)
#
# end_time = time.time()
# elapsed_time = end_time - start_time
# time_per_game = elapsed_time / n_games
# print("time per random game: ", time_per_game)
#
#
#
# # test wins
# board = connect4.BitBoard()
# board.play_move(3)
# board.play_move(0)
# board.play_move(4)
# board.play_move(0)
# board.play_move(5)
# board.play_move(0)
# board.play_move(6)
# board.print()
# print(board.terminal)

#test the reward
mat = np.array(
    [[0, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0],
     [1, 2, 2, 2, 0, 1, 0],
     [2, 2, 1, 2, 1, 2, 1],
     [1, 1, 2, 2, 1, 1, 2]]
)
board = connect4.BitBoard()
board.from_board_matrix(mat)
board.play_move(4)
board.print()
print(board.training_reward())



# test policy
best_net = data_storage.load_net("networks/network_gen_37.pt", torch.device('cpu'))
random_net = data_storage.load_net("networks/network_gen_0.pt", torch.device('cpu'))

mat = np.array(
    [[0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 2, 2, 0, 0],
     [2, 0, 0, 1, 1, 1, 0]]
)
board = connect4.BitBoard()
board.from_board_matrix(mat)


# board.play_move(6)
print(board.terminal)

c_puct = 4
temp = 1
mcts_sim_count = 800
mcts_player = mcts.MCTS(c_puct)
policy_best = mcts_player.policy_values(board, {}, best_net, mcts_sim_count, temp, Globals.evaluation_device)
policy_random = mcts_player.policy_values(board, {}, random_net, mcts_sim_count, temp, Globals.evaluation_device)

print("best: ", policy_best)
print("random: ", policy_random)
