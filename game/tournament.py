from globals import CONST
from game import connect4
import numpy as np
import copy

import mcts


class RandomPlayer:
    """
    player that makes a random legal move
    """
    def __init__(self):
        pass


    def play_move(self, board):
        move = board.random_move()
        board.play_move(move)


class AlphaZeroPlayer:
    """
    player that makes a move according to the alpha zero algorithm
    """

    def __init__(self, net, c_puct, mcts_sim_count, temp):
        """
        :param net:                 alpha zero network
        :param c_puct:              constant that controls the exploration
        :param mcts_sim_count:      the number of monte-carlo simulation counts
        :param temp:                the temperature
        """
        self.net = net
        self.mcts_player = mcts.MCTS(c_puct)
        self.mcts_sim_count = mcts_sim_count
        self.temp = temp
        self.position_cache = {}


    def play_move(self, board):
        policy = self.mcts_player.policy_values(board, self.position_cache, self.net, self.mcts_sim_count, self.temp)
        if self.temp == 0:
            move = np.where(policy == 1)[0]
        else:
            move = np.random.choice(len(policy), p=policy)

        board.play_move(move)


def play_one_color(game_count, player1, color1, player2):
    """
    lets the two passed players play against each other. the players will play all
    games with the same colors.
    the players will get the following scores:
    loss:  0
    draw:  0.5
    win:   1
    :param game_count:  the number of games per match
    :param player1:     player 1
    :param color1       the color of player 1
    :param player2:     player 2
    :return:            average score of the player1 between 0 and 1
    """
    score_player1 = 0

    for _ in range(game_count):
        # play half the games where player1 is white
        board = connect4.BitBoard()
        while not board.terminal:
            if board.player == color1:
                player1.play_move(board)
            else:
                player2.play_move(board)

        score = board.white_score() if color1 == CONST.WHITE else board.black_score()
        score_player1 += score

    return score_player1 / game_count



def play_match(game_count, player1, player2):
    """
    lets the two passed players play against each other. the number of matches need to be even
    or the total number of games will be the next lower even number
    each player will play half the games as white and half the games as black
    the players will get the following scores:
    loss:  0
    draw:  0.5
    win:   1
    :param game_count:  the number of games per match
    :param player1:     player 1
    :param player2:     player 2
    :return:            average score of the player1 between 0 and 1
    """
    half_game_count = int(game_count / 2)
    score_player1 = 0

    # clone the players in order to play have the same match conditions if the colors are switched
    player1_clone = copy.deepcopy(player1)
    player2_clone = copy.deepcopy(player2)

    for _ in range(half_game_count):
        # play half the games where player1 is white
        board = connect4.BitBoard()
        while not board.terminal:
            if board.player == CONST.WHITE:
                player1_clone.play_move(board)
            else:
                player2_clone.play_move(board)

        score_player1 += board.white_score()

        # play half the games where player1 is black
        board = connect4.BitBoard()
        while not board.terminal:
            if board.player == CONST.WHITE:
                player2.play_move(board)
            else:
                player1.play_move(board)

        score_player1 += board.black_score()

    return score_player1 / (2*half_game_count)
