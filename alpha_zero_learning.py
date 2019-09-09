from multiprocessing import Manager
import numpy as np
import logging

import torch

from game import connect4
from globals import CONST, Config, Globals
from game import tournament
from mcts import MCTS
import data_storage

logger = logging.getLogger('Connect4')



class Agent:
    def __init__(self, network, mcts_sim_count, c_puct, temp, batch_size, exp_buffer_size):
        """
        :param learning_rate:       learning rate for the neural network
        :param n_filters:           the number of filters in the convolutional layers
        :param mcts_sim_count:      the number of simulations for the monte-carlo tree search
        :param c_puct:              the higher this constant the more the mcts explores
        :param temp:                the temperature, controls the policy value distribution
        :param batch_size:          the experience buffer batch size to train the training network
        :param exp_buffer_size:     the size of the experience replay buffer
        """

        self.network = network                                       # the network
        self.mcts_sim_count = mcts_sim_count                         # the number of simulations for the monte-carlo tree search
        self.c_puct = c_puct                                         # the higher this constant the more the mcts explores
        self.temp = temp                                             # the temperature, controls the policy value distribution
        self.batch_size = batch_size                                 # the size of the experience replay buffer

        self.board = connect4.BitBoard()                             # connect4 board
        self.exp_buffer_size = exp_buffer_size                       # the size of the experience buffer
        self.experience_buffer = ExperienceBuffer(exp_buffer_size)   # buffer that saves all experiences

        # activate the evaluation mode of the networks
        self.network = data_storage.net_to_device(self.network, Config.evaluation_device)
        self.network.eval()


    def clear_experience_buffer(self):
        self.experience_buffer = ExperienceBuffer(self.exp_buffer_size)


    # from utils import utils
    # @utils.profile
    def play_self_play_games(self, network_path, game_count, temp_threshold, alpha_dirich=0):
        """
        plays some games against itself and adds the experience into the experience buffer
        :param network_path     the path of the current network
        :param game_count:      the number of games to play
        :param temp_threshold:  up to this move the temp will be temp, after the threshold it will be set to 0
                                plays a game against itself with some exploratory moves in it
        :param alpha_dirich     alpha parameter for the dirichlet noise that is added to the root node
        :return:                the average nu,ber of moves played in a game
        """

        if Globals.pool is None:
            position_cache = {}
            self_play_results = [__self_play_worker__(network_path, position_cache, self.mcts_sim_count, self.c_puct,
                                                      temp_threshold, self.temp, alpha_dirich, game_count)]
        else:
            # create the shared dict for the position cache
            manager = Manager()
            position_cache = manager.dict()

            games_per_process = int(game_count / Globals.n_pool_processes)
            self_play_results = Globals.pool.starmap(__self_play_worker__,
                                                      zip([network_path] * Globals.n_pool_processes,
                                                          [position_cache] * Globals.n_pool_processes,
                                                          [self.mcts_sim_count] * Globals.n_pool_processes,
                                                          [self.c_puct] * Globals.n_pool_processes,
                                                          [temp_threshold] * Globals.n_pool_processes,
                                                          [self.temp] * Globals.n_pool_processes,
                                                          [alpha_dirich] * Globals.n_pool_processes,
                                                          [games_per_process] * Globals.n_pool_processes))

        # add the training examples to the experience buffer
        logger.info("start to augment the training examples by using the game symmetry")
        tot_moves_played = 0
        for sample in self_play_results:
            state = torch.Tensor(sample[0]).to(Config.training_device)
            policy = torch.Tensor(sample[1]).reshape(-1, CONST.BOARD_WIDTH).to(Config.training_device)
            value = torch.Tensor(sample[2]).unsqueeze(1).to(Config.training_device)
            tot_moves_played += state.shape[0]

            # add the original data
            self.experience_buffer.add_batch(state, policy, value)

            # flip it vertically
            new_state = torch.flip(state, [3])
            new_policy = torch.flip(policy, [1])
            self.experience_buffer.add_batch(new_state, new_policy, value)

        avg_game_length = tot_moves_played / game_count
        return avg_game_length



    def nn_update(self, epoch_count):
        """
        updates the neural network by picking a random batch form the experience replay
        :param epoch_count:     number of times to pass all training examples through the network
        :return:                average policy and value loss over all mini batches
        """

        # activate the training mode
        self.network = data_storage.net_to_device(self.network, Config.training_device)
        self.network.train()

        avg_loss_p = 0
        avg_loss_v = 0
        tot_batch_count = 0
        for epoch in range(epoch_count):
            # shuffle all training examples
            num_training_examples = self.experience_buffer.size - (self.experience_buffer.size % self.batch_size)
            states, policies, values = self.experience_buffer.random_batch(num_training_examples)

            # train the network with all training examples
            num_batches = int(num_training_examples / self.batch_size)
            for batch_num in range(num_batches):
                start_batch = batch_num * self.batch_size
                end_batch = (batch_num+1) * self.batch_size
                loss_p, loss_v = self.network.train_step(states[start_batch:end_batch],
                                                         policies[start_batch:end_batch],
                                                         values[start_batch:end_batch])
                avg_loss_p += loss_p
                avg_loss_v += loss_v
                tot_batch_count += 1

        # calculate the mean of the loss
        avg_loss_p /= tot_batch_count
        avg_loss_v /= tot_batch_count

        # activate the evaluation mode
        self.network = data_storage.net_to_device(self.network, Config.evaluation_device)
        self.network.eval()

        return avg_loss_p.item(), avg_loss_v.item()

    
    
    def play_against_random(self, color, game_count):
        """
        lets the agent play against a random player
        :param color:           the color of the agent
        :param game_count:      the number of games that are played
        :return:                the mean score against the random player 0: lose, 0.5 draw, 1: win
        """

        az_player = tournament.AlphaZeroPlayer(self.network, self.c_puct, self.mcts_sim_count, 0)
        random_player = tournament.RandomPlayer()
        score = tournament.play_one_color(game_count, az_player, color, random_player)
        return score


class ExperienceBuffer:
    def __init__(self, max_size):
        self.max_size = max_size

        self.state = torch.empty(max_size, 2, 6, 7).to(Config.training_device)
        self.policy = torch.empty(max_size, CONST.BOARD_WIDTH).to(Config.training_device)
        self.value = torch.empty(max_size, 1).to(Config.training_device)

        self.size = 0  # size of the buffer
        self.ring_index = 0  # current index of where the next sample is added


    def add_batch(self, states, policies, values):
        """
        adds the multiple experiences to the buffer
        :param states:           the state s_t
        :param policies:         probability value for all actions
        :param values:           value of the current state
        :return:
        :return:
        """

        sample_count = values.shape[0]
        start_index = self.ring_index
        end_index = self.ring_index + sample_count

        # check if the index is not too large
        if end_index > self.max_size:
            end_index = self.max_size
            batch_end_index = end_index - start_index

            # add all elements until the end of the ring buffer array
            self.add_batch(states[0:batch_end_index, :],
                           policies[0:batch_end_index, :],
                           values[0:batch_end_index])

            # add the rest of the elements at the beginning of the buffer
            self.add_batch(states[batch_end_index:, :],
                           policies[batch_end_index:, :],
                           values[batch_end_index:])
            return

        # add the elements into the ring buffer
        self.state[start_index:end_index, :] = states
        self.policy[start_index:end_index, :] = policies
        self.value[start_index:end_index] = values

        # update indices and size
        self.ring_index += sample_count
        self.ring_index = self.ring_index % self.max_size

        if self.size < self.max_size:
            self.size += sample_count


    def random_batch(self, batch_size):
        """
        returns a random batch of the experience buffer
        :param batch_size:   the size of the batch
        :return:             state, policy, value
        """

        sample_size = batch_size if self.size > batch_size else self.size
        idx = np.random.choice(self.size, sample_size, replace=False)
        return self.state[idx, :], self.policy[idx, :], self.value[idx]


def net_vs_net(net1, net2, game_count, mcts_sim_count, c_puct, temp):
    """
    lets two alpha zero networks play against each other
    :param net1:            net for player 1
    :param net2:            net for player 2
    :param game_count:      total games to play
    :param mcts_sim_count   number of monte carlo simulations
    :param c_puct           constant that controls the exploration
    :param temp             the temperature
    :return:                score of network1
    """

    az_player1 = tournament.AlphaZeroPlayer(net1, c_puct, mcts_sim_count, temp)
    az_player2 = tournament.AlphaZeroPlayer(net2, c_puct, mcts_sim_count, temp)
    score1 = tournament.play_match(game_count, az_player1, az_player2)
    return score1


def __self_play_worker__(network_path, position_cache, mcts_sim_count, c_puct, temp_threshold, temp, alpha_dirich, game_count):
    """
    plays a number of self play games
    :param network_path:        path of the network
    :param position_cache:      holds positions already evaluated by the network
    :param mcts_sim_count:      the monte carlo simulation count
    :param c_puct:              constant that controls the exploration
    :param temp_threshold:      up to this move count the temperature will be temp, later it will be 0
    :param temp:                the temperature
    :param alpha_dirich:        dirichlet parameter alpha
    :param game_count:          the number of self-play games to play
    :return:                    state_list, policy_list, value_list
    """

    # load the network
    net = data_storage.load_net(network_path, Config.evaluation_device)

    state_list = []
    policy_list = []
    value_list = []
    q_list = []
    # position_cache = {}   # give each simulation its own state dict
    # position_count_dict = {}

    for i in range(game_count):
        board = connect4.BitBoard()
        mcts = MCTS(c_puct)  # reset the search tree

        # reset the players list
        player_list = []

        move_count = 0
        while not board.terminal:
            state, player = board.white_perspective()
            temp = 0 if move_count >= temp_threshold else temp
            policy = mcts.policy_values(board, position_cache, net, mcts_sim_count, temp, alpha_dirich)
            s = board.state_id()

            # sample from the policy to determine the move to play
            move = np.random.choice(len(policy), p=policy)
            q_value = mcts.Q.get((s, move), 0)
            board.play_move(move)

            # save the training example
            state_list.append(state)
            player_list.append(player)
            policy_list.append(policy)
            q_list.append(q_value)
            move_count += 1

        # calculate the values from the perspective of the player who's move it is
        # modification to alpha-zero: average value and q_value
        reward = board.training_reward()
        for idx, player in enumerate(player_list):
            value = (reward + q_list[idx]) / 2
            value = value if player == CONST.WHITE else -value
            value_list.append(value)

    # delete the network
    del net
    torch.cuda.empty_cache()

    return state_list, policy_list, value_list
