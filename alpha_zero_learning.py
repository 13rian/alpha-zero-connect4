from multiprocessing import Manager
import numpy as np
import logging
from operator import itemgetter

import torch

from game import connect4
from globals import CONST, Config, Globals
from game import tournament
from mcts import MCTS
import data_storage

logger = logging.getLogger('Connect4')



class Agent:
    def __init__(self, network):
        """
        :param network:       alpha zero network that is used for training and evaluation
        """

        self.network = network                        # the network
        self.experience_buffer = ExperienceBuffer()   # buffer that saves all experiences

        # activate the evaluation mode of the networks
        self.network = data_storage.net_to_device(self.network, Config.evaluation_device)
        self.network.eval()


    # from utils import utils
    # @utils.profile
    def play_self_play_games(self, network_path):
        """
        plays some games against itself and adds the experience into the experience buffer
        :param network_path     the path of the current network
        :param game_count:      the number of games to play
        :return:                the average nu,ber of moves played in a game
        """

        if Globals.pool is None:
            position_cache = {}
            self_play_results = [__self_play_worker__(network_path, position_cache, Config.episode_count)]
        else:
            # create the shared dict for the position cache
            manager = Manager()
            position_cache = manager.dict()

            games_per_process = int(Config.episode_count / Globals.n_pool_processes)
            self_play_results = Globals.pool.starmap(__self_play_worker__,
                                                      zip([network_path] * Globals.n_pool_processes,
                                                          [position_cache] * Globals.n_pool_processes,
                                                          [games_per_process] * Globals.n_pool_processes))

        # add the training examples to the experience buffer
        logger.info("start to prepare the training data")
        self.experience_buffer.add_new_cycle()
        tot_moves_played = 0
        for sample in self_play_results:
            tot_moves_played += len(sample) / 2            # divide by two as symmetric positions were added

            # add the original data
            self.experience_buffer.add_data(sample)

        self.experience_buffer.prepare_data(Config.window_size)

        avg_game_length = tot_moves_played / Config.episode_count
        return avg_game_length


    def nn_update(self):
        """
        updates the neural network by picking a random batch form the experience replay
        :return:                average policy and value loss over all mini batches
        """

        # activate the training mode
        self.network = data_storage.net_to_device(self.network, Config.training_device)
        self.network.train()

        avg_loss_p = 0
        avg_loss_v = 0
        tot_batch_count = 0
        for epoch in range(Config.epoch_count):
            # shuffle all training examples
            num_training_examples = self.experience_buffer.data_size - (self.experience_buffer.data_size % Config.batch_size)
            states, policies, values = self.experience_buffer.random_batch(num_training_examples)

            # train the network with all training examples
            num_batches = int(num_training_examples / Config.batch_size)
            for batch_num in range(num_batches):
                start_batch = batch_num * Config.batch_size
                end_batch = (batch_num+1) * Config.batch_size
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

        az_player = tournament.AlphaZeroPlayer(self.network, Config.c_puct, Config.mcts_sim_count, 0)
        random_player = tournament.RandomPlayer()
        score = tournament.play_one_color(game_count, az_player, color, random_player)
        return score


class ExperienceBuffer:
    def __init__(self):
        self.training_cycles = []       # holds a list with the training data of different cycles

        # to save the current training data
        self.data_size = 0
        self.state = None
        self.policy = None
        self.value = None


    def add_new_cycle(self):
        """
        adds a new training cycle to the training cycle list
        :return:
        """
        self.training_cycles.append([])


    def add_data(self, training_examples):
        """
        adds the passed training examples to the experience buffer
        :param training_examples:    list containing the training examples
        :return:
        """

        self.training_cycles[-1] += training_examples


    def prepare_data(self, window_size):
        """
        prepares the training data for training
        :param window_size:     the size of the window (number of cycles)
        :return:
        """

        # get rid of old data
        while len(self.training_cycles) > window_size:
            self.training_cycles.pop(0)

        # get the whole training data
        training_data = []
        for sample in self.training_cycles:
            training_data += sample

        # average the positions (early positions are overrepresented)
        training_data = self.__average_positions__(training_data)


        # prepare the training data
        self.data_size = len(training_data)
        self.state = torch.empty(self.data_size, 2, CONST.BOARD_HEIGHT, CONST.BOARD_WIDTH)
        self.policy = torch.empty(self.data_size, CONST.BOARD_WIDTH).to(Config.evaluation_device)
        self.value = torch.empty(self.data_size, 1)

        for idx, sample in enumerate(training_data):
            self.state[idx, :] = torch.Tensor(sample.get("state"))
            self.policy[idx, :] = torch.Tensor(sample.get("policy"))
            self.value[idx, :] = sample.get("value")

        # # copy everything to the training device
        # self.state = self.state.to(Config.training_device)
        # self.policy = self.policy.to(Config.training_device)
        # self.value = self.value.to(Config.training_device)


    def __average_positions__(self, training_data):
        """
        calculates the average over same positions, since connect4 only has a few reasonable starting
        lines the position at the beginning are overrepresented in the data set.
        :param training_data:   list of training samples
        :return:                list of averaged training samples that does not contain position duplicates
        """
        training_data = sorted(training_data, key=itemgetter('state_id'))

        training_data_avg = []
        state_id = training_data[0].get("state_id")
        state = training_data[0].get("state")
        position_count = 0
        policy = 0
        value = 0
        for position in training_data:
            if state_id == position.get("state_id"):
                policy = np.add(policy, position.get("policy"))
                value += position.get("value")
                position_count += 1

            else:
                policy = np.divide(policy, np.sum(policy))     # normalize the policy

                averaged_sample = {
                    "state_id": state_id,
                    "state": state,
                    "policy": policy,
                    "value": value / position_count
                }
                training_data_avg.append(averaged_sample)

                state_id = position.get("state_id")
                state = position.get("state")
                policy = position.get("policy")
                value = position.get("value")
                position_count = 1

        size_reduction = (len(training_data) - len(training_data_avg)) / len(training_data)
        logger.debug("size reduction due to position averaging: {}".format(size_reduction))
        return training_data_avg


    def random_batch(self, batch_size):
        """
        returns a random batch of the experience buffer
        :param batch_size:   the size of the batch
        :return:             state, policy, value
        """
        sample_size = batch_size if self.data_size > batch_size else self.data_size
        idx = np.random.choice(self.data_size, sample_size, replace=False)
        return self.state[idx, :], self.policy[idx, :], self.value[idx]


def net_vs_net_mcts(net1, net2, game_count, mcts_sim_count, c_puct, temp):
    """
    lets two alpha zero networks play against each other using the network and mcts
    :param net1:            net for player 1
    :param net2:            net for player 2
    :param game_count:      total games to play
    :param mcts_sim_count   number of monte carlo simulations
    :param c_puct           constant that controls the exploration
    :param temp             the temperature
    :return:                score of network1
    """

    az_player1 = tournament.AlphaZeroPlayerMcts(net1, c_puct, mcts_sim_count, temp)
    az_player2 = tournament.AlphaZeroPlayerMcts(net2, c_puct, mcts_sim_count, temp)
    score1 = tournament.play_match(game_count, az_player1, az_player2)
    return score1


def net_vs_net(net1, net2, game_count):
    """
    lets two alpha zero networks play against each other using only the network policy without mcts
    :param net1:            net for player 1
    :param net2:            net for player 2
    :param game_count:      total games to play
    :return:                score of network1
    """

    az_player1 = tournament.AlphaZeroPlayer(net1)
    az_player2 = tournament.AlphaZeroPlayer(net2)
    score1 = tournament.play_match(game_count, az_player1, az_player2)
    return score1


def __self_play_worker__(network_path, position_cache, game_count):
    """
    plays a number of self play games
    :param network_path:        path of the network
    :param position_cache:      holds positions already evaluated by the network
    :param game_count:          the number of self-play games to play
    :return:                    a list of dictionaries with all training examples
    """

    # load the network
    net = data_storage.load_net(network_path, Config.evaluation_device)

    training_expl_list = []
    # q_list = []
    # position_cache = {}   # give each simulation its own state dict
    # position_count_dict = {}

    for i in range(game_count):
        board = connect4.BitBoard()
        mcts = MCTS(Config.c_puct)  # reset the search tree

        # reset the lists
        player_list = []
        state_list = []
        state_id_list = []
        policy_list = []

        move_count = 0
        while not board.terminal:
            # add regular board
            state, player = board.white_perspective()
            state_id = board.state_id()
            state_list.append(state)
            state_id_list.append(state_id)
            player_list.append(player)

            # add mirrored board
            board_mirrored = board.mirror()
            state_m, player_m = board_mirrored.white_perspective()
            state_id_m = board_mirrored.state_id()
            state_list.append(state_m)
            state_id_list.append(state_id_m)
            player_list.append(player_m)

            # get the policy from the mcts
            temp = 0 if move_count >= Config.temp_threshold else Config.temp
            policy = mcts.policy_values(board, position_cache, net, Config.mcts_sim_count, temp, Config.alpha_dirich)
            policy_list.append(policy)
            # s = board.state_id()

            # add the mirrored policy as well
            policy_m = np.flip(policy)
            policy_list.append(policy_m)

            # sample from the policy to determine the move to play
            move = np.random.choice(len(policy), p=policy)
            # q_value = mcts.Q.get((s, move), 0)
            board.play_move(move)

            # save the training example
            # q_list.append(q_value)
            move_count += 1

        # calculate the values from the perspective of the player who's move it is
        # modification to alpha-zero: average value and q_value
        reward = board.training_reward()
        for idx, player in enumerate(player_list):
            # value = (reward + q_list[idx]) / 2
            # value = value if player == CONST.WHITE else -value
            value = reward if player == CONST.WHITE else -reward

            # save the training example
            training_expl_list.append({
                "state": state_list[idx],
                "state_id": state_id_list[idx],
                "player": player_list[idx],
                "policy": policy_list[idx],
                "value": value
            })

    # delete the network
    del net
    torch.cuda.empty_cache()

    return training_expl_list
