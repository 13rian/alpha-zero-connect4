import matplotlib.pyplot as plt
import torch
import random
import time
import logging
import torch.multiprocessing as mp

from utils import utils

from game.globals import Globals
import alpha_zero_learning
import data_storage
import networks


# @utils.profile
def mainTrain():
    # The logger
    utils.init_logger(logging.DEBUG, file_name="log/connect4.log")
    logger = logging.getLogger('Connect4')


    # set the random seed
    random.seed(a=None, version=2)

    # initialize the pool
    Globals.n_pool_processes = mp.cpu_count()
    Globals.pool = mp.Pool(processes=Globals.n_pool_processes)


    # define the parameters
    cycle_count = 20                        # the number of alpha zero cycles
    episode_count = 8                    # the number of games that are self-played in one cycle
    epoch_count = 2                         # the number of times all training examples are passed through the network 10
    mcts_sim_count = 200                    # the number of simulations for the monte-carlo tree search 800
    c_puct = 4                              # the higher this constant the more the mcts explores 4
    temp = 1                                # the temperature, controls the policy value distribution
    temp_threshold = 42                     # up to this move the temp will be temp, otherwise 0 (deterministic play)
    alpha_dirich = 1     # alpha parameter for the dirichlet noise (0.03 - 0.3 az paper, 10/ avg n_moves) 0.3
    n_filters = 64                         # the number of filters in the conv layers 128
    learning_rate = 0.01                   # the learning rate of the neural network
    dropout = 0.2                           # dropout probability for the fully connected layers 0.3
    n_blocks = 5                            # number of residual blocks
    batch_size = 256                         # the batch size of the experience buffer for the neural network training 64
    exp_buffer_size = 3*2*42*episode_count    # the size of the experience replay buffer


    # define the devices for the training and the evaluation cpu or cuda
    Globals.evaluation_device = torch.device('cpu')
    Globals.training_device = torch.device('cuda')

    # create the storage object
    training_data = data_storage.load_data()

    # create the agent
    # network = networks.ConvNet(learning_rate, n_filters, dropout)
    network = networks.ResNet(learning_rate, n_blocks, n_filters)
    agent = alpha_zero_learning.Agent(network, mcts_sim_count, c_puct, temp, batch_size, exp_buffer_size)

    if training_data.cycle == 0:
        logger.debug("create a new agent")
        training_data.save_data(agent.network)      # save the generation 0 network
    else:
        # load the current network
        logger.debug("load an old network")
        agent.network = training_data.load_current_net()
        agent.experience_buffer = training_data.experience_buffer


    start_training = time.time()
    for i in range(training_data.cycle, cycle_count, 1):
        # ###### play against random
        # white_score = agent.play_against_random(CONST.WHITE, evaluation_game_count)
        # logger.info("white score vs random: {}".format(white_score))
        #
        # black_score = agent.play_against_random(CONST.BLACK, evaluation_game_count)
        # logger.info("black score vs random: {}".format(black_score))


        ###### self play and update: create some game data through self play
        logger.info("start playing games in cycle {}".format(i))
        avg_moves_played = agent.play_self_play_games(episode_count, temp_threshold, alpha_dirich)
        training_data.avg_moves_played.append(avg_moves_played)
        print("average moves played: ", avg_moves_played)


        ###### training, train the training network and use the target network for predictions
        logger.info("start updates in cycle {}".format(i))
        loss_p, loss_v = agent.nn_update(epoch_count)
        training_data.policy_loss.append(loss_p)
        training_data.value_loss.append(loss_v)
        print("policy loss: ", loss_p)
        print("value loss: ", loss_v)


        ###### save the new network
        logger.info("save check point to file in cycle {}".format(i))
        training_data.cycle += 1
        training_data.experience_buffer = agent.experience_buffer
        training_data.save_data(agent.network)


    end_training = time.time()
    training_time = end_training - start_training
    logger.info("elapsed time whole training process {}".format(training_time))



    # plot the value training loss
    fig1 = plt.figure(1)
    plt.plot(training_data.value_loss)
    plt.title("Average Value Training Loss")
    plt.xlabel("Iteration Cycle")
    plt.ylabel("Value Loss")
    fig1.show()

    # plot the training policy loss
    fig2 = plt.figure(2)
    plt.plot(training_data.policy_loss)
    plt.title("Average Policy Training Loss")
    plt.xlabel("Iteration Cycle")
    plt.ylabel("Policy Loss")
    fig2.show()

    # plot the average number of moves played in the self-play games
    fig3 = plt.figure(3)
    plt.plot(training_data.avg_moves_played)
    plt.title("Average Moves in Self-Play Games")
    plt.xlabel("Iteration Cycle")
    plt.ylabel("Move Count")
    fig3.show()

    plt.show()


if __name__ == '__main__':
    mainTrain()


