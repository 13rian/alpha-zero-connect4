import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import random
import time
import logging

from utils import utils

import alpha_zero_learning
import data_storage
import networks
from globals import Globals, Config


# @utils.profile
def mainTrain():
    # The logger
    utils.init_logger(logging.DEBUG, file_name="log/connect4.log")
    logger = logging.getLogger('Connect4')

    # set the random seed
    random.seed(a=None, version=2)

    # initialize the pool
    Globals.pool = mp.Pool(processes=Globals.n_pool_processes)

    # create the storage object
    training_data = data_storage.load_data()

    # create the agent
    # network = networks.ConvNet(learning_rate, n_filters, dropout)
    network = networks.ResNet(Config.learning_rate, Config.n_blocks, Config.n_filters)
    agent = alpha_zero_learning.Agent(network)

    if training_data.cycle == 0:
        logger.debug("create a new agent")
        training_data.save_data(agent.network)      # save the generation 0 network
    else:
        # load the current network
        logger.debug("load an old network")
        agent.network = training_data.load_current_net()
        agent.experience_buffer = training_data.experience_buffer


    start_training = time.time()
    for i in range(training_data.cycle, Config.cycle_count, 1):
        ###### self play and update: create some game data through self play
        logger.info("start playing games in cycle {}".format(i))
        avg_moves_played = agent.play_self_play_games(training_data.network_path)
        training_data.avg_moves_played.append(avg_moves_played)
        print("average moves played: ", avg_moves_played)


        ###### training, train the training network and use the target network for predictions
        logger.info("start updates in cycle {}".format(i))
        loss_p, loss_v = agent.nn_update()
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


