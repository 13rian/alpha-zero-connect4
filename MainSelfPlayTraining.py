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

# @utils.profile()
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
    epoch_count = 6                         # the number of epochs to train the neural network
    episode_count = 8                    # the number of games that are self-played in one epoch
    update_count = 8                      # the number the neural net is updated  in one epoch with the experience data
    mcts_sim_count = 80                     # the number of simulations for the monte-carlo tree search
    c_puct = 4                              # the higher this constant the more the mcts explores
    temp = 1                                # the temperature, controls the policy value distribution
    temp_threshold = 36                     # up to this move the temp will be temp, otherwise 0 (deterministic play)
    alpha_dirich = 1  # alpha parameter for the dirichlet noise (0.03 - 0.3 az paper, 10/ avg n_moves)
    n_filters = 128                         # the number of filters in the conv layers
    learning_rate = 0.005                   # the learning rate of the neural network
    batch_size = 256                        # the batch size of the experience buffer for the neural network training
    exp_buffer_size = 2*36*episode_count    # the size of the experience replay buffer

    # define the devices for the training and the evaluation
    Globals.evaluation_device = torch.device('cpu')
    Globals.training_device = torch.device('cpu')

    # create the storage object
    training_data = data_storage.load_data()

    # create the agent
    agent = alpha_zero_learning.Agent(learning_rate, n_filters, mcts_sim_count, c_puct, temp, batch_size, exp_buffer_size)

    if training_data.epoch == 0:
        logger.debug("create a new agent")
        training_data.save_data(agent.network)      # save the generation 0 network
    else:
        # load the current network
        logger.debug("load an old network")
        agent.network = training_data.load_current_net()
        agent.experience_buffer = training_data.experience_buffer


    start_training = time.time()
    for i in range(training_data.epoch, epoch_count, 1):
        ###### self play and update: create some game data through self play
        logger.info("start playing games in epoch {}".format(i))
        agent.play_self_play_games(episode_count, temp_threshold, alpha_dirich)


        ###### training, train the training network and use the target network for predictions
        logger.info("start updates in epoch {}".format(i))
        loss_p, loss_v = agent.nn_update(update_count)
        training_data.policy_loss.append(loss_p)
        training_data.value_loss.append(loss_v)
        print("policy loss: ", loss_p)
        print("value loss: ", loss_v)


        ###### save the new network
        logger.info("save check point to file in epoch {}".format(i))
        training_data.epoch += 1
        training_data.experience_buffer = agent.experience_buffer
        training_data.save_data(agent.network)


    end_training = time.time()
    training_time = end_training - start_training
    logger.info("elapsed time whole training process {}".format(training_time))



    # plot the value training loss
    fig1 = plt.figure(1)
    plt.plot(training_data.value_loss)
    plt.title("Average Value Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Value Loss")
    fig1.show()

    # plot the training policy loss
    fig2 = plt.figure(2)
    plt.plot(training_data.policy_loss)
    plt.title("Average Policy Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Policy Loss")
    fig2.show()


    plt.show()


if __name__ == '__main__':
    mainTrain()


