import random
import pickle
import time
import logging

import torch

from utils import utils
import alpha_zero_learning
import networks
from globals import Config


def main():
    # The logger
    utils.init_logger(logging.DEBUG, file_name="log/connect4.log")
    logger = logging.getLogger('Connect4')

    # set the random seed
    random.seed(a=None, version=2)


    # create the configuration values for a random network
    network_path = "self-play-net.pt"
    Config.n_blocks = 1
    Config.n_filters = 1
    Config.mcts_sim_count = 200
    loops = 10
    games_per_loop = 1000


    # create the agent
    logger.info("create a new random network for the self play")
    network = networks.ResNet(Config.learning_rate, Config.n_blocks, Config.n_filters, Config.weight_decay)
    torch.save({'state_dict': network.state_dict()}, network_path)


    # play self-play games
    logger.info("start to create self-play games")
    start = time.time()
    training_examples = []

    for i in range(loops):
        new_examples = alpha_zero_learning.__self_play_worker__(network_path, games_per_loop)
        training_examples.extend(new_examples)
        logger.debug("finished creating games in loop {}".format(i))

    # save the training examples
    with open("initial_training_data.pkl", 'wb') as output:
        pickle.dump(training_examples, output, pickle.HIGHEST_PROTOCOL)

    logger.info("finished creating the initial training examples, length: {}".format(len(training_examples)))
    average_length = 0.5*len(training_examples) / (games_per_loop*loops) # 0.5 as symmetric positions are included as well
    logger.debug("average moves per game: {}".format(average_length))
    logger.debug("elapsed time: {}".format(time.time() - start))


if __name__ == '__main__':
    main()


