import pickle
import logging
import os
import shutil

import torch

from game.globals import Globals

logger = logging.getLogger('TrainingData')
storage_path = "training_data.pkl"
network_dir = "networks"


class TrainingData:
    def __init__(self):
        """
        holds the current state of the training progress
        """
        self.epoch = 0
        self.policy_loss = []
        self.value_loss = []
        self.experience_buffer = None


    def save_data(self, network):
        """
        saves the current training state
        :param network:     current network
        :return:
        """

        # save the current network
        torch.save(network, "{}/network_gen_{}.pt".format(network_dir, self.epoch))

        # dump the storage object to file
        with open(storage_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


    def load_current_net(self):
        """
        loads the most recent network
        :return:
        """

        net_path = "{}/network_gen_{}.pt".format(network_dir, self.epoch)
        net = torch.load(net_path).to(Globals.evaluation_device)
        logger.debug("network loaded from path {}".format(net_path))
        return net


def load_data():
    """
    loads the training data from the state file
    :return:
    """

    # create a new storage object
    if not os.path.exists(storage_path):
        logger.info("create a new data storage object")

        if not os.path.exists(network_dir):
            os.makedirs(network_dir)

        shutil.rmtree(network_dir)
        os.makedirs(network_dir)
        return TrainingData()

    # load an old storage object with the current training data
    with open(storage_path, 'rb') as input:
        training_data = pickle.load(input)
        return training_data



