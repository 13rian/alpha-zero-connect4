import pickle
import logging
import os
import shutil

import torch

from globals import Config

logger = logging.getLogger('TrainingData')
storage_path = "training_data.pkl"
network_dir = "networks"
temp_dir = "temp_net"


class TrainingData:
    def __init__(self):
        """
        holds the current state of the training progress
        """
        self.cycle = 0                  # iteration cycles
        self.avg_moves_played = []      # the average number of moves played in the self-play games
        self.policy_loss = []           # policy training loss
        self.value_loss = []            # value training loss
        self.network_path = None        # path of the current best network
        self.experience_buffer = None   # buffer with the self-play game data


    def save_data(self, network):
        """
        saves the current training state
        :param network:     current network
        :return:
        """

        # save the current network
        self.network_path = "{}/network_gen_{}.pt".format(network_dir, self.cycle)
        torch.save(network, self.network_path)

        # dump the storage object to file
        with open(storage_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)



    def load_current_net(self):
        """
        loads the most recent network
        :return:
        """

        net_path = "{}/network_gen_{}.pt".format(network_dir, self.cycle)
        net = load_net(net_path, Config.evaluation_device)
        net.eval()
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


def net_to_device(net, device):
    """
    sends the network to the passed device
    :param net:     the network to transfer into the cpu
    :param device:  the device to which the network is sent
    :return:
    """
    net_path = "{}/temp_net.pt".format(temp_dir)

    # ensure that the temp dir exists and is empty and
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)


    # put the model on the gpu
    if device.type == "cuda":
        torch.save(net, net_path)
        cuda_net = torch.load(net_path, map_location='cuda')
        shutil.rmtree(temp_dir)
        return cuda_net

    # put the model on the cpu
    if device.type == "cpu":
        torch.save(net, net_path)
        cpu_net = torch.load(net_path, map_location='cpu')
        shutil.rmtree(temp_dir)
        return cpu_net

    logger.error("device type {} is not known".format(device.type))
    return None


def load_net(net_path, device):
    """
    loads the network to the passed device
    :param net_path:    the path of the network to load
    :param device:      the device to which the network is loaded
    :return:
    """

    # put the model on the gpu
    if device.type == "cuda":
        gpu_net = torch.load(net_path, map_location='cuda')
        return gpu_net

    # put the model on the cpu
    if device.type == "cpu":
        cpu_net = torch.load(net_path, map_location='cpu')
        return cpu_net

