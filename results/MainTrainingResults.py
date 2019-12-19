import matplotlib.pyplot as plt
import torch
import pandas as pd
import random
import numpy as np
import os
import logging
import mcts
import evaluation

from utils import utils
from game import connect4

from globals import Config
import data_storage


# The logger
utils.init_logger(logging.DEBUG, file_name="log/connect4.log")
logger = logging.getLogger('Evaluation')

np.set_printoptions(suppress=True, precision=6)


# set the random seed
random.seed(a=None, version=2)
np.random.seed(seed=None)

c_puct = 4
temp = 0
mcts_sim_count = 200
test_set_path = "../test_set/training_set.csv"
network_dir = "../networks/"                   # directory in which the networks are saved

print("pytorch version: ", torch.__version__)


# load the network
path_list = os.listdir(network_dir)
path_list.sort(key=utils.natural_keys)
net_path = network_dir + path_list[-1]
net = data_storage.load_net(net_path, evaluation.torch_device)



# load the test set with the solved positions
test_set = pd.read_csv(test_set_path, sep=",")



# calculate the prediciton error of the networks
generation = []
net_prediciton_error = []
net_value_error = []
mcts_prediciton_error_200 = []
mcts_prediciton_error_800 = []
path_list = os.listdir(network_dir)
path_list.sort(key=utils.natural_keys)


# get the prediction error of all networks
interval = 15
path_list = path_list[0::interval]
for i in range(len(path_list)):
    generation.append(i*interval)
    net_path = network_dir + path_list[i]
    net = data_storage.load_net(net_path, evaluation.torch_device)

    policy_error, value_error = evaluation.net_prediction_error(net, test_set)
    net_prediciton_error.append(policy_error)
    net_value_error.append(value_error)
    logger.debug("prediction-error: {}, value-error: {}, network: {}".format(policy_error, value_error, net_path))

np.save("net_prediciton_error.npy", np.array(net_prediciton_error))
np.save("net_value_error.npy", np.array(net_value_error))
np.save("net_generation.npy", np.array(generation))


# get the mcts prediction error of all networks with 200 and 800 simulations
temp = 1
alpha_dirich = 0
for i in range(len(path_list)):
    net_path = network_dir + path_list[i]
    net = data_storage.load_net(net_path, Config.evaluation_device)

    prediction_error_200 = evaluation.mcts_prediction_error(net, test_set, 200, alpha_dirich, temp)
    mcts_prediciton_error_200.append(prediction_error_200)

    prediction_error_800 = evaluation.mcts_prediction_error(net, test_set, 800, alpha_dirich, temp)
    mcts_prediciton_error_800.append(prediction_error_800)

    logger.debug("mcts-prediction-error sim200: {}, sim800: {}, network: {}".format(prediction_error_200, prediction_error_800, net_path))


np.save("mcts_prediciton_error_200.npy", np.array(mcts_prediciton_error_200))
np.save("mcts_prediciton_error_800.npy", np.array(mcts_prediciton_error_800))

