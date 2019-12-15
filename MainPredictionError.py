import random
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

from utils import utils
from game import connect4
from globals import Config
import data_storage
import mcts
import evaluation


# The logger
utils.init_logger(logging.DEBUG, file_name="log/connect4.log")
logger = logging.getLogger('evaluation')

np.set_printoptions(suppress=True, precision=6)


# set the random seed
random.seed(a=None, version=2)
np.random.seed(seed=None)

c_puct = 4
temp = 0
mcts_sim_count = 200
test_set_path = "test_set/positions.csv"
network_dir = "networks/"                   # directory in which the networks are saved

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
mcts_prediciton_error = []
path_list = os.listdir(network_dir)
path_list.sort(key=utils.natural_keys)

# test the best network to quickly get a result
policy_error, value_error = evaluation.net_prediction_error(net, test_set)
logger.debug("prediction-error: {}, value-error: {}, network: {}".format(policy_error, value_error, net_path))


# empty board test
board = connect4.BitBoard()
batch, _ = board.white_perspective()
batch = torch.Tensor(batch).unsqueeze(0).to(evaluation.torch_device)
policy, value = net(batch)
logger.debug("empty board policy:      {}".format(policy.detach().cpu().squeeze().numpy()))
net = data_storage.load_net(net_path, Config.evaluation_device)
policy = mcts.mcts_policy(board, 800, net, 1, 0)
logger.debug("empty board mcts-policy: {}".format(policy))


# get the prediction error of all networks
for i in range(len(path_list)):
    generation.append(i)
    net_path = network_dir + path_list[i]
    net = data_storage.load_net(net_path, evaluation.torch_device)

    policy_error, value_error = evaluation.net_prediction_error(net, test_set)
    net_prediciton_error.append(policy_error)
    net_value_error.append(value_error)
    logger.debug("prediction-error: {}, value-error: {}, network: {}".format(policy_error, value_error, net_path))



# get the mcts prediction error of all networks
temp = 1
alpha_dirich = 0
# path_list = [path_list[0], path_list[-1]]
path_list = [path_list[-1]]
for i in range(len(path_list)):
    net_path = network_dir + path_list[i]
    net = data_storage.load_net(net_path, Config.evaluation_device)

    prediction_error = evaluation.mcts_prediction_error(net, test_set, 800, alpha_dirich, temp)
    mcts_prediciton_error.append(prediction_error)
    logger.debug("mcts-prediction-error: {}, network: {}".format(prediction_error, net_path))



# plot the network prediction error
fig1 = plt.figure(1)
plt.plot(generation, net_prediciton_error)
axes = plt.gca()
axes.set_ylim([0, 80])
plt.title("Network Optimal Move Prediction Error")
plt.xlabel("Generation")
plt.ylabel("Prediction Error")
fig1.show()


# plot the network value error
fig2 = plt.figure(2)
plt.plot(generation, net_value_error)
axes = plt.gca()
axes.set_ylim([0, 1.5])
plt.title("Network Value Error")
plt.xlabel("Generation")
plt.ylabel("MSE Value")
fig1.show()



# # plot the mcts prediction error
# fig3 = plt.figure(3)
# plt.plot(generation, mcts_prediciton_error)
# axes = plt.gca()
# axes.set_ylim([0, 80])
# plt.title("MCTS Optimal Move Prediction Error")
# plt.xlabel("Generation")
# plt.ylabel("Prediction Error")
# fig3.show()


plt.show()
