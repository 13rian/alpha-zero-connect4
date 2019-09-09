import matplotlib.pyplot as plt
import torch
import pandas as pd
import random
import numpy as np
import os
import mcts

from utils import utils
from game import connect4

from globals import Config
import data_storage


# set the random seed
random.seed(a=None, version=2)

c_puct = 4
temp = 0
mcts_sim_count = 200
test_set_path = "test_set/positions.csv"
network_dir = "networks/"           # directory in which the networks are saved


def net_prediction_error(net, test_set):
    """
    returns the error percentage of the optimal move prediction by the network
    only the network is used to predict the correct move
    :param net:         the network
    :param test_set:    the test set
    :return:            error percentage
    """

    tot_positions = test_set.shape[0]   # the total number of positions in the test set
    correct_predictions = 0             # the correctly predicted positions in the test set

    for i in range(tot_positions):
        # load the board
        board = connect4.BitBoard()
        board.from_position(test_set["position"][i], test_set["disk_mask"][i])

        # get the white perspective
        board.white_perspective()
        batch, _ = board.white_perspective()
        batch = torch.Tensor(batch).unsqueeze(0).to(Config.evaluation_device)

        # find the predicted move
        policy, _ = net(batch)
        _, move = policy.max(1)
        move = move.item()

        # check if the move is part of the optimal moves
        if str(move) in test_set["moves"][i]:
            correct_predictions += 1

    # calculate the prediction error
    error = (tot_positions - correct_predictions) / tot_positions * 100
    return error



def mcts_prediction_error(net, test_set, mcts_sim_count, temp):
    """
    returns the error percentage of the optimal move prediction by the network
    the network and the mcts are used to predict the move to play
    :param net:         the network
    :param test_set:    the test set
    :return:            error percentage
    """

    tot_positions = test_set.shape[0]   # the total number of positions in the test set
    correct_predictions = 0             # the correctly predicted positions in the test set

    # mcts
    for i in range(test_set.shape[0]):
        mcts_net = mcts.MCTS(c_puct)

        # load the board
        board = connect4.BitBoard()
        board.from_position(test_set["position"][i], test_set["disk_mask"][i])

        # find the predicted move
        policy = mcts_net.policy_values(board, {}, net, mcts_sim_count, temp)
        move = np.where(policy == 1)[0][0]

        if str(move) in test_set["moves"][i]:
            correct_predictions += 1

    # calculate the prediction error
    error = (tot_positions - correct_predictions) / tot_positions * 100
    return error




# load the test set with the solved positions
test_set = pd.read_csv(test_set_path, sep=",")


# calculate the prediciton error of the networks
generation = []
net_prediciton_error = []
mcts_prediciton_error = []
path_list = os.listdir(network_dir)
path_list.sort(key=utils.natural_keys)
path_list = path_list[0:2]


# get the prediction error of all networks
for i in range(len(path_list)):
    generation.append(i)
    net_path = network_dir + path_list[i]
    net = data_storage.load_net(net_path, Config.evaluation_device)

    error = net_prediction_error(net, test_set)
    net_prediciton_error.append(error)
    print("error: ", error, "network: ", net_path)




# get the mcts prediction error of all networks
for i in range(len(path_list)):
    net_path = network_dir + path_list[i]
    net = data_storage.load_net(net_path, Config.evaluation_device)

    error = mcts_prediction_error(net, test_set, mcts_sim_count, temp)
    mcts_prediciton_error.append(error)
    print("mcts-error: ", error, "network: ", net_path)



# plot the network prediction error
fig1 = plt.figure(1)
plt.plot(generation, net_prediciton_error)
axes = plt.gca()
axes.set_ylim([0, 80])
plt.title("Network Optimal Move Prediction Error")
plt.xlabel("Generation")
plt.ylabel("Prediction Error")
fig1.show()



# plot the network prediction error
fig2 = plt.figure(2)
plt.plot(generation, net_prediciton_error)
axes = plt.gca()
axes.set_ylim([0, 80])
plt.title("MCTS Optimal Move Prediction Error")
plt.xlabel("Generation")
plt.ylabel("Prediction Error")
fig1.show()

plt.show()
