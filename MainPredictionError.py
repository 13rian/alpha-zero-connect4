import matplotlib.pyplot as plt
import torch
import math
import pandas as pd
import random
import numpy as np
import os
import mcts

from utils import utils
from game import connect4

from globals import Config
import data_storage

np.set_printoptions(suppress=True, precision=6)


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

    tot_positions = test_set.shape[0]   # test_set.shape[0]   # the total number of positions in the test set
    correct_predictions = 0             # the correctly predicted positions in the test set
    tot_predictions = 0

    for j in range(tot_positions):      # tot_positions
        # ignore losing and drawing positions
        if test_set["weak_score"][j] <= 0:
            continue


        # load the board
        board = connect4.BitBoard()
        board.from_position(test_set["position"][j], test_set["disk_mask"][j])

        # get the white perspective
        batch, _ = board.white_perspective()
        batch = torch.Tensor(batch).unsqueeze(0).to(Config.evaluation_device)

        # find the predicted move
        policy, value = net(batch)
        #print(value)
        #print(policy)
        _, move = policy.max(1)
        move = move.item()

        # check if the move is part of the optimal moves
        if str(move) in str(test_set["weak_moves"][j]):
            correct_predictions += 1
        else:
            print("pred: {} wrong pos".format(move))
            print("v: ", value.item())
            print("p: ", policy.squeeze().detach().numpy())
            board.print()
            print(" ")


        tot_predictions += 1

    # calculate the prediction error
    pred_error = (tot_predictions - correct_predictions) / tot_predictions * 100
    return pred_error



def mcts_prediction_error(net, test_set, mcts_sim_count, temp):
    """
    returns the error percentage of the optimal move prediction by the network
    the network and the mcts are used to predict the move to play
    :param net:         the network
    :param test_set:    the test set
    :return:            error percentage
    """

    tot_positions = test_set.shape[0]        # test_set.shape[0]   # the total number of positions in the test set
    correct_predictions = 0             # the correctly predicted positions in the test set
    tot_predictions = 0

    # mcts
    for j in range(tot_positions):      # tot_positions
        # ignore losing and drawing positions
        if test_set["weak_score"][j] <= 0:
            continue

        mcts_net = mcts.MCTS(c_puct)

        # load the board
        board = connect4.BitBoard()
        board.from_position(test_set["position"][j], test_set["disk_mask"][j])

        # find the predicted move
        policy = mcts_net.policy_values(board, {}, net, mcts_sim_count, temp)
        move = np.where(policy == 1)[0][0]

        if str(move) in test_set["weak_moves"][j]:
            correct_predictions += 1
        # else:
        #     print("wrong pos:" + j)

        tot_predictions += 1

    # calculate the prediction error
    pred_error = (tot_predictions - correct_predictions) / tot_predictions * 100
    return pred_error




# load the test set with the solved positions
test_set = pd.read_csv(test_set_path, sep=",")





# # compare the mcts and the network policy of the best network
# path_list = os.listdir(network_dir)
# path_list.sort(key=utils.natural_keys)
# network_path = network_dir + "/" + path_list[-1]
# net = data_storage.load_net(network_path, Config.evaluation_device)
#
# i = 2003
#
#
# # MCTS
# mcts_net = mcts.MCTS(c_puct)
#
# # load the board
# board = connect4.BitBoard()
# board.from_position(test_set["position"][i], test_set["disk_mask"][i])
# board.print()
#
# # find the predicted move
# policy = mcts_net.policy_values(board, {}, net, mcts_sim_count, temp)
# print("mcts-pol: ", policy)
# move = np.where(policy == 1)[0][0]
#
# if str(move) in test_set["moves"][i]:
#     print("mcts ok")
#
#
# # NET prediction
# # load the board
# board = connect4.BitBoard()
# board.from_position(test_set["position"][i], test_set["disk_mask"][i])
#
# # get the white perspective
# board.white_perspective()
# batch, _ = board.white_perspective()
# batch = torch.Tensor(batch).unsqueeze(0).to(Config.evaluation_device)
#
# # find the predicted move
# policy, _ = net(batch)
# print("net-pol: ", policy)
# _, move = policy.max(1)
# move = move.item()
#
# # check if the move is part of the optimal moves
# if str(move) in test_set["moves"][i]:
#     print("network prediction ok")
#
# print("moves: ", test_set["moves"][i])











# calculate the prediciton error of the networks
generation = []
net_prediciton_error = []
mcts_prediciton_error = []
path_list = os.listdir(network_dir)
path_list.sort(key=utils.natural_keys)


# # random mcts prediction
# net_path = network_dir + path_list[0]
# net = data_storage.load_net(net_path, Config.evaluation_device)
#
# error = mcts_prediction_error(net, test_set, mcts_sim_count, temp)
# mcts_prediciton_error.append(error)
# print("mcts-error: ", error, "network: ", net_path)
#



net_path = network_dir + path_list[-1]
net = data_storage.load_net(net_path, Config.evaluation_device)
error = net_prediction_error(net, test_set)
print("error: ", error, "network: ", net_path)

# mcts search test
mcts_net = mcts.MCTS(c_puct)
board = connect4.BitBoard()

# empty board test
batch, _ = board.white_perspective()
batch = torch.Tensor(batch).unsqueeze(0).to(Config.evaluation_device)
policy, value = net(batch)
policy = mcts_net.policy_values(board, {}, net, 800, 1)

board.from_position(50099824841353, 279245752885183)
policy = mcts_net.policy_values(board, {}, net, mcts_sim_count, 0)
move = np.where(policy == 1)[0][0]
board.print()



# get the prediction error of all networks
for i in range(len(path_list)):
    generation.append(i)
    net_path = network_dir + path_list[i]
    net = data_storage.load_net(net_path, Config.evaluation_device)

    error = net_prediction_error(net, test_set)
    net_prediciton_error.append(error)
    print("error: ", error, "network: ", net_path)




# get the mcts prediction error of all networks
path_list = [path_list[-1]]
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



# # plot the network prediction error
# fig2 = plt.figure(2)
# plt.plot(generation, net_prediciton_error)
# axes = plt.gca()
# axes.set_ylim([0, 80])
# plt.title("MCTS Optimal Move Prediction Error")
# plt.xlabel("Generation")
# plt.ylabel("Prediction Error")
# fig2.show()

plt.show()
