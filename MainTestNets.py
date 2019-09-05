import matplotlib.pyplot as plt
import torch
import random
import os

from utils import utils

from game.globals import Globals
import alpha_zero_learning
import data_storage


# set the random seed
random.seed(a=None, version=2)


# define the parameters
network_duel_game_count = 40        # number of games that are played between the old and the new network
mcts_sim_count = 100                # the number of simulations for the monte-carlo tree search
c_puct = 4                          # the higher this constant the more the mcts explores
network_dir = "networks/"           # directory in which the networks are saved

# define the devices for the training and the target networks     cpu or cuda, here cpu is way faster for small nets
Globals.evaluation_device = torch.device('cpu')



# let the different networks play against each other
generation = []
avg_score = []
path_list = os.listdir(network_dir)
path_list.sort(key=utils.natural_keys)

# get the best network
best_network_path = network_dir + path_list[0]
best_net = data_storage.load_net(best_network_path, Globals.evaluation_device)
for i in range(len(path_list)):
    generation.append(i)
    net_path = network_dir + path_list[i]
    net = data_storage.load_net(net_path, Globals.evaluation_device)


    print("play {} against the random network {}".format(net_path, best_network_path))
    # random_net = alpha_zero_learning.Network(learning_rate)
    net_score = alpha_zero_learning.net_vs_net(net, best_net, network_duel_game_count, mcts_sim_count, c_puct, 0)
    print("score: ", net_score)
    avg_score.append(net_score)



# plot the score of the different generation network against the best network
fig1 = plt.figure(1)
plt.plot(generation, avg_score, color="#9ef3f3")
axes = plt.gca()
axes.set_ylim([0, 1])
plt.title("Average Score Against Random Network")
plt.xlabel("Generation")
plt.ylabel("Average Score")
fig1.show()

plt.show()
