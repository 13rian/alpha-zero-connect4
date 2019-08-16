import matplotlib.pyplot as plt
import torch
import random
import os

from utils import utils

from game.globals import Globals
import alpha_zero_learning


# set the random seed
random.seed(a=None, version=2)


# define the parameters
epoch_count = 90                   # the number of epochs to train the neural network
episode_count = 100                 # the number of games that are self-played in one epoch
update_count = 10                   # the number the neural net is updated  in one epoch with the experience data
network_duel_game_count = 40        # number of games that are played between the old and the new network
mcts_sim_count = 15                 # the number of simulations for the monte-carlo tree search
c_puct = 4                          # the higher this constant the more the mcts explores
temp = 1                            # the temperature, controls the policy value distribution
temp_threshold = 5                  # up to this move the temp will be temp, otherwise 0 (deterministic play)
new_net_win_rate = 0.55             # win rate of the new network in order to replace the old one
learning_rate = 0.005                 # the learning rate of the neural network
batch_size = 128                    # the batch size of the experience buffer for the neural network training
exp_buffer_size = 2*9*episode_count   # the size of the experience replay buffer
network_dir = "networks/"           # directory in which the networks are saved

# define the devices for the training and the target networks     cpu or cuda, here cpu is way faster for small nets
Globals.evaluation_device = torch.device('cpu')
Globals.training_device = torch.device('cuda')




# let the different networks play against each other
generation = []
avg_score = []
path_list = os.listdir(network_dir)
path_list.sort(key=utils.natural_keys)

# get the best network
best_network_path = network_dir + path_list[-1]
best_net = torch.load(best_network_path).to(Globals.evaluation_device)
for i in range(len(path_list)):
    generation.append(i)
    net_path = network_dir + path_list[i]
    net = torch.load(net_path).to(Globals.evaluation_device)

    print("play {} against the best network {}".format(net_path, best_network_path))
    # random_net = alpha_zero_learning.Network(learning_rate)
    best_net_score, net_score = alpha_zero_learning.net_vs_net(best_net, net, network_duel_game_count, mcts_sim_count, c_puct, 0)
    avg_score.append(net_score/network_duel_game_count)



# plot the score of the different generation network against the best network
fig1 = plt.figure(1)
plt.plot(generation, avg_score, color="#9ef3f3")
axes = plt.gca()
axes.set_ylim([0, 1])
plt.title("Average Score Against Best Network")
plt.xlabel("Generation")
plt.ylabel("Average Score")
fig1.show()

plt.show()
