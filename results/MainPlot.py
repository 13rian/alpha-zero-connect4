import matplotlib.pyplot as plt
import numpy as np


# define the styles for the plot
plt.style.use('seaborn-dark-palette')

# plot the lr test for the policy
learning_rates = np.load("learning_rates.npy")
lr_policy_error = np.load("lr_policy_error.npy")


fig1 = plt.figure(1)
plt.semilogx(learning_rates, lr_policy_error)
axes = plt.gca()
axes.grid(True, color=(0.9, 0.9, 0.9))
plt.title("Policy Error After 8 Epochs")
plt.xlabel("Learning Rate")
plt.ylabel("Policy Error (%)")
fig1.show()


# plot the lr test for the value
lr_value_error = np.load("lr_value_error.npy")


fig2 = plt.figure(2)
plt.semilogx(learning_rates, lr_value_error)
axes = plt.gca()
axes.grid(True, color=(0.9, 0.9, 0.9))
plt.title("Value Error After 8 Epochs")
plt.xlabel("Learning Rate")
plt.ylabel("MSE Value")
fig2.show()


# plot the policy loss
policy_loss = np.load("policy_loss.npy")

spacing = 5
generation = np.arange(0, len(policy_loss), spacing)

fig3 = plt.figure(3)
plt.plot(generation, policy_loss[0::spacing])
axes = plt.gca()
axes.grid(True, color=(0.9, 0.9, 0.9))
plt.title("Policy Loss")
plt.xlabel("Generation")
plt.ylabel("Cross-Entropy Loss")
fig3.show()


# plot the value loss
value_loss = np.load("value_loss.npy")

fig4 = plt.figure(4)
plt.plot(generation, value_loss[0::spacing])
axes = plt.gca()
axes.grid(True, color=(0.9, 0.9, 0.9))
plt.title("Value Loss")
plt.xlabel("Generation")
plt.ylabel("MSE Loss")
fig4.show()


# plot the policy loss
avg_moves = np.load("avg_moves.npy")

spacing = 5
generation = np.arange(0, len(policy_loss), spacing)

fig5 = plt.figure(5)
plt.plot(generation, avg_moves[0::spacing])
axes = plt.gca()
axes.grid(True, color=(0.9, 0.9, 0.9))
plt.title("Average Moves Played")
plt.xlabel("Generation")
plt.ylabel("Move Count")
fig5.show()



# plot the network value error
net_generation = np.load("net_generation.npy")
net_value_error = np.load("net_value_error.npy")

fig6 = plt.figure(6)
plt.plot(net_generation, net_value_error)
axes = plt.gca()
axes.set_ylim([0, 1.5])
axes.grid(True, color=(0.9, 0.9, 0.9))
plt.title("Network Value Error")
plt.xlabel("Generation")
plt.ylabel("MSE Value")
fig6.show()



# plot the mcts prediction error
net_prediciton_error = np.load("net_prediciton_error.npy")
mcts_prediciton_error_200 = np.load("mcts_prediciton_error_200.npy")
mcts_prediciton_error_800 = np.load("mcts_prediciton_error_800.npy")

fig7 = plt.figure(7)
plt.plot(net_generation, net_prediciton_error, label="net only")
plt.plot(net_generation, mcts_prediciton_error_200, label="200 simulations")
plt.plot(net_generation, mcts_prediciton_error_800, label="800 simulations")
axes = plt.gca()
axes.set_ylim([0, 20])
# axes.set_facecolor((0.8, 0.8, 0.8))
axes.grid(True, color=(0.9, 0.9, 0.9))
plt.legend()
plt.title("Optimal Move Prediction Error")
plt.xlabel("Generation")
plt.ylabel("Prediction Error (%)")
fig7.show()


plt.show()