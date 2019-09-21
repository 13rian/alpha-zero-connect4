import torch

class CONST:
	WHITE = 0				# white disk (red)
	BLACK = 1 				# black disk (yellow)

	BOARD_WIDTH = 7 	    				 # the width of the board (number of columns)
	BOARD_HEIGHT = 6						 # the height of the board
	BOARD_SIZE = BOARD_WIDTH * BOARD_HEIGHT  # the size of the board


class Globals:
	n_pool_processes = 7
	pool = None


class Config:
	# torch devices for training and evaluation
	evaluation_device = torch.device('cpu')		# the pytorch device that is used for evaluation
	training_device = torch.device('cuda') 		# the pytorch device that is used for training

	# hyperparameters
	cycle_count = 1000  	# the number of alpha zero cycles
	episode_count = 210  	# the number of games that are self-played in one cycle 2000
	epoch_count = 2  		# the number of times all training examples are passed through the network 10
	mcts_sim_count = 200  	# the number of simulations for the monte-carlo tree search 800
	c_puct = 4 	 			# the higher this constant the more the mcts explores 4
	temp = 1  				# the temperature, controls the policy value distribution
	temp_threshold = 42  	# up to this move the temp will be temp, otherwise 0 (deterministic play)
	alpha_dirich = 1  		# alpha parameter for the dirichlet noise (0.03 - 0.3 az paper, 10/ avg n_moves) 0.3
	n_filters = 32  		# the number of filters in the conv layers 128
	learning_rate = 0.001  	# the learning rate of the neural network
	dropout = 0.2  			# dropout probability for the fully connected layers 0.3
	n_blocks = 15  			# number of residual blocks
	batch_size = 256  		# the batch size of the experience buffer for the neural network training
	window_size = 50 		# size of the training window (number of cycles for the training data)