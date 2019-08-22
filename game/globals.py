
class CONST:
	WHITE = 0				# white disk (red)
	BLACK = 1 				# black disk (yellow)

	NN_INPUT_SIZE = 42  	# size of the neural network input
	NN_POLICY_SIZE = 7 	    # the length of the policy vector, 36 actions are possible for the player to move


class Globals:
	evaluation_device = None		# the pytorch device that is used for evaluation
	training_device = None 			# the pytorch device that is used for training

	# pool for the multi processing
	pool = None
	n_pool_processes = None 		# the number of parallel processes, usually the number of cores or one less
