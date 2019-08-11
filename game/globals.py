
class CONST:
	EMPTY = 0 				# no disk
	WHITE = 1				# white disk (cross, x)
	BLACK = 2 				# black disk (circle, o)

	NN_INPUT_SIZE = 36  	# size of the neural network input
	NN_POLICY_SIZE = 36 	# the length of the policy vector, 36 actions are possible for the player to move

	WHITE_MOVE = 0			# white's move constant
	BLACK_MOVE = 1 			# black's move constant


class Globals:
	evaluation_device = None		# the pytorch device that is used for evaluation
	training_device = None 			# the pytorch device that is used for training

	# pool for the multi processing
	pool = None
	n_pool_processes = None 		# the number of parallel processes, usually the number of cores or one less
