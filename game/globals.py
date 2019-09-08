
class CONST:
	WHITE = 0				# white disk (red)
	BLACK = 1 				# black disk (yellow)

	BOARD_WIDTH = 7 	    				 # the width of the board (number of columns)
	BOARD_HEIGHT = 6						 # the height of the board
	BOARD_SIZE = BOARD_WIDTH * BOARD_HEIGHT  # the size of the board


class Globals:
	evaluation_device = None		# the pytorch device that is used for evaluation
	training_device = None 			# the pytorch device that is used for training

	# pool for the multi processing
	pool = None
	n_pool_processes = None 		# the number of parallel processes, usually the number of cores or one less

	# number of cpu and gpu workers for the self play games
	n_cpu_workers = None
	n_gpu_workers = None
