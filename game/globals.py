
class CONST:
	EMPTY = 0 				# no disk
	WHITE = 1				# white disk (cross, x)
	BLACK = 2 				# black disk (circle, o)

	NN_POLICY_SIZE = 36 	# the length of the policy vector, 36 actions are possible for the player to move

	WHITE_MOVE = 0			# white's move constant
	BLACK_MOVE = 1 			# black's move constant


class Globals:
	device = None 			# the pytorch device that is used for training
