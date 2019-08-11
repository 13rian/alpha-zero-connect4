import random
import copy

import numpy as np

from utils import utils
from game.globals import CONST
from game import connect4
import mcts


move_list = None                  # holds all moves in the same order as the policy vector from the network
move_to_policy_dict = None        # key: move, value: policy index


class BitBoard:
    """
    each player gets a separate board representation
    0  1  2  3  4  5
    7  8  9  10 11 12
    14 15 16 17 18 19
    ...
    if a stone is set for a player the bit string will have a 1 on the correct position
    a move is defined by a number, e.g. 4 (this represents setting a stone on the board position 4)
    one row is skipped in order to detect 4 in a row faster
    """

    def __init__(self):
        self.white_player = 0
        self.black_player = 0
        
        self.player = CONST.WHITE                   # disk of the player to move
        self.terminal = False                       # is the game finished
        self.score = 0                              # -1 if black wins, 0 if it is a tie and 1 if white wins
        self.legal_moves = []                       # holds all legal moves of the current board position
                
        # calculate all legal moves and disks to flip
        self.__calc_legal_moves__()
        
        
    def clone(self):
        """
        returns a new board with the same state
        :return:
        """
        board = copy.deepcopy(self)
        return board
        
    
    #################################################################################################
    #                                 board representation                                          #
    #################################################################################################

    def from_board_matrix(self, board):
        """
        creates the bit board from the passed board representation
        :param board:   game represented as one board
        :return:
        """
        pad = np.zeros((6, 1))
        board = np.hstack((board, pad))

        white_board = board == CONST.WHITE
        white_board = white_board.astype(int)
        self.white_player = self.board_to_int(white_board)

        black_board = board == CONST.BLACK
        black_board = black_board.astype(int)
        self.black_player = self.board_to_int(black_board)

        # calculate all legal moves and disks to flip
        self.__calc_legal_moves__()

        # check the game states
        self.swap_players()
        self.check_win()
        self.swap_players()
    

    def print(self):
        """
        prints the current board configuration
        :return:
        """

        # create the board representation form the bit strings
        print(self.get_board_matrix())


    def get_board_matrix(self):
        """
        :return:  human readable game board representation
        """

        white_board = self.int_to_board(self.white_player)
        black_board = self.int_to_board(self.black_player)
        board = np.add(white_board * CONST.WHITE, black_board * CONST.BLACK)
        return board
    
    
    def white_perspective(self):
        """
        returns the board from the white perspective. If it is white's move the normal board representation is returned.
        If it is black's move the white and the black pieces are swapped.
        :return:
        """

        white_board = self.int_to_board(self.white_player)
        black_board = self.int_to_board(self.black_player)
        if self.player == CONST.WHITE:
            bit_board = np.stack((white_board, black_board), axis=0)
            player = CONST.WHITE_MOVE
        else:
            bit_board = np.stack((black_board, white_board), axis=0)
            player = CONST.BLACK_MOVE

        return bit_board, player
    

    def int_to_board(self, number):
        """
        creates the 3x3 bitmask that is represented by the passed integer
        :param number:      move on the board
        :return:            x3 matrix representing the board
        """

        mask = 0x03F
        row_number = 0
        for i in range(6):
            row = number & (mask << (i*7))
            row_number = row_number + (row << i)

        byte_arr = np.array([row_number], dtype=np.uint64).view(np.uint8)
        board_mask = np.unpackbits(byte_arr).reshape(-1, 8)[0:6, ::-1][:, 0:6]
        return board_mask
        

    def board_to_int(self, board):
        """
        converts the passed board mask (6x6) to an integer
        :param board:    binary board representation 6x6
        :return:         integer representing the passed board
        """
        bit_arr = np.reshape(board, -1).astype(np.uint64)
        number = bit_arr.dot(1 << np.arange(bit_arr.size, dtype=np.uint64))
        return int(number)
    

    def move_to_board_mask(self, move):
        """
        :param move:    integer defining a move on the board
        :return:        the move represented as a mask on the 3x3 board
        """

        mask = 1 << move
        board_mask = self.int_to_board(mask)
        return board_mask
    
    
    def state_id(self):
        """
        uses the cantor pairing function to create one unique id for the state form the two integers representing the
        board state
        """
        state = "{}_{}".format(self.white_player, self.black_player)
        return state
    
    
    
    #################################################################################################
    #                                 game play management                                          #
    #################################################################################################

    def play_move(self, move):
        """
        plays the passed move on the board
        :param move:    integer that defines the position to set the stone
        :return:
        """

        # set the token
        move = int(move)
        if self.player == CONST.WHITE:
            self.white_player = self.white_player + (1 << move)
        else:
            self.black_player = self.black_player + (1 << move)
      
        # check if the player won
        self.check_win()
        
        # swap the active player and calculate the legal moves
        self.swap_players()
        self.__calc_legal_moves__()
        

    def check_win(self):
        """
        checks if the current player has won the game
        :return:
        """

        if self.four_in_a_row(self.player):
            self.terminal = True
            self.score = 1 if self.player == CONST.WHITE else -1        
        

    def swap_players(self):
        self.player = CONST.WHITE if self.player == CONST.BLACK else CONST.BLACK
        

    def random_move(self):
        if len(self.legal_moves) > 0:
            index = random.randint(0, len(self.legal_moves) - 1)
            return self.legal_moves[index]
        else:
            return None
            

    def __calc_legal_moves__(self):
        # define the mask with all legal moves
        move_mask = utils.bit_not(self.white_player ^ self.black_player, 41)    # this is basically an xnor (only 1 if both are 0)
        
        self.legal_moves = []
        for move in range(41):
            if move % 7 == 6:
                continue

            if (1 << move) & move_mask > 0:
                self.legal_moves.append(move)
                
        # if there are no legal moves the game is drawn
        if len(self.legal_moves) == 0:
            self.terminal = True
     

    def four_in_a_row(self, player):
        """
        checks if the passed player has a row of four
        :param player:      the player for which 4 in a row is checked
        :return:
        """

        board = self.white_player if player == CONST.WHITE else self.black_player
        
        # horizontal check
        if board & (board << 1) & (board << 2) & (board << 3):
            return True 
        
        # vertical check
        if board & (board << 7) & (board << 14) & (board << 21):
            return True

        # diagonal check /
        if board & (board << 6) & (board << 12) & (board << 18):
            return True
        
        # diagonal check \
        if board & (board << 8) & (board << 16) & (board << 24):
            return True
        
        # nothing found
        return False


    def set_player_white(self):
        self.player = CONST.WHITE
        

    def set_player_black(self):
        self.player = CONST.BLACK
       
       
       
       
    #################################################################################################
    #                               network training methods                                        #
    #################################################################################################
    def reward(self):
        """
        :return:    -1 if black has won
                    0 if the game is drawn or the game is still running
                    1 if white has won
        """

        if not self.terminal:
            return 0        
        else:
            return self.score


    def white_score(self):
        reward = self.reward()
        return (reward + 1) / 2


    def black_score(self):
        reward = self.reward()
        return (-reward + 1) / 2
        



#################################################################################################
#                                       test methods                                            #
#################################################################################################
def move_to_policy_idx(move):
    """
    returns the policy index of the passed move
    :param move:   move
    :return:
    """
    global move_to_policy_dict
    return move_to_policy_dict.get(move)


def policy_idx_to_move(index):
    """
    returns the move that corresponds to the passed policy index
    :param index:   policy value index
    :return:
    """
    global policy_to_move_dict
    return policy_to_move_dict.get(index)


def define_policy_move_dicts():
    """
    defines the list that holds the move in the same order as the policy output from the network
    :return:
    """
    global move_to_policy_dict
    global policy_to_move_dict
    global move_list

    move_to_policy_dict = {}
    move_list = []

    policy_idx = 0
    for i in range(41):
        if i % 7 is not 6:
            move_to_policy_dict[i] = policy_idx
            move_list.append(i)
            policy_idx += 1

    move_list = np.array(move_list)




define_policy_move_dicts()
