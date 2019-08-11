import math

import numpy as np
import torch

from game.globals import CONST, Globals
from game import connect4



class MCTS:
    """
    handles the monte-carlo tree search
    """
    
    def __init__(self, c_puct):
        self.c_puct = c_puct        # a tuneable hyperparameter. The larger this number, the more the model explores
        
        self.P = {}                 # holds the policies for a game state, key: s, value: policy
        self.Q = {}                 # action value, key; (s,a)
        self.N_s = {}               # number of times the state s was visited, key: s
        self.N_sa = {}              # number of times action a was chosen in state s, key: (s,a)
        
        
    
    def policy_values(self, board, net, mc_sim_count, temp, alpha_dirich=0):
        """
        executes mc_sim_count number of monte-carlo simulations to obtain the probability
        vector of the current game position
        :param board:            the game board
        :param net:              neural network that approximates the policy and the value
        :param mc_sim_count:     number of monte-carlo simulations to perform
        :param temp:             the temperature, determines the degree of exploration
                                 temp = 0 means that we only pick the best move
                                 temp = 1 means that we pick the move proportional to the count the state was visited
        :param alpha_dirich:     alpha parameter for the dirichlet noise that is added to the root node probabilities
        :return:                 policy where the probability of an action is proportional to 
                                 N_sa**(1/temp)
        """
               
        # perform the tree search
        for _ in range(mc_sim_count):
            sim_board = board.clone()
            self.tree_search(sim_board, net, alpha_dirich)

        s = board.state_id()
        counts = [self.N_sa[(s, a)] if (s, a) in self.N_sa else 0 for a in connect4.move_list]

        # in order to learn something set the probabilities of the best action to 1 and all other action to 0
        if temp == 0:
            policy_idx = np.argmax(counts)
            probs = [0]*CONST.NN_POLICY_SIZE
            probs[policy_idx] = 1
            return np.array(probs)
        
        else:
            counts = [c**(1./temp) for c in counts]
            probs = [c / float(sum(counts)) for c in counts]
            return np.array(probs)    
    
        
        
    def tree_search(self, board, net, alpha_dirich=0):
        """
        Performs one iteration of the monte-carlo tree search.
        The method is recursively called until a leaf node is found. This is a game
        state from wich no simulation (playout) has yet been initiated. If the leaf note
        is a terminal state, the reward is returned. If the leaf node is not a terminal node
        the value is estimated with the neural network. 
        The move (action) with the highest upper confidence bound is chosen. The fewer a move was
        chosen in a certain position the higher his upper confidence bound.
        
        The method returns the estimated value of the current game state. The sign of the value
        is flipped because the value of the game for the other player is the negative value of
        the state of the current player          
        :param net:             neural network that approximates the policy and the value
        :param board:           represents the game
        :param alpha_dirich:    alpha parameter for the dirichlet noise that is added to the root node probabilities
        :return: 
        """
    
        # check if the game is terminal    
        if board.terminal:
            return board.reward()
    
        # check if we are on a leaf node (state form which no simulation was played so far)
        s = board.state_id()
        player = board.player
        if s not in self.P:  
            batch, _ = board.white_perspective()
            batch = torch.Tensor(batch).unsqueeze(0).to(Globals.evaluation_device)
            self.P[s], v = net(batch)
            self.P[s] = self.P[s].detach().squeeze().numpy()
            v = v.item()
            
            # ensure that the summed probability of all valid moves is 1
            legal_moves = np.array(board.legal_moves)
            legal_move_indices = np.zeros(CONST.NN_POLICY_SIZE)
            for move in legal_moves:
                idx = connect4.move_to_policy_idx(move)
                legal_move_indices[idx] = 1

            self.P[s] = self.P[s] * legal_move_indices
            total_prob = np.sum(self.P[s])
            if total_prob > 0:
                self.P[s] /= total_prob    # normalize the probabilities
            
            else:
                # the network did not choose any legal move, make all moves equally probable
                print("warning: total probabilities estimated by the network for all legal moves is smaller than 0") 
                self.P[s] = 1 / len(legal_moves)*legal_move_indices
            
            self.N_s[s] = 0
            return v

        # add dirichlet noise for the root node
        p_s = self.P[s]
        if alpha_dirich > 0:
            p_s = np.copy(p_s)
            alpha_params = alpha_dirich * np.ones(len(board.legal_moves))
            dirichlet_noise = np.random.dirichlet(alpha_params)
            for counter, move in enumerate(board.legal_moves):
                idx = connect4.move_to_policy_idx(move)
                p_s[idx] = 0.75 * p_s[idx] + 0.25 * dirichlet_noise[counter]

            # normalize the probabilities again
            total_prob = np.sum(p_s)
            p_s /= total_prob

        # choose the action with the highest upper confidence bound
        max_ucb = -float("inf")
        action = -1
        for a in board.legal_moves:
            policy_idx = connect4.move_to_policy_idx(a)
            if (s, a) in self.Q:
                u = self.Q[(s, a)] + self.c_puct*p_s[policy_idx]*math.sqrt(self.N_s[s]) / (1+self.N_sa[(s, a)])
            else:
                u = self.c_puct*p_s[policy_idx]*math.sqrt(self.N_s[s] + 1e-8)  # avoid division by 0

            if u > max_ucb:
                max_ucb = u
                action = a
        
        a = action
        board.play_move(a)
        v = self.tree_search(board, net)
        
        
        # update the Q and N values
        v_true = v
        if player == CONST.BLACK:
            v_true *= -1     # flip the value for the black player since the game is always viewed from the white perspective
        
        if (s, a) in self.Q:
            self.Q[(s, a)] = (self.N_sa[(s, a)]*self.Q[(s, a)] + v_true) / (self.N_sa[(s, a)] + 1)
            self.N_sa[(s, a)] += 1
        else:   
            self.Q[(s, a)] = v_true
            self.N_sa[(s, a)] = 1
        
        self.N_s[s] += 1
        return v

