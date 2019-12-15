import math

import torch
import numpy as np

from globals import CONST, Config
from game import connect4


class MCTS:
    def __init__(self, board=None):
        # defines the board with the current games state
        if board is None:
            self.board = connect4.BitBoard()
        else:
            self.board = board

        self.P = {}                 # holds the policies for a game state, key: s, value: policy
        self.Q = {}                 # action value, key; (s,a)
        self.N_s = {}               # number of times the state s was visited, key: s
        self.N_sa = {}              # number of times action a was chosen in state s, key: (s,a)

        # lists for one simulation
        self.states = []
        self.players = []
        self.actions = []


    def policy_from_state(self, s, temp):
        """
        returns the policy of the passed state id. it only makes sense to call this methods if a few
        monte carlo simulations were executed previously.
        :param s:       state id of the board
        :param temp:    the temperature
        :return:        vector containing the policy value for the moves
        """
        counts = [self.N_sa[(s, a)] if (s, a) in self.N_sa else 0 for a in range(CONST.BOARD_WIDTH)]

        # in order to learn something set the probabilities of the best action to 1 and all other action to 0
        if temp == 0:
            action = np.argmax(counts)
            probs = [0] * CONST.BOARD_WIDTH
            probs[action] = 1
            return np.array(probs)

        else:
            counts = [c ** (1. / temp) for c in counts]
            probs = [c / float(sum(counts)) for c in counts]
            return np.array(probs)


    def exec_simulation(self, board, alpha_dirich=0):
        """
        executes one Monte-Carlo Tree search simulation. the simulation always starts a the rootk node and ends
        when a leaf node is reached. this is a game state from which no simulation (playout) was started so far.
        if the leaf node is a terminal state, the reward is returned. if the leaf node is not a terminal node
        the value and the policy are estimated with the neural network. the value is the result of the simulation.
        if a node is reached with known value and policy the move (action) with the highest upper confidence bound
        is chosen. the upper confidence bound increases with larger probability and if the moves was not chose often
        in previous simulation. the upper confidence bound holds the balance between exploreation and exploitation.
        :param board:           represents the game
        :param alpha_dirich:    alpha parameter for the dirichlet noise that is added to the root node probabilities
        :return:                the board if it needs to be analyzed  by the network or None if the simulation
                                completed
        """
        self.states.clear()
        self.players.clear()
        self.actions.clear()
        while True:
            s = board.state_id()
            self.states.append(s)
            player = board.player
            self.players.append(player)

            # check if we are on a leaf node (state form which no simulation was played so far)
            if s not in self.P:
                # return the board for the network evaluation
                return board

            # add dirichlet noise to the root node
            legal_moves = board.legal_moves()
            p_s = self.P[s]
            if alpha_dirich > 0:
                p_s = np.copy(p_s)
                alpha_params = alpha_dirich * np.ones(len(legal_moves))
                dirichlet_noise = np.random.dirichlet(alpha_params)
                p_s[legal_moves] = 0.75 * p_s[legal_moves] + 0.25 * dirichlet_noise

                # normalize the probabilities again
                p_s /= np.sum(p_s)

                # set the dirichlet noise to 0 in order to only add it to the root node
                alpha_dirich = 0

            # choose the action with the highest upper confidence bound
            max_ucb = -float("inf")
            action = -1
            for a in legal_moves:
                if (s, a) in self.Q:
                    u = self.Q[(s, a)] + Config.c_puct * p_s[a] * math.sqrt(self.N_s[s]) / (1 + self.N_sa[(s, a)])
                else:
                    u = Config.c_puct * p_s[a] * math.sqrt(self.N_s[s] + 1e-8)  # avoid division by 0

                if u > max_ucb:
                    max_ucb = u
                    action = a

            self.N_s[s] += 1
            self.actions.append(action)
            board = board.clone()
            board.play_move(action)

            # check if the game is terminal
            if board.terminal:
                v = board.training_reward()
                self.finish_simulation(board, v)
                return None


    def finish_simulation(self, board, value_white, policy=None):
        """
        ends one monte-carlo simulation with the terminal game value of the network evaluation
        :param board:           the board
        :param value_white:     the simulated value from the white perspective
        :param policy:          the policy if the simulation requires network inference
        :return:
        """
        if policy is not None:
            s = board.state_id()
            self.P[s] = policy

            # ensure that the summed probability of all valid moves is 1
            self.P[s][board.illegal_moves()] = 0
            total_prob = np.sum(self.P[s])
            if total_prob > 0:
                self.P[s] /= total_prob  # normalize the probabilities

            else:
                # the network did not choose any legal move, make all moves equally probable
                print("warning: network probabilities for all legal moves are 0, choose a equal distribution")
                legal_moves = board.legal_moves()
                self.P[s][legal_moves] = 1 / len(legal_moves)

            self.N_s[s] = 0


        # back up the tree by updating the Q and the N values
        for i in range(len(self.actions)):
            # flip the value for the black player since the game is always viewed from the white perspective
            v_true = value_white if self.players[i] == CONST.WHITE else -value_white

            s = self.states[i]
            a = self.actions[i]
            if (s, a) in self.Q:
                self.Q[(s, a)] = (self.N_sa[(s, a)] * self.Q[(s, a)] + v_true) / (self.N_sa[(s, a)] + 1)
                self.N_sa[(s, a)] += 1
            else:
                self.Q[(s, a)] = v_true
                self.N_sa[(s, a)] = 1



def run_simulations(mcts_list, mcts_sim_count, net, alpha_dirich):
    """
    runs a bunch of mcts simulations in parallel
    :param mcts_list:           list containing all the mcts objects
    :param mcts_sim_count:      the number of mcts simulations to perform
    :param net:                 the network that is used for evaluation
    :param alpha_dirich:        the dirichlet parameter alpha
    :return:
    """
    batch_list = []
    leaf_board_list = len(mcts_list) * [None]

    for sim in range(mcts_sim_count):
        batch_list.clear()
        for i_mcts_ctx, mcts_ctx in enumerate(mcts_list):
            # skip finished games
            if mcts_ctx.board.terminal:
                leaf_board_list[i_mcts_ctx] = None
                continue

            # execute one simulation
            leaf_board = mcts_list[i_mcts_ctx].exec_simulation(mcts_ctx.board, alpha_dirich)
            leaf_board_list[i_mcts_ctx] = leaf_board

            if leaf_board is not None:
                batch, _ = leaf_board.white_perspective()
                batch_list.append(batch)

        # pass all samples through the network
        if len(batch_list) > 0:
            batch_bundle = torch.Tensor(batch_list).to(Config.evaluation_device)
            policy, value = net(batch_bundle)
            policy = policy.detach().cpu().numpy()

        # finish the simulation of the states that need a result from the network
        i_sample = 0
        for i_mcts_ctx in range(len(mcts_list)):
            leaf_board = leaf_board_list[i_mcts_ctx]
            if leaf_board is not None:
                # get the value from the white perspective
                value_white = value[i_sample].item() if leaf_board.player == CONST.WHITE else -value[i_sample].item()

                # finish the simulation with the simulation with the network evaluation
                mcts_list[i_mcts_ctx].finish_simulation(leaf_board, value_white, policy[i_sample])
                i_sample += 1


def mcts_policy(board, mcts_sim_count, net, temp, alpha_dirich):
    """
    calculates the mcts policy for one board state
    :param board:               the board of which the policy should be calculated
    :param mcts_sim_count:      the number of mcts simulations
    :param net:                 the network
    :param temp:                the temperature
    :param alpha_dirich:        the dirichlet parameter alpha
    :return:                    policy vector for all next moves
    """

    mcts_list = [MCTS(board)]
    run_simulations(mcts_list, mcts_sim_count, net, alpha_dirich)
    policy = mcts_list[0].policy_from_state(mcts_list[0].board.state_id(), temp)
    return policy
