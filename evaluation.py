import torch
import numpy as np

from game import connect4
import mcts
from mcts import MCTS


torch_device = torch.device('cuda')         # the torch device for the network evaluation


def net_prediction_error(net, test_set):
    """
    returns the error percentage of the optimal move prediction by the network
    only the network is used to predict the correct move
    :param net:         the network
    :param test_set:    the test set
    :return:            error percentage, mean squared value error
    """
    tot_positions = test_set.shape[0]   # test_set.shape[0]   # the total number of positions in the test set
    correct_predictions = 0             # the correctly predicted positions in the test set
    tot_predictions = 0
    avg_mse_value = 0                   # to track the mean squared value error

    batch_size = 512
    batch_list = []
    idx_list = []
    board_list = []
    for j in range(tot_positions):
        # ignore losing positions
        if test_set["weak_score"][j] < 0:
            continue

        # load the board
        board = connect4.BitBoard()
        board.from_position(test_set["position"][j], test_set["disk_mask"][j])
        board_list.append(board)

        # get the white perspective
        sample, _ = board.white_perspective()
        batch_list.append(sample)
        idx_list.append(j)

        if len(batch_list) == batch_size or j == tot_positions - 1:
            batch = torch.Tensor(batch_list).to(torch_device)
            policy, value = net(batch)

            for i_batch in range(policy.shape[0]):
                _, move = policy[i_batch].max(0)
                move = move.item()

                # check if the move is part of the optimal moves
                idx = idx_list[i_batch]
                if str(move) in str(test_set["weak_moves"][idx]):
                    correct_predictions += 1

                tot_predictions += 1

                # value error
                correct_value = test_set["weak_score"][idx]
                net_value = value[i_batch].item()
                mse_value = (correct_value - net_value)**2
                avg_mse_value += mse_value

            batch_list = []
            idx_list = []
            board_list = []


    # calculate the prediction error
    pred_error = (tot_predictions - correct_predictions) / tot_predictions * 100
    avg_mse_value /= tot_predictions
    return pred_error, avg_mse_value



def mcts_prediction_error(net, test_set, mcts_sim_count, alpha_dirich, temp):
    """
    returns the error percentage of the optimal move prediction by the network
    the network and the mcts are used to predict the move to play
    :param net:             the network
    :param test_set:        the test set
    :param mcts_sim_count:  the number of monte carlo simulations
    :param alpha_dirich:    dirichlet noise parameter
    :param temp:            the temperature
    :return:                error percentage
    """
    tot_positions = test_set.shape[0]   # test_set.shape[0]   # the total number of positions in the test set
    correct_predictions = 0             # the correctly predicted positions in the test set
    tot_predictions = 0

    batch_size = 512
    idx_list = []
    mcts_list = []

    for j in range(tot_positions):
        # ignore losing positions
        if test_set["weak_score"][j] < 0:
            continue

        # load the board
        board = connect4.BitBoard()
        board.from_position(test_set["position"][j], test_set["disk_mask"][j])
        mcts_list.append(MCTS(board))

        # save the index
        idx_list.append(j)

        if len(mcts_list) == batch_size or j == tot_positions - 1:
            # =========================================== execute the mcts simulations for all boards
            mcts.run_simulations(mcts_list, mcts_sim_count, net, alpha_dirich)

            # ===========================================  get the policy from the mcts
            for i_mcts_ctx, mcts_ctx in enumerate(mcts_list):
                policy = mcts_list[i_mcts_ctx].policy_from_state(mcts_ctx.board.state_id(), temp)
                move = np.argmax(policy)

                # check if the move is part of the optimal moves
                idx = idx_list[i_mcts_ctx]
                if str(move) in str(test_set["weak_moves"][idx]):
                    correct_predictions += 1

                tot_predictions += 1

            idx_list = []
            mcts_list = []

    # calculate the prediction error
    error = (tot_predictions - correct_predictions) / tot_predictions * 100
    return error
