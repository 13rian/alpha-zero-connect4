import torch
import torch.multiprocessing as mp
import os
from utils import utils
import data_storage
import pandas as pd
from game import connect4
import time




def __mp_worker__(network_dir, test_set_path, torch_device):
    """
    defines the multiprocessing test worker
    """
    print("worker started")

    # load the network
    path_list = os.listdir(network_dir)
    path_list.sort(key=utils.natural_keys)
    net_path = network_dir + path_list[-1]
    net = data_storage.load_net(net_path, torch_device)

    # load the test_set
    test_set = pd.read_csv(test_set_path, sep=",")
    tot_positions = test_set.shape[0]   # the total number of positions in the test set
    correct_predictions = 0             # the correctly predicted positions in the test set

    # calculate the prediction error
    for i in range(tot_positions):
        # load the board
        board = connect4.BitBoard()
        board.from_position(test_set["position"][i], test_set["disk_mask"][i])

        # get the white perspective
        board.white_perspective()
        batch, _ = board.white_perspective()
        batch = torch.Tensor(batch).unsqueeze(0).to(torch_device)

        # find the predicted move
        policy, _ = net(batch)
        _, move = policy.max(1)
        move = move.item()

        # check if the move is part of the optimal moves
        if str(move) in test_set["moves"][i]:
            correct_predictions += 1

    # calculate the prediction error
    error = (tot_positions - correct_predictions) / tot_positions * 100
    return error


def main():
    # create the multiprocessing pool
    n_pool_processes = 7  # mp.cpu_count()
    pool = mp.Pool(processes=n_pool_processes)

    test_set_path = "test_set/positions.csv"
    network_dir = "networks/"
    torch_device = torch.device('cpu')

    # print(__mp_worker__(network_dir, test_set_path, torch_device))

    start_time = time.time()
    prediction_errors = pool.starmap(__mp_worker__, zip([network_dir] * n_pool_processes,
                                                        [test_set_path] * n_pool_processes,
                                                        [torch_device] * n_pool_processes))

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(prediction_errors)
    print("elapsed time: ", elapsed_time)



if __name__ == '__main__':
    main()

