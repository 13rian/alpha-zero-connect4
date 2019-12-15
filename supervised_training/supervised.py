import torch
from torch.utils import data

from globals import CONST
from game import connect4


class Dataset(data.Dataset):
    def __init__(self, states, values, policies):
        """
        data set for the neural network training
        :param states:      the board state list
        :param values:      the value list
        :param policies:    the policy list
        """
        self.states = torch.Tensor(states)
        self.values = torch.Tensor(values).unsqueeze(1)
        self.policies = torch.Tensor(policies)


    def __len__(self):
        return len(self.values)


    def __getitem__(self, index):
        """
        returns a sample with the passed index
        :param index:   index of the sample
        :return:        state, value, policy
        """
        return self.states[index], self.values[index], self.policies[index]


def weak_sample_size(data_set):
    """
    returns the number of weak samples in the passed data set
    :param data_set:    csv data set
    :return:
    """
    sample_count = 0
    for i in range(data_set.shape[0]):
        if data_set["weak_score"][i] >= 0:
            sample_count += 1

    return sample_count


def create_training_set(data_set):
    """
    creates the pytorch data set from the passed csv data set
    :param data_set:    csv data set
    :return:
    """
    # parse the data set
    states = []
    values = []
    policies = []
    for i in range(data_set.shape[0]):
        if data_set["weak_score"][i] >= 0:
            # state
            board = connect4.BitBoard()
            position = data_set["position"][i]
            disk_mask = data_set["disk_mask"][i]
            board.from_position(position, disk_mask)
            state, _ = board.white_perspective()
            states.append(state)

            # values
            value = data_set["weak_score"][i]
            values.append(value)

            # policy
            policy = CONST.BOARD_WIDTH * [0]
            move_str = data_set["weak_moves"][i]
            moves = move_str.split('-')
            probability = 1 / len(moves)
            for move in moves:
                policy[int(move)] = probability
            policies.append(policy)

    return Dataset(states, values, policies)
