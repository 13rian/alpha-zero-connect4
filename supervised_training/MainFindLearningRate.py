import matplotlib.pyplot as plt
from torch.utils import data
import pandas as pd
import random
import numpy as np
import logging


from utils import utils
from globals import Config
from supervised_training import supervised
import networks
import data_storage
import evaluation


def main_lr():
    """
    finds the min and the max learning rate. train the network for a few epochs and plot the
    prediction accuracy vs the learning rate. min learning rate is the rate where the prediction accuracy
    starts to increase and the max learning rate is the lr where the prediction accuracy slows down
    or even deteriorates. The batch size and all other hyperparameters should be the same for
    this test and the actual training. This test can be done with a smaller subset of the data set
    if the full data set is too large.
    :return:
    """
    # The logger
    utils.init_logger(logging.DEBUG, file_name="log/connect4.log")
    logger = logging.getLogger('Sup Learning')

    np.set_printoptions(suppress=True, precision=6)


    # set the random seed
    random.seed(a=None, version=2)
    np.random.seed(seed=None)


    # parameters
    Config.batch_size = 256
    Config.weight_decay = 1e-4
    Config.n_blocks = 10
    Config.n_filters = 128
    epochs = 8
    csv_training_set_path = "../data_sets/training_set.csv"
    csv_test_set_path = "../data_sets/training_set.csv"


    # load the test set
    csv_test_set = pd.read_csv(csv_test_set_path, sep=",")


    # load the training set
    csv_training_set = pd.read_csv(csv_training_set_path, sep=",")
    sample_count = supervised.weak_sample_size(csv_training_set)
    logger.info("the training set contains {} weak training examples".format(sample_count))

    # create the training data
    training_set = supervised.create_training_set(csv_training_set)
    logger.info("finished parsing the training set, size: {}".format(training_set.__len__()))


    # define the parameters for the training
    params = {
        'batch_size': Config.batch_size,
        'shuffle': True,
        'num_workers': 2,
        'pin_memory': True,
        'drop_last': True,
    }


    # generators
    training_generator = data.DataLoader(training_set, **params)


    # train the neural network
    learning_rates = []
    prediction_errors = []
    value_errors = []
    for power in np.arange(-6, 0.1, 0.25):
        Config.learning_rate = 10**power
        prediction_error, value_error = train_net(epochs, training_generator, csv_test_set)

        learning_rates.append(Config.learning_rate)
        prediction_errors.append(prediction_error)
        value_errors.append(value_error)


    # save the results
    np.save("learning_rates.npy", np.array(learning_rates))
    np.save("lr_policy_error.npy", np.array(prediction_errors))
    np.save("lr_value_error.npy", np.array(value_errors))

    # set the style of the plot
    plt.style.use('seaborn-dark-palette')

    # policy prediction error
    fig1 = plt.figure(1)
    plt.semilogx(learning_rates, prediction_errors)
    axes = plt.gca()
    axes.grid(True, color=(0.9, 0.9, 0.9))
    plt.title("Move Prediction Error")
    plt.xlabel("Learning Rate")
    plt.ylabel("Prediciton Error")
    fig1.show()


    # value prediction error
    fig2 = plt.figure(2)
    plt.semilogx(learning_rates, value_errors)
    axes = plt.gca()
    axes.grid(True, color=(0.9, 0.9, 0.9))
    plt.title("Position Value Error")
    plt.xlabel("Learning Rate")
    plt.ylabel("Value Error")
    fig2.show()

    plt.show()



def train_net(epoch_count, training_generator, csv_test_set):
    """
    trains the neural network a few times and returns the value and prediciton error
    :param epoch_count:             the number of epochs to train the neural network
    :param training_generator:      the torch training generator
    :param csv_test_set:            the test set on which the network is tested
    :return:                        prediction error, value error
    """
    logger = logging.getLogger('Sup Learning')

    # create a new network to train
    network = networks.ResNet(Config.learning_rate, Config.n_blocks, Config.n_filters, Config.weight_decay)
    network = data_storage.net_to_device(network, Config.training_device)



    # execute the training by looping over all epochs
    network.train()
    for epoch in range(epoch_count):
        # training
        for state_batch, value_batch, policy_batch in training_generator:
            # send the data to the gpu
            state_batch = state_batch.to(Config.training_device)
            value_batch = value_batch.to(Config.training_device)
            policy_batch = policy_batch.to(Config.training_device)

            # execute one training step
            _, _ = network.train_step(state_batch, policy_batch, value_batch)


    # evaluation
    pred_error, val_error = evaluation.net_prediction_error(network, csv_test_set)
    logger.debug("learning rate {}, prediction error: {}, value-error: {}".format(Config.learning_rate, pred_error, val_error))
    return pred_error, val_error



if __name__ == '__main__':
    main_lr()
