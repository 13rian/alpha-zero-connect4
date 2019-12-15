import torch
import numpy as np

import logging

from utils import utils
import networks
from gui import connect4_gui
from globals import Config



def mainGui():
    # The logger
    utils.init_logger(logging.DEBUG, file_name="log/gui.log")
    logger = logging.getLogger('Gui')

    net_path = "network_gen_148.pt"
    n_blocks = 10
    n_filters = 128

    np.set_printoptions(suppress=True, precision=2)


    # load the network
    Config.evaluation_device = torch.device('cpu')
    cpu_net = networks.ResNet(1e-4, n_blocks, n_filters, 1e-4)
    checkpoint = torch.load(net_path, map_location='cpu')
    cpu_net.load_state_dict(checkpoint['state_dict'])
    logger.debug("network loaded")


    # execute the game
    gui = connect4_gui.GUI(cpu_net)
    gui.execute_game()


if __name__ == '__main__':
    mainGui()
