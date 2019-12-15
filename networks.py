import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from globals import CONST, Config


###############################################################################################################
#                                           ResNet                                                            #
###############################################################################################################
class ConvBlock(nn.Module):
    """
    define one convolutional block
    """

    def __init__(self, n_filters):
        super(ConvBlock, self).__init__()
        self.action_size = 7
        self.conv1 = nn.Conv2d(2, n_filters, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class ResBlock(nn.Module):
    """
    defines the residual block of the ResNet
    """

    def __init__(self, n_filters):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)  #bias=False
        self.bn1 = nn.BatchNorm2d(n_filters)

        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        # save the input for the skip connection
        residual = x

        # conv1
        out = F.relu(self.bn1(self.conv1(x)))

        # conv2 with the skip connection
        out = F.relu(self.bn2(self.conv2(out)) + residual)

        return out



class OutBlock(nn.Module):
    """
    define the alpha zero output block with the value and the policy head
    """

    def __init__(self, n_filters):
        super(OutBlock, self).__init__()
        self.conv1_v = nn.Conv2d(n_filters, 32, kernel_size=1)  # value head
        self.bn1_v = nn.BatchNorm2d(32)
        self.fc1_v = nn.Linear(32 * CONST.BOARD_SIZE, 256)
        self.fc2_v = nn.Linear(256, 1)

        self.conv1_p = nn.Conv2d(n_filters, 32, kernel_size=1)  # policy head
        self.bn1_p = nn.BatchNorm2d(32)
        self.fc1_p = nn.Linear(32 * CONST.BOARD_SIZE, 256)
        self.fc2_p = nn.Linear(256, 7)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        # value head
        v = F.relu(self.bn1_v(self.conv1_v(x)))
        v = v.view(-1, 32 * CONST.BOARD_SIZE)  # channels*board size
        v = F.relu(self.fc1_v(v))
        v = self.fc2_v(v)
        v = torch.tanh(v)

        # policy head
        p = F.relu(self.bn1_p(self.conv1_p(x)))
        p = p.view(-1, 32*CONST.BOARD_SIZE)
        p = F.relu(self.fc1_p(p))
        p = self.fc2_p(p)
        p = self.logsoftmax(p).exp()
        return p, v


class ResNet(nn.Module):
    """
    defines a residual neural network that ends in fully connected layers
    the network has a policy and a value head
    """

    def __init__(self, learning_rate, n_blocks, n_filters, weight_decay=0):
        super(ResNet, self).__init__()

        self.n_channels = n_filters
        self.n_blocks = n_blocks

        # initial convolutional block
        self.conv = ConvBlock(n_filters)

        # residual blocks
        for i in range(n_blocks):
            setattr(self, "res{}".format(i), ResBlock(n_filters))

        # output block with the policy and the value head
        self.outblock = OutBlock(n_filters)

        # define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # self.optimizer = optim.SGD(self.parameters(), momentum=0.9, lr=learning_rate, weight_decay=weight_decay)

        # define the scheduler
        self.scheduler = None   # the scheduler to adapt the learning rate


    def forward(self, x):
        # initial convolutional block
        out = self.conv(x)

        # residual blocks
        for i in range(self.n_blocks):
            out = getattr(self, "res{}".format(i))(out)

        # output block with the policy and value head
        out = self.outblock(out)
        return out


    def train_step(self, batch, target_p, target_v):
        """
        executes one training step of the neural network
        :param batch:           tensor with data [batchSize, nn_input_size]
        :param target_p:        policy target
        :param target_v:        value target
        :return:                policy loss, value loss
        """

        # send the tensors to the used device
        data = batch.to(Config.training_device)

        self.optimizer.zero_grad()               # reset the gradients to zero in every epoch
        prediction_p, prediction_v = self(data)  # pass the data through the network to get the prediction

        # create the label
        criterion_v = nn.MSELoss()

        # define the loss
        loss_p = -torch.sum(target_p * torch.log(1e-8 + prediction_p), 1).mean()
        loss_v = criterion_v(prediction_v, target_v)
        loss = loss_p + loss_v
        loss.backward()             # back propagation
        self.optimizer.step()       # make one optimization step
        return loss_p, loss_v


    def update_scheduler(self, step_size):
        """
        initializes a new scheduler, this method is needed when the size of the training data changes
        :param step_size:   step size of the cyclical learning rate
        :return:
        """
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            step_size_up=step_size,
            base_lr=Config.min_lr,
            max_lr=Config.max_lr,
            cycle_momentum=False)


    def train_cyclical_step(self, batch, target_p, target_v):
        """
        executes one training step of the neural network with a cyclic learning rate
        :param batch:           tensor with data [batchSize, nn_input_size]
        :param target_p:        policy target
        :param target_v:        value target
        :return:                policy loss, value loss
        """

        # send the tensors to the used device
        data = batch.to(Config.training_device)

        self.optimizer.zero_grad()               # reset the gradients to zero in every epoch
        prediction_p, prediction_v = self(data)  # pass the data through the network to get the prediction

        # create the label
        target_p = target_p.to(Config.training_device)
        target_v = target_v.to(Config.training_device)
        criterion_v = nn.MSELoss()


        # define the loss
        loss_p = - torch.sum(target_p * torch.log(1e-8 + prediction_p), 1).mean()
        loss_v = criterion_v(prediction_v, target_v)
        loss = loss_p + loss_v
        loss.backward()                          # back propagation

        # make one optimization step
        self.optimizer.step()
        self.scheduler.step()
        return loss_p, loss_v
