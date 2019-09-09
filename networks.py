import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from globals import CONST, Config



class ConvNet(nn.Module):
    """
    defines a convolutional neural network that ends in fully connected layers
    the network has a policy and a value head
    """

    def __init__(self, learning_rate, n_filters, dropout):
        super(ConvNet, self).__init__()

        self.n_channels = n_filters
        self.dropout = dropout

        # convolutional layers
        self.conv1 = nn.Conv2d(2, n_filters, kernel_size=3, padding=(1, 1), stride=1)           # baord 6x7
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=(1, 1), stride=1)   # baord 6x7
        self.conv3 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1)                   # baord 4x5
        self.conv4 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1)                   # baord 2x3

        # use batch normalization to improve stability and learning rate
        self.conv_bn1 = nn.BatchNorm2d(n_filters)
        self.conv_bn2 = nn.BatchNorm2d(n_filters)
        self.conv_bn3 = nn.BatchNorm2d(n_filters)
        self.conv_bn4 = nn.BatchNorm2d(n_filters)

        # fully connected layers
        self.fc1 = nn.Linear(n_filters * 2 * 3, 256)
        self.fc_bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.fc_bn2 = nn.BatchNorm1d(128)

        # policy head
        self.fc3p = nn.Linear(128, CONST.BOARD_WIDTH)  # approximation for the action value function Q(s, a)

        # value head
        self.fc3v = nn.Linear(128, 1)  # approximation for the value function V(s)

        # define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)



    def forward(self, x):
        # conv1
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = F.relu(x)

        # conv2
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = F.relu(x)

        # conv3
        x = self.conv3(x)
        x = self.conv_bn3(x)
        x = F.relu(x)

        # conv4
        x = self.conv4(x)
        x = self.conv_bn4(x)
        x = F.relu(x)

        # fc layer 1
        x = x.view(-1, self.n_channels * 2 * 3)  # transform to a vector
        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # fc layer 2
        x = self.fc2(x)
        x = self.fc_bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # policy
        p = self.fc3p(x)
        p = F.softmax(p, dim=1)  # values between 0 and 1

        # value
        v = self.fc3v(x)
        v = torch.tanh(v)  # values between -1 and 1

        return p, v


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
        target_p = target_p.to(Config.training_device)
        target_v = target_v.to(Config.training_device)
        criterion_p = nn.MSELoss()
        criterion_v = nn.MSELoss()

        # define the loss
        loss_p = criterion_p(prediction_p, target_p)
        loss_v = criterion_v(prediction_v, target_v)
        loss = loss_p + loss_v
        loss.backward()  # back propagation
        self.optimizer.step()  # make one optimization step
        return loss_p, loss_v


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
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters)

        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        # save the input for the skip connection
        residual = x

        # conv1
        out = F.relu(self.bn1(self.conv1(x)))

        # conv2 with the skip connection
        out = F.relu(self.bn2(self.conv2(out)) + residual)

        # skip connection
        return out



class OutBlock(nn.Module):
    """
    define the alpha zero output block with the value and the policy head
    """

    def __init__(self, n_filters):
        super(OutBlock, self).__init__()
        self.conv1_v = nn.Conv2d(n_filters, 3, kernel_size=1)  # value head
        self.bn1_v = nn.BatchNorm2d(3)
        self.fc1_v = nn.Linear(3 * 6 * 7, 32)
        self.fc2_v = nn.Linear(32, 1)

        self.conv1_p = nn.Conv2d(n_filters, 32, kernel_size=1)  # policy head
        self.bn1_p = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1_p = nn.Linear(6 * 7 * 32, 7)


    def forward(self, x):
        # value head
        v = F.relu(self.bn1_v(self.conv1_v(x)))
        v = v.view(-1, 3 * CONST.BOARD_SIZE)  # channels*board size
        v = F.relu(self.fc1_v(v))
        v = torch.tanh(self.fc2_v(v))

        # policy head
        p = F.relu(self.bn1_p(self.conv1_p(x)))
        p = p.view(-1, CONST.BOARD_SIZE * 32)
        p = self.fc1_p(p)
        p = self.logsoftmax(p).exp()
        return p, v



class ResNet(nn.Module):
    """
    defines a resudual neural network that ends in fully connected layers
    the network has a policy and a value head
    """

    def __init__(self, learning_rate, n_blocks, n_filters):
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
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)


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

        self.optimizer.zero_grad()  # reset the gradients to zero in every epoch
        prediction_p, prediction_v = self(data)  # pass the data through the network to get the prediction

        # create the label
        target_p = target_p.to(Config.training_device)
        target_v = target_v.to(Config.training_device)
        criterion_p = nn.MSELoss()
        criterion_v = nn.MSELoss()

        # define the loss
        loss_p = criterion_p(prediction_p, target_p)
        loss_v = criterion_v(prediction_v, target_v)
        loss = loss_p + loss_v
        loss.backward()  # back propagation
        self.optimizer.step()  # make one optimization step
        return loss_p, loss_v
