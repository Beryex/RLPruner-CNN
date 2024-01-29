from torch.nn import Module as Module_Base
from torch import nn
import torch
import math

# define model parameters

input_side_length = 28  # assume input image is square
# conv1_kernel_num = 6
conv1_kernel_size = 5
conv1_pool_size = 2
# conv2_kernel_num = 16
conv2_kernel_size = 5
conv2_pool_size = 2
# fc1_kernel_num = 120
# fc2_kernel_num = 84
fc3_kernel_num = 10     # this can not be changed as it must equal to the number output classes

class LeNet(Module_Base):
    # define internal methods inside the module
    def __init__(self, conv1_kernel_num_i = 6, conv2_kernel_num_i = 16):
        global fc1_kernel_num
        global fc2_kernel_num
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_kernel_num_i, conv1_kernel_size)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(conv1_pool_size)
        self.conv2 = nn.Conv2d(conv1_kernel_num_i, conv2_kernel_num_i, conv2_kernel_size)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(conv2_pool_size)
        fc1_kernel_num = math.ceil(conv2_kernel_num_i * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2) / 2)
        fc2_kernel_num = math.ceil(conv2_kernel_num_i * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2) / 3)
        fc1_input_features = conv2_kernel_num_i * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2)
        self.fc1 = nn.Linear(fc1_input_features, fc1_kernel_num)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(fc1_kernel_num, fc2_kernel_num)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(fc2_kernel_num, fc3_kernel_num)
        self.relu5 = nn.ReLU()
        print('%d, %d' %(self.conv1.out_channels, self.conv2.out_channels))
    
    # redefine the execution of propagate forward 
    def forward(self, x):
        # input feture is tensor [batchsize, # channel, height, width] = [256, 1, 28, 28]
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y
    
    # define the function to resize the architecture kernel number
    def resize_kernel_num(self):
        new_conv1_kernel_num = self.conv1.out_channels
        new_conv2_kernel_num = self.conv2.out_channels - 1
        if new_conv2_kernel_num < self.conv1.out_channels * 2.2:
            new_conv1_kernel_num = self.conv1.out_channels - 1
        new_fc1_kernel_num = math.ceil(new_conv2_kernel_num * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2) / 2)
        new_fc2_kernel_num = math.ceil(new_conv2_kernel_num * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2) / 3)
        # update conv1 if necessary
        if new_conv1_kernel_num != self.conv1.out_channels:
            new_conv1 = nn.Conv2d(self.conv1.in_channels, new_conv1_kernel_num, conv1_kernel_size)
            with torch.no_grad():
                new_conv1.weight.data = self.conv1.weight.data[:new_conv1_kernel_num, :, :, :]
                new_conv1.bias.data = self.conv1.bias.data[:new_conv1_kernel_num]
            self.conv1 = new_conv1

        # update conv2
        new_conv2 = nn.Conv2d(self.conv1.out_channels, new_conv2_kernel_num, conv2_kernel_size)
        with torch.no_grad():
            new_conv2.weight.data = self.conv2.weight.data[:new_conv2_kernel_num, :self.conv1.out_channels, :, :]
            new_conv2.bias.data = self.conv2.bias.data[:new_conv2_kernel_num]
        self.conv2 = new_conv2

        # update fc1 if necessary
        if new_fc1_kernel_num != self.fc1.out_features:
            new_fc1_input_features = new_conv2_kernel_num * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2)
            new_fc1 = nn.Linear(new_fc1_input_features, new_fc1_kernel_num)
            with torch.no_grad():
                new_fc1.weight.data = self.fc1.weight.data[:new_fc1_kernel_num, :new_fc1_input_features]
                new_fc1.bias.data = self.fc1.bias.data[:new_fc1_kernel_num]
            self.fc1 = new_fc1
        # update fc2 and fc3 if necessary
        if new_fc2_kernel_num != self.fc2.out_features:
            new_fc2 = nn.Linear(self.fc1.out_features, new_fc2_kernel_num)
            new_fc3 = nn.Linear(new_fc2_kernel_num, fc3_kernel_num)
            with torch.no_grad():
                new_fc2.weight.data = self.fc2.weight.data[:new_fc2_kernel_num, :self.fc1.out_features]
                new_fc2.bias.data = self.fc2.bias.data[:new_fc2_kernel_num]
                new_fc3.weight.data = self.fc3.weight.data[:, :new_fc2_kernel_num]
                new_fc3.bias.data = self.fc3.bias.data[:]
            self.fc2 = new_fc2
            self.fc3 = new_fc3
        print('%d, %d' %(self.conv1.out_channels, self.conv2.out_channels))

    # define the helper function
    def get_weights(self):
        cur_weights = []
        cur_weights.append(self.conv1.weight.data)
        cur_weights.append(self.conv2.weight.data)
        return cur_weights
    
    def get_bias(self):
        cur_bias = []
        cur_bias.append(self.conv1.bias.data)
        cur_bias.append(self.conv2.bias.data)
        return cur_bias