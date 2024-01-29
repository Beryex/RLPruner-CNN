from torch.nn import Module as Module_Base
from torch import nn
import math

# define model parameters
input_side_length = 28  # assume input image is square
conv1_kernel_num = 1
conv1_kernel_size = 5
conv1_pool_size = 2
conv2_kernel_num = 1
conv2_kernel_size = 5
conv2_pool_size = 2
# fc1_kernel_num = 120
# fc2_kernel_num = 84
fc3_kernel_num = 10     # this can not be changed as it must equal to the number output classes

class LeNet(Module_Base):
    # define internal methods inside the module
    def __init__(self, conv1_kernel_num_i = 1, conv2_kernel_num_i = 1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_kernel_num_i, conv1_kernel_size)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(conv1_pool_size)
        self.conv2 = nn.Conv2d(conv1_kernel_num_i, conv2_kernel_num_i, conv2_kernel_size)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(conv2_pool_size)
        fc1_kernel_num = math.ceil(conv2_kernel_num_i * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2) / 2)
        fc2_kernel_num = math.ceil(conv2_kernel_num_i * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2) / 3)
        self.fc1 = nn.Linear(conv2_kernel_num_i * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2), fc1_kernel_num)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(fc1_kernel_num, fc2_kernel_num)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(fc2_kernel_num, fc3_kernel_num)
        self.relu5 = nn.ReLU()
        print('%d, %d' %(conv1_kernel_num_i, conv2_kernel_num_i))
    
    # define the execution of propagate forward 
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
    