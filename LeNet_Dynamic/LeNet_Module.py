from torch.nn import Module as Module_Base
from torch import nn
import torch
import math

# define model parameters

input_side_length = 28  # assume input image is square
input_channel = 1
conv1_kernel_size = 5
conv1_pool_size = 2
conv2_kernel_size = 5
conv2_pool_size = 2
fc3_kernel_num = 10     # this can not be changed as it must equal to the number output classes
prone_probability = 0.7
propotion = 3 / 8

class LeNet(Module_Base):
    # define internal methods inside the module
    def __init__(self, conv1_kernel_num_i = 6, conv2_kernel_num_i = 16):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, conv1_kernel_num_i, conv1_kernel_size)
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
    def update_architecture(self):
        if torch.rand(1).item() < prone_probability:
            self.neuron_prune()
        else:
            self.add_neuron()

    def neuron_prune(self):
        if torch.rand(1).item() < propotion:
            new_conv1_kernel_num = self.conv1.out_channels - 1
            # update conv1
            new_conv1 = nn.Conv2d(self.conv1.in_channels, new_conv1_kernel_num, conv1_kernel_size)
            # prone the neuron with least variance weights
            weight_variances = torch.var(self.conv1.weight.data, dim = [1, 2, 3])
            target_neuron = torch.argmin(weight_variances).item()
            with torch.no_grad():
                new_conv1.weight.data = torch.cat([self.conv1.weight.data[:target_neuron], self.conv1.weight.data[target_neuron+1:]], dim=0)
                new_conv1.bias.data = torch.cat([self.conv1.bias.data[:target_neuron], self.conv1.bias.data[target_neuron+1:]], dim=0)
            self.conv1 = new_conv1

            # update conv2
            new_conv2 = nn.Conv2d(self.conv1.out_channels, self.conv2.out_channels, conv2_kernel_size)
            with torch.no_grad():
                kept_indices = [i for i in range(self.conv2.in_channels) if i != target_neuron]
                new_conv2.weight.data = self.conv2.weight.data[:, kept_indices, :, :]
                new_conv2.bias.data = self.conv2.bias.data
            self.conv2 = new_conv2
        else:
            new_conv2_kernel_num = self.conv2.out_channels - 1
            # new_fc1_kernel_num = math.ceil(new_conv2_kernel_num * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2) / 2)
            # update conv2
            new_conv2 = nn.Conv2d(self.conv1.out_channels, new_conv2_kernel_num, conv2_kernel_size)
            # prone the neuron with least variance weights
            weight_variances = torch.var(self.conv2.weight.data, dim = [1, 2, 3])
            target_neuron = torch.argmin(weight_variances).item()
            with torch.no_grad():
                new_conv2.weight.data = torch.cat([self.conv2.weight.data[:target_neuron], self.conv2.weight.data[target_neuron+1:]], dim=0)
                new_conv2.bias.data = torch.cat([self.conv2.bias.data[:target_neuron], self.conv2.bias.data[target_neuron+1:]], dim=0)
            self.conv2 = new_conv2

            # update fc1
            new_fc1_input_features = new_conv2_kernel_num * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2)
            output_length = int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size)
            new_fc1 = nn.Linear(new_fc1_input_features, self.fc1.out_features)
            with torch.no_grad():
                start_index = target_neuron * output_length ** 2
                end_index = start_index + output_length ** 2
                new_fc1.weight.data = torch.cat((self.fc1.weight.data[:, :start_index], self.fc1.weight.data[:, end_index:]), dim=1)
                new_fc1.bias.data = self.fc1.bias.data
            self.fc1 = new_fc1
        print('%d, %d' %(self.conv1.out_channels, self.conv2.out_channels))
    

    def add_neuron(self):
        if torch.rand(1).item() < propotion:
            # add one neuron to convolution layer 1
            new_conv1_kernel_num = self.conv1.out_channels + 1
            # update conv1
            new_conv1 = nn.Conv2d(self.conv1.in_channels, new_conv1_kernel_num, conv1_kernel_size)
            with torch.no_grad():
                # initial the new value channel weight and bias value with the average of original values
                weight_mean = self.conv1.weight.data.mean(dim = 0, keepdim = True)
                bias_mean = self.conv1.bias.data.mean(dim = 0, keepdim = True)
                new_conv1.weight.data = torch.cat((self.conv1.weight.data, weight_mean), dim=0)
                new_conv1.bias.data = torch.cat((self.conv1.bias.data, bias_mean), dim=0)
            self.conv1 = new_conv1

            # update conv2
            new_conv2 = nn.Conv2d(self.conv1.out_channels, self.conv2.out_channels, conv2_kernel_size)
            with torch.no_grad():
                new_conv2.weight.data = self.conv2.weight.data[:self.conv2.out_channels, :self.conv2.in_channels, :, :]
                # initial the new value channel weight and bias value with the average of original values
                weight_mean = self.conv2.weight.data.mean(dim = 1, keepdim = True)
                new_conv2.weight.data = torch.cat((self.conv2.weight.data, weight_mean), dim = 1)
                new_conv2.bias.data = self.conv2.bias.data
            self.conv2 = new_conv2
        else:
            # add one neuron to convolution layer 2
            new_conv2_kernel_num = self.conv2.out_channels + 1

            # update conv2
            new_conv2 = nn.Conv2d(self.conv1.out_channels, new_conv2_kernel_num, conv2_kernel_size)
            with torch.no_grad():
                # initial the new value channel weight and bias value with the average of original values
                weight_mean = self.conv2.weight.data.mean(dim = 0, keepdim = True)
                bias_mean = self.conv2.bias.data.mean(dim = 0, keepdim = True)
                new_conv2.weight.data = torch.cat((self.conv2.weight.data, weight_mean), dim=0)
                new_conv2.bias.data = torch.cat((self.conv2.bias.data, bias_mean), dim=0)
            self.conv2 = new_conv2

            # update fc1
            new_fc1_input_features = new_conv2_kernel_num * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2)
            new_fc1 = nn.Linear(new_fc1_input_features, self.fc1.out_features)
            with torch.no_grad():
                new_fc1.weight.data[:, :self.fc1.in_features] = self.fc1.weight.data[:, :self.fc1.in_features]
                new_fc1.bias.data = self.fc1.bias.data
                # initial the new value channel weight and bias value with the average of original values
                weight_mean = self.fc1.weight.data.mean(dim = 1, keepdim = True)
                for i in range(self.fc1.in_features, new_fc1_input_features):
                    new_fc1.weight.data[:, i] = weight_mean.squeeze() + torch.randn_like(weight_mean).squeeze() * 0.01
            self.fc1 = new_fc1
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
    
    # define evaluating method
    def get_FLOPs(self):
        # ignore the computation inside the relu function, left term denotes multiple, right term denotes addition, comparison is consideration as addition
        output_side_length_1 = input_side_length - self.conv1.kernel_size[0] + 1
        Conv1_FLOPs = output_side_length_1 ** 2 * self.conv1.out_channels * self.conv1.in_channels * self.conv1.kernel_size[0] ** 2 + output_side_length_1 ** 2 * self.conv1.out_channels * (self.conv1.in_channels * (self.conv1.kernel_size[0] ** 2 - 1) + self.conv1.in_channels - 1 + 1) # plus 1 as bias term for convolution kernel
        MaxPool1_FLOPs = 0 + (output_side_length_1 / self.pool1.kernel_size) ** 2 * self.conv1.out_channels * (self.pool1.kernel_size ** 2 - 1)
        output_side_length_1 /= self.pool1.kernel_size
        output_side_length_2 = output_side_length_1 - self.conv2.kernel_size[0] + 1
        Conv2_FLOPs = output_side_length_2 ** 2 * self.conv2.out_channels * self.conv2.in_channels * self.conv2.kernel_size[0] ** 2 + output_side_length_2 ** 2 * self.conv2.out_channels * (self.conv2.in_channels * (self.conv2.kernel_size[0] ** 2 - 1) + self.conv2.in_channels - 1 + 1)
        MaxPool2_FLOPs = 0 + (output_side_length_2 / self.pool2.kernel_size) ** 2 * self.conv2.out_channels * (self.pool2.kernel_size ** 2 - 1)
        FC1_FLOPs = self.fc1.in_features * self.fc1.out_features + ((self.fc1.in_features - 1) * self.fc1.out_features + self.fc1.out_features)
        FC2_FLOPs = self.fc2.in_features * self.fc2.out_features + ((self.fc2.in_features - 1) * self.fc2.out_features + self.fc2.out_features)
        FC3_FLOPs = self.fc3.in_features * self.fc3.out_features + ((self.fc3.in_features - 1) * self.fc3.out_features + self.fc3.out_features)
        return Conv1_FLOPs + MaxPool1_FLOPs + Conv2_FLOPs + MaxPool2_FLOPs + FC1_FLOPs + FC2_FLOPs + FC3_FLOPs
    
    def get_parameter_num(self):
        Conv1_Para_num = (self.conv1.kernel_size[0] ** 2 * self.conv1.in_channels + 1) * self.conv1.out_channels
        Conv2_Para_num = (self.conv2.kernel_size[0] ** 2 * self.conv2.in_channels + 1) * self.conv2.out_channels
        FC1_Para_num = (self.fc1.in_features + 1) * self.fc1.out_features
        FC2_Para_num = (self.fc2.in_features + 1) * self.fc2.out_features
        FC3_Para_num = (self.fc3.in_features + 1) * self.fc3.out_features
        return Conv1_Para_num + Conv2_Para_num + FC1_Para_num + FC2_Para_num + FC3_Para_num