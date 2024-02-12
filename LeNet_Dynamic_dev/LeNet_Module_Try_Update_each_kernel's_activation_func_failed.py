from torch.nn import Module as Module_Base
from torch import nn
import torch.nn.functional as F
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
prune_probability = 0.7
kernel_proportion = 3 / 8
neuron_proportion = 2 / 3
update_activitation_probability = 1

class LeNet(Module_Base):
    # define internal methods inside the module
    def __init__(self, conv1_kernel_num_i = 6, conv2_kernel_num_i = 16):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, conv1_kernel_num_i, conv1_kernel_size)
        self.pool1 = nn.MaxPool2d(conv1_pool_size)
        self.conv2 = nn.Conv2d(conv1_kernel_num_i, conv2_kernel_num_i, conv2_kernel_size)
        self.pool2 = nn.MaxPool2d(conv2_pool_size)
        fc1_kernel_num = math.ceil(conv2_kernel_num_i * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2) / 2)
        fc2_kernel_num = math.ceil(conv2_kernel_num_i * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2) / 3)
        fc1_input_features = conv2_kernel_num_i * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2)
        self.fc1 = nn.Linear(fc1_input_features, fc1_kernel_num)
        self.fc2 = nn.Linear(fc1_kernel_num, fc2_kernel_num)
        self.fc3 = nn.Linear(fc2_kernel_num, fc3_kernel_num)

        self.conv1_activation_func_list = [F.relu for _ in range(self.conv1.out_channels)]
        self.conv2_activation_func_list = [F.relu for _ in range(self.conv2.out_channels)]
        self.fc1_activation_func_list = [F.relu for _ in range(self.fc1.out_features)]
        self.fc2_activation_func_list = [F.relu for _ in range(self.fc2.out_features)]
        self.fc3_activation_func_list = [F.relu for _ in range(self.fc3.out_features)]
        # self.active_forward = self.original_forward
    

    # define execution of propagate forward using active_forward
    def forward(self, x):
        #return self.active_forward(x)
        y = self.conv1(x)
        y1 = (self.conv1_activation_func_list[0])(y[:, 0, :, :])
        for i in range(1, self.conv1.out_channels):
            y2 = (self.conv1_activation_func_list[i])(y[:, i, :, :])
            y1 = torch.cat((y1, y2), dim = 1)
        y = y1
        y = self.pool1(y)
        y = self.conv2(y)
        y1 = (self.conv2_activation_func_list[0])(y[:, 0, :, :])
        for i in range(1, self.conv2.out_channels):
            y2 = (self.conv2_activation_func_list[i])(y[:, i, :, :])
            y1 = torch.cat((y1, y2), dim = 1)
        y = y1
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y1 = (self.fc1_activation_func_list[0])(y[:, 0, :, :])
        for i in range(1, self.fc1.out_features):
            y2 = (self.fc1_activation_func_list[i])(y[:, i, :, :])
            y1 = torch.cat((y1, y2), dim = 1)
        y = y1
        y = self.fc2(y)
        y1 = (self.fc2_activation_func_list[0])(y[:, 0, :, :])
        for i in range(1, self.fc2.out_features):
            y2 = (self.fc2_activation_func_list[i])(y[:, i, :, :])
            y1 = torch.cat((y1, y2), dim = 1)
        y = y1
        y = self.fc3(y)
        y1 = (self.fc3_activation_func_list[0])(y[:, 0, :, :])
        for i in range(1, self.fc3.out_features):
            y2 = (self.fc3_activation_func_list[i])(y[:, i, :, :])
            y1 = torch.cat((y1, y2), dim = 1)
        y = y1
        return y
    
    '''
    # define the default execution of propagate forward 
    def original_forward(self, x):
        # input feture is tensor [batchsize, # channel, height, width] = [256, 1, 28, 28]
        y = self.conv1(x)
        y = F.relu(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = F.relu(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = F.relu(y)
        y = self.fc3(y)
        y = F.relu(y)
        return y
    '''
    

    # update the conv1 activation function
    def change_activation_function(self):
        p = torch.rand(1).item()
        if p < 0.2:
            # change conv1
            weight_variances = torch.var(self.conv1.weight.data, dim = [1, 2, 3])
            target_kernel = torch.argmin(weight_variances).item()
            self.conv1_activation_func_list[target_kernel] = torch.tanh
        elif p < 0.4:
            # change conv2
            weight_variances = torch.var(self.conv2.weight.data, dim = [1, 2, 3])
            target_kernel = torch.argmin(weight_variances).item()
            self.conv2_activation_func_list[target_kernel] = torch.tanh
            '''
            def new_forward(x):
                y = self.conv1(x)
                y = F.relu(y)
                y = self.pool1(y)
                y = self.conv2(y)
                y1 = F.relu(y[:, 0: target_kernel, :, :])
                y2 = torch.tanh(y[:, target_kernel: target_kernel + 1, :, :])
                y3 = F.relu(y[:, target_kernel + 1:, :, :])
                y = torch.cat((y1, y2, y3), dim = 1)
                y = self.pool2(y)
                y = y.view(y.shape[0], -1)
                y = self.fc1(y)
                y = F.relu(y)
                y = self.fc2(y)
                y = F.relu(y)
                y = self.fc3(y)
                y = F.relu(y)
                return y
            self.active_forward = new_forward
            return
            '''
        elif p < 0.6:
            # change fc1
            weight_variances = torch.var(self.fc1.weight.data, dim = 1)
            target_kernel = torch.argmin(weight_variances).item()
            self.fc1_activation_func_list[target_kernel] = torch.tanh
        elif p < 0.8:
            # change fc2
            weight_variances = torch.var(self.fc2.weight.data, dim = 1)
            target_kernel = torch.argmin(weight_variances).item()
            self.fc2_activation_func_list[target_kernel] = torch.tanh
        else:
            # change fc3
            weight_variances = torch.var(self.fc3.weight.data, dim = 1)
            target_kernel = torch.argmin(weight_variances).item()
            self.fc3_activation_func_list[target_kernel] = torch.tanh



    # define the function to resize the architecture kernel number
    def update_architecture(self):
        if torch.rand(1).item() < prune_probability:
            if torch.rand(1).item() < 0.3:
                self.prune_kernel()
            else:
                self.prune_neuron()
        else:
            if torch.rand(1).item() < 0.3:
                self.add_kernel()
            else:
                self.add_neuron()
        if torch.rand(1).item() < update_activitation_probability:
            self.change_activation_function()

    def prune_kernel(self):
        if torch.rand(1).item() < kernel_proportion:
            new_conv1_kernel_num = self.conv1.out_channels - 1
            # update conv1
            new_conv1 = nn.Conv2d(self.conv1.in_channels, new_conv1_kernel_num, conv1_kernel_size)
            # prone the kernel with least variance weights
            weight_variances = torch.var(self.conv1.weight.data, dim = [1, 2, 3])
            target_kernel = torch.argmin(weight_variances).item()
            with torch.no_grad():
                new_conv1.weight.data = torch.cat([self.conv1.weight.data[:target_kernel], self.conv1.weight.data[target_kernel+1:]], dim=0)
                new_conv1.bias.data = torch.cat([self.conv1.bias.data[:target_kernel], self.conv1.bias.data[target_kernel+1:]], dim=0)
            self.conv1 = new_conv1
            del self.conv1_activation_func_list[target_kernel]

            # update conv2
            new_conv2 = nn.Conv2d(self.conv1.out_channels, self.conv2.out_channels, conv2_kernel_size)
            with torch.no_grad():
                kept_indices = [i for i in range(self.conv2.in_channels) if i != target_kernel]
                new_conv2.weight.data = self.conv2.weight.data[:, kept_indices, :, :]
                new_conv2.bias.data = self.conv2.bias.data
            self.conv2 = new_conv2
        else:
            new_conv2_kernel_num = self.conv2.out_channels - 1
            # new_fc1_kernel_num = math.ceil(new_conv2_kernel_num * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2) / 2)
            # update conv2
            new_conv2 = nn.Conv2d(self.conv1.out_channels, new_conv2_kernel_num, conv2_kernel_size)
            # prone the kernel with least variance weights
            weight_variances = torch.var(self.conv2.weight.data, dim = [1, 2, 3])
            target_kernel = torch.argmin(weight_variances).item()
            with torch.no_grad():
                new_conv2.weight.data = torch.cat([self.conv2.weight.data[:target_kernel], self.conv2.weight.data[target_kernel+1:]], dim=0)
                new_conv2.bias.data = torch.cat([self.conv2.bias.data[:target_kernel], self.conv2.bias.data[target_kernel+1:]], dim=0)
            self.conv2 = new_conv2
            del self.conv2_activation_func_list[target_kernel]

            # update fc1
            new_fc1_input_features = new_conv2_kernel_num * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2)
            output_length = int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size)
            new_fc1 = nn.Linear(new_fc1_input_features, self.fc1.out_features)
            with torch.no_grad():
                start_index = target_kernel * output_length ** 2
                end_index = start_index + output_length ** 2
                new_fc1.weight.data = torch.cat((self.fc1.weight.data[:, :start_index], self.fc1.weight.data[:, end_index:]), dim=1)
                new_fc1.bias.data = self.fc1.bias.data
            self.fc1 = new_fc1
    

    def prune_neuron(self):
        if torch.rand(1).item() < neuron_proportion:
            new_fc1_output_features = self.fc1.out_features - 1
            # update fc1
            new_fc1 = nn.Linear(self.fc1.in_features, new_fc1_output_features)
            # prune the neuron with least variance weights
            weight_variances = torch.var(self.fc1.weight.data, dim = 1)
            target_neuron = torch.argmin(weight_variances).item()
            with torch.no_grad():
                new_fc1.weight.data = torch.cat([self.fc1.weight.data[:target_neuron], self.fc1.weight.data[target_neuron+1:]], dim=0)
                new_fc1.bias.data = torch.cat([self.fc1.bias.data[:target_neuron], self.fc1.bias.data[target_neuron+1:]])
            self.fc1 = new_fc1
            del self.fc1_activation_func_list[target_neuron]

            # update fc2
            new_fc2 = nn.Linear(self.fc1.out_features, self.fc2.out_features)
            with torch.no_grad():
                new_fc2.weight.data = torch.cat([self.fc2.weight.data[:, :target_neuron], self.fc2.weight.data[:, target_neuron+1:]], dim=1)
                new_fc2.bias.data = self.fc2.bias.data
            self.fc2 = new_fc2
        else:
            new_fc2_output_features = self.fc2.out_features - 1
            # update fc2
            new_fc2 = nn.Linear(self.fc2.in_features, new_fc2_output_features)
            # prune the neuron with least variance weights
            weight_variances = torch.var(self.fc2.weight.data, dim = 1)
            target_neuron = torch.argmin(weight_variances).item()
            with torch.no_grad():
                new_fc2.weight.data = torch.cat([self.fc2.weight.data[:target_neuron], self.fc2.weight.data[target_neuron+1:]], dim=0)
                new_fc2.bias.data = torch.cat([self.fc2.bias.data[:target_neuron], self.fc2.bias.data[target_neuron+1:]])
            self.fc2 = new_fc2
            del self.fc2_activation_func_list[target_neuron]

            # update fc3
            new_fc3 = nn.Linear(self.fc2.out_features, self.fc3.out_features)
            with torch.no_grad():
                new_fc3.weight.data = torch.cat([self.fc3.weight.data[:, :target_neuron], self.fc3.weight.data[:, target_neuron+1:]], dim=1)
                new_fc3.bias.data = self.fc3.bias.data
            self.fc3 = new_fc3


    def add_kernel(self):
        if torch.rand(1).item() < kernel_proportion:
            # add one kernel to convolution layer 1
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
            self.conv1_activation_func_list.append(F.relu)

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
            # add one kernel to convolution layer 2
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
            self.conv2_activation_func_list.append(F.relu)

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


    def add_neuron(self):
        if torch.rand(1).item() < neuron_proportion:
            # add one neuron to fc1
            new_fc1_output_features = self.fc1.out_features + 1
            # update fc1
            new_fc1 = nn.Linear(self.fc1.in_features, new_fc1_output_features)
            with torch.no_grad():
                # initial the new value channel weight and bias value with the average of original values
                weight_mean = self.fc1.weight.data.mean(dim = 0, keepdim = True)
                bias_mean = self.fc1.bias.data.mean().unsqueeze(0)
                new_fc1.weight.data = torch.cat((self.fc1.weight.data, weight_mean), dim=0)
                new_fc1.bias.data = torch.cat((self.fc1.bias.data, bias_mean), dim=0)
            self.fc1 = new_fc1
            self.fc1_activation_func_list.append(F.relu)

            # update fc2
            new_fc2 = nn.Linear(self.fc1.out_features, self.fc2.out_features)
            with torch.no_grad():
                # initial the new value channel weight and bias value with the average of original values
                weight_mean = self.fc2.weight.data.mean(dim = 1, keepdim = True)
                new_fc2.weight.data = torch.cat((self.fc2.weight.data, weight_mean), dim=1)
                new_fc2.bias.data = self.fc2.bias.data
            self.fc2 = new_fc2
        else:
            # add one neuron to fc2
            new_fc2_output_features = self.fc2.out_features + 1
            # update fc2
            new_fc2 = nn.Linear(self.fc2.in_features, new_fc2_output_features)
            with torch.no_grad():
                # initial the new value channel weight and bias value with the average of original values
                weight_mean = self.fc2.weight.data.mean(dim = 0, keepdim = True)
                bias_mean = self.fc2.bias.data.mean().unsqueeze(0)
                new_fc2.weight.data = torch.cat((self.fc2.weight.data, weight_mean), dim=0)
                new_fc2.bias.data = torch.cat((self.fc2.bias.data, bias_mean), dim=0)
            self.fc2 = new_fc2
            self.fc2_activation_func_list.append(F.relu)

            # update fc2
            new_fc3 = nn.Linear(self.fc2.out_features, self.fc3.out_features)
            with torch.no_grad():
                # initial the new value channel weight and bias value with the average of original values
                weight_mean = self.fc3.weight.data.mean(dim = 1, keepdim = True)
                new_fc3.weight.data = torch.cat((self.fc3.weight.data, weight_mean), dim=1)
                new_fc3.bias.data = self.fc3.bias.data
            self.fc3 = new_fc3


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