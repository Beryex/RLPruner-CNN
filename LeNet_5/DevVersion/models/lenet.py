from torch.nn import Module as Module_Base
from torch import nn
import torch.nn.functional as F
import torch
import math

# define model parameters

input_side_length = 32  # assume input image is square
input_channel = 1
conv1_kernel_size = 5
conv1_pool_size = 2
conv2_kernel_size = 5
conv2_pool_size = 2
fc3_kernel_num = 10     # this can not be changed as it must equal to the number output classes
prune_probability = 1
kernel_proportion = 3 / 8
neuron_proportion = 2 / 3
update_activitation_probability = 0.5
max_modification_num = 20   # for each update, at most how many kernels/neurons are we deleting/adding

class LeNet(Module_Base):
    # define internal methods inside the module
    def __init__(self, conv1_kernel_num_i = 6, conv2_kernel_num_i = 16):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, conv1_kernel_num_i, conv1_kernel_size)
        self.pool1 = nn.MaxPool2d(conv1_pool_size)
        self.conv2 = nn.Conv2d(conv1_kernel_num_i, conv2_kernel_num_i, conv2_kernel_size)
        self.pool2 = nn.MaxPool2d(conv2_pool_size)
        fc1_input_features = conv2_kernel_num_i * (int(((input_side_length - conv1_kernel_size + 1) / conv1_pool_size - conv2_kernel_size + 1) / conv2_pool_size) ** 2)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, fc3_kernel_num)

        self.conv1_activation_func = torch.nn.ReLU()
        self.conv2_activation_func = torch.nn.ReLU()
        self.fc1_activation_func = torch.nn.ReLU()
        self.fc2_activation_func = torch.nn.ReLU()
        self.fc3_activation_func = torch.nn.ReLU()
    

    # define execution of propagate forward using active_forward
    def forward(self, x):
        #return self.active_forward(x)
        y = self.conv1(x)
        y = (self.conv1_activation_func)(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = (self.conv2_activation_func)(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = (self.fc1_activation_func)(y)
        y = self.fc2(y)
        y = (self.fc2_activation_func)(y)
        y = self.fc3(y)
        y = (self.fc3_activation_func)(y)
        return y


    # define the function to resize the architecture kernel number
    def update_architecture(self):
        update_times = torch.randint(low=int(max_modification_num / 3), high=max_modification_num + 1, size=(1,))
        if torch.rand(1).item() < prune_probability:
            for update_id in range(update_times):
                if torch.rand(1).item() < 0.10:
                    self.prune_kernel()
                if torch.rand(1).item() > 0.10:
                    self.prune_neuron()
        else:
            for update_id in range(update_times):
                if torch.rand(1).item() < 0.10:
                    self.add_kernel()
                if torch.rand(1).item() > 0.10:
                    self.add_neuron()
        if torch.rand(1).item() < update_activitation_probability:
            self.change_activation_function()


    def prune_kernel(self):
        if torch.rand(1).item() < kernel_proportion and self.conv1.out_channels - 1 > 0:
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

            # update conv2
            new_conv2 = nn.Conv2d(self.conv1.out_channels, self.conv2.out_channels, conv2_kernel_size)
            with torch.no_grad():
                kept_indices = [i for i in range(self.conv2.in_channels) if i != target_kernel]
                new_conv2.weight.data = self.conv2.weight.data[:, kept_indices, :, :]
                new_conv2.bias.data = self.conv2.bias.data
            self.conv2 = new_conv2
        else:
            new_conv2_kernel_num = self.conv2.out_channels - 1
            if new_conv2_kernel_num == 0:
                return
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
        if torch.rand(1).item() < neuron_proportion and self.fc1.out_features - 1 > 0:
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

            # update fc2
            new_fc2 = nn.Linear(self.fc1.out_features, self.fc2.out_features)
            with torch.no_grad():
                new_fc2.weight.data = torch.cat([self.fc2.weight.data[:, :target_neuron], self.fc2.weight.data[:, target_neuron+1:]], dim=1)
                new_fc2.bias.data = self.fc2.bias.data
            self.fc2 = new_fc2
        else:
            new_fc2_output_features = self.fc2.out_features - 1
            if new_fc2_output_features == 0:
                return
            # update fc2
            new_fc2 = nn.Linear(self.fc2.in_features, new_fc2_output_features)
            # prune the neuron with least variance weights
            weight_variances = torch.var(self.fc2.weight.data, dim = 1)
            target_neuron = torch.argmin(weight_variances).item()
            with torch.no_grad():
                new_fc2.weight.data = torch.cat([self.fc2.weight.data[:target_neuron], self.fc2.weight.data[target_neuron+1:]], dim=0)
                new_fc2.bias.data = torch.cat([self.fc2.bias.data[:target_neuron], self.fc2.bias.data[target_neuron+1:]])
            self.fc2 = new_fc2

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

            # update fc2
            new_fc3 = nn.Linear(self.fc2.out_features, self.fc3.out_features)
            with torch.no_grad():
                # initial the new value channel weight and bias value with the average of original values
                weight_mean = self.fc3.weight.data.mean(dim = 1, keepdim = True)
                new_fc3.weight.data = torch.cat((self.fc3.weight.data, weight_mean), dim=1)
                new_fc3.bias.data = self.fc3.bias.data
            self.fc3 = new_fc3
    

    # update the conv1 activation function
    def change_activation_function(self):
        p1 = torch.rand(1).item()
        p2 = torch.rand(1).item()
        if p1 < 0.2:
            # change conv1
            if p2 < 0.2:
                self.conv1_activation_func = torch.nn.ReLU()
            elif p2 < 0.4:
                self.conv1_activation_func = torch.nn.Tanh()
            elif p2 < 0.6:
                self.conv1_activation_func = torch.nn.LeakyReLU()
            elif p2 < 0.8:
                self.conv1_activation_func = torch.nn.Sigmoid()
            else:
                self.conv1_activation_func = torch.nn.ELU()
        elif p1 < 0.4:
            # change conv2
            if p2 < 0.2:
                self.conv2_activation_func = torch.nn.ReLU()
            elif p2 < 0.4:
                self.conv2_activation_func = torch.nn.Tanh()
            elif p2 < 0.6:
                self.conv2_activation_func = torch.nn.LeakyReLU()
            elif p2 < 0.8:
                self.conv2_activation_func = torch.nn.Sigmoid()
            else:
                self.conv2_activation_func = torch.nn.ELU()
        elif p1 < 0.6:
            # change fc1
            if p2 < 0.2:
                self.fc1_activation_func = torch.nn.ReLU()
            elif p2 < 0.4:
                self.fc1_activation_func = torch.nn.Tanh()
            elif p2 < 0.6:
                self.fc1_activation_func = torch.nn.LeakyReLU()
            elif p2 < 0.8:
                self.fc1_activation_func = torch.nn.Sigmoid()
            else:
                self.fc1_activation_func = torch.nn.ELU()
        elif p1 < 0.8:
            # change fc2
            if p2 < 0.2:
                self.fc2_activation_func = torch.nn.ReLU()
            elif p2 < 0.4:
                self.fc2_activation_func = torch.nn.Tanh()
            elif p2 < 0.6:
                self.fc2_activation_func = torch.nn.LeakyReLU()
            elif p2 < 0.8:
                self.fc2_activation_func = torch.nn.Sigmoid()
            else:
                self.fc2_activation_func = torch.nn.ELU()
        else:
            # change fc3
            if p2 < 0.2:
                self.fc3_activation_func = torch.nn.ReLU()
            elif p2 < 0.4:
                self.fc3_activation_func = torch.nn.Tanh()
            elif p2 < 0.6:
                self.fc3_activation_func = torch.nn.LeakyReLU()
            elif p2 < 0.8:
                self.fc3_activation_func = torch.nn.Sigmoid()
            else:
                self.fc3_activation_func = torch.nn.ELU()
                