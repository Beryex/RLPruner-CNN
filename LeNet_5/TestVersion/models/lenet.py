import torch
import torch.nn as nn
class LeNet(nn.Module):
    # define internal methods inside the module
    def __init__(self, num_class=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)

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
    def update_architecture(self, modification_num):
        update_times = int(modification_num + 1)
        for update_id in range(update_times):
            if torch.rand(1).item() < 0.1:
                self.prune_kernel()
            else:
                self.prune_neuron()


    def prune_kernel(self):
        if torch.rand(1).item() < 0.3 and self.conv1.out_channels - 1 > 0:
            new_conv1_kernel_num = self.conv1.out_channels - 1
            # update conv1
            new_conv1 = nn.Conv2d(self.conv1.in_channels, new_conv1_kernel_num, kernel_size=5)
            # prone the kernel with least variance weights
            weight_variances = torch.var(self.conv1.weight.data, dim = [1, 2, 3])
            target_kernel = torch.argmin(weight_variances).item()
            with torch.no_grad():
                new_conv1.weight.data = torch.cat([self.conv1.weight.data[:target_kernel], self.conv1.weight.data[target_kernel+1:]], dim=0)
                new_conv1.bias.data = torch.cat([self.conv1.bias.data[:target_kernel], self.conv1.bias.data[target_kernel+1:]], dim=0)
            self.conv1 = new_conv1

            # update conv2
            new_conv2 = nn.Conv2d(self.conv1.out_channels, self.conv2.out_channels, kernel_size=5)
            with torch.no_grad():
                kept_indices = [i for i in range(self.conv2.in_channels) if i != target_kernel]
                new_conv2.weight.data = self.conv2.weight.data[:, kept_indices, :, :]
                new_conv2.bias.data = self.conv2.bias.data
            self.conv2 = new_conv2
        else:
            new_conv2_kernel_num = self.conv2.out_channels - 1
            if new_conv2_kernel_num == 0:
                return
            # update conv2
            new_conv2 = nn.Conv2d(self.conv1.out_channels, new_conv2_kernel_num, kernel_size=5)
            # prone the kernel with least variance weights
            weight_variances = torch.var(self.conv2.weight.data, dim = [1, 2, 3])
            target_kernel = torch.argmin(weight_variances).item()
            with torch.no_grad():
                new_conv2.weight.data = torch.cat([self.conv2.weight.data[:target_kernel], self.conv2.weight.data[target_kernel+1:]], dim=0)
                new_conv2.bias.data = torch.cat([self.conv2.bias.data[:target_kernel], self.conv2.bias.data[target_kernel+1:]], dim=0)
            self.conv2 = new_conv2

            # update fc1
            output_length = 5 # gained by printing "output"
            new_fc1_input_features = new_conv2_kernel_num * output_length ** 2
            new_fc1 = nn.Linear(new_fc1_input_features, self.fc1.out_features)
            with torch.no_grad():
                start_index = target_kernel * output_length ** 2
                end_index = start_index + output_length ** 2
                new_fc1.weight.data = torch.cat((self.fc1.weight.data[:, :start_index], self.fc1.weight.data[:, end_index:]), dim=1)
                new_fc1.bias.data = self.fc1.bias.data
            self.fc1 = new_fc1
    

    def prune_neuron(self):
        if torch.rand(1).item() < 0.6 and self.fc1.out_features - 1 > 0:
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
                