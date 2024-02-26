"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
from thop import profile

# define model parameters

input_side_length = 32  # assume input image is square
input_channel = 1
conv1_kernel_size = 3
conv1_pool_size = 2
conv2_kernel_size = 3
conv2_pool_size = 2
fc3_kernel_num = 10     # this can not be changed as it must equal to the number output classes
prune_probability = 1
kernel_neuron_proportion = 0.7
neuron_proportion = 0.5
update_activitation_probability = 0.5
max_modification_num = 500

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, num_class=100):
        super().__init__()
        self.features = nn.ModuleDict({
            'Conv1': nn.Conv2d(3, 64, kernel_size=3, padding=1),
            'bn1': nn.BatchNorm2d(64),
            'activation1': nn.ReLU(inplace=True),
            'Conv2': nn.Conv2d(64, 64, kernel_size=3, padding=1),
            'bn2': nn.BatchNorm2d(64),
            'activation2': nn.ReLU(inplace=True),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),
            'Conv3': nn.Conv2d(64, 128, kernel_size=3, padding=1),
            'bn3': nn.BatchNorm2d(128),
            'activation3': nn.ReLU(inplace=True),
            'Conv4': nn.Conv2d(128, 128, kernel_size=3, padding=1),
            'bn4': nn.BatchNorm2d(128),
            'activation4': nn.ReLU(inplace=True),
            'pool2': nn.MaxPool2d(kernel_size=2, stride=2),
            'Conv5': nn.Conv2d(128, 256, kernel_size=3, padding=1),
            'bn5': nn.BatchNorm2d(256),
            'activation5': nn.ReLU(inplace=True),
            'Conv6': nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'bn6': nn.BatchNorm2d(256),
            'activation6': nn.ReLU(inplace=True),
            'Conv7': nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'bn7': nn.BatchNorm2d(256),
            'activation7': nn.ReLU(inplace=True),
            'pool3': nn.MaxPool2d(kernel_size=2, stride=2),
            'Conv8': nn.Conv2d(256, 512, kernel_size=3, padding=1),
            'bn8': nn.BatchNorm2d(512),
            'activation8': nn.ReLU(inplace=True),
            'Conv9': nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'bn9': nn.BatchNorm2d(512),
            'activation9': nn.ReLU(inplace=True),
            'Conv10': nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'bn10': nn.BatchNorm2d(512),
            'activation10': nn.ReLU(inplace=True),
            'pool4': nn.MaxPool2d(kernel_size=2, stride=2),
            'Conv11': nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'bn11': nn.BatchNorm2d(512),
            'activation11': nn.ReLU(inplace=True),
            'Conv12': nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'bn12': nn.BatchNorm2d(512),
            'activation12': nn.ReLU(inplace=True),
            'Conv13': nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'bn13': nn.BatchNorm2d(512),
            'activation13': nn.ReLU(inplace=True),
            'pool5': nn.MaxPool2d(kernel_size=2, stride=2),
        })

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features['Conv1'](x)
        output = self.features['bn1'](output)
        output = self.features['activation1'](output)
        output = self.features['Conv2'](output)
        output = self.features['bn2'](output)
        output = self.features['activation2'](output)
        output = self.features['pool1'](output)
        output = self.features['Conv3'](output)
        output = self.features['bn3'](output)
        output = self.features['activation3'](output)
        output = self.features['Conv4'](output)
        output = self.features['bn4'](output)
        output = self.features['activation4'](output)
        output = self.features['pool2'](output)
        output = self.features['Conv5'](output)
        output = self.features['bn5'](output)
        output = self.features['activation5'](output)
        output = self.features['Conv6'](output)
        output = self.features['bn6'](output)
        output = self.features['activation6'](output)
        output = self.features['Conv7'](output)
        output = self.features['bn7'](output)
        output = self.features['activation7'](output)
        output = self.features['pool3'](output)
        output = self.features['Conv8'](output)
        output = self.features['bn8'](output)
        output = self.features['activation8'](output)
        output = self.features['Conv9'](output)
        output = self.features['bn9'](output)
        output = self.features['activation9'](output)
        output = self.features['Conv10'](output)
        output = self.features['bn10'](output)
        output = self.features['activation10'](output)
        output = self.features['pool4'](output)
        output = self.features['Conv11'](output)
        output = self.features['bn11'](output)
        output = self.features['activation11'](output)
        output = self.features['Conv12'](output)
        output = self.features['bn12'](output)
        output = self.features['activation12'](output)
        output = self.features['Conv13'](output)
        output = self.features['bn13'](output)
        output = self.features['activation13'](output)
        output = self.features['pool5'](output)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output


    # define the function to resize the architecture kernel number
    def update_architecture(self):
        update_times = torch.randint(low=int(max_modification_num / 3), high=max_modification_num + 1, size=(1,))
        if torch.rand(1).item() < prune_probability:
            for update_id in range(update_times):
                if torch.rand(1).item() < kernel_neuron_proportion:
                    self.prune_kernel()
                if torch.rand(1).item() > kernel_neuron_proportion:
                    self.prune_neuron()
        else:
            for update_id in range(update_times):
                if torch.rand(1).item() < kernel_neuron_proportion:
                    self.add_kernel()
                if torch.rand(1).item() > kernel_neuron_proportion:
                    self.add_neuron()
        if torch.rand(1).item() < update_activitation_probability:
            self.change_activation_function()


    def prune_kernel(self):
        conv_num = 13
        target_layer = torch.randint(low=1, high=conv_num + 1, size=(1,)).item()

        new_conv1_kernel_num = self.features['Conv' + str(target_layer)].out_channels - 1
        if new_conv1_kernel_num == 0:
            return
        # update conv1
        new_conv1 = nn.Conv2d(self.features['Conv' + str(target_layer)].in_channels, new_conv1_kernel_num, kernel_size=3, padding=1)
        # prone the kernel with least variance weights
        weight_variances = torch.var(self.features['Conv' + str(target_layer)].weight.data, dim = [1, 2, 3])
        target_kernel = torch.argmin(weight_variances).item()
        with torch.no_grad():
            new_conv1.weight.data = torch.cat([self.features['Conv' + str(target_layer)].weight.data[:target_kernel], self.features['Conv' + str(target_layer)].weight.data[target_kernel+1:]], dim=0)
            new_conv1.bias.data = torch.cat([self.features['Conv' + str(target_layer)].bias.data[:target_kernel], self.features['Conv' + str(target_layer)].bias.data[target_kernel+1:]], dim=0)
        self.features['Conv' + str(target_layer)] = new_conv1

        # update bn1
        new_bn1 = nn.BatchNorm2d(new_conv1_kernel_num)
        with torch.no_grad():
            kept_indices = [i for i in range(self.features['bn' + str(target_layer)].num_features) if i != target_kernel]
            new_bn1.weight.data = self.features['bn' + str(target_layer)].weight.data[kept_indices]
            new_bn1.bias.data = self.features['bn' + str(target_layer)].bias.data[kept_indices]
            new_bn1.running_mean = self.features['bn' + str(target_layer)].running_mean[kept_indices]
            new_bn1.running_var = self.features['bn' + str(target_layer)].running_var[kept_indices]
        self.features['bn' + str(target_layer)] = new_bn1

        if target_layer < 13:
            # if not last Conv layer, update conv2
            new_conv2 = nn.Conv2d(self.features['Conv' + str(target_layer)].out_channels, self.features['Conv' + str(target_layer + 1)].out_channels, kernel_size=3, padding=1)
            with torch.no_grad():
                kept_indices = [i for i in range(self.features['Conv' + str(target_layer + 1)].in_channels) if i != target_kernel]
                new_conv2.weight.data = self.features['Conv' + str(target_layer + 1)].weight.data[:, kept_indices, :, :]
                new_conv2.bias.data = self.features['Conv' + str(target_layer + 1)].bias.data
            self.features['Conv' + str(target_layer + 1)] = new_conv2
        else:
            # if so, update first FC layer
            new_fc1_intput_features = self.features['Conv' + str(target_layer)].out_channels
            # update fc1
            new_fc1 = nn.Linear(new_fc1_intput_features, self.classifier[0].out_features)
            output_length = 1 # gained by printing "output"
            # prune the neuron with least variance weights
            with torch.no_grad():
                start_index = target_kernel * output_length ** 2
                end_index = start_index + output_length ** 2
                new_fc1.weight.data = torch.cat([self.classifier[0].weight.data[:, :start_index], self.classifier[0].weight.data[:, end_index:]], dim=1)
                new_fc1.bias.data = self.classifier[0].bias.data
            self.classifier[0] = new_fc1
    

    def prune_neuron(self):
        if torch.rand(1).item() < neuron_proportion and self.classifier[0].out_features - 1 > 0:
            new_fc1_output_features = self.classifier[0].out_features - 1
            # update fc1
            new_fc1 = nn.Linear(self.classifier[0].in_features, new_fc1_output_features)
            # prune the neuron with least variance weights
            weight_variances = torch.var(self.classifier[0].weight.data, dim = 1)
            target_neuron = torch.argmin(weight_variances).item()
            with torch.no_grad():
                new_fc1.weight.data = torch.cat([self.classifier[0].weight.data[:target_neuron], self.classifier[0].weight.data[target_neuron+1:]], dim=0)
                new_fc1.bias.data = torch.cat([self.classifier[0].bias.data[:target_neuron], self.classifier[0].bias.data[target_neuron+1:]])
            self.classifier[0] = new_fc1

            # update fc2
            new_fc2 = nn.Linear(self.classifier[0].out_features, self.classifier[3].out_features)
            with torch.no_grad():
                new_fc2.weight.data = torch.cat([self.classifier[3].weight.data[:, :target_neuron], self.classifier[3].weight.data[:, target_neuron+1:]], dim=1)
                new_fc2.bias.data = self.classifier[3].bias.data
            self.classifier[3] = new_fc2
        else:
            new_fc2_output_features = self.classifier[3].out_features - 1
            if new_fc2_output_features == 0:
                return
            # update fc2
            new_fc2 = nn.Linear(self.classifier[3].in_features, new_fc2_output_features)
            # prune the neuron with least variance weights
            weight_variances = torch.var(self.classifier[3].weight.data, dim = 1)
            target_neuron = torch.argmin(weight_variances).item()
            with torch.no_grad():
                new_fc2.weight.data = torch.cat([self.classifier[3].weight.data[:target_neuron], self.classifier[3].weight.data[target_neuron+1:]], dim=0)
                new_fc2.bias.data = torch.cat([self.classifier[3].bias.data[:target_neuron], self.classifier[3].bias.data[target_neuron+1:]])
            self.classifier[3] = new_fc2

            # update fc3
            new_fc3 = nn.Linear(self.classifier[3].out_features, self.classifier[6].out_features)
            with torch.no_grad():
                new_fc3.weight.data = torch.cat([self.classifier[6].weight.data[:, :target_neuron], self.classifier[6].weight.data[:, target_neuron+1:]], dim=1)
                new_fc3.bias.data = self.classifier[6].bias.data
            self.classifier[6] = new_fc3


    def add_kernel(self):
        conv_num = 12 # exclude the final convolutional layer
        target_layer = torch.randint(low=1, high=conv_num + 1, size=(1,)).item()
        # add one kernel to convolution layer 1
        new_conv1_kernel_num = self.features['Conv' + str(target_layer)].out_channels + 1
        # update conv1
        new_conv1 = nn.Conv2d(self.features['Conv' + str(target_layer)].in_channels, new_conv1_kernel_num, kernel_size=3, padding=1)
        with torch.no_grad():
            # initial the new value channel weight and bias value with the average of original values
            weight_mean = self.features['Conv' + str(target_layer)].weight.data.mean(dim = 0, keepdim = True)
            bias_mean = self.features['Conv' + str(target_layer)].bias.data.mean(dim = 0, keepdim = True)
            new_conv1.weight.data = torch.cat((self.features['Conv' + str(target_layer)].weight.data, weight_mean), dim=0)
            new_conv1.bias.data = torch.cat((self.features['Conv' + str(target_layer)].bias.data, bias_mean), dim=0)
        self.features['Conv' + str(target_layer)] = new_conv1

        # update bn1
        new_bn1 = nn.BatchNorm2d(new_conv1_kernel_num)
        with torch.no_grad():
            new_bn1.weight.data = torch.cat((self.features['bn' + str(target_layer)].weight.data, self.features['bn' + str(target_layer)].weight.data.mean(dim=0, keepdim = True)), dim=0)
            new_bn1.bias.data = torch.cat((self.features['bn' + str(target_layer)].bias.data, self.features['bn' + str(target_layer)].bias.data.mean(dim=0, keepdim=True)), dim=0)
            new_bn1.running_mean = torch.cat((self.features['bn' + str(target_layer)].running_mean, self.features['bn' + str(target_layer)].running_mean.mean(dim=0, keepdim=True)), dim=0)
            new_bn1.running_var = torch.cat((self.features['bn' + str(target_layer)].running_var, self.features['bn' + str(target_layer)].running_var.mean(dim=0, keepdim=True)), dim=0)
        self.features['bn' + str(target_layer)] = new_bn1

        # update conv2
        new_conv2 = nn.Conv2d(self.features['Conv' + str(target_layer)].out_channels, self.features['Conv' + str(target_layer + 1)].out_channels, kernel_size=3, padding=1)
        with torch.no_grad():
            new_conv2.weight.data = self.features['Conv' + str(target_layer + 1)].weight.data[:self.features['Conv' + str(target_layer + 1)].out_channels, :self.features['Conv' + str(target_layer + 1)].in_channels, :, :]
            # initial the new value channel weight and bias value with the average of original values
            weight_mean = self.features['Conv' + str(target_layer + 1)].weight.data.mean(dim = 1, keepdim = True)
            new_conv2.weight.data = torch.cat((self.features['Conv' + str(target_layer + 1)].weight.data, weight_mean), dim = 1)
            new_conv2.bias.data = self.features['Conv' + str(target_layer+ 1)].bias.data
        self.features['Conv' + str(target_layer + 1)] = new_conv2


    def add_neuron(self):
        if torch.rand(1).item() < neuron_proportion:
            # add one neuron to fc1
            new_fc1_output_features = self.classifier[0].out_features + 1
            # update fc1
            new_fc1 = nn.Linear(self.classifier[0].in_features, new_fc1_output_features)
            with torch.no_grad():
                # initial the new value channel weight and bias value with the average of original values
                weight_mean = self.classifier[0].weight.data.mean(dim = 0, keepdim = True)
                bias_mean = self.classifier[0].bias.data.mean().unsqueeze(0)
                new_fc1.weight.data = torch.cat((self.classifier[0].weight.data, weight_mean), dim=0)
                new_fc1.bias.data = torch.cat((self.classifier[0].bias.data, bias_mean), dim=0)
            self.classifier[0] = new_fc1

            # update fc2
            new_fc2 = nn.Linear(self.classifier[0].out_features, self.classifier[3].out_features)
            with torch.no_grad():
                # initial the new value channel weight and bias value with the average of original values
                weight_mean = self.classifier[3].weight.data.mean(dim = 1, keepdim = True)
                new_fc2.weight.data = torch.cat((self.classifier[3].weight.data, weight_mean), dim=1)
                new_fc2.bias.data = self.classifier[3].bias.data
            self.classifier[3] = new_fc2
        else:
            # add one neuron to fc2
            new_fc2_output_features = self.classifier[3].out_features + 1
            # update fc2
            new_fc2 = nn.Linear(self.classifier[3].in_features, new_fc2_output_features)
            with torch.no_grad():
                # initial the new value channel weight and bias value with the average of original values
                weight_mean = self.classifier[3].weight.data.mean(dim = 0, keepdim = True)
                bias_mean = self.classifier[3].bias.data.mean().unsqueeze(0)
                new_fc2.weight.data = torch.cat((self.classifier[3].weight.data, weight_mean), dim=0)
                new_fc2.bias.data = torch.cat((self.classifier[3].bias.data, bias_mean), dim=0)
            self.classifier[3] = new_fc2

            # update fc2
            new_fc3 = nn.Linear(self.classifier[3].out_features, self.classifier[6].out_features)
            with torch.no_grad():
                # initial the new value channel weight and bias value with the average of original values
                weight_mean = self.classifier[6].weight.data.mean(dim = 1, keepdim = True)
                new_fc3.weight.data = torch.cat((self.classifier[6].weight.data, weight_mean), dim=1)
                new_fc3.bias.data = self.classifier[6].bias.data
            self.classifier[6] = new_fc3
    

    # update the conv1 activation function
    def change_activation_function(self):
        activation_funcs = 15
        target_layer = torch.randint(low=1, high=activation_funcs + 1, size=(1,)).item()
        p1 = torch.rand(1).item()
        if target_layer <= 13:
            # change convolution layer
            if p1 < 0.2:
                self.features['activation' + str(target_layer)] = torch.nn.ReLU(inplace=True)
            elif p1 < 0.4:
                self.features['activation' + str(target_layer)] = torch.nn.Tanh()
            elif p1 < 0.6:
                self.features['activation' + str(target_layer)] = torch.nn.LeakyReLU(inplace=True)
            elif p1 < 0.8:
                self.features['activation' + str(target_layer)] = torch.nn.Sigmoid()
            else:
                self.features['activation' + str(target_layer)] = torch.nn.ELU(inplace=True)
        elif target_layer == 14:
            # change fully connection layer
            if p1 < 0.2:
                self.classifier[1] = torch.nn.ReLU(inplace=True)
            elif p1 < 0.4:
                self.classifier[1] = torch.nn.Tanh()
            elif p1 < 0.6:
                self.classifier[1] = torch.nn.LeakyReLU(inplace=True)
            elif p1 < 0.8:
                self.classifier[1] = torch.nn.Sigmoid()
            else:
                self.classifier[1] = torch.nn.ELU(inplace=True)
        else:
            if p1 < 0.2:
                self.classifier[4] = torch.nn.ReLU(inplace=True)
            elif p1 < 0.4:
                self.classifier[4] = torch.nn.Tanh()
            elif p1 < 0.6:
                self.classifier[4] = torch.nn.LeakyReLU(inplace=True)
            elif p1 < 0.8:
                self.classifier[4] = torch.nn.Sigmoid()
            else:
                self.classifier[4] = torch.nn.ELU(inplace=True)
