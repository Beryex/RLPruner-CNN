import torch
from torch import Tensor
import torch.nn as nn

from utils import Prune_agent

class VGG16(nn.Module):
    def __init__(self, 
                 in_channels: int =3, 
                 num_class: int =100):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, 
                x: Tensor):
        x = self.conv_layers(x)
        x = x.view(x.size()[0], -1)
        x = self.linear_layers(x)
        return x

    def get_prune_distribution_and_filter_num(self):
        prune_distribution = torch.zeros(self.prune_choices_num)
        for idx, layer_idx in enumerate(self.prune_choices):
            if idx <= self.last_conv_layer_idx:
                layer = self.conv_layers[layer_idx]
                prune_distribution[idx] = layer.out_channels
            else:
                layer = self.linear_layers[layer_idx]
                prune_distribution[idx] = layer.out_features
        filter_num = torch.sum(prune_distribution)
        prune_distribution = prune_distribution / filter_num
        return prune_distribution, filter_num

    def update_architecture(self, 
                            prune_agent: Prune_agent,
                            probability_lower_bound: float = 0.005):
        prune_counter = torch.zeros(self.prune_choices_num)
        noise = torch.randn(self.prune_choices_num) * prune_agent.noise_var * torch.rand(1).item()
        noised_distribution = prune_agent.prune_distribution + noise
        noised_distribution = torch.clamp(noised_distribution, min=probability_lower_bound)
        noised_distribution = noised_distribution / torch.sum(noised_distribution)
        prune_counter = torch.round(noised_distribution * prune_agent.modification_num)

        for target_layer_idx, count in enumerate(prune_counter):
            target_layer = self.prune_choices[target_layer_idx].item()
            for _ in range(int(count.item())):
                if target_layer_idx <= self.last_conv_layer_idx:
                    self.prune_filter_conv(target_layer)
                else:
                    self.prune_filter_linear(target_layer)
        
        return noised_distribution, prune_counter


    def prune_filter_conv(self, 
                          target_layer: int):
        # prune kernel
        target_kernel = self.conv_layers[target_layer].prune_filter()
        if target_kernel is None:
            return

        # update following BN layer
        target_bn = self.conv_layers[target_layer + 1]
        with torch.no_grad():
            kept_indices = [i for i in range(target_bn.num_features) if i != target_kernel]
            target_bn.weight.data = target_bn.weight.data[kept_indices]
            target_bn.bias.data = target_bn.bias.data[kept_indices]
            target_bn.running_mean = target_bn.running_mean[kept_indices]
            target_bn.running_var = target_bn.running_var[kept_indices]
        target_bn.num_features -= 1
        if target_bn.num_features != target_bn.weight.shape[0]:
            raise ValueError(f'BatchNorm layer number_features {target_bn.num_features} and weight dimension {target_bn.weight.shape[0]} mismath')

        # update next layer input
        if target_layer < self.prune_choices[self.last_conv_layer_idx]:
            target_layer += 1
            while (not isinstance(self.conv_layers[target_layer], Custom_Conv2d)):
                target_layer += 1
            self.conv_layers[target_layer].decre_input(target_kernel)
        else:
            # update first FC layer
            output_length = 1 # gained by printing "output"
            new_in_features = self.conv_layers[target_layer].out_channels * output_length ** 2
            start_index = target_kernel * output_length ** 2
            end_index = start_index + output_length ** 2
            self.linear_layers[0].decre_input(new_in_features, start_index, end_index)

    def prune_filter_linear(self, 
                            target_layer: int):
        target_neuron = self.linear_layers[target_layer].prune_filter()
        if target_neuron is None:
            return
        
        new_in_features = self.linear_layers[target_layer].out_features
        start_index = target_neuron
        end_index = target_neuron + 1
        self.linear_layers[target_layer + 3].decre_input(new_in_features, start_index, end_index)
