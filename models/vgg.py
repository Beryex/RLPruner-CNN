import torch
from torch import Tensor
import torch.nn as nn

from utils import Custom_Conv2d, Custom_Linear, Prune_agent

class VGG16(nn.Module):
    def __init__(self, 
                 in_channels: int =3, 
                 num_class: int =100):
        super().__init__()
        self.conv_layers = nn.Sequential(
            Custom_Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Custom_Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Custom_Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Custom_Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Custom_Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Custom_Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Custom_Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Custom_Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Custom_Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Custom_Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Custom_Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Custom_Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Custom_Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            Custom_Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            Custom_Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            Custom_Linear(4096, num_class)
        )

        self.prune_choices_num = 15
        self.last_conv_layer_idx = 12
        self.prune_choices = torch.tensor([0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40, 0, 3])

    def forward(self, 
                x: Tensor):
        x = self.conv_layers(x)
        x = x.view(x.size()[0], -1)
        x = self.linear_layers(x)
        return x


    def update_architecture(self, 
                            prune_agent: Prune_agent,
                            probability_lower_bound: float = 0.005):
        prune_counter = torch.zeros(self.prune_choices_num)
        noise = torch.randn(self.prune_choices_num) * prune_agent.noise_var * (torch.rand(1).item() * 0.5 + 0.5)
        noised_distribution = prune_agent.prune_distribution + noise
        noised_distribution = torch.clamp(noised_distribution, min=probability_lower_bound)
        noised_distribution = noised_distribution / torch.sum(noised_distribution)
        prune_counter = torch.round(noised_distribution * prune_agent.modification_num)
        
        conv_action, linear_action = {
            'prune_filter': (self.prune_filter_conv, self.prune_filter_linear),
            'weight_sharing': (self.weight_sharing_conv, self.weight_sharing_linear)
        }.get(prune_agent.strategy, (None, None))

        if conv_action and linear_action:
            for target_layer_idx, count in enumerate(prune_counter):
                target_layer = self.prune_choices[target_layer_idx].item()
                for _ in range(int(count.item())):
                    if target_layer_idx <= self.last_conv_layer_idx:
                        conv_action(target_layer)
                    else:
                        linear_action(target_layer)
        else:
            raise ValueError('Invalid strategy provided')
        
        return noised_distribution, prune_counter


    def prune_filter_conv(self, 
                   target_layer: int):
        # prune kernel
        target_kernel = self.conv_layers[target_layer].prune_filter()
        if target_kernel is None:
            return

        # update bn1
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
    
    def weight_sharing_conv(self, 
                      target_layer: int):
        self.conv_layers[target_layer].weight_sharing()
    
    def weight_sharing_linear(self, 
                        target_layer: int):
        self.linear_layers[target_layer].weight_sharing()
