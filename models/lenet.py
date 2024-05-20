import torch
from torch import Tensor
import torch.nn as nn

from utils import Custom_Conv2d, Custom_Linear

class LeNet5(nn.Module):
    def __init__(self, in_channels: int=1, num_class: int=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            Custom_Conv2d(in_channels, 6, 5, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            Custom_Conv2d(6, 16, 5, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.linear_layers = nn.Sequential(
            Custom_Linear(400, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            Custom_Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            Custom_Linear(84, num_class)
        )
        
        self.prune_choices_num = 4
        self.prune_choices = torch.tensor([0, 4, 0, 3])
        self.prune_distribution = torch.tensor([0.1, 0.3, 0.3, 0.3])
    
    def forward(self, x: Tensor):
        x = self.conv_layers(x)
        x = x.view(x.size()[0], -1)
        x = self.linear_layers(x)
        return x


    def update_architecture(self, modification_num: int, strategy: str, noise_var: float = 0.01, probability_lower_bound: float = 0.005):
        prune_counter = torch.zeros(self.prune_choices_num)
        noise = torch.randn(self.prune_choices_num) * noise_var
        noised_distribution = self.prune_distribution + noise
        noised_distribution = torch.clamp(noised_distribution, min=probability_lower_bound)
        noised_distribution = noised_distribution / torch.sum(noised_distribution)
        prune_counter = noised_distribution * modification_num
        
        conv_action, linear_action = {
            'prune': (self.prune_conv, self.prune_linear),
            'quantize': (self.quantize_conv, self.quantize_linear)
        }.get(strategy, (None, None))

        if conv_action and linear_action:
            for target_layer_idx, count in enumerate(prune_counter):
                target_layer = self.prune_choices[target_layer_idx].item()
                count = int(count.item())
                for _ in range(count):
                    if target_layer_idx < 13:
                        conv_action(target_layer)
                    else:
                        linear_action(target_layer)
        else:
            raise ValueError('Invalid strategy provided')
        
        return noised_distribution
    
    
    def update_prune_distribution(self,top1_pretrain_accuracy_tensors: Tensor, prune_distribution_tensors: Tensor, step_length: float, probability_lower_bound: float, ppo_clip: float):
        distribution_weight = (top1_pretrain_accuracy_tensors - torch.min(top1_pretrain_accuracy_tensors)) / torch.sum(top1_pretrain_accuracy_tensors - torch.min(top1_pretrain_accuracy_tensors))
        distribution_weight = distribution_weight.unsqueeze(1)
        weighted_distribution = torch.sum(prune_distribution_tensors * distribution_weight, dim=0)
        new_prune_distribution = self.prune_distribution + step_length * (weighted_distribution - self.prune_distribution)
        ratio = new_prune_distribution / self.prune_distribution
        clipped_ratio = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip)
        self.prune_distribution *= clipped_ratio
        self.prune_distribution = torch.clamp(self.prune_distribution, min=probability_lower_bound)
        self.prune_distribution /= torch.sum(self.prune_distribution)


    def prune_conv(self, target_layer: int):
        # prune kernel
        target_kernel = self.conv_layers[target_layer].prune_kernel()
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

        if target_layer < 4:
            target_layer += 1
            while (not isinstance(self.conv_layers[target_layer], Custom_Conv2d)):
                target_layer += 1
            self.conv_layers[target_layer].decre_input(target_kernel)
        else:
            # update first FC layer
            output_length = 5 # gained by printing "output"
            new_in_features = self.conv_layers[target_layer].out_channels * output_length ** 2
            start_index = target_kernel * output_length ** 2
            end_index = start_index + output_length ** 2
            self.linear_layers[0].decre_input(new_in_features, start_index, end_index)

    def prune_linear(self, target_layer: int):
        target_neuron = self.linear_layers[target_layer].prune_neuron()
        if target_neuron is None:
            return
        
        new_in_features = self.linear_layers[target_layer].out_features
        start_index = target_neuron
        end_index = target_neuron + 1
        self.linear_layers[target_layer + 3].decre_input(new_in_features, start_index, end_index)
    
    def quantize_conv(self, target_layer: int):
        self.conv_layers[target_layer].quantization_hash_weights()
    
    def quantize_linear(self, target_layer: int):
        self.linear_layers[target_layer].quantization_hash_weights()