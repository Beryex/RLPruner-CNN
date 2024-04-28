import torch
import torch.nn as nn

from utils import Custom_Conv2d, Custom_Linear

class VGG16(nn.Module):
    def __init__(self, in_channels=3, num_class=100):
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
        self.prune_choices = torch.tensor([0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40, 0, 3])
        self.prune_distribution = torch.tensor([0.002, 0.002, 0.002, 0.002, 0.002, 0.06125, 0.06125, 0.06125, 0.06125, 0.06125, 0.06125, 0.06125, 0.06125, 0.25, 0.25])

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size()[0], -1)
        x = self.linear_layers(x)
        return x


    # define the function to resize the architecture kernel number
    def update_architecture(self, modification_num, strategy, noise_var=0.01, probability_lower_bound=0.005):
        prune_counter = torch.zeros(self.prune_choices_num)
        noise = torch.randn(self.prune_choices_num) * noise_var
        noised_distribution = self.prune_distribution + noise
        noised_distribution = torch.clamp(noised_distribution, min=probability_lower_bound)
        noised_distribution = noised_distribution / torch.sum(noised_distribution)
        prune_counter = noised_distribution * modification_num
        if strategy == 'prune':
            for target_layer_idx, count in enumerate(prune_counter):
                target_layer = self.prune_choices[target_layer_idx].item()
                count = int(count.item())
                for _ in range(count):
                    if target_layer_idx < 13:
                        self.prune_conv(target_layer)
                    else:
                        self.prune_linear(target_layer)
        elif strategy == 'quantize':
            for update_id in range(modification_num):
                if torch.rand(1).item() < 0.5:
                    self.quantize_conv()
                else:
                    self.quantize_linear()
        return noised_distribution
    
    
    def update_prune_distribution(self,top1_pretrain_accuracy_tensors, prune_distribution_tensors, step_length, probability_lower_bound, ppo_clip):
        distribution_weight = (top1_pretrain_accuracy_tensors - torch.min(top1_pretrain_accuracy_tensors)) / torch.sum(top1_pretrain_accuracy_tensors - torch.min(top1_pretrain_accuracy_tensors))
        distribution_weight = distribution_weight.unsqueeze(1)
        weighted_distribution = torch.sum(prune_distribution_tensors * distribution_weight, dim=0)
        new_prune_distribution = self.prune_distribution + step_length * (weighted_distribution - self.prune_distribution)
        ratio = new_prune_distribution / self.prune_distribution
        clipped_ratio = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip)
        self.prune_distribution *= clipped_ratio
        self.prune_distribution = torch.clamp(self.prune_distribution, min=probability_lower_bound)
        self.prune_distribution /= torch.sum(self.prune_distribution)


    def prune_conv(self, target_layer):
        # prune kernel
        target_kernel = self.conv_layers[target_layer].prune_kernel()
        if target_kernel == None:
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
            raise ValueError("BatchNorm layer number_features and weight dimension mismath")

        if target_layer < 40:
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

    def prune_linear(self, target_layer):
        target_neuron = self.linear_layers[target_layer].prune_neuron()
        if target_neuron == None:
            return
        
        new_in_features = self.linear_layers[target_layer].out_features
        start_index = target_neuron
        end_index = target_neuron + 1
        self.linear_layers[target_layer + 3].decre_input(new_in_features, start_index, end_index)
    
    def quantize_conv(self):
        if torch.rand(1).item() < 0.3:
            # low probabilit to quantize sensitive layer
            layer_choices =  torch.tensor([0, 3, 7, 10])
        else:
            layer_choices =  torch.tensor([14, 17, 20, 24, 27, 30, 34, 37, 40])
        target_layer = torch.randint(0, len(layer_choices), (1,)).item()
        target_layer = layer_choices[target_layer].item()

        # prune kernel
        self.conv_layers[target_layer].quantization_hash_weights()
    
    def quantize_linear(self):
        layer_choices =  torch.tensor([0, 3])
        target_layer = torch.randint(0, len(layer_choices), (1,)).item()
        target_layer = layer_choices[target_layer].item()

        self.linear_layers[target_layer].quantization_hash_weights()
