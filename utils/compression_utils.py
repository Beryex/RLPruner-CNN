import torch
from torch import Tensor
import math
import torch.nn as nn

from conf import settings

class Prune_agent():
    def __init__(self,
                 prune_distribution: Tensor,
                 ReplayBuffer: Tensor, 
                 filter_num: int,
                 cur_top1_acc: float,
                 prune_filter_max_ratio: float = settings.C_PRUNE_FILTER_MAX_RATIO,
                 prune_filter_min_ratio: float = settings.C_PRUNE_FILTER_MIN_RATIO,
                 noise_var: float = settings.RL_PRUNE_FILTER_NOISE_VAR):
        self.modification_max_num = int(filter_num * prune_filter_max_ratio)
        self.modification_min_num = int(filter_num * prune_filter_min_ratio)
        self.prune_distribution = prune_distribution
        self.noise_var = noise_var
        self.modification_num = self.modification_max_num
        self.T_max = settings.C_COS_PRUNE_EPOCH
        self.ReplayBuffer = ReplayBuffer
        self.Reward_cache = {}
        self.net_list = [None] * settings.RL_MAX_GENERATE_NUM
        self.cur_single_step_acc_threshold = settings.C_SINGLE_STEP_ACCURACY_CHANGE_THRESHOLD
        self.lr_epoch = settings.RL_LR_EPOCH
        self.cur_Q_value_max = (cur_top1_acc * settings.RL_CUR_ACC_TO_CUR_Q_VALUE_COEFFICIENT + 
                                cur_top1_acc * settings.RL_CUR_ACC_TO_CUR_Q_VALUE_COEFFICIENT ** 2 * settings.RL_DISCOUNT_FACTOR)

    def step(self, 
             optimal_net_index: int, 
             epoch: int,
             cur_top1_acc: float):
        if optimal_net_index == 1:
            # means generated net is better, reset counter then clear the ReplayBuffer, Reward_cache and net_list
            self.ReplayBuffer.zero_()
            self.Reward_cache = {}
            self.net_list = [None] * settings.RL_MAX_GENERATE_NUM
            self.cur_single_step_acc_threshold = settings.C_SINGLE_STEP_ACCURACY_CHANGE_THRESHOLD
            self.cur_Q_value_max = (cur_top1_acc * settings.RL_CUR_ACC_TO_CUR_Q_VALUE_COEFFICIENT + 
                                    cur_top1_acc * settings.RL_CUR_ACC_TO_CUR_Q_VALUE_COEFFICIENT ** 2 * settings.RL_DISCOUNT_FACTOR)
        else:
            self.cur_single_step_acc_threshold += settings.C_SINGLE_STEP_ACCURACY_CHANGE_THRESHOLD_INCRE
        # update modification_num using method similiar to CosineAnnealingLR
        if epoch < self.T_max:
            self.modification_num = int(self.modification_min_num + (self.modification_max_num - self.modification_min_num) * 0.5 * (math.cos(math.pi * epoch / self.T_max) + 1))

    def update_prune_distribution(self, step_length: float, probability_lower_bound: float, ppo_clip: float, ppo_enable: bool):
        ReplayBuffer = self.ReplayBuffer
        original_prune_distribution = self.prune_distribution
        _, optimal_idx = torch.max(ReplayBuffer[:, 0], dim=0)
        optimal_distribution = ReplayBuffer[optimal_idx, 1:]
        
        updated_prune_distribution = original_prune_distribution + step_length * (optimal_distribution - original_prune_distribution) 
        updated_prune_distribution = torch.clamp(updated_prune_distribution, min=probability_lower_bound)
        updated_prune_distribution /= torch.sum(updated_prune_distribution)
        
        if ppo_enable == True:
            ratio = updated_prune_distribution / original_prune_distribution
            updated_prune_distribution = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * original_prune_distribution
            updated_prune_distribution = torch.clamp(updated_prune_distribution, min=probability_lower_bound)
            updated_prune_distribution /= torch.sum(updated_prune_distribution)
        self.prune_distribution = updated_prune_distribution
        return updated_prune_distribution - original_prune_distribution

    def update_architecture(self,
                            prunable_layers: list,
                            next_layers: list,
                            probability_lower_bound: float = settings.RL_PROBABILITY_LOWER_BOUND):
        prune_counter = torch.zeros(len(self.prune_distribution))
        noise = torch.randn(len(self.prune_distribution)) * self.noise_var * torch.rand(1).item()
        noised_distribution = self.prune_distribution + noise
        noised_distribution = torch.clamp(noised_distribution, min=probability_lower_bound)
        noised_distribution = noised_distribution / torch.sum(noised_distribution)
        prune_counter = torch.round(noised_distribution * self.modification_num)

        for target_layer_idx, count in enumerate(prune_counter):
            target_layer = prunable_layers[target_layer_idx]
            for _ in range(int(count.item())):
                if isinstance(target_layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                    self.prune_filter_conv(target_layer_idx, target_layer, next_layers)
                elif isinstance(target_layer, (nn.Linear)):
                    self.prune_filter_linear(target_layer_idx, target_layer, next_layers)
        
        return noised_distribution, prune_counter

    def prune_filter_conv(self,
                          target_layer_idx: int,
                          target_layer: nn.Module, 
                          next_layers: list):
        # prune kernel
        if target_layer.out_channels - 1 == 0:
            return
        weight_variances = torch.var(target_layer.weight.data, dim = [1, 2, 3])
        # weight_L2norm = torch.norm(target_layer.weight.data, p=2, dim= [1, 2, 3])
        target_kernel = torch.argmin(weight_variances).item()
        with torch.no_grad():
            target_layer.weight.data = torch.cat([target_layer.weight.data[:target_kernel], target_layer.weight.data[target_kernel+1:]], dim=0)
            if target_layer.bias is not None:
                target_layer.bias.data = torch.cat([target_layer.bias.data[:target_kernel], self.bias.data[target_kernel+1:]])
        target_layer.out_channels -= 1
        if target_layer.out_channels != target_layer.weight.shape[0]:
            raise ValueError(f'Conv2d layer out_channels {target_layer.out_channels} and weight dimension {target_layer.weight.shape[0]} mismath')

        # update following layers
        for next_layer_info in next_layers[target_layer_idx]:
            next_layer = next_layer_info[0]
            offset = next_layer_info[1]
            if isinstance(next_layer, nn.BatchNorm2d):
                # case 1: BatchNorm
                target_bn = next_layer
                with torch.no_grad():
                    kept_indices = [i for i in range(target_bn.num_features) if i != target_kernel + offset]
                    target_bn.weight.data = target_bn.weight.data[kept_indices]
                    target_bn.bias.data = target_bn.bias.data[kept_indices]
                    target_bn.running_mean = target_bn.running_mean[kept_indices]
                    target_bn.running_var = target_bn.running_var[kept_indices]
                target_bn.num_features -= 1
                if target_bn.num_features != target_bn.weight.shape[0]:
                    raise ValueError(f'BatchNorm layer number_features {target_bn.num_features} and weight dimension {target_bn.weight.shape[0]} mismath')
            elif isinstance(next_layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                # case 2: Conv
                with torch.no_grad():
                    kept_indices = [i for i in range(next_layer.in_channels) if i != target_kernel + offset]
                    next_layer.weight.data = next_layer.weight.data[:, kept_indices, :, :]
                next_layer.in_channels -= 1
                if next_layer.in_channels != next_layer.weight.shape[1]:
                    raise ValueError(f'Conv2d layer in_channels {next_layer.in_channels} and weight dimension {next_layer.weight.shape[1]} mismath')
            elif isinstance(next_layer, (nn.Linear)):
                # case 3: Linear
                output_area = 1 # default for most CNNs
                new_in_features = target_layer.out_channels * output_area ** 2
                start_index = (target_kernel + offset) * output_area ** 2
                end_index = start_index + output_area ** 2
                with torch.no_grad():
                    next_layer.weight.data = torch.cat([next_layer.weight.data[:, :start_index], next_layer.weight.data[:, end_index:]], dim=1)
                    if next_layer.bias is not None:
                        next_layer.bias.data = next_layer.bias.data
                next_layer.in_features = new_in_features
                if next_layer.in_features != next_layer.weight.shape[1]:
                    raise ValueError(f'Linear layer in_channels {next_layer.in_features} and weight dimension {next_layer.weight.shape[1]} mismath')

    def prune_filter_linear(self, 
                            target_layer_idx: int,
                            target_layer: nn.Linear,
                            next_layers: list):
        if target_layer.out_features - 1 == 0:
            return
        weight_variances = torch.var(target_layer.weight.data, dim = 1)
        # weight_L2norm = torch.norm(target_layer.weight.data, p=2, dim=1)
        target_neuron = torch.argmin(weight_variances).item()
        with torch.no_grad():
            target_layer.weight.data = torch.cat([target_layer.weight.data[:target_neuron], target_layer.weight.data[target_neuron+1:]], dim=0)
            if target_layer.bias is not None:
                target_layer.bias.data = torch.cat([target_layer.bias.data[:target_neuron], target_layer.bias.data[target_neuron+1:]])
        target_layer.out_features -= 1
        if target_layer.out_features != target_layer.weight.shape[0]:
            raise ValueError(f'Linear layer out_channels {target_layer.out_features} and weight dimension {target_layer.weight.shape[0]} mismath')
        
        for next_layer_info in next_layers[target_layer_idx]:
            next_layer = next_layer_info[0]
            offset = next_layer_info[1]
            if isinstance(next_layer, (nn.Linear)):
                # case 1: Linear
                new_in_features = target_layer.out_features
                start_index = target_neuron + offset
                end_index = start_index + 1
                with torch.no_grad():
                    next_layer.weight.data = torch.cat([next_layer.weight.data[:, :start_index], next_layer.weight.data[:, end_index:]], dim=1)
                    if next_layer.bias is not None:
                        next_layer.bias.data = next_layer.bias.data
                next_layer.in_features = new_in_features
                if next_layer.in_features != next_layer.weight.shape[1]:
                    raise ValueError(f'Linear layer in_channels {next_layer.in_features} and weight dimension {next_layer.weight.shape[1]} mismath')
