import torch
from torch import Tensor
from typing import List
import math
import torch.nn as nn

from conf import settings
from utils import adjust_prune_distribution_for_cluster, CONV_LAYERS


class Prune_agent():
    """ Agent used to prune architecture and maintain prune_distribution """
    def __init__(self,
                 prune_distribution: Tensor,
                 layer_cluster_mask: List,
                 ReplayBuffer: Tensor, 
                 filter_num: int,
                 cur_top1_acc: float,
                 prune_filter_max_ratio: float = settings.C_PRUNE_FILTER_MAX_RATIO,
                 prune_filter_min_ratio: float = settings.C_PRUNE_FILTER_MIN_RATIO,
                 noise_var: float = settings.RL_PRUNE_FILTER_NOISE_VAR):
        # Static Data: These attributes do not change after initialization
        self.modification_max_num = int(filter_num * prune_filter_max_ratio)
        self.modification_min_num = int(filter_num * prune_filter_min_ratio)
        self.layer_cluster_mask = layer_cluster_mask
        self.noise_var = noise_var
        self.lr_epoch = settings.RL_LR_EPOCH
        self.lr_tolerance_time = settings.RL_LR_TOLERANCE
        self.lr_change_threshold = settings.RL_REWARD_CHANGE_THRESHOLD
        self.T_max = settings.C_COS_PRUNE_EPOCH
        self.cur_single_step_acc_threshold = settings.C_ACC_CHANGE_THRESHOLD
        
        # Dynamic Data: These attributes may change during the object's lifetime
        self.modification_num = self.modification_max_num
        self.prune_distribution = prune_distribution
        self.ReplayBuffer = ReplayBuffer
        self.model_info_list = [None] * settings.RL_MAX_SAMPLE_NUM
        self.cur_Q_value_max = (cur_top1_acc * settings.RL_CUR_ACC_TO_CUR_Q_VALUE_COEFFICIENT + 
                                cur_top1_acc * settings.RL_CUR_ACC_TO_CUR_Q_VALUE_COEFFICIENT ** 2 * settings.RL_DISCOUNT_FACTOR)


    def step(self, optimal_model_index: int, epoch: int, cur_top1_acc: float) -> None:
        """ clear the ReplayBuffer and model_info_list if generated model is better, then decrease modification num """
        if optimal_model_index == 1:
            # means generated model is better, reset counter then clear the ReplayBuffer and model_info_list
            self.ReplayBuffer.zero_()
            self.model_info_list = [None] * settings.RL_MAX_SAMPLE_NUM
            self.cur_single_step_acc_threshold = settings.C_ACC_CHANGE_THRESHOLD_INCRE
            self.cur_Q_value_max = (cur_top1_acc * settings.RL_CUR_ACC_TO_CUR_Q_VALUE_COEFFICIENT + 
                                    cur_top1_acc * settings.RL_CUR_ACC_TO_CUR_Q_VALUE_COEFFICIENT ** 2 * settings.RL_DISCOUNT_FACTOR)
        else:
            self.cur_single_step_acc_threshold += settings.C_ACC_CHANGE_THRESHOLD_INCRE
        if epoch < self.T_max:
            self.modification_num = int(self.modification_min_num + 
                                        (self.modification_max_num - self.modification_min_num) * 0.5 * (math.cos(math.pi * epoch / self.T_max) + 1))


    def update_prune_distribution(self, 
                                  step_length: float,  
                                  ppo_clip: float, 
                                  ppo_enable: bool) -> Tensor:
        """ Update prune distribution and return its change """
        P_lower_bound = settings.RL_PROBABILITY_LOWER_BOUND
        original_PD = self.prune_distribution
        _, optimal_idx = torch.max(self.ReplayBuffer[:, 0], dim=0)
        optimal_PD = self.ReplayBuffer[optimal_idx, 1:]
        
        updated_PD = original_PD + step_length * (optimal_PD - original_PD) 
        updated_PD = torch.clamp(updated_PD, min=P_lower_bound)
        updated_PD /= torch.sum(updated_PD)
        updated_PD = adjust_prune_distribution_for_cluster(updated_PD, 
                                                           self.layer_cluster_mask)
        
        if ppo_enable == True:
            # apply PPO to make PD changes stably
            ratio = updated_PD / original_PD
            updated_PD = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * original_PD
            updated_PD = torch.clamp(updated_PD, min=P_lower_bound)
            updated_PD /= torch.sum(updated_PD)
            updated_PD = adjust_prune_distribution_for_cluster(updated_PD, 
                                                               self.layer_cluster_mask)
        self.prune_distribution = updated_PD
        return updated_PD - original_PD


    def prune_architecture(self,
                            prunable_layers: List,
                            next_layers: List) -> Tensor:
        """ Generate new noised PD and prune architecture based on noised PD """
        P_lower_bound = settings.RL_PROBABILITY_LOWER_BOUND
        prune_counter = torch.zeros(len(self.prune_distribution))
        noise = torch.randn(len(self.prune_distribution)) * self.noise_var * torch.rand(1).item()
        noised_PD = self.prune_distribution + noise
        noised_PD = torch.clamp(noised_PD, min=P_lower_bound)
        noised_PD = noised_PD / torch.sum(noised_PD)
        noised_PD = adjust_prune_distribution_for_cluster(noised_PD, self.layer_cluster_mask)
        prune_counter = torch.round(noised_PD * self.modification_num).int().tolist()
        decred_layer = {}

        for target_layer_idx, count in enumerate(prune_counter):
            target_layer = prunable_layers[target_layer_idx]
            for _ in range(count):
                if isinstance(target_layer, CONV_LAYERS):
                    prune_conv_filter(target_layer_idx, target_layer, next_layers, decred_layer)
                elif isinstance(target_layer, (nn.Linear)):
                    prune_linear_filter(target_layer_idx, target_layer, next_layers, decred_layer)
        
        return noised_PD


def prune_conv_filter(target_layer_idx: int,
                        target_layer: nn.Module, 
                        next_layers: List,
                        decred_layer: dict) -> None:
    """ Prune one conv filter and decrease next layers' input dim """
    if target_layer.out_channels - 1 == 0:
        return
    weight_variances = torch.var(target_layer.weight.data, dim = [1, 2, 3])
    target_kernel = torch.argmin(weight_variances).item()
    with torch.no_grad():
        target_layer.weight.data = torch.cat([target_layer.weight.data[:target_kernel], 
                                              target_layer.weight.data[target_kernel+1:]], 
                                              dim=0)
        if target_layer.bias is not None:
            target_layer.bias.data = torch.cat([target_layer.bias.data[:target_kernel], 
                                                target_layer.bias.data[target_kernel+1:]])
    target_layer.out_channels -= 1
    if target_layer.out_channels != target_layer.weight.shape[0]:
        raise ValueError(f'Conv2d layer out_channels {target_layer.out_channels} and '
                         f'weight dimension {target_layer.weight.shape[0]} mismatch')

    for next_layer_info in next_layers[target_layer_idx]:
        next_layer = next_layer_info[0]
        if next_layer_info[1] == -1:
            if next_layer not in decred_layer:
                # means this layer involved in residual connection 
                # and can only be decreased input once
                decred_layer[next_layer] = target_layer
            elif decred_layer[next_layer] != target_layer:
                continue
        offset = max(next_layer_info[1], 0)
        
        if isinstance(next_layer, nn.BatchNorm2d):
            # case 1: BatchNorm
            target_bn = next_layer
            with torch.no_grad():
                kept_indices = [i for i in range(target_bn.num_features) 
                                if i != target_kernel + offset]
                target_bn.weight.data = target_bn.weight.data[kept_indices]
                target_bn.bias.data = target_bn.bias.data[kept_indices]
                target_bn.running_mean = target_bn.running_mean[kept_indices]
                target_bn.running_var = target_bn.running_var[kept_indices]
            target_bn.num_features -= 1
            decrease_offset(next_layers, next_layer, offset, 1)
            if target_bn.num_features != target_bn.weight.shape[0]:
                raise ValueError(f'BatchNorm layer number_features {target_bn.num_features} and '
                                 f'weight dimension {target_bn.weight.shape[0]} mismatch')
        
        elif isinstance(next_layer, CONV_LAYERS):
            # case 2: Conv
            with torch.no_grad():
                kept_indices = [i for i in range(next_layer.in_channels) 
                                if i != target_kernel + offset]
                next_layer.weight.data = next_layer.weight.data[:, kept_indices, :, :]
            next_layer.in_channels -= 1
            decrease_offset(next_layers, next_layer, offset, 1)
            if next_layer.in_channels != next_layer.weight.shape[1]:
                raise ValueError(f'Conv2d layer in_channels {next_layer.in_channels} and '
                                 f'weight dimension {next_layer.weight.shape[1]} mismatch')
        
        elif isinstance(next_layer, (nn.Linear)):
            # case 3: Linear
            output_area = 1 # default for most CNNs
            start_index = (target_kernel + offset) * output_area ** 2
            end_index = start_index + output_area ** 2
            with torch.no_grad():
                next_layer.weight.data = torch.cat([next_layer.weight.data[:, :start_index], 
                                                    next_layer.weight.data[:, end_index:]], 
                                                    dim=1)
                if next_layer.bias is not None:
                    next_layer.bias.data = next_layer.bias.data
            next_layer.in_features -= output_area ** 2
            decrease_offset(next_layers, next_layer, offset, output_area ** 2)
            if next_layer.in_features != next_layer.weight.shape[1]:
                raise ValueError(f'Linear layer in_channels {next_layer.in_features} and '
                                 f'weight dimension {next_layer.weight.shape[1]} mismatch')


def prune_linear_filter(target_layer_idx: int,
                        target_layer: nn.Linear,
                        next_layers: List, 
                        decred_layer: dict) -> None:
    """ Prune one linear filter and decrease next layers' input dim """
    if target_layer.out_features - 1 == 0:
        return
    weight_variances = torch.var(target_layer.weight.data, dim = 1)
    target_neuron = torch.argmin(weight_variances).item()
    with torch.no_grad():
        target_layer.weight.data = torch.cat([target_layer.weight.data[:target_neuron], 
                                              target_layer.weight.data[target_neuron+1:]], 
                                              dim=0)
        if target_layer.bias is not None:
            target_layer.bias.data = torch.cat([target_layer.bias.data[:target_neuron], 
                                                target_layer.bias.data[target_neuron+1:]])
    target_layer.out_features -= 1
    if target_layer.out_features != target_layer.weight.shape[0]:
        raise ValueError(f'Linear layer out_channels {target_layer.out_features} and '
                         f'weight dimension {target_layer.weight.shape[0]} mismatch')
    
    # update following layers
    for next_layer_info in next_layers[target_layer_idx]:
        next_layer = next_layer_info[0]
        if next_layer_info[1] == -1:
            if next_layer not in decred_layer:
                # means this layer involved in residual connection 
                # and can only be decreased input once
                decred_layer[next_layer] = target_layer
            elif decred_layer[next_layer] != target_layer:
                continue
        offset = max(next_layer_info[1], 0)
        if isinstance(next_layer, (nn.Linear)):
            # case 1: Linear
            start_index = target_neuron + offset
            end_index = start_index + 1
            with torch.no_grad():
                next_layer.weight.data = torch.cat([next_layer.weight.data[:, :start_index], 
                                                    next_layer.weight.data[:, end_index:]], 
                                                    dim=1)
                if next_layer.bias is not None:
                    next_layer.bias.data = next_layer.bias.data
            next_layer.in_features -= 1
            decrease_offset(next_layers, next_layer, offset, 1)
            if next_layer.in_features != next_layer.weight.shape[1]:
                raise ValueError(f'Linear layer in_channels {next_layer.in_features} and '
                                 f'weight dimension {next_layer.weight.shape[1]} mismatch')


def decrease_offset(next_layers: List, 
                    target_layer: nn.Module, 
                    target_offset: int, 
                    decrement: int) -> None:
    """ decrease offset for each next layer after we decre next layer input dim """
    for ith_next_layers in next_layers:
        for next_layer_info in ith_next_layers:
            next_layer = next_layer_info[0]
            offset = next_layer_info[1]
            if id(target_layer) == id(next_layer) and offset > target_offset:
                next_layer_info[1] -= decrement
