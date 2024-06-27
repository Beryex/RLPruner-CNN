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
                 cur_top1_acc: float):
        self.modification_max_num = int(filter_num * settings.C_PRUNE_FILTER_MAX_RATIO)
        self.modification_min_num = int(filter_num * settings.C_PRUNE_FILTER_MIN_RATIO)
        self.prune_distribution = prune_distribution
        self.noise_var = settings.RL_PRUNE_FILTER_NOISE_VAR
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
             cur_top1_acc: float,
             target_net: nn.Module):
        if optimal_net_index == 1:
            # means generated net is better, reset counter then clear the ReplayBuffer, Reward_cache and net_list
            self.ReplayBuffer.zero_()
            self.Reward_cache = {}
            self.net_list = [None] * settings.RL_MAX_GENERATE_NUM
            self.cur_single_step_acc_threshold = settings.C_SINGLE_STEP_ACCURACY_CHANGE_THRESHOLD
            self.cur_Q_value_max = (cur_top1_acc * settings.RL_CUR_ACC_TO_CUR_Q_VALUE_COEFFICIENT + 
                                    cur_top1_acc * settings.RL_CUR_ACC_TO_CUR_Q_VALUE_COEFFICIENT ** 2 * settings.RL_DISCOUNT_FACTOR)
            # reinitialize prune distribution
            for idx, layer_idx in enumerate(target_net.prune_choices):
                if idx <= target_net.last_conv_layer_idx:
                    layer = target_net.conv_layers[layer_idx]
                    self.prune_distribution[idx] = layer.out_channels
                else:
                    layer = target_net.linear_layers[layer_idx]
                    self.prune_distribution[idx] = layer.out_features
            filter_num = torch.sum(self.prune_distribution)
            self.prune_distribution /= filter_num
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
