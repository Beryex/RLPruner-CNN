import torch
from torch import Tensor
import math

from conf import settings

class Prune_agent():
    def __init__(self,
                 prune_distribution: Tensor,
                 ReplayBuffer: Tensor, 
                 filter_num: int):
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
        self.prev_sampled_Q_value_mean = -1

    def step(self, optimal_net_index: int, epoch: int):
        if optimal_net_index == 1:
            # means generated net is better, reset counter then clear the ReplayBuffer, Reward_cache and net_list
            self.ReplayBuffer.zero_()
            self.Reward_cache = {}
            self.net_list = [None] * settings.RL_MAX_GENERATE_NUM
            self.cur_single_step_acc_threshold = settings.C_SINGLE_STEP_ACCURACY_CHANGE_THRESHOLD
            self.prev_sampled_Q_value_mean = -1
        else:
            self.cur_single_step_acc_threshold += 0.001
        # update modification_num using method similiar to CosineAnnealingLR
        if epoch < self.T_max:
            self.modification_num = int(self.modification_min_num + (self.modification_max_num - self.modification_min_num) * 0.5 * (math.cos(torch.pi * epoch / self.T_max) + 1))

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
