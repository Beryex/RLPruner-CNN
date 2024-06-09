import torch
from torch import Tensor
import math

from conf import settings

class Prune_agent():
    def __init__(self, strategy: str,
                 prune_distribution: Tensor,
                 ReplayBuffer: Tensor, 
                 filter_num: int,
                 prune_choices_num: int,
                 T_max: int, 
                 single_step_acc_threshold: float):
        if strategy == "prune_filter":
            self.strategy = "prune_filter"
            self.modification_max_num = filter_num * settings.C_PRUNE_FILTER_MAX_RATIO
            self.modification_min_num = filter_num * settings.C_PRUNE_FILTER_MIN_RATIO
            self.prune_distribution = prune_distribution
            self.noise_var = settings.RL_PRUNE_FILTER_NOISE_VAR
        elif strategy == "weight_sharing":
            self.strategy = "weight_sharing"
            self.modification_max_num = prune_choices_num * settings.C_WEIGHT_SHARING_MAX_RATIO
            self.modification_min_num = prune_choices_num * settings.C_WEIGHT_SHARING_MIN_RATIO
            self.prune_distribution = torch.ones_like(prune_distribution) / prune_distribution.numel()
            self.noise_var = settings.RL_WEIGHT_SHARING_NOISE_VAR
        elif strategy == "finished":
            self.strategy = "finished"
            self.modification_max_num = -1
            self.modification_min_num = 0
            self.prune_distribution = None
            self.noise_var = 0
        else:
            raise TypeError(f"Invalid strategy input {strategy}")
        self.modification_num = self.modification_max_num
        self.T_max = T_max
        self.ReplayBuffer = ReplayBuffer
        self.Reward_cache = {}
        self.original_single_step_acc_threshold = single_step_acc_threshold
        self.cur_single_step_acc_threshold = single_step_acc_threshold

    def change_strategy(self, 
                     strategy: str, 
                     prune_choices_num: int):
        if self.strategy == "prune_filter" and strategy == "weight_sharing":
            self.strategy = "weight_sharing"
            self.modification_max_num = prune_choices_num * settings.C_WEIGHT_SHARING_MAX_RATIO
            self.modification_min_num = prune_choices_num * settings.C_WEIGHT_SHARING_MIN_RATIO
            self.prune_distribution = torch.ones_like(self.prune_distribution) / self.prune_distribution.numel()
            self.noise_var = settings.RL_WEIGHT_SHARING_NOISE_VAR
        elif self.strategy == "weight_sharing" and strategy == "finished":
            self.strategy = "finished"
            self.modification_max_num = -1
            self.modification_min_num = 0
            self.prune_distribution = None
            self.noise_var = 0
        else:
            raise TypeError(f"Unsupport strategy change from {self.strategy} to {strategy}")
        self.modification_num = self.modification_max_num


    def step(self, optimal_net_index: int, epoch: int):
        if optimal_net_index == 1:
            # means generated net is better, reset counter then clear the ReplayBuffer and Reward_cache
            self.ReplayBuffer.zero_()
            self.Reward_cache = {}
            self.cur_single_step_acc_threshold = self.original_single_step_acc_threshold
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
