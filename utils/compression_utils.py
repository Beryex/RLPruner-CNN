import torch
from torch import Tensor

from conf import settings

class PR_scheduler():
    def __init__(self, strategy: str,
                 prune_distribution: Tensor,
                 ReplayBuffer: Tensor):
        self.prune_distribution = prune_distribution
        self.ReplayBuffer = ReplayBuffer
        self.tolerance_ct = settings.TOLERANCE_CT
        if strategy == "prune_filter":
            self.strategy = "prune_filter"
            self.modification_num = settings.PRUNE_FILTER_MAX_NUM
            self.modification_min_num = settings.PRUNE_FILTER_MIN_NUM
        elif strategy == "weight_sharing":
            self.strategy = "weight_sharing"
            self.modification_num = settings.WEIGHT_SHARING_MAX_NUM
            self.modification_min_num = settings.WEIGHT_SHARING_MIN_NUM
        elif strategy == "finished":
            self.strategy = "finished"
            self.modification_num = -1
            self.modification_min_num = 0
        else:
            self.strategy = ""
            self.modification_num = -1
            self.modification_min_num = 0
            print("ERROR: Invalid strategy input!")

    def step(self, model_index: int):
        if model_index == 0:
            # means original net is better
            self.tolerance_ct -= 1
        else:
            # means generated net is better, reset counter and clear the ReplayBuffer
            self.tolerance_ct = settings.TOLERANCE_CT
            self.ReplayBuffer.zero_()
        if self.tolerance_ct <= 0:
            self.modification_num //= 2
            self.tolerance_ct = settings.TOLERANCE_CT
        if self.modification_num < self.modification_min_num:
            return None
        else:
            return self.modification_num

    def update_prune_distribution(self, step_length: float, probability_lower_bound: float, ppo_clip: float):
        ReplayBuffer = self.ReplayBuffer
        prune_distribution = self.prune_distribution
        top1_accuracy_cache = ReplayBuffer[:, 0]
        _, indices = torch.sort(top1_accuracy_cache, dim=0, descending=False)
        mid_idx = top1_accuracy_cache.shape[0] // 2
        negative_samples = ReplayBuffer[indices[:mid_idx]]
        positive_samples = ReplayBuffer[indices[mid_idx:]]
        
        positive_weight = positive_samples[:, 0] / torch.sum(positive_samples[:, 0], dim=0, keepdim=True)
        positive_distribution = torch.sum(positive_samples[:, 1:] * positive_weight.unsqueeze(1), dim=0)
        negative_weight = (negative_samples[:, 0] - 1) / torch.sum(negative_samples[:, 0] - 1, dim=0, keepdim=True)
        negative_distribution = torch.sum(negative_samples[:, 1:] * negative_weight.unsqueeze(1), dim=0)
        updated_prune_distribution = prune_distribution + (step_length * (positive_distribution - prune_distribution) 
                                                        - step_length * (negative_distribution - prune_distribution))
        
        ratio = updated_prune_distribution / prune_distribution
        clipped_ratio = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip)
        updated_prune_distribution = prune_distribution * clipped_ratio
        updated_prune_distribution = torch.clamp(updated_prune_distribution, min=probability_lower_bound)
        updated_prune_distribution /= torch.sum(updated_prune_distribution)
        self.prune_distribution = updated_prune_distribution
