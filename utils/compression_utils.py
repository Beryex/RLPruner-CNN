import torch
from torch import Tensor

from conf import settings

class PR_scheduler():
    def __init__(self, modification_num: int):
        self.modification_num = modification_num
        self.tolerance_ct = settings.TOLERANCE_CT
    
    def reset(self, modification_num: int):
        self.modification_num = modification_num
        self.tolerance_ct = settings.TOLERANCE_CT

    def step(self, model_index: int):
        if model_index == 0:
            # means original net is better
            self.tolerance_ct -= 1
        else:
            # means generated net is better, reset counter
            self.tolerance_ct = settings.TOLERANCE_CT
        if self.tolerance_ct <= 0:
            self.modification_num //= 2
            self.tolerance_ct = settings.TOLERANCE_CT
        if self.modification_num < settings.MODIFICATION_MIN_NUM:
            return None
        else:
            return self.modification_num

def update_prune_distribution(ReplayBuffer: Tensor, prune_distribution: Tensor, step_length: float, probability_lower_bound: float, ppo_clip: float):
    weighted_distribution = ReplayBuffer[torch.argmax(ReplayBuffer[:, 0], dim=0).item(), 1:]
    updated_prune_distribution = prune_distribution + step_length * (weighted_distribution - prune_distribution)
    ratio = updated_prune_distribution / prune_distribution
    clipped_ratio = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip)
    updated_prune_distribution = prune_distribution * clipped_ratio
    updated_prune_distribution = torch.clamp(updated_prune_distribution, min=probability_lower_bound)
    updated_prune_distribution /= torch.sum(updated_prune_distribution)
    return updated_prune_distribution
