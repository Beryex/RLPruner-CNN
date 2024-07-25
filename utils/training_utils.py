import torch
import os
import random
import numpy as np
import wandb
import logging
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler

from conf import settings


def torch_set_random_seed(seed: int) -> None:
    """ Set random seed for reproducible usage """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def torch_resume_random_seed(prev_checkpoint: dict) -> None:
    """ Resume random seed for reproducible usage """
    os.environ['PYTHONHASHSEED'] = prev_checkpoint['python_hash_seed']
    random.setstate(prev_checkpoint['random_state'])
    np.random.set_state(prev_checkpoint['np_random_state'])
    torch.set_rng_state(prev_checkpoint['torch_random_state'])
    torch.cuda.set_rng_state_all(prev_checkpoint['cuda_random_state'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logging(log_dir: str,
                  experiment_id: int, 
                  model_name: str, 
                  dataset_name: str,
                  action: str,
                  use_wandb: bool = False) -> None:
    """ Set up wandb, logging """    
    hyperparams_config = {
        "model": model_name,
        "dataset": dataset_name,
        "action": action, 
        "random_seed": experiment_id
    }
    hyperparams_config.update(settings.__dict__)
    wandb.init(
        project="RLPruner",
        name=f"{action}_{model_name}_on_{dataset_name}_{experiment_id}",
        id=str(experiment_id),
        config=hyperparams_config,
        resume=True,
        mode='online' if use_wandb else 'disabled'
    )

    
    log_filename = f"{log_dir}/log_{action}_{model_name}_{dataset_name}_{experiment_id}.txt"
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_filename)])


class WarmUpLR(_LRScheduler):
    # The fuction is adapted from https://github.com/weiaicunzai/pytorch-cifar100/blob/master/utils.py
    # author: baiyu
    """ Warmup_training learning rate scheduler """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """ Fpr first m batches, and set the learning rate to base_lr * m / total_iters """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def dice_coeff(input: Tensor, 
               target: Tensor, 
               reduce_batch_first: bool = False, 
               epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def multiclass_dice_coeff(input: Tensor, 
                          target: Tensor, 
                          reduce_batch_first: bool = False, 
                          epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input: Tensor, 
              target: Tensor, 
              multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
