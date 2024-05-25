import torch
from conf import settings
import os
import random
import numpy as np
import logging
from torch import Tensor
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


def torch_set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logging(experiment_id: int, 
                  net: str, 
                  dataset: str, 
                  action: str):
    log_directory = "experiment_log"
    if not os.path.isdir(log_directory):
        os.mkdir(log_directory)
    
    if action == 'train':
        log_filename = f"{log_directory}/experiment_log_train_{net}_on_{dataset}_random_seed_{experiment_id}.txt"
    elif action == 'compress':
        log_filename = f"{log_directory}/experiment_log_compress_{net}_on_{dataset}_random_seed_{experiment_id}.txt"
    else:
        log_filename = f"{log_directory}/experiment_log_test_{net}_on_{dataset}_random_seed_{experiment_id}.txt"
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_filename),
                            logging.StreamHandler()
                        ])

    logging.info(f'Logging setup complete for experiment number: {experiment_id}')
    hyperparams_info = "\n".join(f"{key}={value}" for key, value in settings.__dict__.items())
    logging.info(f"Experiment hyperparameters:\n{hyperparams_info}")

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

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, 
                 optimizer: optim.Optimizer, 
                 total_iters: int, 
                 last_epoch:int = -1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
