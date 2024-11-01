import torch
import os
import random
import numpy as np
import wandb
import logging
import argparse
from torch.optim.lr_scheduler import _LRScheduler


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
                  random_seed: int,
                  args: argparse.Namespace,
                  model_name: str, 
                  dataset_name: str,
                  action: str,
                  project_name: str = "RLPruner-CNN",
                  use_wandb: bool = False) -> None:
    """ Set up wandb, logging """    
    hyperparams_config = {
        "model": model_name,
        "dataset": dataset_name,
        "action": action, 
        "random_seed": random_seed
    }
    hyperparams_config.update(vars(args))
    if action == 'compress':
        wandb_run_name = f"{action}_{model_name}_on_{dataset_name}_{args.Q_FLOP_coef}_{args.Q_Para_coef}"
        logging_file_name = f"{log_dir}/log_{action}_{model_name}_{dataset_name}_{args.Q_FLOP_coef}_{args.Q_Para_coef}.txt"
    else:
        wandb_run_name = f"{action}_{model_name}_on_{dataset_name}"
        logging_file_name = f"{log_dir}/log_{action}_{model_name}_{dataset_name}.txt"
    wandb.init(
        project=project_name,
        name=wandb_run_name,
        id=str(experiment_id),
        config=hyperparams_config,
        resume=True,
        mode='online' if use_wandb else 'disabled'
    )

    os.makedirs(f"{log_dir}", exist_ok=True)
    log_filename = logging_file_name
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
