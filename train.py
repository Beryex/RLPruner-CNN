import time
import argparse
import torch
import os
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import logging
from typing import Tuple

from conf import settings
from utils import (get_model, get_dataloader, setup_logging, torch_set_random_seed, 
                   WarmUpLR, MODELS, DATASETS)


def main():
    """ Train model and save model using early stop on test dataset """
    global args
    global epoch
    global warmup_epoch
    args = get_args()
    model_name = args.model
    dataset_name = args.dataset

    random_seed = args.random_seed
    experiment_id = int(time.time())
    
    device = args.device
    setup_logging(log_dir=args.log_dir,
                  experiment_id=experiment_id, 
                  random_seed=random_seed,
                  args=args,
                  model_name=args.model, 
                  dataset_name=args.dataset, 
                  action='train',
                  use_wandb=args.use_wandb)

    torch_set_random_seed(random_seed)
    logging.info(f'Start with random seed: {random_seed}')
    print(f"Start with random seed: {random_seed}")
    
    train_loader, _, test_loader, num_classes = get_dataloader(args.dataset, 
                                                             batch_size=args.batch_size, 
                                                             num_workers=args.num_worker)
    model = get_model(args.model, num_classes).to(device)

    loss_function = nn.CrossEntropyLoss()
    warmup_epoch = int(args.epoch * args.warmup_ratio)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                        T_max=args.epoch - warmup_epoch, 
                                                        eta_min= args.min_lr,
                                                        last_epoch=-1)
    lr_scheduler_warmup = WarmUpLR(optimizer, warmup_epoch * len(train_loader))
    
    best_acc = 0.0
    with tqdm(total=args.epoch, desc=f'Training', unit='epoch') as pbar:
        for epoch in range(1, args.epoch + 1):
            train_loss = train(model, 
                               train_loader, 
                               loss_function, 
                               optimizer,
                               lr_scheduler_warmup,
                               device)
            top1_acc, top5_acc, _ = evaluate(model, 
                                             test_loader, 
                                             loss_function, 
                                             device)
            
            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            if epoch > warmup_epoch:
                lr_scheduler.step()

            if best_acc < top1_acc:
                best_acc = top1_acc
                best_model = copy.deepcopy(model)
            
            logging.info(f'Epoch: {epoch}, Train Loss: {train_loss}, "lr": {lr}, '
                         f'Top1 Accuracy: {top1_acc}, Top5 Accuracy: {top5_acc}, '
                         f'Best top1 acc: {best_acc}')
            wandb.log({"epoch": epoch, "train_loss": train_loss, "lr": lr,
                       "top1_acc": top1_acc, "top5_acc": top5_acc,
                       "best top1 acc": best_acc})
            
            pbar.set_postfix({'Train loss': train_loss, 'Best top1 acc': best_acc, 'Top1 acc': top1_acc})
            pbar.update(1)
    
    os.makedirs(f"{args.output_dir}", exist_ok=True)
    output_pth = f"{args.output_dir}/{model_name}_{dataset_name}_pretrained.pth"
    torch.save(best_model, output_pth)
    logging.info(f"Pretrained model saved at {output_pth}")
    print(f"Pretrained model saved at {output_pth}")
    wandb.finish()


def train(model: nn.Module, 
          train_loader: DataLoader, 
          loss_function: nn.Module,
          optimizer: optim.Optimizer,
          lr_scheduler_warmup: WarmUpLR,
          device: str) -> float:
    """ Train model and save using early stop on test dataset """
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if epoch <= warmup_epoch:
            lr_scheduler_warmup.step()

    train_loss /= len(train_loader)
    return train_loss


@torch.no_grad()
def evaluate(model: nn.Module,
             eval_loader: DataLoader, 
             loss_function: nn.Module, 
             device: str) -> Tuple[float, float, float]:
    """ Evaluate model """
    model.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    eval_loss = 0.0
    for images, labels in eval_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        eval_loss += loss_function(outputs, labels).item()
        _, preds = outputs.topk(5, 1, largest=True, sorted=True)
        correct_1 += (preds[:, :1] == labels.unsqueeze(1)).sum().item()
        top5_correct = labels.view(-1, 1).expand_as(preds) == preds
        correct_5 += top5_correct.any(dim=1).sum().item()
    
    top1_acc = correct_1 / len(eval_loader.dataset)
    top5_acc = correct_5 / len(eval_loader.dataset)
    eval_loss /= len(eval_loader)
    return top1_acc, top5_acc, eval_loss


def get_args():
    parser = argparse.ArgumentParser(description='train given model under given dataset')
    parser.add_argument('--model', '-m', type=str, default=None, 
                        help='the type of model to train')
    parser.add_argument('--dataset', '-ds', type=str, default=None, 
                        help='the dataset to train on')
    parser.add_argument('--lr', type=float, default=settings.T_LR_SCHEDULER_INITIAL_LR,
                        help='initial learning rate')
    parser.add_argument('--min_lr', type=float, default=settings.T_LR_SCHEDULER_MIN_LR,
                        help='minimal learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=settings.T_BATCH_SIZE, 
                        help='batch size for dataloader')
    parser.add_argument('--num_worker', '-n', type=int, default=settings.T_NUM_WORKER, 
                        help='number of workers for dataloader')
    parser.add_argument('--epoch', '-e', type=int, default=settings.T_EPOCH, 
                        help='total epoch to train')
    parser.add_argument('--warmup_ratio', '-wr', type=int, default=settings.T_WARMUP_RATIO, 
                        help='the ratio of warmup epoch number over total epoch number for lr scheduler')
    parser.add_argument('--device', '-dev', type=str, default='cpu', 
                        help='device to use')
    parser.add_argument('--random_seed', '-rs', type=int, default=1, 
                        help='the random seed for the training')
    parser.add_argument('--use_wandb', action='store_true', default=False, 
                        help='use wandb to track the experiment')
    
    parser.add_argument('--log_dir', '-log', type=str, default='log', 
                        help='the directory containing logging text')
    parser.add_argument('--output_dir', '-opth', type=str, default='pretrained_model', 
                        help='the directory to store output model')

    args = parser.parse_args()
    check_args(args)

    return args


def check_args(args: argparse.Namespace):
    if args.model is None:
        raise TypeError(f"the specific type of model should be provided, "
                        f"please select one of {MODELS}")
    elif args.model not in MODELS:
        raise TypeError(f"the specific model {args.model} is not supported, "
                        f"please select one of {MODELS}")
    if args.dataset is None:
        raise ValueError(f"the specific type of dataset to train on should be provided, "
                            f"please select one of {DATASETS}")
    elif args.dataset not in DATASETS:
        raise ValueError(f"the provided dataset {args.dataset} is not supported, "
                            f"please select one of {DATASETS}")


if __name__ == '__main__':
    main()
