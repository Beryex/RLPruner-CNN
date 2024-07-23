import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import logging
from typing import Tuple

from conf import settings
from utils import get_model, get_dataloader, setup_logging, torch_set_random_seed, WarmUpLR


def main():
    """ Train model and save model using early stop on test dataset """
    global args
    global epoch
    args = get_args()
    if args.random_seed is not None:
        random_seed = args.random_seed
        experiment_id = int(time.time())
    else:
        random_seed = int(time.time())
        experiment_id = random_seed
    device = args.device
    setup_logging(experiment_id=experiment_id, 
                  model_name=args.model, 
                  dataset_name=args.dataset, 
                  action='train')

    torch_set_random_seed(random_seed)
    logging.info(f'Start with random seed: {random_seed}')
    
    train_loader, _, test_loader, in_channels, num_class = get_dataloader(args.dataset, 
                                                                          batch_size=args.batch_size, 
                                                                          num_workers=args.num_worker)
    model = get_model(args.model, in_channels, num_class).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                        T_max=args.epoch - args.warmup_epoch - 10, 
                                                        eta_min= args.min_lr,
                                                        last_epoch=-1)
    iter_per_epoch = len(train_loader)
    lr_scheduler_warmup = WarmUpLR(optimizer, iter_per_epoch * args.warmup_epoch)
    
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
            logging.info(f'Epoch: {epoch}, Train Loss: {train_loss}, '
                        f'Top1 Accuracy: {top1_acc}, Top5 Accuracy: {top5_acc}')
            wandb.log({"epoch": epoch, "train_loss": train_loss, 
                    "top1_acc": top1_acc, "top5_acc": top5_acc})

            if epoch > args.warmup_epoch:
                lr_scheduler.step()
            
            for param_group in optimizer.param_groups:
                print('Current learning rate:', param_group['lr'])

            if best_acc < top1_acc:
                best_acc = top1_acc
                torch.save(model, f'models/{args.model}_{args.dataset}_{experiment_id}_original.pth')
            
            pbar.set_postfix({'Train loss': train_loss, 'Top1 acc': top1_acc})
            pbar.update(1)
    
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
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if epoch <= args.warmup_epoch:
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
    parser.add_argument('--lr', type=float, default=settings.T_LR_SCHEDULAR_INITIAL_LR,
                        help='initial learning rate')
    parser.add_argument('--min-lr', type=float, default=settings.T_LR_SCHEDULAR_MIN_LR,
                        help='minimal learning rate')
    parser.add_argument('--warmup-epoch', '-we', type=int, default=settings.T_WARMUP_EPOCH, 
                        help='warmup epoch number for lr schedular')
    parser.add_argument('--batch-size', '-b', type=int, default=settings.T_BATCH_SIZE, 
                        help='batch size for dataloader')
    parser.add_argument('--num-worker', '-n', type=int, default=settings.T_NUM_WORKER, 
                        help='num_workers for dataloader')
    parser.add_argument('--epoch', '-e', type=int, default=settings.T_EPOCH, 
                        help='total epoch to train')
    parser.add_argument('--device', '-dev', type=str, default='cpu', 
                        help='device to use')
    parser.add_argument('--random-seed', '-rs', type=int, default=None, 
                        help='the random seed for the current new compression')

    args = parser.parse_args()
    check_args(args)

    return args


def check_args(args: argparse.Namespace):
    if args.model is None:
        raise TypeError(f"the specific type of model should be provided, "
                        f"please select one of 'lenet5', 'vgg16', 'googlenet', 'resnet50', 'unet'")
    elif args.model not in ['lenet5', 'vgg16', 'googlenet', 'resnet50', 'unet']:
        raise TypeError(f"the specific model {args.net} is not supported, "
                        f"please select one of 'lenet5', 'vgg16', 'googlenet', 'resnet50', 'unet'")
    if args.dataset is None:
        raise TypeError(f"the specific type of dataset to train on should be provided, "
                        f"please select one of 'mnist', 'cifar10', 'cifar100'")
    elif args.dataset not in ['mnist', 'cifar10', 'cifar100']:
        raise TypeError(f"the specific dataset {args.dataset} is not supported, "
                        f"please select one of 'mnist', 'cifar10', 'cifar100'")


if __name__ == '__main__':
    main()
