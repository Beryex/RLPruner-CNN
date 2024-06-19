import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False
import logging

from conf import settings
from utils import get_network, get_dataloader, setup_logging, WarmUpLR, torch_set_random_seed

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.serialization")


def train_network(epoch: int):
    net.train()
    train_loss = 0.0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{settings.T_ORIGINAL_EPOCH}', unit='batch') as pbar:
        for _, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if epoch <= warm:
                warmup_scheduler.step()

            pbar.update(1)
            pbar.set_postfix(**{'loss (batch)': loss.item()})
        train_loss /= len(train_loader)
        pbar.set_postfix(**{'loss (batch)': train_loss})
    return train_loss

@torch.no_grad()
def eval_network(target_eval_loader: torch.utils.data.DataLoader):
    net.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    eval_loss = 0.0
    for (images, labels) in tqdm(target_eval_loader, total=len(target_eval_loader), desc='Testing round', unit='batch', leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = net(images)
        eval_loss += loss_function(outputs, labels).item()
        _, preds = outputs.topk(5, 1, largest=True, sorted=True)
        correct_1 += (preds[:, :1] == labels.unsqueeze(1)).sum().item()
        top5_correct = labels.view(-1, 1).expand_as(preds) == preds
        correct_5 += top5_correct.any(dim=1).sum().item()
    
    top1_acc = correct_1 / len(target_eval_loader.dataset)
    top5_acc = correct_5 / len(target_eval_loader.dataset)
    eval_loss /= len(target_eval_loader)
    return top1_acc, top5_acc, eval_loss


def get_args():
    parser = argparse.ArgumentParser(description='train given model under given dataset')
    parser.add_argument('-net', type=str, default=None, help='the type of model to train')
    parser.add_argument('-dataset', type=str, default=None, help='the dataset to train on')
    parser.add_argument('-lr', type=float, default=settings.T_LR_SCHEDULAR_INITIAL_LR, help='initial learning rate')
    parser.add_argument('-b', type=int, default=settings.T_BATCH_SIZE, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=settings.T_WARM, help='warm up training phase')
    parser.add_argument('-n', type=int, default=settings.T_NUM_WORKERS, help='num_workers for dataloader')
    parser.add_argument('-random_seed', type=int, default=None, help='the random seed for the current new compression')

    args = parser.parse_args()
    check_args(args)

    return args

def check_args(args: argparse.Namespace):
    if args.net is None:
        raise TypeError(f"the specific type of model should be provided, please select one of 'lenet5', 'vgg16', 'googlenet', 'resnet50', 'unet'")
    elif args.net not in ['lenet5', 'vgg16', 'googlenet', 'resnet50', 'unet']:
        raise TypeError(f"the specific model {args.net} is not supported, please select one of 'lenet5', 'vgg16', 'googlenet', 'resnet50', 'unet'")
    if args.dataset is None:
        raise TypeError(f"the specific type of dataset to train on should be provided, please select one of 'mnist', 'cifar10', 'cifar100', 'imagenet'")
    elif args.dataset not in ['mnist', 'cifar10', 'cifar100', 'imagenet']:
        raise TypeError(f"the specific dataset {args.dataset} is not supported, please select one of 'mnist', 'cifar10', 'cifar100', 'imagenet'")


if __name__ == '__main__':
    args = get_args()
    if args.random_seed is not None:
        random_seed = args.random_seed
    else:
        random_seed = int(time.time())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_logging(experiment_id=random_seed, net=args.net, dataset=args.dataset, action='train')

    # initialize random seed
    torch_set_random_seed(random_seed)
    logging.info(f'Start with random seed: {random_seed}')
    
    # process input arguments
    train_loader, _, test_loader, in_channels, num_class = get_dataloader(dataset=args.dataset, batch_size=args.b, num_workers=args.n, pin_memory=True)
    net = get_network(net=args.net, in_channels=in_channels, num_class=num_class).to(device)
    warm = args.warm

    # set experiment parameter
    best_acc = 0.0

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, settings.T_ORIGINAL_EPOCH - 10, eta_min=settings.T_LR_SCHEDULAR_MIN_LR,last_epoch=-1)
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)

    for epoch in range(1, settings.T_ORIGINAL_EPOCH + 1):
        train_loss = train_network(epoch)
        top1_acc, top5_acc, _ = eval_network(target_eval_loader=test_loader)
        logging.info(f'Epoch: {epoch}, Train Loss: {train_loss},Top1 Accuracy: {top1_acc}, Top5 Accuracy: {top5_acc}')
        if wandb_available:
            wandb.log({"epoch": epoch, "train_loss": train_loss,"top1_acc": top1_acc, "top5_acc": top5_acc})

        lr_scheduler.step()

        # start to save best performance model after first training milestone
        if best_acc < top1_acc:
            best_acc = top1_acc
            torch.save(net, f'models/{args.net}_{args.dataset}_{random_seed}_original.pth')
    
    if wandb_available:
        wandb.finish()
