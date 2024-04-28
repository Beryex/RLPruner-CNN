import os
import time
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging

from conf import settings
from utils import get_network, get_dataloader, setup_logging, WarmUpLR


def train(epoch: int):
    net.train()
    with tqdm(total=len(training_loader), desc=f'Epoch {epoch}/{settings.ORIGINAL_EPOCH}', unit='batch') as pbar:
        for _, (images, labels) in enumerate(training_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            if epoch <= warm:
                warmup_scheduler.step()

            pbar.update(1)
            pbar.set_postfix(**{'loss (batch)': loss.item()})

@torch.no_grad()
def eval_training(epoch: int):
    net.eval()
    correct_1 = 0.0
    correct_5 = 0.0
    for (images, labels) in tqdm(test_loader, total=len(test_loader), desc='Testing round', unit='batch', leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = net(images)
        _, preds = outputs.topk(5, 1, largest=True, sorted=True)
        correct_1 += (preds[:, :1] == labels.unsqueeze(1)).sum().item()
        top5_correct = labels.view(-1, 1).expand_as(preds) == preds
        correct_5 += top5_correct.any(dim=1).sum().item()
    
    top1_acc = correct_1 / len(test_loader.dataset)
    top5_acc = correct_5 / len(test_loader.dataset)
    logging.info(f'Epoch: {epoch}, Top1 Accuracy: {top1_acc}, Top5 Accuracy: {top5_acc}')
    return top1_acc, top5_acc


def get_args():
    parser = argparse.ArgumentParser(description='train given model under given dataset')
    parser.add_argument('-net', type=str, default=None, help='the type of model to train')
    parser.add_argument('-dataset', type=str, default=None, help='the dataset to train on')
    parser.add_argument('-lr', type=float, default=settings.INITIAL_LR, help='initial learning rate')
    parser.add_argument('-lr_decay', type=float, default=settings.LR_DECAY, help='learning rate decay rate')
    parser.add_argument('-b', type=int, default=settings.BATCH_SIZE, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=settings.WARM, help='warm up training phase')

    args = parser.parse_args()
    check_args(args)

    return args

def check_args(args: argparse.Namespace):
    if args.net is None:
        logging.error("the specific type of model should be provided, please select one of 'lenet5', 'vgg16', 'googlenet', 'resnet50', 'unet'")
        sys.exit(1)
    elif args.net not in ['lenet5', 'vgg16', 'googlenet', 'resnet50', 'unet']:
        logging.error('the specific model is not supported')
        sys.exit(1)
    if args.dataset is None:
        logging.error("the specific type of dataset to train on should be provided, please select one of 'mnist', 'cifar10', 'cifar100', 'imagenet'")
        sys.exit(1)
    elif args.dataset not in ['mnist', 'cifar10', 'cifar100', 'imagenet']:
        logging.error('the specific dataset is not supported')
        sys.exit(1)


if __name__ == '__main__':
    start_time = int(time.time())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    setup_logging(experiment_id=start_time, net=args.net, dataset=args.dataset, action='train')

    # initialize random seed
    torch.manual_seed(start_time)
    logging.info(f'Start with random seed: {start_time}')
    
    # process input arguments
    training_loader, _, test_loader, in_channels, num_class = get_dataloader(dataset=args.dataset, batch_size=args.b)
    net = get_network(net=args.net, in_channels=in_channels, num_class=num_class).to(device)
    lr = args.lr
    gamma = args.lr_decay
    warm = args.warm

    # set experiment parameter
    current_lr = lr
    best_acc = 0.0

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)

    for epoch in range(1, settings.ORIGINAL_EPOCH + 1):
        if epoch in settings.ORIGINAL_MILESTONES:
            current_lr *= gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        train(epoch)
        top1_acc, _ = eval_training(epoch)

        # start to save best performance model after first training milestone
        if epoch > settings.ORIGINAL_MILESTONES[1] and best_acc < top1_acc:
            best_acc = top1_acc

            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(net, f'models/{args.net}_{args.dataset}_Original_{start_time}.pkl')
    
    end_time = time.time()
    logging.info(f'Original training process takes {(end_time - start_time) / 60} minutes')
